// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_fragment.h
// A1 — Fragment extraction from paired-end BAM for 10x ATAC / ATAC-seq.
//
// Reads a coordinate-sorted BAM produced by STAR in PE-DNA mode and emits Tn5-shifted, deduplicated fragments as
// (chrom_idx, start, end, barcode_idx, read_pairs), the content of Cell Ranger ATAC's fragments.tsv.gz.
//
// One record per pair is used: the leftmost mate (TLEN > 0). The fragment is [pos, pos + TLEN) on the reference
// (0-based, end exclusive), shifted +4 / -5 for the Tn5 9 bp duplication (Buenrostro 2013; Cell Ranger ATAC).
// Both mates of a STAR pair carry the same MAPQ, so one MAPQ check covers the pair.
//
// Barcode origin: QNAME prefix "BC:<raw>_originalName" injected during .1fq decode. With an AtacBarcodeCorrector the raw
// barcode is corrected against the whitelist (atac_barcodes.h) and barcode_idx is the whitelist index; without one,
// the legacy exact-match barcode map is used. Reads with no or all-N barcode are skipped.
//
// Deduplication: identical (start, end, barcode) pairs are one fragment whose read_pairs counts the duplicates (column 5
// of fragments.tsv.gz). Leftmost mates arrive sorted by start in a coordinate-sorted BAM, so the dedup table only holds
// the current start position.

#include <htslib/sam.h>

#include "atac_barcodes.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Public fragment record
// ---------------------------------------------------------------------------
struct ATACFragment {
    int32_t chrom_idx;
    int32_t start;  // Tn5-shifted, 0-based inclusive
    int32_t end;    // Tn5-shifted, 0-based exclusive
    uint32_t barcode_idx;
    uint32_t read_pairs = 1;  // read pairs supporting the fragment (duplicates + 1)
};

// ---------------------------------------------------------------------------
// Deduplication key (16 bytes, fits in one cache line slot)
// ---------------------------------------------------------------------------
struct FragmentKey {
    int32_t chrom_idx;
    int32_t start;
    int32_t end;
    uint32_t barcode_idx;

    bool operator==(const FragmentKey& o) const noexcept {
        return chrom_idx == o.chrom_idx &&
               start == o.start &&
               end == o.end &&
               barcode_idx == o.barcode_idx;
    }
};

struct FragmentKeyHash {
    size_t operator()(const FragmentKey& k) const noexcept {
        // FNV-1a mixing over the 16-byte struct
        uint64_t a = (static_cast<uint64_t>(static_cast<uint32_t>(k.chrom_idx)) << 32) |
                     static_cast<uint32_t>(k.start);
        uint64_t b = (static_cast<uint64_t>(static_cast<uint32_t>(k.end)) << 32) |
                     static_cast<uint64_t>(k.barcode_idx);
        // Murmur-inspired finalizer
        a ^= a >> 33;
        a *= 0xff51afd7ed558ccdULL;
        a ^= a >> 33;
        b ^= b >> 33;
        b *= 0xc4ceb9fe1a85ec53ULL;
        b ^= b >> 33;
        return static_cast<size_t>(a ^ b);
    }
};

// ---------------------------------------------------------------------------
// ATACFragmentExtractor
// ---------------------------------------------------------------------------
class ATACFragmentExtractor {
   public:
    // ------------------------------------------------------------------
    // Configuration — must be called before process_record()
    // ------------------------------------------------------------------

    /// Map from barcode string → integer index (same convention as PileupEngine).
    void set_barcode_map(const std::unordered_map<std::string, uint32_t>& bc_map) {
        barcode_map_ = &bc_map;
    }

    /// Whitelist correction (takes precedence over the barcode map). Caller owns the corrector.
    void set_barcode_corrector(AtacBarcodeCorrector* c) { corrector_ = c; }

    void set_min_fragment_size(int min_size = 10) { min_frag_ = min_size; }
    void set_max_fragment_size(int max_size = 2000) { max_frag_ = max_size; }
    void set_min_mapq(uint8_t min_mapq = 30) { min_mapq_ = min_mapq; }

    // ------------------------------------------------------------------
    // Core processing loop — call once per BAM record
    // ------------------------------------------------------------------
    void process_record(const bam1_t* b, const bam_hdr_t* /*hdr*/) {
        ++stats_.total_reads;

        const uint16_t flag = b->core.flag;

        // Must be a proper pair
        if (!(flag & BAM_FPROPER_PAIR)) return;
        // Skip unmapped / mate unmapped
        if (flag & BAM_FUNMAP) return;
        if (flag & BAM_FMUNMAP) return;
        // Skip secondary / supplementary
        if (flag & BAM_FSECONDARY) return;
        if (flag & BAM_FSUPPLEMENTARY) return;
        // MAPQ filter — exclude ambiguous multi-mappers (standard: MAPQ>=30, matching Cell Ranger ATAC, ArchR, sinto).
        // STAR gives 255 (unique), 3, 1, 0.
        if (b->core.qual < min_mapq_) {
            if (b->core.isize > 0) ++stats_.low_mapq;
            return;
        }

        // One record per pair: the leftmost mate carries TLEN > 0.
        const int32_t tlen = b->core.isize;
        if (tlen <= 0) return;
        if (b->core.mtid != b->core.tid) return;

        ++stats_.proper_pairs;

        if (tlen < min_frag_ || tlen > max_frag_) return;

        int32_t frag_start = b->core.pos;
        int32_t frag_end = frag_start + tlen;

        // Tn5 shift (standard Buenrostro / 10x ATAC convention)
        frag_start += 4;
        frag_end -= 5;

        if (frag_start >= frag_end) return;  // degenerate after shift

        // Extract barcode from QNAME prefix "BC:XXXX_"
        const char* qname = bam_get_qname(b);
        uint32_t bc_idx = 0;
        if (!parse_barcode(qname, bc_idx)) {
            ++stats_.no_barcode;
            return;
        }

        const int32_t tid = b->core.tid;

        // Deduplication: keys only collide at the same start, and starts arrive sorted.
        if (tid != cur_tid_ || frag_start != cur_start_) {
            dedup_.clear();
            cur_tid_ = tid;
            cur_start_ = frag_start;
        }
        FragmentKey key{tid, frag_start, frag_end, bc_idx};
        auto [it, inserted] = dedup_.emplace(key, fragments_.size());
        if (!inserted) {
            ++stats_.duplicate_fragments;
            ++fragments_[it->second].read_pairs;
            return;
        }

        ++stats_.unique_fragments;
        fragments_.push_back(ATACFragment{tid, frag_start, frag_end, bc_idx, 1});
    }

    /// Raw barcode of a record ("BC:<raw>_..." QNAME prefix), or empty. Used for the corrector's counting passes.
    static std::string_view raw_barcode(const bam1_t* b) {
        const char* q = bam_get_qname(b);
        if (q[0] != 'B' || q[1] != 'C' || q[2] != ':') return {};
        const char* s = q + 3;
        const char* u = std::strchr(s, '_');
        if (!u) return {};
        return std::string_view(s, static_cast<size_t>(u - s));
    }

    /// True for the record of a pair that process_record() would use (before barcode and MAPQ checks).
    static bool is_fragment_record(const bam1_t* b) {
        const uint16_t f = b->core.flag;
        if (!(f & BAM_FPROPER_PAIR) || (f & (BAM_FUNMAP | BAM_FMUNMAP | BAM_FSECONDARY | BAM_FSUPPLEMENTARY)))
            return false;
        return b->core.isize > 0;
    }

    // ------------------------------------------------------------------
    // Finalise — sorts fragments by (chrom_idx, start) for downstream use.
    // Call once after all records have been processed.
    // ------------------------------------------------------------------
    void finalise() {
        std::sort(fragments_.begin(), fragments_.end(),
                  [](const ATACFragment& a, const ATACFragment& b) {
                      if (a.chrom_idx != b.chrom_idx) return a.chrom_idx < b.chrom_idx;
                      if (a.start != b.start) return a.start < b.start;
                      if (a.end != b.end) return a.end < b.end;
                      return a.barcode_idx < b.barcode_idx;
                  });
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------
    const std::vector<ATACFragment>& fragments() const { return fragments_; }
    size_t total_reads() const { return stats_.total_reads; }
    size_t proper_pairs() const { return stats_.proper_pairs; }
    size_t low_mapq_reads() const { return stats_.low_mapq; }
    size_t duplicate_fragments() const { return stats_.duplicate_fragments; }
    size_t unique_fragments() const { return stats_.unique_fragments; }
    size_t no_barcode_reads() const { return stats_.no_barcode; }

   private:
    // ------------------------------------------------------------------
    // Internals
    // ------------------------------------------------------------------

    /// Parse "BC:XXXX_..." prefix from QNAME.  Returns false if absent or
    /// all-N barcode (treated as missing).  On success writes the
    /// barcode index into bc_idx_out.
    bool parse_barcode(const char* qname, uint32_t& bc_idx_out) const {
        // Expect prefix "BC:"
        if (qname[0] != 'B' || qname[1] != 'C' || qname[2] != ':') return false;

        const char* bc_start = qname + 3;
        const char* underscore = std::strchr(bc_start, '_');
        if (!underscore) return false;

        // Build barcode string and check for all-N
        std::string bc(bc_start, underscore);
        if (bc.empty()) return false;

        bool all_n = true;
        for (char c : bc) {
            if (c != 'N' && c != 'n') {
                all_n = false;
                break;
            }
        }
        if (all_n) return false;

        if (corrector_) {
            const int64_t idx = corrector_->resolve(bc);
            if (idx < 0) return false;
            bc_idx_out = static_cast<uint32_t>(idx);
            return true;
        }
        if (!barcode_map_) return false;
        auto it = barcode_map_->find(bc);
        if (it == barcode_map_->end()) return false;

        bc_idx_out = it->second;
        return true;
    }

    // Pointer to caller-owned barcode map (not owned here)
    const std::unordered_map<std::string, uint32_t>* barcode_map_ = nullptr;
    AtacBarcodeCorrector* corrector_ = nullptr;
    std::unordered_map<FragmentKey, size_t, FragmentKeyHash> dedup_;
    int32_t cur_tid_ = -1, cur_start_ = -1;

    int min_frag_ = 10;
    int max_frag_ = 2000;
    uint8_t min_mapq_ = 30;

    std::vector<ATACFragment> fragments_;

    struct Stats {
        size_t total_reads = 0;
        size_t proper_pairs = 0;
        size_t low_mapq = 0;
        size_t no_barcode = 0;
        size_t duplicate_fragments = 0;
        size_t unique_fragments = 0;
    } stats_;
};

}  // namespace singlet
