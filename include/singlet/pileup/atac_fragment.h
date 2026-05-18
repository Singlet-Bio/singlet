// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: atac_fragment.h
// A1 — Fragment extraction from paired-end BAM for 10x ATAC / ATAC-seq.
//
// Reads a coordinate-sorted BAM produced by STAR in PE-DNA mode and emits
// Tn5-shifted, deduplicated fragments as (chrom_idx, start, end, barcode_idx).
//
// Barcode origin: QNAME prefix "BC:NNNN_originalName" injected during .1fq
// decode. Reads with no or all-N barcode are silently skipped.
//
// Deduplication: per-chromosome unordered_set on (start, end, barcode_idx).
// The BAM is coordinate-sorted so the caller can flush each chromosome's set
// after the last record for that chromosome to bound peak memory.

#include <htslib/sam.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <string>
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
        // Process only read1 — read2 carries the same fragment info
        if (!(flag & BAM_FREAD1)) return;

        // MAPQ filter — exclude ambiguous multi-mappers (standard: MAPQ≥30,
        // matching CellRanger-ATAC, ArchR, sinto conventions).
        if (b->core.qual < min_mapq_) {
            ++stats_.low_mapq;
            return;
        }

        ++stats_.proper_pairs;

        // Fragment coordinates from TLEN (insert size, signed)
        const int32_t tlen = b->core.isize;
        if (tlen == 0 || std::abs(tlen) < min_frag_ || std::abs(tlen) > max_frag_) return;

        int32_t frag_start = b->core.pos;
        int32_t frag_end = frag_start + std::abs(tlen);

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

        // Per-chromosome deduplication
        FragmentKey key{tid, frag_start, frag_end, bc_idx};
        auto [it, inserted] = dedup_set_.emplace(key);
        if (!inserted) {
            ++stats_.duplicate_fragments;
            return;
        }

        ++stats_.unique_fragments;
        fragments_.push_back(ATACFragment{tid, frag_start, frag_end, bc_idx});
    }

    // ------------------------------------------------------------------
    // Finalise — sorts fragments by (chrom_idx, start) for downstream use.
    // Call once after all records have been processed.
    // ------------------------------------------------------------------
    void finalise() {
        std::sort(fragments_.begin(), fragments_.end(),
                  [](const ATACFragment& a, const ATACFragment& b) {
                      if (a.chrom_idx != b.chrom_idx) return a.chrom_idx < b.chrom_idx;
                      return a.start < b.start;
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

        if (!barcode_map_) return false;
        auto it = barcode_map_->find(bc);
        if (it == barcode_map_->end()) return false;

        bc_idx_out = it->second;
        return true;
    }

    // Pointer to caller-owned barcode map (not owned here)
    const std::unordered_map<std::string, uint32_t>* barcode_map_ = nullptr;

    int min_frag_ = 10;
    int max_frag_ = 2000;
    uint8_t min_mapq_ = 30;

    std::unordered_set<FragmentKey, FragmentKeyHash> dedup_set_;
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
