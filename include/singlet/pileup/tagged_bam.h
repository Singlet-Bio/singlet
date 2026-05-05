#pragma once
// singlet-pileup: tagged_bam.h
// G-TAGGED-BAM — Streaming Cell Ranger-compatible tagged BAM writer.
//
// Writes a coordinate-sorted BAM during the pileup pass, adding CB/CR/CY
// (barcode), UB/UR/UY (UMI), GX/GN (gene), and RE (region type) tags.
// Compatible with downstream tools: cellSNP-lite, velocyto, scVelo, sinto,
// Signac, ArchR.
//
// Design: single-pass, single-threaded (pileup is streaming single-threaded).
// TaggedBamWriter wraps an htslib samFile*. The caller supplies each bam1_t*
// from the STAR pipe together with annotation strings; the writer adds tags and
// writes. BAM records must arrive in coordinate order — same order as STAR
// emits them.
//
// Tag map (Cell Ranger standard):
//   CB:Z  corrected cell barcode (with -1 suffix, e.g. "ACGTACGT-1")
//   CR:Z  raw (uncorrected) cell barcode
//   CY:Z  cell barcode base-quality scores (ASCII Phred+33)
//   UB:Z  corrected UMI
//   UR:Z  raw UMI
//   UY:Z  UMI base-quality scores
//   GX:Z  Ensembl gene ID(s), semicolon-separated if multi-gene
//   GN:Z  gene name(s), semicolon-separated
//   xf:i  feature type flags (1 = GEX, 2 = ADT)
//   RE:A  region type: 'E'=exonic, 'N'=intronic, 'I'=intergenic

#include <htslib/sam.h>
#include <htslib/hts.h>
#include <htslib/bgzf.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <unordered_set>

namespace singlet {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct TaggedBamConfig {
    std::string output_path;          // Output BAM file path
    bool add_barcode_tag = true;      // CB:Z, CR:Z (and CY:Z if quality non-empty)
    bool add_umi_tag     = true;      // UB:Z, UR:Z (and UY:Z if quality non-empty)
    bool add_gene_tag    = true;      // GX:Z, GN:Z, RE:A
    bool add_xf_tag      = true;      // xf:i feature type flags
    bool add_index       = true;      // Create .bai after close()
    int  threads         = 4;         // bgzf compression threads
    /// If non-null, only reads whose corrected barcode is in this set are written.
    /// Barcodes should be stored WITHOUT the "-1" suffix (comparison is done on
    /// the raw corrected barcode string before suffix is appended).
    const std::unordered_set<std::string>* called_barcodes = nullptr;
};

// ---------------------------------------------------------------------------
// TaggedBamWriter
// ---------------------------------------------------------------------------
class TaggedBamWriter {
   public:
    TaggedBamWriter() = default;
    ~TaggedBamWriter() { close(); }

    // Non-copyable (owns file handle)
    TaggedBamWriter(const TaggedBamWriter&)            = delete;
    TaggedBamWriter& operator=(const TaggedBamWriter&) = delete;

    /// Open output BAM and copy header from the source BAM header.
    /// The header is cloned so the caller may destroy the source after open().
    /// Returns false on failure.
    bool open(const TaggedBamConfig& cfg, const sam_hdr_t* hdr) {
        if (!hdr) return false;
        cfg_ = cfg;

        // Open output BAM for writing (bgzf-compressed)
        fp_ = sam_open(cfg_.output_path.c_str(), "wb");
        if (!fp_) return false;

        // Set bgzf compression threads
        if (cfg_.threads > 1) {
            hts_set_threads(fp_, cfg_.threads);
        }

        // Clone header and write it
        hdr_ = sam_hdr_dup(hdr);
        if (!hdr_) {
            sam_close(fp_);
            fp_ = nullptr;
            return false;
        }
        if (sam_hdr_write(fp_, hdr_) < 0) {
            sam_hdr_destroy(hdr_);
            sam_close(fp_);
            hdr_ = nullptr;
            fp_  = nullptr;
            return false;
        }
        return true;
    }

    /// Write a BAM record with Cell Ranger-style tags appended.
    ///
    /// @param b           BAM record (modified in-place; existing tags preserved)
    /// @param barcode     Corrected cell barcode (empty = uncorrected/unknown)
    /// @param raw_barcode Raw (uncorrected) cell barcode
    /// @param bc_qual     Cell barcode base-quality string (may be empty)
    /// @param umi         Corrected UMI (empty = uncorrected/unknown)
    /// @param raw_umi     Raw UMI
    /// @param umi_qual    UMI base-quality string (may be empty)
    /// @param gene_id     Ensembl gene ID (empty if intergenic)
    /// @param gene_name   Gene name / symbol (empty if intergenic)
    /// @param region_type 'E'=exonic, 'N'=intronic, 'I'=intergenic
    /// @param xf          Feature type flag (1=GEX, 2=ADT; 0 = skip xf tag)
    /// @return true on success, false if record was skipped (filter) or error
    bool write_record(bam1_t*            b,
                      const std::string& barcode,
                      const std::string& raw_barcode,
                      const std::string& bc_qual,
                      const std::string& umi,
                      const std::string& raw_umi,
                      const std::string& umi_qual,
                      const std::string& gene_id,
                      const std::string& gene_name,
                      char               region_type,
                      int32_t            xf = 1) {
        if (!fp_ || !b) return false;

        // Apply cell-calling filter (check WITHOUT the "-1" suffix)
        if (cfg_.called_barcodes && !barcode.empty()) {
            if (cfg_.called_barcodes->find(barcode) == cfg_.called_barcodes->end()) {
                ++skipped_;
                return false;
            }
        }

        // ── Barcode tags ───────────────────────────────────────────────────
        if (cfg_.add_barcode_tag) {
            if (!barcode.empty()) {
                // CB:Z — corrected barcode with -1 suffix (Cell Ranger standard)
                std::string cb = barcode + "-1";
                tag_set_str(b, "CB", cb);
            }
            if (!raw_barcode.empty()) {
                tag_set_str(b, "CR", raw_barcode);
            }
            if (!bc_qual.empty()) {
                tag_set_str(b, "CY", bc_qual);
            }
        }

        // ── UMI tags ──────────────────────────────────────────────────────
        if (cfg_.add_umi_tag) {
            if (!umi.empty()) {
                tag_set_str(b, "UB", umi);
            }
            if (!raw_umi.empty()) {
                tag_set_str(b, "UR", raw_umi);
            }
            if (!umi_qual.empty()) {
                tag_set_str(b, "UY", umi_qual);
            }
        }

        // ── Gene annotation tags ──────────────────────────────────────────
        if (cfg_.add_gene_tag) {
            if (!gene_id.empty()) {
                tag_set_str(b, "GX", gene_id);
            }
            if (!gene_name.empty()) {
                tag_set_str(b, "GN", gene_name);
            }
            // RE:A always written when gene tags enabled
            tag_set_char(b, "RE", region_type);
        }

        // ── xf feature type flag ─────────────────────────────────────────
        if (cfg_.add_xf_tag && xf != 0) {
            tag_set_int(b, "xf", xf);
        }

        // Write record
        if (sam_write1(fp_, hdr_, b) < 0) return false;
        ++written_;
        return true;
    }

    /// Convenience overload matching the task specification signature
    /// (no quality strings, no xf parameter).
    bool write_record(bam1_t*            b,
                      const std::string& barcode,
                      const std::string& raw_barcode,
                      const std::string& umi,
                      const std::string& raw_umi,
                      const std::string& gene_name,
                      const std::string& gene_id,
                      char               region_type) {
        return write_record(b, barcode, raw_barcode,
                            /*bc_qual=*/"", umi, raw_umi,
                            /*umi_qual=*/"", gene_id, gene_name,
                            region_type, /*xf=*/1);
    }

    /// Flush, close the BAM file, and optionally build a .bai index.
    void close() {
        if (!fp_) return;
        sam_close(fp_);
        fp_ = nullptr;
        if (hdr_) {
            sam_hdr_destroy(hdr_);
            hdr_ = nullptr;
        }
        if (cfg_.add_index && !cfg_.output_path.empty()) {
            // sam_index_build returns 0 on success; ignore failure gracefully
            [[maybe_unused]] int rc = sam_index_build(cfg_.output_path.c_str(), 0);
        }
    }

    uint64_t records_written() const { return written_; }
    uint64_t records_skipped() const { return skipped_; }

    bool is_open() const { return fp_ != nullptr; }

   private:
    samFile*      fp_      = nullptr;
    sam_hdr_t*    hdr_     = nullptr;
    TaggedBamConfig cfg_;
    uint64_t      written_ = 0;
    uint64_t      skipped_ = 0;

    // ── Low-level tag helpers ──────────────────────────────────────────────
    // Remove existing tag if present, then append new value.
    // htslib bam_aux_update_str / bam_aux_append operate on the aux block.

    static void tag_set_str(bam1_t* b, const char tag[2],
                             const std::string& val) {
        // Remove existing tag to avoid duplicates
        uint8_t* existing = bam_aux_get(b, tag);
        if (existing) bam_aux_del(b, existing);
        // Append new Z-type tag (length includes NUL terminator)
        bam_aux_append(b, tag, 'Z',
                       static_cast<int>(val.size() + 1),
                       reinterpret_cast<const uint8_t*>(val.c_str()));
    }

    static void tag_set_char(bam1_t* b, const char tag[2], char val) {
        uint8_t* existing = bam_aux_get(b, tag);
        if (existing) bam_aux_del(b, existing);
        bam_aux_append(b, tag, 'A', 1,
                       reinterpret_cast<const uint8_t*>(&val));
    }

    static void tag_set_int(bam1_t* b, const char tag[2], int32_t val) {
        uint8_t* existing = bam_aux_get(b, tag);
        if (existing) bam_aux_del(b, existing);
        bam_aux_append(b, tag, 'i', 4,
                       reinterpret_cast<const uint8_t*>(&val));
    }
};

}  // namespace singlet
