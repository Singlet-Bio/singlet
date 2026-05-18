// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: fragments_writer.h
// G-FRAGMENTS — Standard fragments.tsv.gz writer (10x Cell Ranger ATAC / ArchR / Signac compatible).
//
// Format: tab-separated, coordinate-sorted, bgzf-compressed, tabix-indexed.
// Columns: chrom  start  end  barcode  duplicate_count
//   - start/end are 0-based half-open, matching BED convention
//   - barcode has "-1" suffix by default (Cell Ranger convention)
//   - file must be sorted by (chrom, start) before writing for tabix to work
//
// Usage:
//   FragmentsWriter w;
//   w.open({"out.tsv.gz"});
//   w.write_fragments(sorted_frags);  // vector must be coordinate-sorted
//   w.close();  // closes BGZF + calls tbx_index_build

#include <htslib/bgzf.h>
#include <htslib/tbx.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace singlet {

// ---------------------------------------------------------------------------
// Public fragment record (string-based, for TSV output)
// ---------------------------------------------------------------------------
struct Fragment {
    std::string chrom;
    int64_t     start;   // 0-based inclusive
    int64_t     end;     // 0-based exclusive
    std::string barcode;
    int32_t     count;   // duplicate count
};

// ---------------------------------------------------------------------------
// Writer configuration
// ---------------------------------------------------------------------------
struct FragmentsWriteConfig {
    std::string filepath;               // Output path (e.g. "frags.tsv.gz"); written as-is
    bool        compress             = true;   // Use bgzf compression
    bool        index                = true;   // Build tabix index after close
    bool        append_barcode_suffix = true;  // Append "-1" to each barcode
};

// ---------------------------------------------------------------------------
// FragmentsWriter
// ---------------------------------------------------------------------------
class FragmentsWriter {
   public:
    FragmentsWriter() = default;
    ~FragmentsWriter() { close(); }

    // Disable copy; allow move.
    FragmentsWriter(const FragmentsWriter&)            = delete;
    FragmentsWriter& operator=(const FragmentsWriter&) = delete;
    FragmentsWriter(FragmentsWriter&& o) noexcept
        : fp_(o.fp_), cfg_(std::move(o.cfg_)), written_(o.written_) {
        o.fp_      = nullptr;
        o.written_ = 0;
    }

    /// Open the output file for writing.
    /// @return true on success.
    bool open(const FragmentsWriteConfig& cfg) {
        if (fp_) return false;
        cfg_     = cfg;
        written_ = 0;
        // bgzf_open with mode "wg" writes bgzf; plain "w" writes uncompressed gz blocks
        // "w" in htslib bgzf means BGZF-compressed (not raw gzip)
        const char* mode = cfg_.compress ? "w" : "wu";  // "wu" = uncompressed bgzf
        fp_ = bgzf_open(cfg_.filepath.c_str(), mode);
        return fp_ != nullptr;
    }

    /// Write a single fragment.
    bool write_fragment(const Fragment& f) {
        if (!fp_) return false;
        const std::string& bc = cfg_.append_barcode_suffix
                                    ? f.barcode + "-1"
                                    : f.barcode;
        // Build line in a local buffer to avoid repeated small writes
        char buf[4096];
        int  n = std::snprintf(buf, sizeof(buf), "%s\t%lld\t%lld\t%s\t%d\n",
                               f.chrom.c_str(),
                               static_cast<long long>(f.start),
                               static_cast<long long>(f.end),
                               bc.c_str(),
                               f.count);
        if (n <= 0 || static_cast<size_t>(n) >= sizeof(buf)) return false;
        ssize_t written = bgzf_write(fp_, buf, static_cast<size_t>(n));
        if (written != n) return false;
        ++written_;
        return true;
    }

    /// Write a pre-sorted vector of fragments.
    /// The vector MUST be coordinate-sorted (chrom:start) — required for tabix.
    bool write_fragments(const std::vector<Fragment>& frags) {
        for (const auto& f : frags) {
            if (!write_fragment(f)) return false;
        }
        return true;
    }

    /// Close the BGZF file and build the tabix index (if configured).
    /// Safe to call multiple times; second call is a no-op.
    void close() {
        if (!fp_) return;
        bgzf_close(fp_);
        fp_ = nullptr;
        if (cfg_.index && cfg_.compress && !cfg_.filepath.empty()) {
            // tbx_conf_bed: sc=1, bc=2, ec=3 (1-based column numbers)
            // Fragments format col1=chrom col2=start(0-based) col3=end — matches BED
            tbx_index_build(cfg_.filepath.c_str(), 0, &tbx_conf_bed);
        }
    }

    /// Number of fragments successfully written.
    uint64_t fragments_written() const { return written_; }

   private:
    BGZF*               fp_      = nullptr;
    FragmentsWriteConfig cfg_    = {};
    uint64_t            written_ = 0;
};

// ---------------------------------------------------------------------------
// Sort fragments by (chrom, start, end) for correct tabix ordering
// ---------------------------------------------------------------------------
inline void sort_fragments(std::vector<Fragment>& frags) {
    std::sort(frags.begin(), frags.end(),
              [](const Fragment& a, const Fragment& b) {
                  if (a.chrom != b.chrom) return a.chrom < b.chrom;
                  if (a.start != b.start) return a.start < b.start;
                  return a.end < b.end;
              });
}

}  // namespace singlet
