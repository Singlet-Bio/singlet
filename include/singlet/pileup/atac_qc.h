#pragma once
// singlet-pileup: atac_qc.h
// A3 — Per-cell QC metrics for ATAC-seq.
//
// Computes from ATACFragment records (A1):
//   - total_fragments, mito_fragments, mito_fraction
//   - tss_fragments, tss_enrichment (fraction of fragments near any TSS ±1000bp)
//   - median_fragment_size (per cell)
//   - frip_approx (fraction of fragments in "signal" bins, threshold configurable)
//
// TSS positions loaded from GTF (gene feature, strand-aware):
//   '+' strand → TSS = gene start
//   '-' strand → TSS = gene end
//
// All per-cell vectors are indexed by barcode_idx (same convention as
// ATACFragmentExtractor / PileupEngine).

#include <zlib.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "singlet-pileup/atac_bin_counter.h"
#include "singlet-pileup/atac_fragment.h"

namespace singlet {

// ─────────────────────────────────────────────────────────────────────────────
// Per-cell QC record
// ─────────────────────────────────────────────────────────────────────────────
struct ATACCellQC {
    uint32_t total_fragments = 0;
    uint32_t mito_fragments = 0;
    float mito_fraction = 0.0f;
    uint32_t tss_fragments = 0;
    float tss_enrichment = 0.0f;  // tss_fragments / total_fragments
    float median_fragment_size = 0.0f;
    float frip_approx = 0.0f;
};

// ─────────────────────────────────────────────────────────────────────────────
// ATACQCComputer
// ─────────────────────────────────────────────────────────────────────────────
class ATACQCComputer {
   public:
    // ── Configuration ────────────────────────────────────────────────────────

    /// Set mitochondrial chromosome names (idx-based, must match chrom_map).
    /// Defaults applied in compute() if not called explicitly.
    void set_mito_chroms(const std::vector<std::string>& names = {"chrM", "MT", "chrMT"}) {
        mito_chrom_names_ = names;
    }

    /// Set chromosome name → index mapping (from BAM header chrom list).
    void set_chrom_map(const std::unordered_map<std::string, int32_t>& cmap) {
        chrom_map_ = cmap;
    }

    /// Set TSS positions directly as chrom_idx → sorted vector of TSS pos.
    void set_tss_positions(const std::unordered_map<int32_t, std::vector<int32_t>>& tss) {
        tss_positions_ = tss;
        for (auto& [idx, vec] : tss_positions_) {
            std::sort(vec.begin(), vec.end());
        }
    }

    /// Load TSS positions from GTF file.  Extracts gene features only.
    /// '+' strand: TSS = start (0-based).
    /// '-' strand: TSS = end (exclusive, 0-based).
    bool load_tss_from_gtf(const std::string& gtf_path) {
        bool is_gz = gtf_path.size() >= 3 &&
                     gtf_path.substr(gtf_path.size() - 3) == ".gz";

        gzFile gz = nullptr;
        std::ifstream plain;

        if (is_gz) {
            gz = gzopen(gtf_path.c_str(), "r");
            if (!gz) {
                std::cerr << "[atac_qc] Cannot open: " << gtf_path << "\n";
                return false;
            }
            gzbuffer(gz, 256 * 1024);
        } else {
            plain.open(gtf_path);
            if (!plain.is_open()) {
                std::cerr << "[atac_qc] Cannot open: " << gtf_path << "\n";
                return false;
            }
        }

        auto read_line = [&](std::string& line) -> bool {
            if (gz) {
                char buf[8192];
                line.clear();
                while (gzgets(gz, buf, sizeof(buf))) {
                    line += buf;
                    if (!line.empty() && line.back() == '\n') {
                        line.pop_back();
                        if (!line.empty() && line.back() == '\r') line.pop_back();
                        return true;
                    }
                }
                return !line.empty();
            } else {
                return static_cast<bool>(std::getline(plain, line));
            }
        };

        std::unordered_map<int32_t, std::vector<int32_t>> tss_raw;
        size_t loaded = 0;

        std::string line;
        while (read_line(line)) {
            if (line.empty() || line[0] == '#') continue;

            // Tab-split: fields[0]=chrom [1]=src [2]=feat [3]=start [4]=end
            //            [5]=score [6]=strand [7]=frame [8]=attrs
            const char* p = line.c_str();
            const char* fields[9];
            int fi = 0;
            fields[0] = p;
            while (*p && fi < 8) {
                if (*p == '\t') {
                    fi++;
                    fields[fi] = p + 1;
                }
                p++;
            }
            if (fi < 8) continue;

            // Feature must be "gene"
            const char* feat_s = fields[2];
            const char* feat_e = fields[3] - 1;
            if ((feat_e - feat_s) != 4 || strncmp(feat_s, "gene", 4) != 0) continue;

            std::string chrom(fields[0], fields[1] - 1);
            int32_t start = atoi(fields[3]) - 1;  // GTF 1-based → 0-based
            int32_t end = atoi(fields[4]);        // GTF inclusive → exclusive
            char strand = *fields[6];

            auto it = chrom_map_.find(chrom);
            if (it == chrom_map_.end()) continue;

            int32_t chrom_idx = it->second;
            int32_t tss_pos = (strand == '-') ? end : start;
            tss_raw[chrom_idx].push_back(tss_pos);
            ++loaded;
        }

        if (gz) gzclose(gz);

        // Sort each chromosome's TSS list
        for (auto& [idx, vec] : tss_raw) {
            std::sort(vec.begin(), vec.end());
        }
        tss_positions_ = std::move(tss_raw);

        std::cerr << "[atac_qc] Loaded " << loaded << " TSS positions\n";
        return loaded > 0;
    }

    /// TSS window half-width in bp (default ±1000 bp).
    void set_tss_window(int32_t half_width = 1000) {
        tss_window_ = half_width;
    }

    /// FRIP approximation: bins with ≥ frip_min_count are "signal" bins.
    void set_frip_min_count(uint32_t n = 3) {
        frip_min_count_ = n;
    }

    // ── Main computation ─────────────────────────────────────────────────────

    /// Compute all QC metrics.
    ///
    /// @param unique_fragments  Sorted unique fragments from ATACFragmentExtractor.
    /// @param extractor         Used for nothing currently (stats are global);
    ///                          kept for API symmetry with the task spec.
    /// @param num_cells         Number of cells (max barcode_idx + 1).
    /// @param bin_counter       Optional: used to compute FRIP approximation.
    ///                          Pass nullptr to skip FRIP.
    void compute(const std::vector<ATACFragment>& unique_fragments,
                 const ATACFragmentExtractor& /*extractor*/,
                 uint32_t num_cells,
                 const ATACBinCounter* bin_counter = nullptr) {
        if (mito_chrom_names_.empty()) {
            mito_chrom_names_ = {"chrM", "MT", "chrMT"};
        }

        // Build set of mito chrom indices
        std::unordered_map<int32_t, bool> mito_idx;
        for (const auto& name : mito_chrom_names_) {
            auto it = chrom_map_.find(name);
            if (it != chrom_map_.end()) {
                mito_idx[it->second] = true;
            }
        }

        // Per-cell fragment size accumulators (for median)
        std::vector<std::vector<int32_t>> cell_sizes(num_cells);

        // Init QC storage
        cell_qc_.assign(num_cells, ATACCellQC{});

        // Process all fragments
        for (const auto& frag : unique_fragments) {
            if (frag.barcode_idx >= num_cells) continue;

            ATACCellQC& qc = cell_qc_[frag.barcode_idx];
            int32_t frag_size = frag.end - frag.start;

            qc.total_fragments++;

            // Mito
            if (mito_idx.count(frag.chrom_idx)) {
                qc.mito_fragments++;
            }

            // TSS proximity
            if (is_near_tss(frag.chrom_idx, frag.start, frag.end)) {
                qc.tss_fragments++;
            }

            // Fragment size for median
            if (frag.barcode_idx < num_cells) {
                cell_sizes[frag.barcode_idx].push_back(frag_size);
            }

            // Global histogram
            int bin = frag_size / hist_bin_width_;
            if (bin >= static_cast<int>(global_hist_.size())) {
                global_hist_.resize(static_cast<size_t>(bin + 1), 0ULL);
            }
            global_hist_[static_cast<size_t>(bin)]++;
        }

        // Finalise per-cell derived metrics
        for (uint32_t i = 0; i < num_cells; i++) {
            ATACCellQC& qc = cell_qc_[i];
            if (qc.total_fragments > 0) {
                qc.mito_fraction = static_cast<float>(qc.mito_fragments) /
                                   static_cast<float>(qc.total_fragments);
                qc.tss_enrichment = static_cast<float>(qc.tss_fragments) /
                                    static_cast<float>(qc.total_fragments);
            }
            // Median fragment size
            auto& sv = cell_sizes[i];
            if (!sv.empty()) {
                std::sort(sv.begin(), sv.end());
                size_t mid = sv.size() / 2;
                if (sv.size() % 2 == 0) {
                    qc.median_fragment_size =
                        (static_cast<float>(sv[mid - 1]) + static_cast<float>(sv[mid])) / 2.0f;
                } else {
                    qc.median_fragment_size = static_cast<float>(sv[mid]);
                }
            }
        }

        // FRIP approximation using bin matrix
        if (bin_counter) {
            compute_frip(unique_fragments, *bin_counter, num_cells);
        }
    }

    // ── Accessors ──────────────────────────────────────────────────────────

    const std::vector<ATACCellQC>& cell_qc() const { return cell_qc_; }

    /// Return global fragment size histogram.
    /// Each bucket i covers [i*bin_width, (i+1)*bin_width) bp.
    std::vector<uint64_t> fragment_size_histogram(int bin_width = 10) const {
        if (bin_width == hist_bin_width_) return global_hist_;
        // Re-bin on the fly
        std::vector<uint64_t> result;
        for (size_t i = 0; i < global_hist_.size(); i++) {
            int orig_lo = static_cast<int>(i) * hist_bin_width_;
            int new_bin = orig_lo / bin_width;
            if (new_bin >= static_cast<int>(result.size())) {
                result.resize(static_cast<size_t>(new_bin + 1), 0ULL);
            }
            result[static_cast<size_t>(new_bin)] += global_hist_[i];
        }
        return result;
    }

    /// Set internal histogram resolution (before compute()).
    void set_hist_bin_width(int w = 10) { hist_bin_width_ = w; }

    // ── Output ─────────────────────────────────────────────────────────────

    /// Write QC TSV: barcode \t total \t mito \t mito_frac \t tss \t tss_enr
    ///               \t med_frag \t frip_approx
    void write_qc_tsv(const std::string& path,
                      const std::vector<std::string>& barcodes) const {
        std::ofstream f(path);
        if (!f.is_open()) {
            std::cerr << "[atac_qc] Cannot write: " << path << "\n";
            return;
        }
        f << "barcode\ttotal_fragments\tmito_fragments\tmito_fraction"
             "\ttss_fragments\ttss_enrichment\tmedian_fragment_size\tfrip_approx\n";

        for (size_t i = 0; i < cell_qc_.size(); i++) {
            const ATACCellQC& q = cell_qc_[i];
            const std::string& bc = (i < barcodes.size()) ? barcodes[i] : std::to_string(i);
            f << bc << '\t'
              << q.total_fragments << '\t'
              << q.mito_fragments << '\t'
              << q.mito_fraction << '\t'
              << q.tss_fragments << '\t'
              << q.tss_enrichment << '\t'
              << q.median_fragment_size << '\t'
              << q.frip_approx << '\n';
        }
    }

   private:
    // ── Helpers ────────────────────────────────────────────────────────────

    /// True if [frag_start, frag_end) midpoint is within tss_window_ of any TSS.
    bool is_near_tss(int32_t chrom_idx, int32_t frag_start, int32_t frag_end) const {
        auto it = tss_positions_.find(chrom_idx);
        if (it == tss_positions_.end()) return false;

        const auto& tss_vec = it->second;
        if (tss_vec.empty()) return false;

        int32_t mid = (frag_start + frag_end) / 2;
        int32_t lo = mid - tss_window_;
        int32_t hi = mid + tss_window_;

        // Binary search for first TSS >= lo
        auto pos = std::lower_bound(tss_vec.begin(), tss_vec.end(), lo);
        if (pos != tss_vec.end() && *pos <= hi) return true;
        return false;
    }

    /// Compute FRIP approximation: for each fragment, check if the bin
    /// containing its start position has a total count >= frip_min_count_.
    void compute_frip(const std::vector<ATACFragment>& fragments,
                      const ATACBinCounter& bc,
                      uint32_t num_cells) {
        // Build total count per bin from CSC matrix
        // CSCMatrix fields: indptr (col ptrs), indices (row/bin idx), data (values),
        //                   nrows (n_bins), ncols (n_cells)
        auto csc = bc.to_csc(num_cells);
        size_t n_bins = csc.nrows;

        std::vector<uint32_t> bin_total(n_bins, 0);
        for (size_t j = 0; j < csc.data.size(); j++) {
            auto bin_idx = static_cast<size_t>(csc.indices[j]);
            if (bin_idx < n_bins) bin_total[bin_idx] += csc.data[j];
        }

        // Mark signal bins
        std::vector<bool> is_signal(n_bins, false);
        for (size_t b = 0; b < n_bins; b++) {
            if (bin_total[b] >= frip_min_count_) is_signal[b] = true;
        }

        // Per-cell signal fragment counts
        std::vector<uint32_t> cell_signal(num_cells, 0);
        for (const auto& frag : fragments) {
            if (frag.barcode_idx >= num_cells) continue;
            size_t global_bin = bc.fragment_global_bin(frag);
            if (global_bin < n_bins && is_signal[global_bin]) {
                cell_signal[frag.barcode_idx]++;
            }
        }

        for (uint32_t i = 0; i < num_cells; i++) {
            if (cell_qc_[i].total_fragments > 0) {
                cell_qc_[i].frip_approx =
                    static_cast<float>(cell_signal[i]) /
                    static_cast<float>(cell_qc_[i].total_fragments);
            }
        }
    }

    // ── State ──────────────────────────────────────────────────────────────
    std::vector<std::string> mito_chrom_names_;
    std::unordered_map<std::string, int32_t> chrom_map_;
    std::unordered_map<int32_t, std::vector<int32_t>> tss_positions_;

    int32_t tss_window_ = 1000;
    uint32_t frip_min_count_ = 3;
    int hist_bin_width_ = 10;

    std::vector<ATACCellQC> cell_qc_;
    std::vector<uint64_t> global_hist_;
};

}  // namespace singlet
