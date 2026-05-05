#pragma once
// singlet-pileup: atac_bin_counter.h
// A2 — Fixed-width bin matrix generator for ATAC-seq.
//
// Tiles the genome into fixed-width bins (default 500 bp) and counts
// ATACFragment records from A1 (atac_fragment.h) into a sparse
// bins × cells matrix using SparseAccumulator<uint16_t>.
//
// Bin layout:
//   global_bin_index = global_bin_offset[chrom_idx] + position / bin_width
//   A fragment [start, end) overlaps bins [start/W, (end-1)/W] inclusive.
//
// Output:
//   bin_features() returns "chr1:0-500\tchr1:0-500\tPeaks" labels.
//   to_csc(num_cells) returns a SparseAccumulator::CSCMatrix (bins × cells).

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "singlet-pileup/atac_fragment.h"
#include "singlet-pileup/sparse_accumulator.h"

namespace singlet {

class ATACBinCounter {
   public:
    // ----------------------------------------------------------------
    // Configuration
    // ----------------------------------------------------------------

    /// Set bin width (default 500 bp). Must be called before set_chromosome_sizes.
    void set_bin_width(int width = 500) {
        if (width <= 0) throw std::invalid_argument("bin_width must be positive");
        bin_width_ = width;
    }

    /// Provide chromosome sizes as (name, length) pairs.
    /// Call this before count_fragments(); order must match chrom_idx in ATACFragment.
    void set_chromosome_sizes(const std::vector<std::pair<std::string, int32_t>>& chrom_sizes) {
        chrom_names_.clear();
        chrom_sizes_.clear();
        global_bin_offset_.clear();

        int32_t offset = 0;
        for (const auto& [name, size] : chrom_sizes) {
            chrom_names_.push_back(name);
            chrom_sizes_.push_back(size);
            global_bin_offset_.push_back(offset);
            int32_t bins_for_chrom = (size + bin_width_ - 1) / bin_width_;
            offset += bins_for_chrom;
        }
        total_bins_ = static_cast<size_t>(offset);

        acc_.set_n_features(static_cast<uint32_t>(total_bins_));
    }

    // ----------------------------------------------------------------
    // Fragment counting
    // ----------------------------------------------------------------

    /// Accumulate a batch of ATACFragment records.
    /// barcode_idx values must be < num_cells passed to to_csc().
    void count_fragments(const std::vector<ATACFragment>& fragments) {
        for (const auto& frag : fragments) {
            count_one(frag);
        }
    }

    /// Accumulate a single fragment.
    void count_one(const ATACFragment& frag) {
        if (frag.chrom_idx < 0 ||
            static_cast<size_t>(frag.chrom_idx) >= chrom_sizes_.size()) return;

        int32_t chrom_len = chrom_sizes_[static_cast<size_t>(frag.chrom_idx)];
        int32_t clipped_start = std::max(int32_t(0), frag.start);
        int32_t clipped_end = std::min(chrom_len, frag.end);
        if (clipped_start >= clipped_end) return;

        int32_t first_bin = clipped_start / bin_width_;
        int32_t last_bin = (clipped_end - 1) / bin_width_;
        int32_t base_offset = global_bin_offset_[static_cast<size_t>(frag.chrom_idx)];

        for (int32_t b = first_bin; b <= last_bin; ++b) {
            acc_.increment(
                static_cast<uint32_t>(base_offset + b),
                frag.barcode_idx,
                uint16_t(1));
        }
    }

    // ----------------------------------------------------------------
    // Output
    // ----------------------------------------------------------------

    /// Returns total number of genome bins.
    size_t total_bins() const { return total_bins_; }

    /// Returns number of non-zero COO entries accumulated.
    size_t nonzero_bins() const { return acc_.nnz_raw(); }

    /// Generate bin feature labels.
    /// Format per line: "chr1:0-500\tchr1:0-500\tPeaks"
    std::vector<std::string> bin_features() const {
        std::vector<std::string> features;
        features.reserve(total_bins_);

        for (size_t ci = 0; ci < chrom_names_.size(); ++ci) {
            int32_t chrom_len = chrom_sizes_[ci];
            int32_t bins_for_chrom = (chrom_len + bin_width_ - 1) / bin_width_;
            for (int32_t b = 0; b < bins_for_chrom; ++b) {
                int32_t bin_start = b * bin_width_;
                int32_t bin_end = std::min(bin_start + bin_width_, chrom_len);
                std::string label = chrom_names_[ci] + ":" +
                                    std::to_string(bin_start) + "-" +
                                    std::to_string(bin_end);
                features.push_back(label + "\t" + label + "\tPeaks");
            }
        }
        return features;
    }

    /// Export accumulated counts as CSC matrix (bins × cells).
    /// num_cells must equal the barcode count.
    SparseAccumulator<uint16_t>::CSCMatrix to_csc(size_t num_cells) const {
        // Build a copy of the accumulator with the correct barcode count set.
        SparseAccumulator<uint16_t> tmp = acc_;
        // Set n_barcodes via a dummy barcodes vector of the right size
        std::vector<std::string> dummy(num_cells);
        tmp.set_barcodes(dummy);
        return tmp.to_csc();
    }

    /// Compute global bin index for a fragment's start position.
    /// Returns SIZE_MAX if the fragment is out of range.
    size_t fragment_global_bin(const ATACFragment& frag) const {
        if (frag.chrom_idx < 0 ||
            static_cast<size_t>(frag.chrom_idx) >= chrom_sizes_.size()) return SIZE_MAX;
        int32_t clipped_start = std::max(int32_t(0), frag.start);
        int32_t bin_local = clipped_start / bin_width_;
        return static_cast<size_t>(global_bin_offset_[static_cast<size_t>(frag.chrom_idx)] + bin_local);
    }

    /// Reset all accumulated data (keeps chromosome config).
    void clear() {
        acc_.clear();
        acc_.set_n_features(static_cast<uint32_t>(total_bins_));
    }

   private:
    int bin_width_ = 500;
    std::vector<std::string> chrom_names_;
    std::vector<int32_t> chrom_sizes_;
    std::vector<int32_t> global_bin_offset_;
    size_t total_bins_ = 0;
    SparseAccumulator<uint16_t> acc_;
};

}  // namespace singlet
