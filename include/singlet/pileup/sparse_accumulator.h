// SPDX-License-Identifier: MIT
#pragma once
#define SINGLET_SPARSE_ACCUMULATOR_DEFINED
// singlet-pileup: sparse_accumulator.h
// Memory-efficient sparse matrix accumulator for streaming BAM pileup.
//
// Design:
// - Barcodes mapped to dense indices via hash table (typically 1K-50K barcodes)
// - Features (SNPs, exons, editing sites) mapped to dense column indices
// - Accumulates uint8/uint16/float values per (barcode, feature)
// - Exports to CSC sparse format for .1pz writing
//
// Memory: ~20KB per barcode for typical SNP coverage (20K SNPs × 1 byte)
// For 10K barcodes: ~200 MB for SNP AD matrix

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet {

// Sparse matrix accumulator using COO triplets.
// Efficient for streaming insertion, converts to CSC at output time.
template <typename ValT = uint8_t>
class SparseAccumulator {
public:
    SparseAccumulator() = default;

    // Set the barcode count (columns in output)
    void set_barcodes(const std::vector<std::string>& barcodes) {
        n_barcodes_ = barcodes.size();
    }

    // Set number of features (rows in output)
    void set_n_features(uint32_t n) { n_features_ = n; }

    // Get number of features
    uint32_t n_features() const { return n_features_; }

    // Increment value at (feature_idx, barcode_idx) by delta
    void increment(uint32_t feature_idx, uint32_t barcode_idx, ValT delta = 1) {
        rows_.push_back(feature_idx);
        cols_.push_back(barcode_idx);
        vals_.push_back(delta);
    }

    // Return number of accumulated entries (before dedup)
    size_t nnz_raw() const { return rows_.size(); }

    // Export as CSC sparse matrix.
    // Deduplicates and sums values for duplicate (row, col) pairs.
    // Returns: (indptr, indices, data) in CSC format
    // indptr: size n_barcodes+1
    // indices: size nnz (feature indices)
    // data: size nnz (values)
    struct CSCMatrix {
        std::vector<int32_t> indptr;   // column pointers
        std::vector<int32_t> indices;  // row indices
        std::vector<ValT> data;        // values
        uint32_t nrows;                // n_features
        uint32_t ncols;                // n_barcodes
    };

    CSCMatrix to_csc() const {
        CSCMatrix out;
        out.nrows = n_features_;
        out.ncols = n_barcodes_;

        if (rows_.empty()) {
            out.indptr.assign(n_barcodes_ + 1, 0);
            return out;
        }

        const size_t n = rows_.size();

        // ── Phase 1: Count entries per column (O(n)) ──
        // AUTOFIX-INDROP-EXPORT-SEGV: internal scratch arrays use int64 to
        // avoid overflow when total entries exceed 2^31 (e.g. mt_alleles on
        // samples with >2B pileup bases).  Previously int32_t slot arithmetic
        // produced negative indices → OOB write → SIGSEGV.  The output
        // CSCMatrix.indptr remains int32_t; post-dedup nnz per column rarely
        // exceeds 2^31, and we clamp defensively below.
        std::vector<int64_t> col_count(n_barcodes_, 0);
        for (size_t i = 0; i < n; ++i) {
            col_count[cols_[i]]++;
        }

        // Build column start offsets (prefix sum)
        std::vector<int64_t> col_start(n_barcodes_ + 1, 0);
        for (uint32_t c = 0; c < n_barcodes_; ++c) {
            col_start[c + 1] = col_start[c] + col_count[c];
        }

        // ── Phase 2: Scatter into column-grouped arrays (O(n) counting sort) ──
        std::vector<int64_t> col_pos(col_start.begin(), col_start.begin() + n_barcodes_);
        std::vector<uint32_t> grouped_rows(n);
        std::vector<ValT> grouped_vals(n);
        for (size_t i = 0; i < n; ++i) {
            int64_t slot = col_pos[cols_[i]]++;
            grouped_rows[slot] = rows_[i];
            grouped_vals[slot] = vals_[i];
        }

        // ── Phase 3: Dense accumulate + sort unique rows → CSC ──
        // Instead of sorting all entries (including duplicates) per column,
        // accumulate values into a dense array by row index (O(1) per entry),
        // then sort only the unique row indices (typically ~4× fewer).
        out.indptr.assign(n_barcodes_ + 1, 0);
        out.indices.reserve(n / 2);
        out.data.reserve(n / 2);

        // Accumulator type: uint32 for integral ValT (overflow safety), double for float
        using AccT = std::conditional_t<std::is_integral_v<ValT>, uint32_t, double>;
        std::vector<AccT> acc(n_features_, AccT(0));
        std::vector<uint32_t> active_rows;
        active_rows.reserve(4096);

        // Helper: clamp size_t → int32_t for indptr output.
        auto indptr_val = [](size_t ns) -> int32_t {
            if (ns > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
                return std::numeric_limits<int32_t>::max();
            return static_cast<int32_t>(ns);
        };

        for (uint32_t c = 0; c < n_barcodes_; ++c) {
            int64_t start = col_start[c];
            int64_t end = col_start[c + 1];
            if (start == end) {
                out.indptr[c + 1] = indptr_val(out.indices.size());
                continue;
            }

            // Accumulate entries and track active rows: O(entries_in_col)
            active_rows.clear();
            for (int64_t i = start; i < end; ++i) {
                uint32_t row = grouped_rows[i];
                if (acc[row] == AccT(0)) {
                    active_rows.push_back(row);
                }
                acc[row] += static_cast<AccT>(grouped_vals[i]);
            }

            // Sort only unique rows: O(unique × log(unique))
            std::sort(active_rows.begin(), active_rows.end());

            // Emit sorted unique entries and reset accumulator
            for (uint32_t row : active_rows) {
                out.indices.push_back(static_cast<int32_t>(row));
                if constexpr (std::is_integral_v<ValT>) {
                    out.data.push_back(static_cast<ValT>(std::min(
                        static_cast<uint32_t>(acc[row]),
                        static_cast<uint32_t>(std::numeric_limits<ValT>::max()))));
                } else {
                    out.data.push_back(static_cast<ValT>(acc[row]));
                }
                acc[row] = AccT(0);
            }

            out.indptr[c + 1] = indptr_val(out.indices.size());
        }

        return out;
    }

    void clear() {
        rows_.clear();
        cols_.clear();
        vals_.clear();
    }

    // Merge another accumulator's COO data into this one (for parallel workers)
    void merge_from(SparseAccumulator& other) {
        if (other.rows_.empty()) return;
        rows_.insert(rows_.end(), other.rows_.begin(), other.rows_.end());
        cols_.insert(cols_.end(), other.cols_.begin(), other.cols_.end());
        vals_.insert(vals_.end(), other.vals_.begin(), other.vals_.end());
        // Update n_features if other had more
        if (other.n_features_ > n_features_) n_features_ = other.n_features_;
        other.clear();
    }

private:
    uint32_t n_barcodes_ = 0;
    uint32_t n_features_ = 0;

    // COO triplets
    std::vector<uint32_t> rows_;
    std::vector<uint32_t> cols_;
    std::vector<ValT> vals_;
};

} // namespace singlet
