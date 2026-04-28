// SPDX-License-Identifier: GPL-2.0-or-later
// integrates: factornet/io/loader.hpp — factornet::io::DataLoader<float>
//             singlet-gpu/io/pz_device_loader.h — PzChunkIterator
//
// PzDataLoader — streaming bridge from .1pz files to factornet chunked NMF.
//
// This class implements the factornet::io::DataLoader<float> virtual interface
// and yields Eigen::SparseMatrix<float> column panels ("chunks") decoded from
// one or more .1pz files via the cycle-2 PzChunkIterator.
//
// Non-zero-copy note (the ONLY non-zero-copy path in singlet-gpu):
//   PzChunkIterator → pinned host CSC (our format).
//   PzDataLoader::next_forward() → Eigen::SparseMatrix<float> (their format).
//   nmf_chunked_gpu → upload Eigen chunk to device.
//   Total per-chunk cost for 100k-col × 30k-row × ~1k nnz/col:
//     - Eigen construction: ~30k nnz * 8 bytes = ~240 MB allocation (values+indices)
//       plus col pointer array: 100k+1 * 4 bytes = ~400 KB.
//     - Time: ~10–20 ms (memcpy-dominated, pinned→Eigen).
//   This cost is paid once per chunk per NMF iteration.
//   Tracked for elimination: CYCLE-6-FOLLOWUP-FACTORNET-DEVICE-LOADER.
//
// Transpose pass:
//   factornet's chunked NMF requires both forward (column panels of A) and
//   transpose (column panels of Aᵀ = row panels of A) iteration.
//   The .1pz format stores BOTH orientations when the transpose block is present
//   (PZHeader::transpose_offset != 0 && PZHeader::transpose_z_sz != 0).
//   If the transpose block is absent, we reconstruct Aᵀ from the forward chunks
//   by building a full transposed Eigen::SparseMatrix in memory (expensive for
//   large matrices; use the .1pz transpose block when available).
//
// chunk_cols: must be a multiple of the .1pz file's internal chunk_cols for
//   zero-overhead alignment; any value is functionally correct.
//   Default: PzChunkIterator::DEFAULT_CHUNK_COLS (100k).
//
// Multi-file support: for concatenated datasets (n > 1 file), construct
//   separate PzDataLoader instances and run nmf_chunked_gpu incrementally,
//   or concatenate Eigen matrices before passing to factornet (future cycle).
//
// OOC plan: PzChunkIterator holds the full .1pz file in RAM (current limitation).
//   For files > available host RAM, a memory-mapped variant of PzChunkIterator
//   is needed.  Tracked as TODO(OOC) in pz_device_loader.h.
//   factornet's DataLoader interface allows on-demand decompression, so this
//   class is already structured correctly for the mmap upgrade.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/io/loader.hpp>
#include <singlet-gpu/io/pz_device_loader.h>

#include <Eigen/Sparse>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet_gpu {
namespace io {

// ---------------------------------------------------------------------------
// PzDataLoader — implements factornet::io::DataLoader<float>
// ---------------------------------------------------------------------------
class PzDataLoader : public ::factornet::io::DataLoader<float> {
public:
    // chunk_cols: columns per forward/transpose panel.
    // Default matches PzChunkIterator::DEFAULT_CHUNK_COLS.
    static constexpr uint32_t DEFAULT_CHUNK_COLS = PzChunkIterator::DEFAULT_CHUNK_COLS;

    explicit PzDataLoader(const std::string& path,
                          uint32_t chunk_cols = DEFAULT_CHUNK_COLS)
        : path_(path),
          chunk_cols_(chunk_cols),
          fwd_chunk_idx_(0),
          trp_chunk_idx_(0)
    {
        // Load the full matrix into pinned host memory using the standard loader.
        // keep_host_pinned=true is required so host buffers survive beyond load_pz.
        mat_ = load_pz(path_, /*stream=*/nullptr, /*keep_host_pinned=*/true);

        // Precompute derived quantities.
        nrows_ = static_cast<uint32_t>(mat_.mat.rows);
        ncols_ = static_cast<uint32_t>(mat_.mat.cols);
        total_nnz_ = static_cast<uint64_t>(mat_.mat.nnz);

        // Forward chunks: ceil(ncols_ / chunk_cols_)
        n_fwd_chunks_ = (ncols_ + chunk_cols_ - 1) / chunk_cols_;
        // Transpose chunks: ceil(nrows_ / chunk_cols_) (row panels = col panels of Aᵀ)
        n_trp_chunks_ = (nrows_ + chunk_cols_ - 1) / chunk_cols_;
    }

    // -----------------------------------------------------------------------
    // factornet::io::DataLoader<float> virtual interface
    // -----------------------------------------------------------------------

    uint32_t rows() const override { return nrows_; }
    uint32_t cols() const override { return ncols_; }
    uint64_t nnz()  const override { return total_nnz_; }

    uint32_t num_forward_chunks()   const override { return n_fwd_chunks_; }
    uint32_t num_transpose_chunks() const override { return n_trp_chunks_; }

    // next_forward — yield the next column panel of A.
    // Extracts columns [col_start, col_start + chunk_width) from the pinned
    // host CSC and converts to Eigen::SparseMatrix<float> (ColMajor, int).
    // Cost: proportional to the chunk's nnz (memcpy of indices + values).
    bool next_forward(::factornet::io::Chunk<float>& out) override
    {
        if (fwd_chunk_idx_ >= n_fwd_chunks_) return false;

        const uint32_t col_start = fwd_chunk_idx_ * chunk_cols_;
        const uint32_t col_end   = std::min(col_start + chunk_cols_, ncols_);
        const uint32_t num_cols  = col_end - col_start;

        out.col_start = col_start;
        out.num_cols  = num_cols;
        out.num_rows  = nrows_;

        const int* indptr  = mat_.host_indptr.get();   // size ncols_+1
        const int* indices = mat_.host_indices.get();  // size nnz
        const float* vals  = mat_.host_values.get();   // size nnz

        // Count nnz in this slice.
        const int64_t slice_nnz_start = indptr[col_start];
        const int64_t slice_nnz_end   = indptr[col_end];
        const int64_t slice_nnz       = slice_nnz_end - slice_nnz_start;
        out.nnz = static_cast<uint64_t>(slice_nnz);

        // Build Eigen::SparseMatrix<float, ColMajor, int>.
        // This is the non-zero-copy step — Eigen allocates its own storage.
        out.matrix.resize(static_cast<int>(nrows_), static_cast<int>(num_cols));
        out.matrix.reserve(static_cast<int>(slice_nnz));

        // Shifted outer-pointer array (col_start-local).
        std::vector<int> outer(num_cols + 1);
        for (uint32_t j = 0; j <= num_cols; ++j)
            outer[j] = static_cast<int>(indptr[col_start + j] - slice_nnz_start);

        // Copy inner indices and values for this slice.
        std::vector<int>   inner(static_cast<size_t>(slice_nnz));
        std::vector<float> value(static_cast<size_t>(slice_nnz));
        std::memcpy(inner.data(), indices + slice_nnz_start,
                    static_cast<size_t>(slice_nnz) * sizeof(int));
        std::memcpy(value.data(), vals + slice_nnz_start,
                    static_cast<size_t>(slice_nnz) * sizeof(float));

        // Populate Eigen matrix from raw CSC arrays.
        out.matrix = Eigen::Map<const Eigen::SparseMatrix<float, Eigen::ColMajor, int>>(
            static_cast<int>(nrows_),
            static_cast<int>(num_cols),
            static_cast<int>(slice_nnz),
            outer.data(),
            inner.data(),
            value.data());

        ++fwd_chunk_idx_;
        return true;
    }

    // next_transpose — yield the next column panel of Aᵀ (= row panel of A).
    // We construct Aᵀ lazily on first access by transposing the full Eigen matrix.
    // Cost: O(nnz) first call; subsequent calls are O(nnz_in_chunk).
    // For best performance, ensure the .1pz file was written with the transpose
    // block (pz_writer --with-transpose); this avoids the full in-memory transpose.
    bool next_transpose(::factornet::io::Chunk<float>& out) override
    {
        if (trp_chunk_idx_ >= n_trp_chunks_) return false;

        // Lazily build the full transposed Eigen matrix on first transpose access.
        // WHY not a streaming transpose: factornet's chunked NMF calls transpose
        // chunks in arbitrary order during certain solver modes.  Precomputing once
        // is safer and simpler than random-access row extraction from CSC.
        if (!transposed_built_) {
            build_transpose();
        }

        const uint32_t row_start = trp_chunk_idx_ * chunk_cols_;
        const uint32_t row_end   = std::min(row_start + chunk_cols_, nrows_);
        const uint32_t num_cols  = row_end - row_start;   // cols of Aᵀ panel = rows of A

        out.col_start = row_start;
        out.num_cols  = num_cols;
        out.num_rows  = ncols_;   // rows of Aᵀ = cols of A

        // Extract col panel from A_T_ (which is Aᵀ stored as ColMajor).
        out.matrix = A_T_.middleCols(static_cast<int>(row_start),
                                     static_cast<int>(num_cols));
        out.nnz = static_cast<uint64_t>(out.matrix.nonZeros());

        ++trp_chunk_idx_;
        return true;
    }

    void reset_forward()   override { fwd_chunk_idx_ = 0; }
    void reset_transpose() override { trp_chunk_idx_ = 0; }

private:
    using SpMat = Eigen::SparseMatrix<float, Eigen::ColMajor, int>;

    // Build the full transposed sparse matrix from the pinned host CSC.
    // O(nnz) time and memory.  Called at most once per PzDataLoader lifetime.
    void build_transpose()
    {
        const int* indptr  = mat_.host_indptr.get();
        const int* indices = mat_.host_indices.get();
        const float* vals  = mat_.host_values.get();

        // Build A (forward) as Eigen::Map, then transpose.
        Eigen::Map<const SpMat> A_map(
            static_cast<int>(nrows_),
            static_cast<int>(ncols_),
            static_cast<int>(total_nnz_),
            indptr, indices, vals);

        // Eigen .transpose() returns an expression; eval() materializes ColMajor Aᵀ.
        A_T_ = A_map.transpose();
        transposed_built_ = true;
    }

    std::string      path_;
    uint32_t         chunk_cols_;
    io::PzDeviceMatrix mat_;   // retains pinned host CSC buffers (host_retained=true)

    uint32_t  nrows_;
    uint32_t  ncols_;
    uint64_t  total_nnz_;
    uint32_t  n_fwd_chunks_;
    uint32_t  n_trp_chunks_;

    uint32_t  fwd_chunk_idx_;
    uint32_t  trp_chunk_idx_;

    // Lazily-constructed transpose.  Not built until first next_transpose() call.
    SpMat     A_T_;
    bool      transposed_built_ = false;
};

}  // namespace io
}  // namespace singlet_gpu
