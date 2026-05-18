// SPDX-License-Identifier: MIT
// integrates: singlet/gpu/io/pz_device_loader.h — PzChunkIterator
//             singlet/gpu/io/chunk.h             — singlet::gpu::io::Chunk
//
// PzDataLoader — streaming bridge from .1pz files to native chunked NMF.
//
// CYCLE-106: factornet::io::DataLoader<float> base class removed; PzDataLoader
// is now a free-standing class.  The Eigen::SparseMatrix dependency that came
// with factornet::io::Chunk<float> is replaced by singlet::gpu::io::Chunk (raw
// host CSC vectors).  chunked_fit(loader, cfg) in reduce/nmf/chunked.h
// delegates to the PzChunkIterator path using loader.path() — so that overload
// still compiles unchanged despite the interface change.
//
// Non-zero-copy note (the ONLY non-zero-copy path in singlet-gpu):
//   PzChunkIterator → pinned host CSC (our format).
//   PzDataLoader::next_forward() → singlet::gpu::io::Chunk (CSC vectors, copied).
//   upload_chunk() in streamed_pipeline.h → H2D cudaMemcpyAsync.
//   Total per-chunk cost for 100k-col × 30k-row × ~1k nnz/col:
//     - std::vector construction: ~30k nnz * 8 bytes ≈ 240 MB (values+indices)
//       plus col_ptr: (100k+1) * 4 bytes ≈ 400 KB.
//     - Time: ~10–20 ms (memcpy-dominated, pinned→vector).
//   This cost is paid once per chunk per NMF iteration.
//
// Transpose pass:
//   next_transpose() constructs row panels of A by extracting from a lazily
//   built transposed CSC.  The .1pz format always stores a forward CSC; Aᵀ
//   is reconstructed on first transpose access.
//
// chunk_cols: must be a multiple of the .1pz file's internal chunk_cols for
//   zero-overhead alignment; any value is functionally correct.
//   Default: PzChunkIterator::DEFAULT_CHUNK_COLS (100k).

#pragma once

#include <singlet/gpu/io/chunk.h>
#include <singlet/gpu/io/pz_device_loader.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace io {

// ---------------------------------------------------------------------------
// PzDataLoader — free-standing streaming loader over a single .1pz file.
//
// Yields native singlet::gpu::io::Chunk panels (host CSC slabs).
// No virtual interface — callers iterate directly via next_forward / next_transpose.
// ---------------------------------------------------------------------------
class PzDataLoader {
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
        // Load the full matrix into pinned host memory.
        // keep_host_pinned=true is required so host buffers survive beyond load_pz.
        mat_ = load_pz(path_, /*stream=*/nullptr, /*keep_host_pinned=*/true);

        nrows_ = static_cast<uint32_t>(mat_.mat.rows);
        ncols_ = static_cast<uint32_t>(mat_.mat.cols);
        total_nnz_ = static_cast<uint64_t>(mat_.mat.nnz);

        // Forward chunks: ceil(ncols_ / chunk_cols_)
        n_fwd_chunks_ = (ncols_ + chunk_cols_ - 1) / chunk_cols_;
        // Transpose chunks: ceil(nrows_ / chunk_cols_) (row panels = col panels of Aᵀ)
        n_trp_chunks_ = (nrows_ + chunk_cols_ - 1) / chunk_cols_;
    }

    // Accessors (mirror the DataLoader interface for call-site compat).
    uint32_t rows()                  const noexcept { return nrows_; }
    uint32_t cols()                  const noexcept { return ncols_; }
    uint64_t nnz()                   const noexcept { return total_nnz_; }
    uint32_t num_forward_chunks()    const noexcept { return n_fwd_chunks_; }
    uint32_t num_transpose_chunks()  const noexcept { return n_trp_chunks_; }

    // path() — used by chunked_fit(loader, cfg) backward-compat overload.
    const std::string& path() const noexcept { return path_; }

    // -----------------------------------------------------------------------
    // next_forward — yield the next column panel of A as a host CSC Chunk.
    // Extracts columns [col_start, col_start + chunk_width) from pinned host CSC.
    // Cost: proportional to the chunk's nnz (memcpy of indices + values).
    // -----------------------------------------------------------------------
    bool next_forward(singlet::gpu::io::Chunk& out)
    {
        if (fwd_chunk_idx_ >= n_fwd_chunks_) return false;

        const uint32_t col_start = fwd_chunk_idx_ * chunk_cols_;
        const uint32_t col_end   = std::min(col_start + chunk_cols_, ncols_);
        const uint32_t num_cols  = col_end - col_start;

        const int* indptr  = mat_.host_indptr.get();   // size ncols_+1
        const int* indices = mat_.host_indices.get();  // size nnz
        const float* vals  = mat_.host_values.get();   // size nnz

        const int64_t slice_start = indptr[col_start];
        const int64_t slice_end   = indptr[col_end];
        const int64_t slice_nnz   = slice_end - slice_start;

        out.n_rows   = static_cast<int>(nrows_);
        out.n_cols   = static_cast<int>(num_cols);
        out.chunk_id = static_cast<int>(fwd_chunk_idx_);
        out.n_chunks = static_cast<int>(n_fwd_chunks_);

        // Build shifted col_ptr (local to this chunk).
        out.col_ptr.resize(num_cols + 1);
        for (uint32_t j = 0; j <= num_cols; ++j)
            out.col_ptr[j] = static_cast<int>(indptr[col_start + j] - slice_start);

        out.row_indices.resize(static_cast<size_t>(slice_nnz));
        out.values    .resize(static_cast<size_t>(slice_nnz));
        std::memcpy(out.row_indices.data(), indices + slice_start,
                    static_cast<size_t>(slice_nnz) * sizeof(int));
        std::memcpy(out.values.data(),      vals + slice_start,
                    static_cast<size_t>(slice_nnz) * sizeof(float));

        ++fwd_chunk_idx_;
        return true;
    }

    // -----------------------------------------------------------------------
    // next_transpose — yield the next column panel of Aᵀ (= row panel of A).
    // Lazily builds the full transposed CSC on first access — O(nnz) time+memory.
    // WHY lazy: chunked NMF accesses transpose chunks sequentially but in a
    // separate outer loop; building on first call is simpler than random access.
    // -----------------------------------------------------------------------
    bool next_transpose(singlet::gpu::io::Chunk& out)
    {
        if (trp_chunk_idx_ >= n_trp_chunks_) return false;

        if (!transposed_built_) build_transpose();

        const uint32_t row_start = trp_chunk_idx_ * chunk_cols_;
        const uint32_t row_end   = std::min(row_start + chunk_cols_, nrows_);
        const uint32_t num_cols  = row_end - row_start;  // cols of Aᵀ panel = rows of A

        // Extract col panel from A_T_ (stored as transposed CSC).
        const int64_t slice_start = At_indptr_[row_start];
        const int64_t slice_end   = At_indptr_[row_end];
        const int64_t slice_nnz   = slice_end - slice_start;

        out.n_rows   = static_cast<int>(ncols_);   // rows of Aᵀ = cols of A
        out.n_cols   = static_cast<int>(num_cols);
        out.chunk_id = static_cast<int>(trp_chunk_idx_);
        out.n_chunks = static_cast<int>(n_trp_chunks_);

        out.col_ptr.resize(num_cols + 1);
        for (uint32_t j = 0; j <= num_cols; ++j)
            out.col_ptr[j] = static_cast<int>(At_indptr_[row_start + j] - slice_start);

        out.row_indices.resize(static_cast<size_t>(slice_nnz));
        out.values    .resize(static_cast<size_t>(slice_nnz));
        std::memcpy(out.row_indices.data(), At_indices_.data() + slice_start,
                    static_cast<size_t>(slice_nnz) * sizeof(int));
        std::memcpy(out.values.data(),      At_values_.data()  + slice_start,
                    static_cast<size_t>(slice_nnz) * sizeof(float));

        ++trp_chunk_idx_;
        return true;
    }

    void reset_forward()   noexcept { fwd_chunk_idx_ = 0; }
    void reset_transpose() noexcept { trp_chunk_idx_ = 0; }

private:
    // Build Aᵀ in CSC form from the pinned host CSC of A.
    // O(nnz) time and memory.  Called at most once per PzDataLoader lifetime.
    void build_transpose()
    {
        const int m      = static_cast<int>(nrows_);
        const int n      = static_cast<int>(ncols_);
        const int64_t nz = static_cast<int64_t>(total_nnz_);

        const int* a_indptr  = mat_.host_indptr.get();
        const int* a_indices = mat_.host_indices.get();
        const float* a_vals  = mat_.host_values.get();

        // Aᵀ has shape (n × m) — n cols of original become m rows of Aᵀ.
        // CSC of Aᵀ: col dimension = m (original rows become cols of Aᵀ).
        At_indptr_ .assign(static_cast<size_t>(m + 1), 0);
        At_indices_.resize(static_cast<size_t>(nz));
        At_values_ .resize(static_cast<size_t>(nz));

        // Count nnz per row of A (= nnz per col of Aᵀ).
        for (int64_t k = 0; k < nz; ++k)
            At_indptr_[a_indices[k] + 1]++;
        // Prefix-sum to get col_ptr.
        for (int r = 1; r <= m; ++r)
            At_indptr_[r] += At_indptr_[r - 1];

        // Scatter entries.
        std::vector<int> pos(At_indptr_.begin(), At_indptr_.begin() + m);
        for (int j = 0; j < n; ++j) {
            for (int k = a_indptr[j]; k < a_indptr[j + 1]; ++k) {
                const int r = a_indices[k];
                At_indices_[pos[r]] = j;        // row of Aᵀ = col of A
                At_values_ [pos[r]] = a_vals[k];
                ++pos[r];
            }
        }

        transposed_built_ = true;
    }

    std::string      path_;
    uint32_t         chunk_cols_;
    io::PzDeviceMatrix mat_;   // retains pinned host CSC (host_retained=true)

    uint32_t  nrows_;
    uint32_t  ncols_;
    uint64_t  total_nnz_;
    uint32_t  n_fwd_chunks_;
    uint32_t  n_trp_chunks_;

    uint32_t  fwd_chunk_idx_;
    uint32_t  trp_chunk_idx_;

    // Lazily-constructed Aᵀ in host CSC form.
    std::vector<int>   At_indptr_;
    std::vector<int>   At_indices_;
    std::vector<float> At_values_;
    bool transposed_built_ = false;
};

}  // namespace io
}  // namespace singlet::gpu
