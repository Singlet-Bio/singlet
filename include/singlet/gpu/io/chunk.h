// SPDX-License-Identifier: MIT
// singlet/gpu/io/chunk.h
//
// Native host-side CSC slab type for streaming iteration.
// Replaces factornet::io::Chunk<float> as of CYCLE-106.
//
// Chunk carries a CSC column panel (col_ptr, row_indices, values) in host
// std::vector storage.  This avoids the Eigen dependency that factornet::io::Chunk
// required and aligns with PzChunkIterator's native CSC representation.
//
// Field correspondence to factornet::io::Chunk<float>:
//   factornet::col_start  → chunk_id * (n_cols)  (derived from chunk_id)
//   factornet::num_cols   → n_cols
//   factornet::num_rows   → n_rows
//   factornet::nnz        → row_indices.size()
//   factornet::matrix     → (col_ptr, row_indices, values)

#pragma once

#include <cstdint>
#include <vector>

namespace singlet::gpu {
namespace io {

// ---------------------------------------------------------------------------
// Chunk — host-side CSC column panel.
//
// col_ptr:     length n_cols+1 (standard CSC indptr, zero-based).
// row_indices: length nnz, row indices for each stored entry.
// values:      length nnz, fp32 values for each stored entry.
// n_rows:      number of rows in the full matrix (genes).
// n_cols:      number of columns in this chunk (cells).
// chunk_id:    0-based index of this chunk in the file.
// n_chunks:    total number of chunks in the file (forward direction).
// ---------------------------------------------------------------------------
struct Chunk {
    std::vector<int>   col_ptr;      // host-side CSC indptr (length n_cols+1)
    std::vector<int>   row_indices;  // host-side row indices (length nnz)
    std::vector<float> values;       // host-side values (length nnz)
    int n_rows    = 0;
    int n_cols    = 0;
    int chunk_id  = 0;
    int n_chunks  = 1;
};

}  // namespace io
}  // namespace singlet::gpu
