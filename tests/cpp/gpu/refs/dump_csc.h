// SPDX-License-Identifier: MIT
// singlet/gpu/tests/refs/dump_csc.h
//
// Inline helper: copy a device CSC back to host and write a compact binary
// file that lognorm_scanpy_reference.py can ingest via numpy.
//
// Binary format (little-endian, no padding):
//   [0]      uint32_t  magic       = 0x43535343  ("CSCC")
//   [4]      uint32_t  n_rows      (m)
//   [8]      uint32_t  n_cols      (n)
//   [12]     uint64_t  nnz
//   [20]     float[nnz]            values
//   [20+4*nnz]  int32_t[n+1]      indptr
//   [20+4*nnz + 4*(n+1)]  int32_t[nnz]  indices
//
// The Python side reads this with numpy.fromfile — see lognorm_scanpy_reference.py.
//
// Usage (inside a GoogleTest test body):
//   dump_csc(mat, n_rows, n_cols, nnz, stream, "/tmp/test_result.bin");
//
// The function synchronizes `stream` before reading — do NOT pass a stream
// that has in-flight work you want to overlap with something else.

#pragma once

#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace singlet::gpu {
namespace tests {

// ---------------------------------------------------------------------------
// dump_csc
//
// Copies device CSC fields to host and writes the compact binary described
// above. `values_dev`, `indptr_dev`, `indices_dev` are raw device pointers;
// sizes are in elements (not bytes).
//
// Rationale for raw pointers: allows callers to address internals of
// factornet::gpu::SparseMatrixGPU<float> without depending on that type's
// internal accessor names.
// ---------------------------------------------------------------------------
inline void dump_csc(
        const float*   values_dev,   // device float[nnz]
        const int32_t* indptr_dev,   // device int32[n_cols + 1]
        const int32_t* indices_dev,  // device int32[nnz]
        uint32_t       n_rows,
        uint32_t       n_cols,
        uint64_t       nnz,
        cudaStream_t   stream,
        const std::string& path)
{
    // Sync: we must not read device memory before all in-flight work is done.
    cudaError_t err = cudaStreamSynchronize(stream);
    if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("dump_csc: cudaStreamSynchronize failed: ") +
            cudaGetErrorString(err));

    // Allocate host buffers.
    std::vector<float>   h_values (nnz);
    std::vector<int32_t> h_indptr (static_cast<size_t>(n_cols) + 1);
    std::vector<int32_t> h_indices(nnz);

    auto check = [](cudaError_t e, const char* ctx) {
        if (e != cudaSuccess)
            throw std::runtime_error(
                std::string("dump_csc: ") + ctx + ": " + cudaGetErrorString(e));
    };

    check(cudaMemcpy(h_values .data(), values_dev,
                     nnz * sizeof(float),   cudaMemcpyDeviceToHost), "values");
    check(cudaMemcpy(h_indptr .data(), indptr_dev,
                     (static_cast<size_t>(n_cols) + 1) * sizeof(int32_t),
                     cudaMemcpyDeviceToHost), "indptr");
    check(cudaMemcpy(h_indices.data(), indices_dev,
                     nnz * sizeof(int32_t), cudaMemcpyDeviceToHost), "indices");

    // Write binary.
    std::ofstream fout(path, std::ios::binary | std::ios::trunc);
    if (!fout)
        throw std::runtime_error("dump_csc: cannot open output: " + path);

    static constexpr uint32_t kMagic = 0x43535343u;  // "CSCC"
    fout.write(reinterpret_cast<const char*>(&kMagic),  4);
    fout.write(reinterpret_cast<const char*>(&n_rows),  4);
    fout.write(reinterpret_cast<const char*>(&n_cols),  4);
    fout.write(reinterpret_cast<const char*>(&nnz),     8);
    fout.write(reinterpret_cast<const char*>(h_values .data()),
               static_cast<std::streamsize>(nnz * sizeof(float)));
    fout.write(reinterpret_cast<const char*>(h_indptr .data()),
               static_cast<std::streamsize>((static_cast<size_t>(n_cols) + 1) * sizeof(int32_t)));
    fout.write(reinterpret_cast<const char*>(h_indices.data()),
               static_cast<std::streamsize>(nnz * sizeof(int32_t)));
    if (!fout)
        throw std::runtime_error("dump_csc: write error: " + path);
}

}  // namespace tests
}  // namespace singlet::gpu
