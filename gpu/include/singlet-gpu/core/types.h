// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/core/types.h
//
// Native GPU type definitions — DeviceCSC, DeviceDense, DeviceMemory.
// Replaces factornet re-exports as of CYCLE-105 (2026-04-29).
//
// Algorithm credit: CSC layout follows factornet::gpu::SparseMatrixGPU<float>
// conventions (Zach DeBruine, factornet 2021-2026, GPL-2.0).
// Field names preserved for ABI compatibility with existing call sites.

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

// ---------------------------------------------------------------------------
// SINGLET_GPU_CUDA_CHECK — throw on CUDA error; named to avoid collisions.
// WHY defined here: several headers (scale.h, lognorm.h, donor_pseudobulk.h)
// use CUDA_CHECK which was previously supplied by factornet headers (CYCLE-105
// removed that dependency). Define once in the root types header so all
// singlet-gpu headers see it without an extra include.
// ---------------------------------------------------------------------------
#define SINGLET_GPU_CUDA_CHECK(call)                                            \
    do {                                                                        \
        cudaError_t _singlet_err = (call);                                      \
        if (_singlet_err != cudaSuccess) {                                      \
            throw std::runtime_error(                                           \
                std::string("CUDA error in " __FILE__ ": ")                     \
                + cudaGetErrorString(_singlet_err));                            \
        }                                                                       \
    } while (0)

#ifndef CUDA_CHECK
#  define CUDA_CHECK(call) SINGLET_GPU_CUDA_CHECK(call)
#endif

// cuSPARSE status check — throws on any error code other than SUCCESS.
// Same pattern as CUDA_CHECK; named independently to avoid factornet collision.
#define SINGLET_GPU_CUSPARSE_CHECK(call)                                        \
    do {                                                                        \
        cusparseStatus_t _singlet_st = (call);                                  \
        if (_singlet_st != CUSPARSE_STATUS_SUCCESS) {                           \
            throw std::runtime_error(                                           \
                std::string("cuSPARSE error in " __FILE__ ": ")                 \
                + cusparseGetErrorString(_singlet_st));                         \
        }                                                                       \
    } while (0)

#ifndef CUSPARSE_CHECK
#  define CUSPARSE_CHECK(call) SINGLET_GPU_CUSPARSE_CHECK(call)
#endif

// cuBLAS status check — throws on any error code other than SUCCESS.
#define SINGLET_GPU_CUBLAS_CHECK(call)                                          \
    do {                                                                        \
        cublasStatus_t _singlet_cb = (call);                                    \
        if (_singlet_cb != CUBLAS_STATUS_SUCCESS) {                             \
            throw std::runtime_error(                                           \
                std::string("cuBLAS error in " __FILE__));                      \
        }                                                                       \
    } while (0)

#ifndef CUBLAS_CHECK
#  define CUBLAS_CHECK(call) SINGLET_GPU_CUBLAS_CHECK(call)
#endif

namespace singlet_gpu {
namespace core {

// ---------------------------------------------------------------------------
// DeviceMemory<T> — RAII device allocation.
//
// Non-copyable, movable. All device buffers must go through this type
// (Rule 5: no raw cudaMalloc outside core/memory.h).
//
// get() returns the raw device pointer. The caller stream is responsible for
// ensuring work using the buffer is completed before destruction.
// ---------------------------------------------------------------------------
template<typename T>
class DeviceMemory {
public:
    DeviceMemory() noexcept : ptr_(nullptr), n_(0), owns_(true) {}

    explicit DeviceMemory(std::size_t n) : ptr_(nullptr), n_(n), owns_(true) {
        if (n == 0) return;
        cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&ptr_),
                                     n * sizeof(T));
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("DeviceMemory: cudaMalloc failed (")
                + std::to_string(n * sizeof(T)) + " bytes): "
                + cudaGetErrorString(err));
        }
    }

    // Non-owning view constructor — no cudaFree on destruction.
    // Used by DeviceCSC::from_device_ptrs to wrap externally-owned device ptrs.
    DeviceMemory(T* ptr, std::size_t n, bool owns) noexcept
        : ptr_(ptr), n_(n), owns_(owns) {}

    // Non-owning factory — mirrors factornet::gpu::DeviceMemory<T>::wrap() API.
    // WHY static: lets callers migrate from factornet::gpu::DeviceMemory<T>::wrap(...)
    // to singlet_gpu::core::DeviceMemory<T>::wrap(...) without changing call shape.
    static DeviceMemory<T> wrap(T* ptr, std::size_t n) noexcept {
        return DeviceMemory<T>(ptr, n, /*owns=*/false);
    }

    ~DeviceMemory() {
        if (owns_ && ptr_) cudaFree(ptr_);
    }

    // Movable only.
    DeviceMemory(DeviceMemory&& o) noexcept
        : ptr_(o.ptr_), n_(o.n_), owns_(o.owns_)
    { o.ptr_ = nullptr; o.n_ = 0; o.owns_ = false; }

    DeviceMemory& operator=(DeviceMemory&& o) noexcept {
        if (this != &o) {
            if (owns_ && ptr_) cudaFree(ptr_);
            ptr_ = o.ptr_; n_ = o.n_; owns_ = o.owns_;
            o.ptr_ = nullptr; o.n_ = 0; o.owns_ = false;
        }
        return *this;
    }

    DeviceMemory(const DeviceMemory&)            = delete;
    DeviceMemory& operator=(const DeviceMemory&) = delete;

    T*          get()   const noexcept { return ptr_; }
    T*          data()  const noexcept { return ptr_; }  // alias for get(); needed by _bind_kernels.hpp
    std::size_t size()  const noexcept { return n_; }
    std::size_t bytes() const noexcept { return n_ * sizeof(T); }
    bool        owns()  const noexcept { return owns_; }

    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    T*          ptr_;
    std::size_t n_;
    bool        owns_;
};

// ---------------------------------------------------------------------------
// DeviceCSC — CSC sparse matrix on device (genes × cells, fp32).
//
// Layout: col_ptr[n+1] (int32), row_indices[nnz] (int32), values[nnz] (float).
// Field names match factornet::gpu::SparseMatrixGPU<float> for ABI compat.
// algorithm derived from factornet/gpu/types.cuh
// ---------------------------------------------------------------------------
struct DeviceCSC {
    DeviceMemory<int32_t> col_ptr;      // size: cols + 1
    DeviceMemory<int32_t> row_indices;  // size: nnz
    DeviceMemory<float>   values;       // size: nnz

    int rows = 0;
    int cols = 0;
    int nnz  = 0;

    DeviceCSC() = default;

    DeviceCSC(int m, int n, int nz)
        : col_ptr(n + 1)
        , row_indices(nz)
        , values(nz)
        , rows(m)
        , cols(n)
        , nnz(nz)
    {}

    // Non-owning factory — wraps externally-owned device pointers (e.g. from cupy).
    // The caller is responsible for the lifetime of the underlying device buffers.
    // Used by _bind_kernels.hpp to bridge Python-side cupy device arrays to GPU kernels.
    // algorithm derived from factornet/gpu/types.cuh
    static DeviceCSC from_device_ptrs(int m, int n, int nz,
                                      int32_t* d_col_ptr,
                                      int32_t* d_row_indices,
                                      float*   d_values) {
        DeviceCSC result;
        result.col_ptr      = DeviceMemory<int32_t>(d_col_ptr,    static_cast<std::size_t>(n + 1), false);
        result.row_indices  = DeviceMemory<int32_t>(d_row_indices, static_cast<std::size_t>(nz),    false);
        result.values       = DeviceMemory<float>  (d_values,      static_cast<std::size_t>(nz),    false);
        result.rows = m;
        result.cols = n;
        result.nnz  = nz;
        return result;
    }

    // Movable, not copyable.
    DeviceCSC(DeviceCSC&&)            = default;
    DeviceCSC& operator=(DeviceCSC&&) = default;
    DeviceCSC(const DeviceCSC&)            = delete;
    DeviceCSC& operator=(const DeviceCSC&) = delete;
};

// ---------------------------------------------------------------------------
// DeviceDense — column-major dense matrix on device (fp32).
//
// Field names match factornet::gpu::DenseMatrixGPU<float> for ABI compat.
// algorithm derived from factornet/gpu/types.cuh
// ---------------------------------------------------------------------------
struct DeviceDense {
    DeviceMemory<float> data;  // size: rows * cols (col-major)

    int rows = 0;
    int cols = 0;

    DeviceDense() = default;

    DeviceDense(int m, int n)
        : data(static_cast<std::size_t>(m) * n)
        , rows(m)
        , cols(n)
    {}

    // Non-owning factory — wraps an externally-owned device pointer.
    // WHY static: mirrors DeviceMemory<T>::wrap() API for call-site symmetry.
    // Used by _bind_kernels.hpp to wrap cupy device pointers without copying.
    // row_major=true documents the caller's layout; not currently used internally
    // (DeviceDense kernels assume col-major) but recorded for future validation.
    static DeviceDense wrap(float* ptr, int m, int n, bool /*row_major*/ = true) noexcept {
        DeviceDense d;
        d.rows = m;
        d.cols = n;
        d.data = DeviceMemory<float>(ptr, static_cast<std::size_t>(m) * n, /*owns=*/false);
        return d;
    }

    // Movable, not copyable.
    DeviceDense(DeviceDense&&)            = default;
    DeviceDense& operator=(DeviceDense&&) = default;
    DeviceDense(const DeviceDense&)            = delete;
    DeviceDense& operator=(const DeviceDense&) = delete;
};

}  // namespace core
}  // namespace singlet_gpu
