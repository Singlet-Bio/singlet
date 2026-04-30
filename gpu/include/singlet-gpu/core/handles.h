// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/core/handles.h
//
// GPUContext — owns cuBLAS, cuSPARSE, cuSOLVER, cuRAND handles + one CUDA stream.
// Replaces factornet::gpu::GPUContext re-export as of CYCLE-105 (2026-04-29).
//
// Every singlet-gpu kernel receives a GPUContext reference — never creates
// handles internally (Rule 6). Streams are passed in by the caller (Rule 7).
//
// algorithm derived from factornet/gpu/types.cuh (GPUContext pattern)

#pragma once

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusparse.h>
#include <cusolverDn.h>
#include <curand.h>

#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace singlet_gpu {
namespace core {

// ---------------------------------------------------------------------------
// GPUContext — aggregated CUDA library handles + stream.
//
// WHY aggregate: all cuBLAS / cuSPARSE / cuSOLVER APIs require an opaque
// handle created once per context. Bundling them ensures kernels share handles
// and the associated device-side state (algorithm caches, workspace) without
// re-initialization cost on every call.
//
// Ownership: GPUContext is move-only. The destructor tears down all handles
// in the correct order (solver → blas → sparse → rand → stream).
//
// Stream ownership: if the caller passes nullptr, GPUContext creates a
// non-blocking high-priority stream and owns it. If the caller passes an
// existing stream, GPUContext does NOT take ownership; the caller destroys it.
// ---------------------------------------------------------------------------
class GPUContext {
public:
    // Default constructor: creates all handles + a new non-blocking stream.
    GPUContext() : stream_owned_(true), stream_(nullptr) {
        create_stream_();
        create_handles_();
    }

    // External-stream constructor: borrows the caller's stream, no ownership.
    explicit GPUContext(cudaStream_t stream)
        : stream_owned_(false), stream_(stream)
    {
        create_handles_();
    }

    ~GPUContext() {
        if (curand_) curandDestroyGenerator(curand_);
        if (solver_)  cusolverDnDestroy(solver_);
        if (blas_)    cublasDestroy(blas_);
        if (sparse_)  cusparseDestroy(sparse_);
        if (stream_owned_ && stream_) cudaStreamDestroy(stream_);
    }

    // Move-only.
    GPUContext(GPUContext&& o) noexcept
        : stream_owned_(o.stream_owned_)
        , stream_(o.stream_)
        , blas_(o.blas_)
        , sparse_(o.sparse_)
        , solver_(o.solver_)
        , curand_(o.curand_)
    {
        o.stream_owned_ = false;
        o.stream_ = nullptr;
        o.blas_   = nullptr;
        o.sparse_ = nullptr;
        o.solver_ = nullptr;
        o.curand_ = nullptr;
    }

    GPUContext& operator=(GPUContext&&) = delete;
    GPUContext(const GPUContext&)       = delete;
    GPUContext& operator=(const GPUContext&) = delete;

    // Accessors — const because handles are mutable but logically constant.
    cudaStream_t        stream()  const noexcept { return stream_; }
    cublasHandle_t      blas()    const noexcept { return blas_; }
    cusparseHandle_t    sparse()  const noexcept { return sparse_; }
    cusolverDnHandle_t  solver()  const noexcept { return solver_; }
    curandGenerator_t   curand()  const noexcept { return curand_; }

    // Re-seed the cuRAND generator (e.g., per-call deterministic seeding).
    void seed_curand(uint64_t seed) {
        curandStatus_t st = curandSetPseudoRandomGeneratorSeed(curand_, seed);
        if (st != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("GPUContext: curandSetPseudoRandomGeneratorSeed failed");
        curandGenerateSeeds(curand_);
    }

private:
    bool               stream_owned_;
    cudaStream_t       stream_  = nullptr;
    cublasHandle_t     blas_    = nullptr;
    cusparseHandle_t   sparse_  = nullptr;
    cusolverDnHandle_t solver_  = nullptr;
    curandGenerator_t  curand_  = nullptr;

    void create_stream_() {
        int least, greatest;
        cudaDeviceGetStreamPriorityRange(&least, &greatest);
        cudaError_t err = cudaStreamCreateWithPriority(
            &stream_, cudaStreamNonBlocking, greatest);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("GPUContext: cudaStreamCreate failed: ")
                + cudaGetErrorString(err));
    }

    void create_handles_() {
        // cuBLAS
        if (cublasCreate(&blas_) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("GPUContext: cublasCreate failed");
        cublasSetStream(blas_, stream_);

        // cuSPARSE
        if (cusparseCreate(&sparse_) != CUSPARSE_STATUS_SUCCESS)
            throw std::runtime_error("GPUContext: cusparseCreate failed");
        cusparseSetStream(sparse_, stream_);

        // cuSOLVER
        if (cusolverDnCreate(&solver_) != CUSOLVER_STATUS_SUCCESS)
            throw std::runtime_error("GPUContext: cusolverDnCreate failed");
        cusolverDnSetStream(solver_, stream_);

        // cuRAND (Philox — reproducible with explicit seed)
        if (curandCreateGenerator(&curand_, CURAND_RNG_PSEUDO_PHILOX4_32_10)
                != CURAND_STATUS_SUCCESS)
            throw std::runtime_error("GPUContext: curandCreateGenerator failed");
        curandSetStream(curand_, stream_);
    }
};

// ---------------------------------------------------------------------------
// default_context() — process-wide singleton GPUContext (device 0).
//
// Thread-safe via std::call_once. Use only for top-level entry points and
// tests; pass the returned reference down to every kernel.
//
// WHY singleton: singlet-gpu is single-GPU by default. Multi-GPU callers
// construct explicit GPUContext objects per device.
// ---------------------------------------------------------------------------
inline GPUContext& default_context() {
    static std::once_flag flag;
    static std::unique_ptr<GPUContext> ctx;
    std::call_once(flag, [](){
        ctx = std::make_unique<GPUContext>();
    });
    if (!ctx) throw std::runtime_error(
        "singlet_gpu::core::default_context() failed to initialize");
    return *ctx;
}

}  // namespace core
}  // namespace singlet_gpu
