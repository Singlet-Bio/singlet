// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/core/handles.h
//
// Re-exports factornet::gpu::GPUContext as singlet_gpu::core::GPUContext and
// provides a process-wide default context via core::default_context().
//
// GPUContext owns: cuBLAS handle, cuSPARSE handle, cuSOLVER handle, and one
// CUDA stream, all bound together. Every singlet-gpu kernel receives a
// GPUContext reference from its caller — never creates its own handles.
//
// default_context() returns a singleton GPUContext bound to device 0.
// Multi-GPU code should maintain per-device contexts and pass them explicitly.

#pragma once

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/gpu/types.cuh>

#include <mutex>
#include <memory>
#include <stdexcept>

namespace singlet_gpu {
namespace core {

// Direct re-export. All kernel headers use singlet_gpu::core::GPUContext.
using GPUContext = factornet::gpu::GPUContext;

// ---------------------------------------------------------------------------
// default_context() — process-wide singleton GPUContext (device 0).
//
// Initialized on first call, destroyed at program exit. Thread-safe.
// Use this only for top-level user-facing entry points and tests; pass the
// resulting reference down to every kernel so they share handles + stream.
//
// WHY singleton rather than a pool: singlet-gpu is a single-GPU library by
// default. Multi-GPU callers construct explicit GPUContext objects per device
// and pass them in; the singleton is just convenience for the single-GPU path.
// ---------------------------------------------------------------------------
inline GPUContext& default_context() {
    // Guarded by a call-once flag; construction may throw on CUDA errors,
    // which propagates to the first caller. Subsequent callers see the
    // initialized object.
    static std::once_flag flag;
    static std::unique_ptr<GPUContext> ctx;
    std::call_once(flag, [](){
        ctx = std::make_unique<GPUContext>();
    });
    if (!ctx) {
        throw std::runtime_error("singlet_gpu::core::default_context() failed to initialize");
    }
    return *ctx;
}

}  // namespace core
}  // namespace singlet_gpu
