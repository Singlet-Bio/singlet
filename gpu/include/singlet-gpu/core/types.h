// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/core/types.h
//
// Re-exports factornet GPU type aliases under singlet_gpu::core.
// factornet::gpu::SparseMatrixGPU<float> IS our DeviceCSC — no wrapper needed,
// no extra indirection. Downstream kernels and the pz loader both include this
// header; they should never spell out factornet:: directly.

#pragma once

// FACTORNET_HAS_GPU=1 must be defined before any factornet GPU header.
// The CMake INTERFACE target (singlet-gpu::singlet-gpu) sets this globally.
#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif

#include <factornet/gpu/types.cuh>

namespace singlet_gpu {
namespace core {

// Primary sparse input type for all singlet-gpu kernels.
// Layout: CSC, int32 indptr (size n+1) + int32 indices (size nnz) + float values.
using DeviceCSC   = factornet::gpu::SparseMatrixGPU<float>;

// Column-major dense output / factor matrix type.
using DeviceDense = factornet::gpu::DenseMatrixGPU<float>;

// RAII device allocation; used inside DeviceCSC and DeviceDense.
// Exposed here so kernels can allocate scratch buffers without a raw cudaMalloc.
template<typename T>
using DeviceMemory = factornet::gpu::DeviceMemory<T>;

}  // namespace core
}  // namespace singlet_gpu
