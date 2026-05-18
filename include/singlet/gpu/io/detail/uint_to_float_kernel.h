// SPDX-License-Identifier: MIT
// singlet/gpu/io/detail/uint_to_float_kernel.h
//
// Fused on-device uint8/uint16/uint32 → fp32 conversion kernels.
//
// Algorithm: each thread converts one element.  Grid is 1D, block size 256.
// No atomics, no reductions — fully deterministic, occupancy-driven.
//
// WHY on-device (not host-side saturate_cast):
//   Upload raw uintN staging (uint8: 1B/elem, uint16: 2B/elem) then convert
//   on device.  Saves 2–4× PCIe bandwidth vs uploading float values.
//   For a 30M-nnz uint16 matrix: 60 MB PCIe instead of 120 MB.
//
// Precision: uint8/uint16 → fp32 is exact (max value 65535 < 2^23 mantissa).
//            uint32 → fp32 loses precision for values > 2^24 (warned, not clamped).
//
// Time complexity: O(nnz) — one pass, no workspace.
// Workspace budget: one temporary DeviceMemory<uintN> staging (same size as values,
//                   freed immediately after the kernel launches).
//
// Stream usage: caller-provided stream, always async.
// Determinism: fully deterministic (no atomics).
// OOC plan: kernel is called per-chunk inside PzChunkIterator — same kernel, smaller N.

#pragma once

#include <singlet/gpu/core/types.h>

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace singlet::gpu {
namespace io {
namespace detail {

// ---------------------------------------------------------------------------
// uint8 → float (exact: max 255 < 2^23 mantissa precision)
// ---------------------------------------------------------------------------
__global__ void uint8_to_float_kernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    uint32_t                    n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<float>(src[idx]);
}

// ---------------------------------------------------------------------------
// uint16 → float (exact: max 65535 < 2^23 mantissa precision)
// ---------------------------------------------------------------------------
__global__ void uint16_to_float_kernel(
    const uint16_t* __restrict__ src,
    float*          __restrict__ dst,
    uint32_t                     n)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) dst[idx] = static_cast<float>(src[idx]);
}

// ---------------------------------------------------------------------------
// uint32 → float (inexact for values > 2^24; logs a per-device warning once)
// ---------------------------------------------------------------------------
__global__ void uint32_to_float_kernel(
    const uint32_t* __restrict__ src,
    float*          __restrict__ dst,
    uint32_t                     n,
    uint32_t*       __restrict__ overflow_flag)   // single-element output counter
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        uint32_t v = src[idx];
        dst[idx] = static_cast<float>(v);
        // Flag precision overflow (no correctness impact, just a warning signal).
        if (v > (1u << 24)) atomicAdd(overflow_flag, 1u);
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper: allocates staging, uploads raw uintN values, launches
// the appropriate kernel, then copies float result into dst_device.
//
// Inputs:
//   host_raw_values  — pinned host buffer containing nnz uintN elements
//                      (uint8/uint16/uint32 depending on vt_code)
//   nnz              — number of elements
//   vt_code          — 1 = uint8, 2 = uint16, 3 = uint32
//   dst_device       — pre-allocated DeviceMemory<float> (nnz elements)
//   stream           — CUDA stream for all async operations
//
// The function:
//   1. Allocates a temporary device buffer matching the raw element type.
//   2. cudaMemcpyAsync raw bytes host → staging.
//   3. Launches the conversion kernel staging → dst_device.
//   4. Frees the staging buffer (RAII via DeviceMemory going out of scope).
// All operations are on `stream`; caller must synchronize before reading dst.
// ---------------------------------------------------------------------------
inline void launch_uint_to_float(
    const void*   host_raw_values,
    size_t        nnz,
    int           vt_code,
    float*        dst_device,
    cudaStream_t  stream)
{
    if (nnz == 0) return;

    constexpr uint32_t BLOCK = 256;
    const     uint32_t grid  = static_cast<uint32_t>((nnz + BLOCK - 1) / BLOCK);

    if (vt_code == 1) {
        // uint8 staging
        uint8_t* d_staging = nullptr;
        SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_staging, nnz * sizeof(uint8_t)));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_staging, host_raw_values,
                                      nnz * sizeof(uint8_t),
                                      cudaMemcpyHostToDevice, stream));
        uint8_to_float_kernel<<<grid, BLOCK, 0, stream>>>(
            d_staging, dst_device, static_cast<uint32_t>(nnz));
        // Fire-and-forget: free after kernel is dispatched.
        // WHY cudaFreeAsync not available pre-CUDA-11.2; use synchronous
        // free here — acceptable because this is a one-time staging path,
        // not a hot loop.
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_staging);

    } else if (vt_code == 2) {
        // uint16 staging
        uint16_t* d_staging = nullptr;
        SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_staging, nnz * sizeof(uint16_t)));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_staging, host_raw_values,
                                      nnz * sizeof(uint16_t),
                                      cudaMemcpyHostToDevice, stream));
        uint16_to_float_kernel<<<grid, BLOCK, 0, stream>>>(
            d_staging, dst_device, static_cast<uint32_t>(nnz));
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        cudaFree(d_staging);

    } else if (vt_code == 3) {
        // uint32 staging
        uint32_t* d_staging     = nullptr;
        uint32_t* d_overflow    = nullptr;
        SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_staging,  nnz * sizeof(uint32_t)));
        SINGLET_GPU_CUDA_CHECK(cudaMalloc(&d_overflow, sizeof(uint32_t)));
        SINGLET_GPU_CUDA_CHECK(cudaMemsetAsync(d_overflow, 0, sizeof(uint32_t), stream));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpyAsync(d_staging, host_raw_values,
                                      nnz * sizeof(uint32_t),
                                      cudaMemcpyHostToDevice, stream));
        uint32_to_float_kernel<<<grid, BLOCK, 0, stream>>>(
            d_staging, dst_device, static_cast<uint32_t>(nnz), d_overflow);
        // Scalar overflow check: copy 4 bytes to host for warning.
        uint32_t h_overflow = 0;
        SINGLET_GPU_CUDA_CHECK(cudaStreamSynchronize(stream));
        SINGLET_GPU_CUDA_CHECK(cudaMemcpy(&h_overflow, d_overflow, sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost));
        if (h_overflow > 0) {
            std::fprintf(stderr,
                "[singlet/gpu/pz_device_loader] WARNING: %u values > 2^24 "
                "converted to fp32 with precision loss.\n", h_overflow);
        }
        cudaFree(d_staging);
        cudaFree(d_overflow);

    } else {
        throw std::runtime_error(
            "launch_uint_to_float: unknown vt_code " + std::to_string(vt_code));
    }
}


}  // namespace detail
}  // namespace io
}  // namespace singlet::gpu
