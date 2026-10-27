// M1: random-gather and dependent-chase roofline on one GPU (docs/gpu-engine/exploration.md, section 8, M1).
//
// Sketch, not wired into CMake. Build: nvcc -O3 -arch=sm_90 -o m1_gather m1_gather.cu
// Run:   ./m1_gather <footprint_GB> [bytes_per_access=32] [chains_per_thread=1]
//
// Question: how many random 32-byte HBM sectors per second does an H100 NVL sustain when the footprint is the size of
// the STAR index (24-32 GB), and how does that change with access size, footprint (TLB reach) and the number of
// independent chains per thread? The throughput model assumes RS_clean = 23 G sectors/s (bench/gpu/throughput_model.py).
//
// Two kernels:
//   gather: every thread issues `iters` independent loads at hashed addresses (models SAi probes, locator probes,
//           U-array bytes: independent random accesses).
//   chase:  every thread follows `chains` independent pointer chains of `steps` dependent loads (models the SA binary
//           search: each step's address depends on the previous load). Reports ns per dependent step and sectors/s.
// Pass (M1): gather >= 15 G sectors/s at 32 GB and 32 B accesses; kill the model's base scenario if < 8 G/s.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t mix64(uint64_t k) {   // fmix64 (same finalizer as nonhost/min_sketch.h)
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k;
}

// Independent gathers. W words of 8 bytes per access (W=1: 8 B, W=4: one full 32 B sector).
template <int W>
__global__ void gather(const uint64_t* __restrict__ a, uint64_t nSectors, int iters, uint64_t* sink) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, acc = 0;
    for (int i = 0; i < iters; i++) {
        uint64_t s = mix64(tid * 1315423911ULL + i) % nSectors;          // random sector
        const uint64_t* p = a + s * 4;
#pragma unroll
        for (int w = 0; w < W; w++) acc ^= __ldg(p + w);
    }
    if (acc == 0x12345) sink[0] = acc;                                    // keep loads alive
}

// Dependent chase: next index = a[cur] (the array holds a random cycle over sectors, one 8-B word per sector).
__global__ void chase(const uint64_t* __restrict__ a, uint64_t nSectors, int steps, int chains, uint64_t* sink) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t cur[8];
    for (int c = 0; c < chains; c++) cur[c] = mix64(tid * 8 + c) % nSectors;
    for (int s = 0; s < steps; s++)
        for (int c = 0; c < chains; c++) cur[c] = __ldg(a + cur[c] * 4) % nSectors;
    uint64_t acc = 0;
    for (int c = 0; c < chains; c++) acc ^= cur[c];
    if (acc == 0x12345) sink[0] = acc;
}

// Fill: word 0 of every sector holds a pseudo-random successor (not a single cycle; good enough for a roofline).
__global__ void fill(uint64_t* a, uint64_t nSectors) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    for (; i < nSectors; i += (uint64_t)gridDim.x * blockDim.x) {
        a[i * 4] = mix64(i ^ 0x9e3779b97f4a7c15ULL) % nSectors;
        a[i * 4 + 1] = a[i * 4 + 2] = a[i * 4 + 3] = i;
    }
}

int main(int argc, char** argv) {
    double gb = argc > 1 ? atof(argv[1]) : 32.0;
    int bytes = argc > 2 ? atoi(argv[2]) : 32;
    int chains = argc > 3 ? atoi(argv[3]) : 1;
    uint64_t nSectors = (uint64_t)(gb * 1e9) / 32;
    uint64_t* a; uint64_t* sink;
    CK(cudaMalloc(&a, nSectors * 32)); CK(cudaMalloc(&sink, 8));
    fill<<<4096, 256>>>(a, nSectors); CK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    const int blocks = 132 * 16, threads = 256, iters = 256;          // 540k threads: far above Little's-law need
    float ms;
    cudaEventRecord(t0);
    if (bytes <= 8) gather<1><<<blocks, threads>>>(a, nSectors, iters, sink);
    else if (bytes <= 16) gather<2><<<blocks, threads>>>(a, nSectors, iters, sink);
    else gather<4><<<blocks, threads>>>(a, nSectors, iters, sink);
    cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1);
    double n = (double)blocks * threads * iters;
    printf("gather footprint=%.1fGB access=%dB: %.2f G accesses/s (%.0f GB/s of sectors touched)\n",
           gb, bytes, n / ms / 1e6, n * 32 / ms / 1e6);

    const int steps = 64;
    for (int occ : {1, 4, 16}) {                                       // blocks per SM: occupancy sweep
        int b = 132 * occ;
        cudaEventRecord(t0);
        chase<<<b, threads>>>(a, nSectors, steps, chains, sink);
        cudaEventRecord(t1); CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1);
        double loads = (double)b * threads * steps * chains;
        printf("chase  footprint=%.1fGB chains=%d blocks/SM=%d: %.2f G loads/s, %.0f ns per dependent step\n",
               gb, chains, occ, loads / ms / 1e6, ms * 1e6 / steps);
    }
    return 0;
}
