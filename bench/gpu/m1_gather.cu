// M1: random-gather and dependent-chase roofline on one GPU (docs/gpu-engine/exploration.md, section 9, M1).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -o m1_gather m1_gather.cu
// Run:   ./m1_gather [--gb 1,8,16,32,48,64,80] [--reps 3] [--alloc malloc|pool|both]
// Output: one JSON object per line (stdout); a human summary goes to stderr.
//
// Question: how many random 32-byte HBM sectors per second does an H100 NVL sustain when the footprint is the size of
// the STAR index (24-32 GB), and how does that change with access size, footprint (TLB reach), the number of independent
// chains per thread, occupancy, the allocator, and an L2 persistence window? The throughput model assumes
// RS_clean = 23 G sectors/s (bench/gpu/throughput_model.py).
//
// Kernels:
//   gather:   every thread issues `iters` independent loads at hashed addresses (SAi probes, locator probes, U-array bytes).
//             W words of 8 B per access: W=1 -> 8 B, W=2 -> 16 B, W=4 -> one 32 B sector.
//   chase:    every thread follows CH independent chains of `steps` dependent loads (the SA binary search). The next
//             address is a hash of the loaded word, the previous address and the step number, so the walk is a true
//             random walk (no short cycles that would sit in L2).
//   gather_hot: as gather, but 1 access in 4 goes to a small hot region (models the SAi top levels); used for the L2
//             persistence variant (access policy window over the hot region) against the same kernel without a window.
//
// Changes vs the first sketch (all minimal): range reduction by umulhi instead of 64-bit `%`; chase successor depends on the
// step number (the original fill gave a functional graph whose chains fall into ~sqrt(N)-sector cycles that fit in L2);
// chains templated (1,2,4,8); occupancy sweep and 3 timed repeats after a warm-up; cudaMalloc vs cudaMallocAsync pool;
// persistence variant; JSON output; footprint sweep inside one process.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint64_t mix64(uint64_t k) {   // fmix64 (same finalizer as nonhost/min_sketch.h)
    k ^= k >> 33; k *= 0xff51afd7ed558ccdULL; k ^= k >> 33; k *= 0xc4ceb9fe1a85ec53ULL; k ^= k >> 33; return k;
}
__device__ __forceinline__ uint64_t red(uint64_t h, uint64_t n) { return __umul64hi(h, n); }   // h*n >> 64, in [0,n)

template <int W>
__global__ void gather(const uint64_t* __restrict__ a, uint64_t nSectors, int iters, uint64_t* sink) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, acc = 0;
#pragma unroll 8
    for (int i = 0; i < iters; i++) {
        uint64_t s = red(mix64(tid * 0x9e3779b97f4a7c15ULL + i), nSectors);
        const uint64_t* p = a + s * 4;
#pragma unroll
        for (int w = 0; w < W; w++) acc ^= __ldg(p + w);
    }
    if (acc == 0x12345) sink[0] = acc;
}

// 1 access in 4 goes to the first hotSectors sectors, the others over the whole footprint.
__global__ void gather_hot(const uint64_t* __restrict__ a, uint64_t nSectors, uint64_t hotSectors, int iters, uint64_t* sink) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x, acc = 0;
#pragma unroll 8
    for (int i = 0; i < iters; i++) {
        uint64_t h = mix64(tid * 0x9e3779b97f4a7c15ULL + i);
        uint64_t s = (i & 3) == 0 ? red(h, hotSectors) : red(h, nSectors);
        const uint64_t* p = a + s * 4;
        acc ^= __ldg(p) ^ __ldg(p + 1) ^ __ldg(p + 2) ^ __ldg(p + 3);
    }
    if (acc == 0x12345) sink[0] = acc;
}

template <int CH>
__global__ void chase(const uint64_t* __restrict__ a, uint64_t nSectors, int steps, uint64_t* sink) {
    uint64_t tid = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    uint64_t cur[CH];
#pragma unroll
    for (int c = 0; c < CH; c++) cur[c] = red(mix64(tid * 8 + c + 1), nSectors);
    for (int s = 0; s < steps; s++) {
#pragma unroll
        for (int c = 0; c < CH; c++) {
            uint64_t v = __ldg(a + cur[c] * 4);                       // the dependent load
            cur[c] = red(mix64(v + cur[c] * 31 + (uint64_t)s), nSectors);
        }
    }
    uint64_t acc = 0;
#pragma unroll
    for (int c = 0; c < CH; c++) acc ^= cur[c];
    if (acc == 0x12345) sink[0] = acc;
}

__global__ void fill(uint64_t* a, uint64_t nSectors) {
    uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    for (; i < nSectors; i += (uint64_t)gridDim.x * blockDim.x) {
        a[i * 4] = mix64(i ^ 0x9e3779b97f4a7c15ULL);
        a[i * 4 + 1] = a[i * 4 + 2] = a[i * 4 + 3] = i;
    }
}

static int g_sms = 132, g_reps = 3;
static cudaStream_t g_stream;

template <class F> static std::vector<float> timed(F launch) {      // warm-up + g_reps timed runs, ms each
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    launch(); CK(cudaStreamSynchronize(g_stream));
    std::vector<float> v;
    for (int r = 0; r < g_reps; r++) {
        float ms;
        cudaEventRecord(t0, g_stream); launch(); cudaEventRecord(t1, g_stream);
        CK(cudaEventSynchronize(t1)); cudaEventElapsedTime(&ms, t0, t1); v.push_back(ms);
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return v;
}
static std::string arr(const std::vector<double>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); i++) { char b[64]; snprintf(b, 64, "%s%.4f", i ? "," : "", v[i]); s += b; }
    return s + "]";
}
static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

static void chase_run(const char* alloc, double gb, uint64_t* a, uint64_t nS, uint64_t* sink, int ch, int occ) {
    const int threads = 256, steps = 512;
    int b = g_sms * occ;
    auto ms = timed([&] {
        switch (ch) {
            case 1: chase<1><<<b, threads, 0, g_stream>>>(a, nS, steps, sink); break;
            case 2: chase<2><<<b, threads, 0, g_stream>>>(a, nS, steps, sink); break;
            case 4: chase<4><<<b, threads, 0, g_stream>>>(a, nS, steps, sink); break;
            default: chase<8><<<b, threads, 0, g_stream>>>(a, nS, steps, sink); break;
        }
    });
    CK(cudaGetLastError());
    std::vector<double> gl, ns;                                   // G loads/s; ns per dependent step (per chain)
    for (float m : ms) { gl.push_back((double)b * threads * steps * ch / m / 1e6); ns.push_back(m * 1e6 / steps); }
    printf("{\"kind\":\"chase\",\"alloc\":\"%s\",\"gb\":%.1f,\"chains\":%d,\"blocks_per_sm\":%d,\"gloads_s\":%s,\"median_gloads_s\":%.3f,"
           "\"ns_per_step\":%s,\"median_ns_per_step\":%.1f}\n", alloc, gb, ch, occ, arr(gl).c_str(), median(gl), arr(ns).c_str(), median(ns));
    fflush(stdout);
}

static void run_footprint(const char* alloc, double gb, bool persist_tests) {
    uint64_t nS = (uint64_t)(gb * 1e9) / 32;
    uint64_t *a = nullptr, *sink = nullptr;
    CK(cudaMalloc(&sink, 8));
    if (!strcmp(alloc, "pool")) CK(cudaMallocAsync((void**)&a, nS * 32, g_stream)); else CK(cudaMalloc(&a, nS * 32));
    CK(cudaStreamSynchronize(g_stream));
    fill<<<4096, 256, 0, g_stream>>>(a, nS); CK(cudaStreamSynchronize(g_stream));
    const int threads = 256, iters = 1024, blocks = g_sms * 8;     // 8 blocks x 256 = 2048 threads/SM: full occupancy

    for (int bytes : {8, 16, 32}) {
        auto ms = timed([&] {
            if (bytes == 8) gather<1><<<blocks, threads, 0, g_stream>>>(a, nS, iters, sink);
            else if (bytes == 16) gather<2><<<blocks, threads, 0, g_stream>>>(a, nS, iters, sink);
            else gather<4><<<blocks, threads, 0, g_stream>>>(a, nS, iters, sink);
        });
        CK(cudaGetLastError());
        std::vector<double> g;
        for (float m : ms) g.push_back((double)blocks * threads * iters / m / 1e6);   // G accesses/s (= G sectors/s: 1 per access)
        printf("{\"kind\":\"gather\",\"alloc\":\"%s\",\"gb\":%.1f,\"bytes\":%d,\"gacc_s\":%s,\"median_gacc_s\":%.3f}\n",
               alloc, gb, bytes, arr(g).c_str(), median(g));
        fflush(stdout);
    }
    for (int ch : {1, 2, 4, 8}) for (int occ : {1, 4, 16}) chase_run(alloc, gb, a, nS, sink, ch, occ);

    if (persist_tests) {                                          // L2 persistence variant (malloc allocation only)
        int maxPersist = 0, maxWin = 0, l2 = 0;
        cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize, 0);
        cudaDeviceGetAttribute(&maxWin, cudaDevAttrMaxAccessPolicyWindowSize, 0);
        cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0);
        size_t hotBytes = std::min<size_t>(std::min<size_t>(maxPersist, maxWin), 32u << 20);
        uint64_t hotS = hotBytes / 32;
        for (int mode = 0; mode < 2; mode++) {                    // 0: no window, 1: persisting window over the hot region
            if (mode == 1) {
                CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, hotBytes));
                cudaStreamAttrValue at; memset(&at, 0, sizeof(at));
                at.accessPolicyWindow.base_ptr = a; at.accessPolicyWindow.num_bytes = hotBytes;
                at.accessPolicyWindow.hitRatio = 1.0f;
                at.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
                at.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
                CK(cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &at));
            }
            auto ms = timed([&] { gather_hot<<<blocks, threads, 0, g_stream>>>(a, nS, hotS, iters, sink); });
            CK(cudaGetLastError());
            std::vector<double> g;
            for (float m : ms) g.push_back((double)blocks * threads * iters / m / 1e6);
            printf("{\"kind\":\"gather_hot\",\"alloc\":\"%s\",\"gb\":%.1f,\"persist_window\":%d,\"hot_mb\":%.1f,\"hot_share\":0.25,"
                   "\"l2_mb\":%.1f,\"max_persist_mb\":%.1f,\"gacc_s\":%s,\"median_gacc_s\":%.3f}\n",
                   alloc, gb, mode, hotBytes / 1048576.0, l2 / 1048576.0, maxPersist / 1048576.0, arr(g).c_str(), median(g));
            fflush(stdout);
        }
        cudaStreamAttrValue off; memset(&off, 0, sizeof(off));      // clear the window and reset persisting lines
        cudaStreamSetAttribute(g_stream, cudaStreamAttributeAccessPolicyWindow, &off);
        cudaCtxResetPersistingL2Cache();
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0);
    }
    if (!strcmp(alloc, "pool")) { CK(cudaFreeAsync(a, g_stream)); CK(cudaStreamSynchronize(g_stream)); } else CK(cudaFree(a));
    CK(cudaFree(sink));
}

int main(int argc, char** argv) {
    std::vector<double> gbs = {1, 8, 16, 32, 48, 64, 80};
    std::string alloc = "both";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--gb") && i + 1 < argc) {
            gbs.clear(); char* s = strdup(argv[++i]);
            for (char* t = strtok(s, ","); t; t = strtok(nullptr, ",")) gbs.push_back(atof(t));
        } else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--alloc") && i + 1 < argc) alloc = argv[++i];
    }
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
    g_sms = p.multiProcessorCount;
    int rt = 0, drv = 0; cudaRuntimeGetVersion(&rt); cudaDriverGetVersion(&drv);
    CK(cudaStreamCreate(&g_stream));
    int poolSupported = 0; cudaDeviceGetAttribute(&poolSupported, cudaDevAttrMemoryPoolsSupported, 0);
    cudaMemPool_t pool; cudaDeviceGetDefaultMemPool(&pool, 0);
    uint64_t thr = ~0ULL; cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
    printf("{\"kind\":\"device\",\"name\":\"%s\",\"sms\":%d,\"l2_bytes\":%d,\"mem_bytes\":%zu,\"mem_clock_khz\":%d,\"sm_clock_khz\":%d,"
           "\"bus_bits\":%d,\"runtime\":%d,\"driver\":%d,\"pools\":%d,\"reps\":%d}\n", p.name, g_sms, p.l2CacheSize, p.totalGlobalMem,
           p.memoryClockRate, p.clockRate, p.memoryBusWidth, rt, drv, poolSupported, g_reps);
    fflush(stdout);
    for (double gb : gbs) {
        if (alloc == "malloc" || alloc == "both") run_footprint("malloc", gb, true);
        if ((alloc == "pool" || alloc == "both") && poolSupported) run_footprint("pool", gb, false);
    }
    return 0;
}
