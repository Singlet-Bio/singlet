// M5: STAR's mapping after the seed search, on the GPU, checked byte for byte against STAR (stardump M5 truth).
// docs/gpu-engine/exploration.md §12.13. The port itself: stitch.cuh.
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -Xcompiler -fopenmp -o m5 m5.cu
// Run:   m5 --genome-dir STAR_INDEX --truth PREFIX [--reps 3] [--json OUT] [--blocks-per-sm 4] [--threads 128]
//          PREFIX: stardump's STARDUMP_M5 prefix (PREFIX.<tid>.bin, PREFIX.params.txt, PREFIX.genome.bin; m5_dump.h)
//
// Every truth read (a distinct cDNA sequence after clipping, as STAR mapped it) runs through mapRead<LaneA> on the GPU (overflow: mapRead<LaneB>),
// one read per thread. Reads that exceed a main-lane capacity run through mapRead<Heavy> (STAR's own limits) on the host
// with the same source. The result (the sequence cache's encoding of STAR's mapping state, plus unmapType) is compared
// with STAR's: identical or not, per read and weighted by the read multiplicity.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <glob.h>
#include <sys/stat.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <numeric>
#include "stitch.cuh"
#ifndef M5_GIT
#define M5_GIT "unknown"
#endif

using namespace m5;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

//                 LREAD NSPLIT    NP     NW  NWA      NWB NEX NWTR NKEEP    NBLOB
#ifndef LA_NW
#define LA_NW 16
#define LA_NWA 16
#define LA_NWB 256
#endif
typedef Caps<      160,     4,   32, LA_NW, LA_NWA, LA_NWB, 8,   8,    8,     640> LaneA;   // per thread, every read
typedef Caps<      160,     4,   32,    64,   8,    1024,  8,   8,   32,    2048> LaneM;   // A's overflow: many windows, few aligns each
typedef Caps<      160,     8,  128,   128,  50,    4096, 20,  32,   64,    4096> LaneB;   // A's overflow; NWA = seedPerWindowNmax
typedef Caps<      650,    50, 1000, 10000,  50, 1 << 20, 20, 100, 4096, 1 << 20> Heavy;   // host: STAR's limits (seedPerReadNmax,
                                                                                            // alignWindowsPerReadNmax, MAX_N_EXONS, alignTranscriptsPerWindowNmax)
// ---------------------------------------------------------------------------------------------------------------------
__device__ unsigned long long g_clk[8], g_scan;
template <class C>
__device__ void lane_loop(const Params& P, const GenomeView& g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 n,
                          u64 tid, u64 nt, Work<C>* wp, ReadBuf<C::LREAD>* bp, uint8_t* out, uint16_t* outLen, uint8_t* status, uint8_t* why, int8_t* unmap);
// one lane: reads idx[k] (or k), work either per thread in global memory or in local memory (LOCAL)
template <class C, bool LOCAL>
__global__ void k_lane(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 n,
                       Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* out, uint16_t* outLen, uint8_t* status, uint8_t* why, int8_t* unmap) {
    u64 tid = blockIdx.x * (u64)blockDim.x + threadIdx.x, nt = (u64)gridDim.x * blockDim.x;
    if (tid >= n) return;
    if constexpr (LOCAL) {
        Work<C> wl_; ReadBuf<C::LREAD> bl_;
        lane_loop<C>(P, g, rb, roff, rlen, idx, n, tid, nt, &wl_, &bl_, out, outLen, status, why, unmap);
    } else {
        lane_loop<C>(P, g, rb, roff, rlen, idx, n, tid, nt, &work[tid], &rbuf[tid], out, outLen, status, why, unmap);
    }
}
template <class C>
__device__ void lane_loop(const Params& P, const GenomeView& g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 n,
                          u64 tid, u64 nt, Work<C>* wp, ReadBuf<C::LREAD>* bp, uint8_t* out, uint16_t* outLen, uint8_t* status, uint8_t* why, int8_t* unmap) {
    wp->wbEpoch = 0;
    for (int c = 0; c < 8; c++) wp->clk[c] = 0;
    wp->nScan = 0;
    for (u64 k = tid; k < n; k += nt) {
        u64 i = idx ? idx[k] : k;
        Blob bl{out + k * C::NBLOB, 0, C::NBLOB, false};
        Status s = mapRead(P, g, *wp, *bp, rb + roff[i], rlen[i], bl);
        status[k] = (uint8_t)s; why[k] = (uint8_t)wp->ov; outLen[k] = (uint16_t)bl.n; unmap[k] = (int8_t)wp->unmapType;
    }
    for (int c = 0; c < 8; c++) atomicAdd(&g_clk[c], (unsigned long long)wp->clk[c]);
    atomicAdd(&g_scan, (unsigned long long)wp->nScan);
}

struct LaneOut { std::vector<uint8_t> out, status, why; std::vector<uint16_t> len; std::vector<int8_t> unmap; double ms = 0; std::vector<double> runs; double phaseMs[3] = {0, 0, 0};
                 double tail[8] = {0, 0, 0, 0, 0, 0, 0, 0}; double workGB = 0, usedGB = 0; u64 workThreads = 0, fallback = 0; int occBlocks = 0; std::string mode; };

static double medianv(std::vector<double> v) { std::sort(v.begin(), v.end()); return v.empty() ? 0 : v[v.size() / 2]; }

template <class C>
static LaneOut runLane(const Params& P, const GenomeView& D, const char* dRb, const u64* dRoff, const uint16_t* dRlen, const uint32_t* dIdx,
                       u64 n, int blocks, int nthr, int reps, bool local, cudaEvent_t e0, cudaEvent_t e1, const char* name) {
    LaneOut L;
    u64 nt = (u64)blocks * nthr;
    if (nt > n) { nt = n; blocks = (int)((n + nthr - 1) / nthr); nt = (u64)blocks * nthr; }
    uint8_t* dOut; CK(cudaMalloc(&dOut, n * (u64)C::NBLOB));
    uint16_t* dLen; CK(cudaMalloc(&dLen, n * 2)); uint8_t* dSt; CK(cudaMalloc(&dSt, n)); uint8_t* dWhy; CK(cudaMalloc(&dWhy, n));
    int8_t* dUn; CK(cudaMalloc(&dUn, n));
    Work<C>* dWork = nullptr; ReadBuf<C::LREAD>* dRbuf = nullptr;
    if (!local) { CK(cudaMalloc(&dWork, nt * sizeof(Work<C>))); CK(cudaMalloc(&dRbuf, nt * sizeof(ReadBuf<C::LREAD>)));
                  CK(cudaMemset(dWork, 0, nt * sizeof(Work<C>))); }
    fprintf(stderr, "lane %s: %llu reads, %d blocks x %d threads, work %zu B/thread (%s, %.2f GB)\n", name, n, blocks, nthr, sizeof(Work<C>),
            local ? "local" : "global", nt * (double)sizeof(Work<C>) / 1e9);
    unsigned long long clk[8] = {0};
    for (int r = 0; r < reps + 1; r++) {
        unsigned long long z[8] = {0}; CK(cudaMemcpyToSymbol(g_clk, z, sizeof z)); CK(cudaMemcpyToSymbol(g_scan, z, 8));
        cudaEventRecord(e0);
        if constexpr (sizeof(Work<C>) < 200000) {
            if (local) k_lane<C, true><<<blocks, nthr>>>(P, D, dRb, dRoff, dRlen, dIdx, n, dWork, dRbuf, dOut, dLen, dSt, dWhy, dUn);
        }
        if (!local)       k_lane<C, false><<<blocks, nthr>>>(P, D, dRb, dRoff, dRlen, dIdx, n, dWork, dRbuf, dOut, dLen, dSt, dWhy, dUn);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float m; cudaEventElapsedTime(&m, e0, e1); if (r > 0) L.runs.push_back(m);
        fprintf(stderr, "  lane %s run %d: %.1f ms\n", name, r, m);
    }
    CK(cudaMemcpyFromSymbol(clk, g_clk, sizeof clk));
    { unsigned long long sc; CK(cudaMemcpyFromSymbol(&sc, g_scan, 8)); fprintf(stderr, "  lane %s: %.1f SA entries scanned per read\n", name, (double)sc / n); }
    { double t = (double)clk[0] + clk[1] + clk[2] + clk[3]; if (t > 0)
        fprintf(stderr, "  lane %s thread cycles: seeds %.1f%%, windows %.1f%%, stitching %.1f%%, select+encode %.1f%%\n", name,
                100 * clk[0] / t, 100 * clk[1] / t, 100 * clk[2] / t, 100 * clk[3] / t); }
    L.ms = medianv(L.runs);
    L.out.resize(n * (u64)C::NBLOB); L.status.resize(n); L.why.resize(n); L.len.resize(n); L.unmap.resize(n);
    CK(cudaMemcpy(L.out.data(), dOut, L.out.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.len.data(), dLen, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.status.data(), dSt, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.why.data(), dWhy, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.unmap.data(), dUn, n, cudaMemcpyDeviceToHost));
    cudaFree(dOut); cudaFree(dLen); cudaFree(dSt); cudaFree(dWhy); cudaFree(dUn); if (dWork) cudaFree(dWork); if (dRbuf) cudaFree(dRbuf);
    return L;
}

// ---------------------------------------------------------------------------------------------------------------------
// staged lane: per batch of reads (one Work per read), phase 1 (thread per read), the assignment scan (warp per read,
// lanes over consecutive SA entries: coalesced) marking the entries that do something, phase 3 (thread per read:
// assignments in STAR's order from the marks, then stitching, selection, encoding)
template <class C>
__global__ void k_p1(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 b0, u64 n,
                     Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* status, u64* nEnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    u64 i = idx ? idx[b0 + k] : b0 + k;
    Work<C>& w = work[k];
    Status s = mapPhase1(P, g, w, rbuf[k], rb + roff[i], rlen[i]);
    status[k] = (uint8_t)s;
    nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
}
template <class C>
__global__ void k_p2(Params P, GenomeView g, u64 n, const Work<C>* work, const u64* nEnt, const u64* base, uint32_t* bits) {
    u64 wid = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= n || nEnt[wid] == 0) return;
    const Work<C>& w = work[wid];
    u64 off = base[wid];
    for (m5::uint iP = 0; iP < w.nP; iP++) {
        const u64 s0 = w.PC[iP][PC_SAstart], cnt = w.PC[iP][PC_SAend] - s0 + 1;
        for (u64 j = lane; j < cnt; j += 32)
            if (entryHit(P, g, w, iP, (m5::uint)(s0 + j))) atomicOr(&bits[(off + j) >> 5], 1u << ((off + j) & 31));
        off += cnt;
    }
}
// ---- E4 (docs/gpu-engine/fable/E4-order.md): scheduling instrumentation. %globaltimer is one clock for all SMs (ns).
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); return t; }
// per launch (batch): block start/end, warp end, the threads' busy time (us, summed), when the last read was taken.
// Arrays are indexed by blkOff + blockIdx.x (blocks of all batches of one run side by side); disabled when blkStart == nullptr.
struct Tail { unsigned long long *blkStart, *blkEnd, *warpEnd, *busyUs, *lastGrab; u64 blkOff, batch; };
__device__ __forceinline__ void tailBegin(const Tail& tl, unsigned long long& t0) {
    if (!tl.blkStart) return;
    t0 = gtime();
    if (threadIdx.x == 0) tl.blkStart[tl.blkOff + blockIdx.x] = t0;
}
__device__ __forceinline__ void tailEnd(const Tail& tl, unsigned long long busyNs) {
    if (!tl.blkStart) return;
    unsigned long long t1 = gtime();
    atomicMax(&tl.blkEnd[tl.blkOff + blockIdx.x], t1);
    atomicMax(&tl.warpEnd[(tl.blkOff + blockIdx.x) * (blockDim.x >> 5) + (threadIdx.x >> 5)], t1);
    unsigned m = __activemask();
    unsigned us = __reduce_add_sync(m, (unsigned)(busyNs / 1000));
    if ((threadIdx.x & 31) == __ffs(m) - 1) atomicAdd(&tl.busyUs[tl.batch], (unsigned long long)us);
}
// phase 3 of one read (k = index in the batch)
template <class C>
__device__ __forceinline__ void p3one(const Params& P, const GenomeView& g, u64 b0, u64 k, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf,
                                      const u64* nEnt, const u64* base, const uint32_t* bits, uint8_t* status, uint8_t* out, uint16_t* outLen,
                                      uint8_t* why, int8_t* unmap) {
    Work<C>& w = work[k];
    w.stepBudget = budget;
    u64 o = b0 + k;
    Status s = (Status)status[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
    status[k] = (uint8_t)s; why[o] = (uint8_t)w.ov; outLen[o] = (uint16_t)bl.n; unmap[o] = (int8_t)w.unmapType;
    for (int c = 0; c < 8; c++) { atomicAdd(&g_clk[c], (unsigned long long)w.clk[c]); w.clk[c] = 0; }
    atomicAdd(&g_scan, (unsigned long long)nEnt[k]);
}
template <class C>
__global__ void k_p3(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* nEnt, const u64* base,
                     const uint32_t* bits, uint8_t* status, uint8_t* out, uint16_t* outLen, uint8_t* why, int8_t* unmap, const uint32_t* perm, Tail tl) {
    unsigned long long t0 = 0; tailBegin(tl, t0);
    u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (t < n) {
        u64 k = perm ? perm[t] : t;   // --bin / --p3order: threads take the batch's reads in another order (each read is independent)
        p3one<C>(P, g, b0, k, budget, work, rbuf, nEnt, base, bits, status, out, outLen, why, unmap);
    }
    if (tl.blkStart) tailEnd(tl, t < n ? gtime() - t0 : 0);
}
// --queue: persistent phase 3. A grid of resident blocks; each warp takes the next reads of the batch (in perm order) from an
// atomic counter until none are left, so the batch no longer waits for its slowest block. Results are written by read index.
template <class C>
__global__ void k_p3q(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* nEnt, const u64* base,
                      const uint32_t* bits, uint8_t* status, uint8_t* out, uint16_t* outLen, uint8_t* why, int8_t* unmap, const uint32_t* perm,
                      unsigned long long* ctr, Tail tl) {
    unsigned long long t0 = 0; tailBegin(tl, t0);
    const unsigned lane = threadIdx.x & 31;
    unsigned long long busy = 0;
    for (;;) {
        unsigned m = __activemask();
        int leader = __ffs(m) - 1;
        unsigned long long b = 0;
        if ((int)lane == leader) b = atomicAdd(ctr, (unsigned long long)__popc(m));
        b = __shfl_sync(m, b, leader);
        u64 j = b + __popc(m & ((1u << lane) - 1));
        if (j >= n) break;
        unsigned long long ts = 0;
        if (tl.blkStart) { ts = gtime(); if (j == n - 1) tl.lastGrab[tl.batch] = ts; }
        u64 k = perm ? perm[j] : j;
        p3one<C>(P, g, b0, k, budget, work, rbuf, nEnt, base, bits, status, out, outLen, why, unmap);
        if (tl.blkStart) busy += gtime() - ts;
    }
    tailEnd(tl, busy);
}
// --fused: one persistent kernel per lane, Work per resident thread (not per read). Each warp takes the next 32 reads (in the
// lane's order) from an atomic counter: phase 1 (thread per read), the assignment scan of each of the 32 reads by the whole warp
// (k_p2's loop, marks in a per-thread bit buffer of capw words), phase 3 (thread per read). A read with more SA entries than the
// buffer holds assigns every entry itself (assignAll, the same result).
template <class C>
__global__ void k_fused(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 n, u64 budget,
                        Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint32_t* wbits, u64 capw, uint8_t* status, uint8_t* out, uint16_t* outLen,
                        uint8_t* why, int8_t* unmap, unsigned long long* ctr, unsigned long long* nFallback, Tail tl) {
    unsigned long long t0 = 0; tailBegin(tl, t0);
    const unsigned lane = threadIdx.x & 31;
    const u64 gt = blockIdx.x * (u64)blockDim.x + threadIdx.x, wb0 = gt - lane;
    Work<C>& w = work[gt];
    ReadBuf<C::LREAD>& rbk = rbuf[gt];
    uint32_t* myBits = wbits + gt * capw;
    const u64 capBits = capw * 32;
    for (int c = 0; c < 8; c++) w.clk[c] = 0;
    unsigned long long busy = 0, fb = 0, scan = 0;
    for (;;) {
        __syncwarp();
        unsigned long long b = 0;
        if (lane == 0) b = atomicAdd(ctr, 32ULL);
        b = __shfl_sync(0xffffffffu, b, 0);
        if (b >= n) break;
        unsigned long long ts = 0;
        if (tl.blkStart) { ts = gtime(); if (lane == 0 && n - 1 < b + 32) tl.lastGrab[tl.batch] = ts; }
        const u64 k = b + lane;
        const bool have = k < n;
        Status s = Status::Overflow; u64 nE = 0;
        if (have) {
            u64 i = idx ? idx[k] : k;
            s = mapPhase1(P, g, w, rbk, rb + roff[i], rlen[i]);
            nE = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
        }
        __syncwarp();
        for (int l = 0; l < 32; l++) {
            const u64 nEl = __shfl_sync(0xffffffffu, nE, l);
            if (nEl == 0 || nEl > capBits) continue;
            const Work<C>& wl = work[wb0 + l];
            uint32_t* bl = wbits + (wb0 + l) * capw;
            for (u64 q = lane; q < (nEl + 31) / 32; q += 32) bl[q] = 0;
            __syncwarp();
            u64 off = 0;
            for (m5::uint iP = 0; iP < wl.nP; iP++) {
                const u64 s0 = wl.PC[iP][PC_SAstart], cnt = wl.PC[iP][PC_SAend] - s0 + 1;
                for (u64 j0 = 0; j0 < cnt; j0 += 32) {
                    const u64 j = j0 + lane;
                    const bool hit = j < cnt && entryHit(P, g, wl, iP, (m5::uint)(s0 + j));
                    const unsigned mask = __ballot_sync(0xffffffffu, hit);
                    if (lane == 0 && mask) {
                        const u64 p = off + j0; const unsigned sh = (unsigned)(p & 31);
                        bl[p >> 5] |= mask << sh;
                        if (sh && (mask >> (32 - sh))) bl[(p >> 5) + 1] |= mask >> (32 - sh);
                    }
                }
                off += cnt;
            }
            __syncwarp();
        }
        if (have) {
            if (s == Status::Ok && w.stitched) {
                bool ok;
                if (nE <= capBits) ok = assignHits(P, g, w, myBits, 0);
                else { ok = assignAll(P, g, w); fb++; }
                if (!ok) s = Status::Overflow;
            }
            scan += nE;
            w.stepBudget = budget;
            Blob bl{out + k * C::NBLOB, 0, C::NBLOB, false};
            if (s == Status::Ok) s = mapFinish(P, g, w, rbk, bl);
            status[k] = (uint8_t)s; why[k] = (uint8_t)w.ov; outLen[k] = (uint16_t)bl.n; unmap[k] = (int8_t)w.unmapType;
            if (tl.blkStart) busy += gtime() - ts;
        }
    }
    for (int c = 0; c < 8; c++) atomicAdd(&g_clk[c], (unsigned long long)w.clk[c]);
    atomicAdd(&g_scan, scan);
    if (fb) atomicAdd(nFallback, fb);
    tailEnd(tl, busy);
}
// cost key of each read after phase 1, for --bin: windows, then SA entries to assign (reads that need no phase 3 first)
template <class C>
__global__ void k_cost(u64 n, const Work<C>* work, const u64* nEnt, u64* key, uint32_t* perm) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    key[k] = nEnt[k] ? ((u64)work[k].nW << 40) | min(nEnt[k], (u64)0xFFFFFFFFFFULL) : 0;
    perm[k] = (uint32_t)k;
}
// --p3order: other keys for the order in which phase-3 threads take a batch's reads (stable sort: ties keep the batch order)
enum P3Order { P3_NONE = 0, P3_COST_ASC, P3_COST_DESC, P3_LOCUS, P3_ANCHOR, P3_NWLOCUS, P3_NW_DESC };
static const char* p3orderName[] = {"none", "cost-asc", "cost-desc", "locus", "anchor", "nwlocus", "nw-desc"};
template <class C>
__global__ void k_key(int mode, Params P, GenomeView g, u64 n, const Work<C>* work, const u64* nEnt, u64* key, uint32_t* perm) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    perm[k] = (uint32_t)k;
    const Work<C>& w = work[k];
    const u64 ne = nEnt[k], nW = ne ? (u64)w.nW : 0;
    const u64 cost = ne ? (nW << 40) | min(ne, (u64)0xFFFFFFFFFFULL) : 0;
    u64 v = 0;
    if (mode == P3_COST_ASC) v = cost;
    else if (mode == P3_COST_DESC) v = ~cost;                                // most windows first, reads without phase 3 last
    else if (mode == P3_NW_DESC) v = ~(nW << 40);                            // windows only (ties: batch order)
    else if (mode == P3_LOCUS) v = nW ? (u64)w.WC[0][WC_gStart] : ~0ULL;     // genome bin of the first window
    else if (mode == P3_ANCHOR) {                                            // genome position of the first anchor's first locus
        v = ~0ULL;
        if (ne) for (m5::uint iP = 0; iP < w.nP; iP++) if (w.PC[iP][PC_Nrep] <= P.winAnchorMultimapNmax) {
            u64 a1 = packed_get(g.SA, w.PC[iP][PC_SAstart], g.saBits);
            u64 aStr = a1 >> g.GstrandBit; a1 &= g.GstrandMask;
            if (aStr == 1) a1 = g.nGenome - (w.PC[iP][PC_Length] + a1);
            v = a1; break;
        }
    } else if (mode == P3_NWLOCUS) {                                         // window-count class (1, 2, 3-4, 5-8, ...) descending, then bin
        u64 cls = nW ? 1 + (64 - __clzll((long long)(nW - 1))) : 0;
        v = ((15 - cls) << 44) | (nW ? (u64)w.WC[0][WC_gStart] : 0xFFFFFFFFFFFULL);
    }
    key[k] = v;
}
static bool g_bin = false;
static int g_p3order = P3_NONE;      // --p3order
static bool g_queue = false;         // --queue: persistent phase 3 per batch
static unsigned g_fused = 0;         // --fused A|M|B...: lanes run as one persistent kernel (bit 0 A, 1 M, 2 B)
static int g_fusedBps[3] = {0, 0, 0};// --fused-bps-a/m/b: resident blocks per SM (0 = occupancy limit)
static u64 g_fusedBits = 16384;      // --fused-bits: SA entries per read marked by the warp scan (more: assignAll)
static int g_queueBps = 0;           // --queue-bps: resident blocks per SM of k_p3q (0 = occupancy limit)
static bool g_tail = false;          // --tail: per-block / per-warp timing of phase 3
static int g_nSM = 132;
// per-lane scheduling statistics (the last timed run)
struct TailStats { double spanMs = 0, fill = 0, warpBal = 0, laneBal = 0, drain = 0, util = 0; int wmax = 0; u64 launches = 0; };
static TailStats tailStats(const std::vector<unsigned long long>& bs, const std::vector<unsigned long long>& be, const std::vector<unsigned long long>& we,
                           const std::vector<unsigned long long>& busyUs, const std::vector<unsigned long long>& lastGrab,
                           const std::vector<std::pair<u64, u64>>& launchBlocks, int wpb, int wmax, bool persistent) {
    // launchBlocks[b] = (first block, number of blocks) of launch b
    TailStats T; T.wmax = wmax; T.launches = launchBlocks.size();
    double span = 0, slot = 0, warpLife = 0, busy = 0, drain = 0;
    for (size_t b = 0; b < launchBlocks.size(); b++) {
        u64 f = launchBlocks[b].first, nbk = launchBlocks[b].second;
        unsigned long long s0 = ~0ULL, s1 = 0, lastStart = 0;
        for (u64 i = f; i < f + nbk; i++) {
            if (!be[i]) continue;
            s0 = std::min(s0, bs[i]); s1 = std::max(s1, be[i]); lastStart = std::max(lastStart, bs[i]);
            slot += (double)(be[i] - bs[i]) * wpb;
            for (int q = 0; q < wpb; q++) { u64 wi = i * wpb + q; if (we[wi] > bs[i]) warpLife += (double)(we[wi] - bs[i]); }
        }
        if (s1 <= s0) continue;
        span += (double)(s1 - s0);
        busy += (double)busyUs[b] * 1000.0;
        unsigned long long lg = persistent ? lastGrab[b] : lastStart;
        if (lg >= s0 && lg <= s1) drain += (double)(s1 - lg);
    }
    T.spanMs = span / 1e6;
    if (span > 0) { T.fill = slot / (wmax * span); T.drain = drain / span; T.util = busy / (32.0 * wmax * span); }
    if (slot > 0) T.warpBal = warpLife / slot;
    if (warpLife > 0) T.laneBal = busy / (32.0 * warpLife);
    return T;
}
static void printTail(const char* name, const TailStats& T) {
    fprintf(stderr, "  lane %s phase-3 schedule: %llu launches, span %.1f ms, resident warps %d; slot fill %.3f (tail+underfill %.1f%%), "
                    "drain after last read taken %.1f%%, warp balance in blocks %.3f, lane balance in warps %.3f, thread-slot utilisation %.3f\n",
            name, T.launches, T.spanMs, T.wmax, T.fill, 100 * (1 - T.fill), 100 * T.drain, T.warpBal, T.laneBal, T.util);
}
__global__ void k_copy_status(const uint8_t* s, uint8_t* dst, u64 n) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k < n) dst[k] = s[k];
}

static double devUsedGB() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); return (double)(t - f) / 1e9; }

template <class C>
static LaneOut runStaged(const Params& P, const GenomeView& D, const char* dRb, const u64* dRoff, const uint16_t* dRlen, const uint32_t* dIdx,
                         u64 n, u64 NB, u64 budget, int reps, cudaEvent_t e0, cudaEvent_t e1, const char* name) {
    LaneOut L;
    const int li = name[0] == 'A' ? 0 : name[0] == 'M' ? 1 : 2;
    const bool fused = (g_fused >> li) & 1;
    NB = std::min<u64>(NB, std::max<u64>(n, 1));
    const int T = 128, wpb = T / 32;
    // resident blocks per SM of the phase-3 kernel in use (grid of the persistent kernels; tail statistics)
    int occ = 0;
    if (fused) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_fused<C>, T, 0));
    else if (g_queue) CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_p3q<C>, T, 0));
    else CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k_p3<C>, T, 0));
    int bps = fused ? (g_fusedBps[li] ? g_fusedBps[li] : occ) : (g_queueBps ? g_queueBps : occ);
    bps = std::max(1, std::min(bps, std::max(occ, 1)));
    const u64 persistBlocks = (u64)g_nSM * bps;
    const u64 fusedBlocks = std::max<u64>(1, std::min<u64>(persistBlocks, (n + T - 1) / T));
    const u64 nWork = fused ? fusedBlocks * T : NB;     // Work per read of a batch (staged) or per resident thread (fused)
    uint8_t* dOut; CK(cudaMalloc(&dOut, n * (u64)C::NBLOB));
    uint16_t* dLen; CK(cudaMalloc(&dLen, n * 2)); uint8_t* dSt; CK(cudaMalloc(&dSt, n)); uint8_t* dWhy; CK(cudaMalloc(&dWhy, n));
    int8_t* dUn; CK(cudaMalloc(&dUn, n));
    Work<C>* dWork; CK(cudaMalloc(&dWork, nWork * sizeof(Work<C>))); CK(cudaMemset(dWork, 0, nWork * sizeof(Work<C>)));
    ReadBuf<C::LREAD>* dRbuf; CK(cudaMalloc(&dRbuf, nWork * sizeof(ReadBuf<C::LREAD>)));
    uint8_t* dBst = nullptr; u64* dNent = nullptr; u64* dBase = nullptr; uint32_t* dBits = nullptr; u64 bitsCap = 0;
    u64* dKey = nullptr; uint32_t* dPerm = nullptr; uint32_t* dWbits = nullptr;
    const u64 capw = (g_fusedBits + 31) / 32;
    if (!fused) {
        CK(cudaMalloc(&dBst, NB)); CK(cudaMalloc(&dNent, NB * 8)); CK(cudaMalloc(&dBase, NB * 8));
        bitsCap = 1 << 20; CK(cudaMalloc(&dBits, bitsCap * 4));
        CK(cudaMalloc(&dKey, NB * 8)); CK(cudaMalloc(&dPerm, NB * 4));
    } else CK(cudaMalloc(&dWbits, nWork * capw * 4));
    unsigned long long* dCtr; CK(cudaMalloc(&dCtr, 16)); CK(cudaMemset(dCtr, 0, 16));
    // scheduling statistics: the blocks of every launch of one run side by side
    const u64 nBatches = fused ? 1 : (n + NB - 1) / NB;
    const u64 blocksPerLaunch = fused ? fusedBlocks : g_queue ? std::min<u64>(persistBlocks, (NB + T - 1) / T) : (NB + T - 1) / T;
    const u64 tailBlocks = nBatches * blocksPerLaunch;
    unsigned long long *dTbs = nullptr, *dTbe = nullptr, *dTwe = nullptr, *dTbusy = nullptr, *dTlast = nullptr;
    if (g_tail) {
        CK(cudaMalloc(&dTbs, tailBlocks * 8)); CK(cudaMalloc(&dTbe, tailBlocks * 8)); CK(cudaMalloc(&dTwe, tailBlocks * wpb * 8));
        CK(cudaMalloc(&dTbusy, nBatches * 8)); CK(cudaMalloc(&dTlast, nBatches * 8));
    }
    auto mkTail = [&](u64 bi) { Tail t{dTbs, dTbe, dTwe, dTbusy, dTlast, bi * blocksPerLaunch, bi}; return t; };
    L.workGB = nWork * (double)(sizeof(Work<C>) + sizeof(ReadBuf<C::LREAD>)) / 1e9; L.workThreads = nWork; L.occBlocks = occ;
    L.usedGB = devUsedGB();
    L.mode = fused ? "fused" : g_queue ? "queue" : "staged";
    fprintf(stderr, "lane %s (%s): %llu reads, %s %llu, work %zu B (%llu x, %.2f GB with read buffers); phase-3 kernel: %d blocks/SM resident, "
                    "persistent grid %d/SM; device memory in use %.2f GB; p3order %s%s\n",
            name, L.mode.c_str(), n, fused ? "resident threads" : "batches of", fused ? nWork : NB, sizeof(Work<C>), nWork, L.workGB, occ, bps, L.usedGB,
            g_bin ? "cost-asc (--bin)" : p3orderName[g_p3order], fused ? " (not used: fused)" : "");
    cudaEvent_t q[4]; for (auto& e : q) cudaEventCreate(&e);
    unsigned long long clk[8] = {0}, scan = 0; double phMs[3] = {0, 0, 0};
    for (int r = 0; r < reps + 1; r++) {
        unsigned long long z[8] = {0}; CK(cudaMemcpyToSymbol(g_clk, z, sizeof z)); CK(cudaMemcpyToSymbol(g_scan, z, 8));
        if (g_tail) {
            CK(cudaMemset(dTbs, 0, tailBlocks * 8)); CK(cudaMemset(dTbe, 0, tailBlocks * 8)); CK(cudaMemset(dTwe, 0, tailBlocks * wpb * 8));
            CK(cudaMemset(dTbusy, 0, nBatches * 8)); CK(cudaMemset(dTlast, 0, nBatches * 8));
        }
        CK(cudaMemset(dCtr + 1, 0, 8));
        std::vector<std::pair<u64, u64>> launches;
        double ph[3] = {0, 0, 0};
        cudaEventRecord(e0);
        if (fused) {
            CK(cudaMemsetAsync(dCtr, 0, 8));
            cudaEventRecord(q[0]);
            k_fused<C><<<(unsigned)fusedBlocks, T>>>(P, D, dRb, dRoff, dRlen, dIdx, n, budget, dWork, dRbuf, dWbits, capw, dSt, dOut, dLen, dWhy, dUn,
                                                    dCtr, dCtr + 1, mkTail(0));
            cudaEventRecord(q[3]);
            CK(cudaEventSynchronize(q[3])); CK(cudaGetLastError());
            float m; cudaEventElapsedTime(&m, q[0], q[3]); ph[2] += m;
            launches.push_back({0, fusedBlocks});
        } else {
            u64 bi = 0;
            for (u64 b0 = 0; b0 < n; b0 += NB, bi++) {
                u64 nb = std::min(NB, n - b0);
                unsigned G = (unsigned)((nb + T - 1) / T);
                cudaEventRecord(q[0]);
                k_p1<C><<<G, T>>>(P, D, dRb, dRoff, dRlen, dIdx, b0, nb, dWork, dRbuf, dBst, dNent);
                cudaEventRecord(q[1]);
                thrust::exclusive_scan(thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dBase));
                u64 lastB, lastN; CK(cudaMemcpy(&lastB, dBase + nb - 1, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lastN, dNent + nb - 1, 8, cudaMemcpyDeviceToHost));
                u64 words = (lastB + lastN) / 32 + 2;
                if (words > bitsCap) { CK(cudaFree(dBits)); bitsCap = words * 2; CK(cudaMalloc(&dBits, bitsCap * 4)); }
                CK(cudaMemsetAsync(dBits, 0, words * 4));
                k_p2<C><<<(unsigned)((nb * 32 + 255) / 256), 256>>>(P, D, nb, dWork, dNent, dBase, dBits);
                cudaEventRecord(q[2]);
                if (g_bin) {
                    k_cost<C><<<G, T>>>(nb, dWork, dNent, dKey, dPerm);
                    thrust::sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm));
                } else if (g_p3order) {
                    k_key<C><<<G, T>>>(g_p3order, P, D, nb, dWork, dNent, dKey, dPerm);
                    thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm));
                }
                const uint32_t* perm = (g_bin || g_p3order) ? dPerm : nullptr;
                if (g_queue) {
                    unsigned Gq = (unsigned)std::min<u64>(persistBlocks, G);
                    CK(cudaMemsetAsync(dCtr, 0, 8));
                    k_p3q<C><<<Gq, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dNent, dBase, dBits, dBst, dOut, dLen, dWhy, dUn, perm, dCtr, mkTail(bi));
                    launches.push_back({bi * blocksPerLaunch, Gq});
                } else {
                    k_p3<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dNent, dBase, dBits, dBst, dOut, dLen, dWhy, dUn, perm, mkTail(bi));
                    launches.push_back({bi * blocksPerLaunch, G});
                }
                k_copy_status<<<G, T>>>(dBst, dSt + b0, nb);
                cudaEventRecord(q[3]);
                CK(cudaEventSynchronize(q[3])); CK(cudaGetLastError());
                float m; cudaEventElapsedTime(&m, q[0], q[1]); ph[0] += m; cudaEventElapsedTime(&m, q[1], q[2]); ph[1] += m;
                cudaEventElapsedTime(&m, q[2], q[3]); ph[2] += m;
            }
        }
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float m; cudaEventElapsedTime(&m, e0, e1); if (r > 0) { L.runs.push_back(m); for (int k = 0; k < 3; k++) phMs[k] += ph[k] / reps; }
        fprintf(stderr, "  lane %s run %d: %.1f ms (phase 1 %.1f, scan %.1f, phase 3 %.1f)\n", name, r, m, ph[0], ph[1], ph[2]);
        if (g_tail && r == reps) {
            std::vector<unsigned long long> bs(tailBlocks), be(tailBlocks), we(tailBlocks * wpb), bu(nBatches), lg(nBatches);
            CK(cudaMemcpy(bs.data(), dTbs, tailBlocks * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(be.data(), dTbe, tailBlocks * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(we.data(), dTwe, tailBlocks * wpb * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bu.data(), dTbusy, nBatches * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(lg.data(), dTlast, nBatches * 8, cudaMemcpyDeviceToHost));
            const bool persistent = fused || g_queue;
            const int wmax = (int)((persistent ? (fused ? fusedBlocks : std::min<u64>(persistBlocks, (NB + T - 1) / T)) : (u64)occ * g_nSM) * wpb);
            TailStats ts = tailStats(bs, be, we, bu, lg, launches, wpb, wmax, persistent);
            printTail(name, ts);
            L.tail[0] = ts.spanMs; L.tail[1] = ts.fill; L.tail[2] = ts.warpBal; L.tail[3] = ts.laneBal; L.tail[4] = ts.drain; L.tail[5] = ts.util;
            L.tail[6] = ts.wmax; L.tail[7] = (double)ts.launches;
        }
    }
    { unsigned long long fb = 0; CK(cudaMemcpy(&fb, dCtr + 1, 8, cudaMemcpyDeviceToHost)); L.fallback = fb;
      if (fused) fprintf(stderr, "  lane %s: %llu reads assigned every SA entry themselves (more than %llu entries)\n", name, fb, g_fusedBits); }
    CK(cudaMemcpyFromSymbol(clk, g_clk, sizeof clk)); CK(cudaMemcpyFromSymbol(&scan, g_scan, 8));
    { double t = (double)clk[0] + clk[1] + clk[2] + clk[3]; if (t > 0)
        fprintf(stderr, "  lane %s thread cycles: seeds %.1f%%, windows %.1f%%, stitching %.1f%%, select+encode %.1f%%; %.1f SA entries per read\n",
                name, 100 * clk[0] / t, 100 * clk[1] / t, 100 * clk[2] / t, 100 * clk[3] / t, (double)scan / n); }
    if (clk[2] > 0) fprintf(stderr, "  lane %s inside stitching: stitchAlignToTranscript %.1f%%, finalization %.1f%% (of which recording %.1f%%), copies %.1f%%\n",
                            name, 100.0 * clk[4] / clk[2], 100.0 * clk[5] / clk[2], 100.0 * clk[7] / clk[2], 100.0 * clk[6] / clk[2]);
    L.ms = medianv(L.runs);
    L.phaseMs[0] = phMs[0]; L.phaseMs[1] = phMs[1]; L.phaseMs[2] = phMs[2];
    L.out.resize(n * (u64)C::NBLOB); L.status.resize(n); L.why.resize(n); L.len.resize(n); L.unmap.resize(n);
    CK(cudaMemcpy(L.out.data(), dOut, L.out.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.len.data(), dLen, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.status.data(), dSt, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.why.data(), dWhy, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.unmap.data(), dUn, n, cudaMemcpyDeviceToHost));
    cudaFree(dOut); cudaFree(dLen); cudaFree(dSt); cudaFree(dWhy); cudaFree(dUn); cudaFree(dWork); cudaFree(dRbuf);
    cudaFree(dBst); cudaFree(dNent); cudaFree(dBase); cudaFree(dBits); cudaFree(dKey); cudaFree(dPerm); cudaFree(dWbits); cudaFree(dCtr);
    cudaFree(dTbs); cudaFree(dTbe); cudaFree(dTwe); cudaFree(dTbusy); cudaFree(dTlast);
    return L;
}

// ---------------------------------------------------------------------------------------------------------------------
static u64 fileSize(const std::string& p) { struct stat st; if (stat(p.c_str(), &st)) { perror(p.c_str()); exit(2); } return st.st_size; }
static void readInto(const std::string& path, u64 off, char* dst, u64 n) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { perror(path.c_str()); exit(2); }
    fseeko(f, (off_t)off, SEEK_SET);
    u64 got = 0; while (got < n) { size_t k = fread(dst + got, 1, std::min<u64>(n - got, 1ULL << 30), f); if (!k) break; got += k; }
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
}
template <class T> static T* toDev(const T* h, u64 n) {
    T* d; CK(cudaMalloc(&d, std::max<u64>(1, n) * sizeof(T)));
    if (n) CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static T* toDev(const std::vector<T>& v) { return toDev(v.data(), (u64)v.size()); }

struct ParamFile {
    std::map<std::string, std::string> kv;
    void load(const std::string& p) {
        FILE* f = fopen(p.c_str(), "r"); if (!f) { perror(p.c_str()); exit(2); }
        char line[8192];
        while (fgets(line, sizeof line, f)) {
            std::string s(line); while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
            size_t sp = s.find(' '); if (sp == std::string::npos) continue;
            kv[s.substr(0, sp)] = s.substr(sp + 1);
        }
        fclose(f);
    }
    const std::string& str(const char* k) { auto it = kv.find(k); if (it == kv.end()) { fprintf(stderr, "param %s missing\n", k); exit(2); } return it->second; }
    u64 u(const char* k) { return strtoull(str(k).c_str(), nullptr, 10); }
    long long i(const char* k) { return strtoll(str(k).c_str(), nullptr, 10); }
    double d(const char* k) { return strtod(str(k).c_str(), nullptr); }
};

static std::vector<char> loadFile(const std::string& p) { std::vector<char> b(fileSize(p)); readInto(p, 0, b.data(), b.size()); return b; }
template <class T> static std::vector<T> takeArr(const std::vector<char>& b, u64& o) {
    u64 n; memcpy(&n, b.data() + o, 8); o += 8;
    std::vector<T> v(n); if (n) memcpy(v.data(), b.data() + o, n * sizeof(T)); o += n * sizeof(T);
    return v;
}

struct Truth {
    std::vector<char> rb; std::vector<u64> roff{0}; std::vector<uint16_t> rlen; std::vector<uint32_t> count;
    std::vector<uint8_t> blob; std::vector<u64> boff{0}; std::vector<int8_t> unmap; std::vector<uint8_t> flags; std::vector<uint32_t> marker;
};
#pragma pack(push, 1)
struct M5Rec { uint64_t iReadAll; uint32_t count; uint16_t Lread; uint8_t flags, pad; int32_t unmapType; uint32_t mapMarker, blobLen; };
#pragma pack(pop)
static void loadTruth(const std::string& prefix, Truth& T) {
    glob_t g; std::string pat = prefix + ".*.bin";
    if (glob(pat.c_str(), 0, nullptr, &g)) { fprintf(stderr, "no truth files %s\n", pat.c_str()); exit(2); }
    for (size_t fi = 0; fi < g.gl_pathc; fi++) {
        std::string path = g.gl_pathv[fi];
        if (path.size() >= 11 && path.compare(path.size() - 11, 11, ".genome.bin") == 0) continue;
        std::vector<char> b = loadFile(path);
        if (b.size() < 8 || memcmp(b.data(), "STARM5T1", 8)) { fprintf(stderr, "bad header %s\n", g.gl_pathv[fi]); exit(2); }
        u64 o = 8;
        while (o < b.size()) {
            M5Rec r; memcpy(&r, b.data() + o, sizeof r); o += sizeof r;
            T.rb.insert(T.rb.end(), b.data() + o, b.data() + o + r.Lread); o += r.Lread; T.roff.push_back(T.rb.size());
            T.blob.insert(T.blob.end(), (uint8_t*)b.data() + o, (uint8_t*)b.data() + o + r.blobLen); o += r.blobLen; T.boff.push_back(T.blob.size());
            T.rlen.push_back(r.Lread); T.count.push_back(r.count); T.unmap.push_back((int8_t)r.unmapType); T.flags.push_back(r.flags);
            T.marker.push_back(r.mapMarker);
        }
    }
    globfree(&g);
}

static std::string blobStr(const uint8_t* p, int n) {
    std::string s; const uint8_t* e = p + n;
    while (p < e) {
        u64 u = 0; int sh = 0;
        while (p < e && (*p & 0x80)) { u |= (u64)(*p++ & 0x7f) << sh; sh += 7; }
        if (p < e) u |= (u64)(*p++) << sh;
        long long v = (long long)((u >> 1) ^ (~(u & 1) + 1));
        s += std::to_string(v); s += ' ';
    }
    return s;
}

// value of STAR's genomic length score for gLength x, with the C library's log2 (stitchWindowAligns.cpp)
static int glScore(double scale, u64 x) { return int(std::ceil(std::log2((double)x) * scale - 0.5)); }

int main(int argc, char** argv) {
    std::string gdir, truth, json, input, emit; int reps = 3, bps = 4, nthr = 128, bpsB = 1, nthrB = 64; bool localWork = false, mono = false; long long nbA = 131072, nbB = 32768, nbM = 131072, budgetA = 0, budgetB = 0, budgetM = 0; bool noMid = false;
    bool presort = false; std::string label;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--genome-dir") gdir = argv[++i];
        else if (a == "--truth") truth = argv[++i];
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--json") json = argv[++i];
        else if (a == "--blocks-per-sm") bps = atoi(argv[++i]);
        else if (a == "--threads") nthr = atoi(argv[++i]);
        else if (a == "--blocks-per-sm-b") bpsB = atoi(argv[++i]);
        else if (a == "--threads-b") nthrB = atoi(argv[++i]);
        else if (a == "--local") localWork = true;
        else if (a == "--mono") mono = true;            // one kernel per lane (no staged assignment scan)
        else if (a == "--batch-a") nbA = atoll(argv[++i]);
        else if (a == "--batch-b") nbB = atoll(argv[++i]);
        else if (a == "--budget-a") budgetA = atoll(argv[++i]);   // stitching steps per read before it moves on (0 = none)
        else if (a == "--budget-b") budgetB = atoll(argv[++i]);
        else if (a == "--budget-m") budgetM = atoll(argv[++i]);
        else if (a == "--batch-m") nbM = atoll(argv[++i]);
        else if (a == "--bin") g_bin = true;   // phase 3 threads take the reads of a batch sorted by window count / SA entries
        // E4 (docs/gpu-engine/fable/E4-order.md)
        else if (a == "--presort") presort = true;      // map the reads in the engine's order: stable sort by the first 32 bases
        else if (a == "--p3order") {                    // order of the phase-3 threads within a batch
            std::string v = argv[++i]; g_p3order = -1;
            for (int k = 0; k < (int)(sizeof p3orderName / sizeof *p3orderName); k++) if (v == p3orderName[k]) g_p3order = k;
            if (g_p3order < 0) { fprintf(stderr, "unknown --p3order %s\n", v.c_str()); return 2; }
        }
        else if (a == "--queue") g_queue = true;        // persistent phase 3 (atomic read counter) per batch
        else if (a == "--queue-bps") g_queueBps = atoi(argv[++i]);
        else if (a == "--fused") { std::string v = argv[++i]; for (char c : v) g_fused |= c == 'A' ? 1u : c == 'M' ? 2u : c == 'B' ? 4u : 0u; }
        else if (a == "--fused-bps-a") g_fusedBps[0] = atoi(argv[++i]);
        else if (a == "--fused-bps-m") g_fusedBps[1] = atoi(argv[++i]);
        else if (a == "--fused-bps-b") g_fusedBps[2] = atoi(argv[++i]);
        else if (a == "--fused-bits") g_fusedBits = strtoull(argv[++i], nullptr, 10);
        else if (a == "--tail") g_tail = true;          // phase-3 scheduling statistics (block/warp times)
        else if (a == "--label") label = argv[++i];
        else if (a == "--no-mid") noMid = true;          // lanes A, B only
        else if (a == "--input") input = argv[++i];     // "count sequence" lines (uniq -c): map these instead of the truth reads
        else if (a == "--emit") emit = argv[++i];       // write (sequence, result) for the sequence cache preload (inject/)
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 2; }
    }
    if (gdir.empty() || truth.empty()) { fprintf(stderr, "usage: m5 --genome-dir D --truth PREFIX [--json F]\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    g_nSM = prop.multiProcessorCount;
    if (mono && (g_fused || g_queue || g_p3order || g_bin)) { fprintf(stderr, "--mono excludes the E4 scheduling options\n"); return 2; }
    // the marker that says which build runs and with which scheduling (job 790068 ran a stale binary)
    fprintf(stderr, "m5 E4-order build %s; label %s; presort %d; p3order %s; queue %d (bps %d); fused %s%s%s (bps %d/%d/%d, bits %llu); tail %d; "
                    "batches %lld/%lld/%lld; budgets %lld/%lld/%lld\n", M5_GIT, label.c_str(), presort, g_bin ? "cost-asc(--bin)" : p3orderName[g_p3order],
            g_queue, g_queueBps, (g_fused & 1) ? "A" : "", (g_fused & 2) ? "M" : "", (g_fused & 4) ? "B" : "", g_fusedBps[0], g_fusedBps[1], g_fusedBps[2],
            g_fusedBits, g_tail, nbA, nbM, nbB, budgetA, budgetM, budgetB);
    fprintf(stderr, "device %s, %d SMs; sizeof Work: A %zu, B %zu, host %zu\n", prop.name, prop.multiProcessorCount,
            sizeof(Work<LaneA>), sizeof(Work<LaneB>), sizeof(Work<Heavy>));

    // ---- parameters
    ParamFile pf; pf.load(truth + ".params.txt");
    Params P{};
    P.maxNsplit = pf.u("maxNsplit"); P.seedSplitMin = pf.u("seedSplitMin"); P.seedSearchStartLmax = pf.u("seedSearchStartLmax");
    P.seedSearchStartLmaxOverLread = pf.d("seedSearchStartLmaxOverLread"); P.seedMapMin = pf.u("seedMapMin"); P.seedSearchLmax = pf.u("seedSearchLmax");
    P.seedMultimapNmax = pf.u("seedMultimapNmax"); P.seedPerReadNmax = pf.u("seedPerReadNmax"); P.seedPerWindowNmax = pf.u("seedPerWindowNmax");
    P.winBinChrNbits = pf.u("winBinChrNbits"); P.winBinNbits = pf.u("winBinNbits"); P.winAnchorDistNbins = pf.u("winAnchorDistNbins");
    P.winFlankNbins = pf.u("winFlankNbins"); P.winBinN = pf.u("winBinN"); P.winAnchorMultimapNmax = pf.u("winAnchorMultimapNmax");
    P.alignWindowsPerReadNmax = pf.u("alignWindowsPerReadNmax"); P.alignTranscriptsPerWindowNmax = pf.u("alignTranscriptsPerWindowNmax");
    P.alignTranscriptsPerReadNmax = pf.u("alignTranscriptsPerReadNmax");
    P.alignSJoverhangMin = pf.u("alignSJoverhangMin"); P.alignSJDBoverhangMin = pf.u("alignSJDBoverhangMin");
    P.alignSplicedMateMapLmin = pf.u("alignSplicedMateMapLmin"); P.alignSplicedMateMapLminOverLmate = pf.d("alignSplicedMateMapLminOverLmate");
    P.alignIntronMin = pf.u("alignIntronMin"); P.alignIntronMax = pf.u("alignIntronMax"); P.alignMatesGapMax = pf.u("alignMatesGapMax");
    { int k = 0; const char* s = pf.str("alignSJstitchMismatchNmax").c_str(); char* e;
      while (k < 4) { long v = strtol(s, &e, 10); if (e == s) break; P.alignSJstitchMismatchNmax[k++] = (int)v; s = e; }
      if (k != 4) { fprintf(stderr, "alignSJstitchMismatchNmax needs 4 values\n"); return 2; } }
    P.alignSoftClipAtReferenceEnds = pf.u("alignSoftClipAtReferenceEnds.yes");
    P.alignEndsTypeExt[0][0] = pf.u("alignEndsType.ext00"); P.alignEndsTypeExt[0][1] = pf.u("alignEndsType.ext01");
    P.alignEndsTypeExt[1][0] = pf.u("alignEndsType.ext10"); P.alignEndsTypeExt[1][1] = pf.u("alignEndsType.ext11");
    P.alignEndsProtrudeNbasesMax = (int)pf.i("alignEndsProtrude.nBasesMax"); P.flushRight = pf.u("alignInsertionFlush.flushRight");
    std::string im = pf.str("outFilterIntronMotifs");
    P.intronMotifsFilter = im == "None" ? 0 : im == "RemoveNoncanonical" ? 1 : im == "RemoveNoncanonicalUnannotated" ? 2 : -1;
    P.removeInconsistentStrands = pf.str("outFilterIntronStrands") == "RemoveInconsistentStrands";
    P.outSAMstrandFieldType = pf.u("outSAMstrandField.type");
    P.scoreGenomicLengthLog2scale = pf.d("scoreGenomicLengthLog2scale");
    P.outFilterMultimapScoreRange = (int)pf.i("outFilterMultimapScoreRange"); P.outFilterMultimapNmax = pf.u("outFilterMultimapNmax");
    P.multOrderRandom = pf.u("outMultimapperOrder.random"); P.outSAMmultNmax = pf.u("outSAMmultNmax");
    P.primaryAllBest = pf.str("outSAMprimaryFlag") == "AllBestScore";
    P.outFilterScoreMin = (int)pf.i("outFilterScoreMin"); P.outFilterScoreMinOverLread = pf.d("outFilterScoreMinOverLread");
    P.outFilterMatchNmin = pf.u("outFilterMatchNmin"); P.outFilterMatchNminOverLread = pf.d("outFilterMatchNminOverLread");
    P.outFilterMismatchNmax = pf.u("outFilterMismatchNmax"); P.outFilterMismatchNoverLmax = pf.d("outFilterMismatchNoverLmax");
    P.outFilterMismatchNoverReadLmax = pf.d("outFilterMismatchNoverReadLmax");
    P.scoreGap = (int)pf.i("scoreGap"); P.scoreGapNoncan = (int)pf.i("scoreGapNoncan"); P.scoreGapGCAG = (int)pf.i("scoreGapGCAG");
    P.scoreGapATAC = (int)pf.i("scoreGapATAC"); P.scoreDelBase = (int)pf.i("scoreDelBase"); P.scoreDelOpen = (int)pf.i("scoreDelOpen");
    P.scoreInsBase = (int)pf.i("scoreInsBase"); P.scoreInsOpen = (int)pf.i("scoreInsOpen"); P.scoreStitchSJshift = (int)pf.i("scoreStitchSJshift");
    P.sjdbScore = (int)pf.i("pGe.sjdbScore"); P.gSAsparseD = pf.u("pGe.gSAsparseD"); P.gSAindexNbases = pf.u("pGe.gSAindexNbases");
    // what the port does not cover (it would need the code paths named here)
    std::vector<std::string> unsupported;
    if (pf.u("readNmates") != 1) unsupported.push_back("readNmates != 1 (mates)");
    if (pf.u("pCh.segmentMin") != 0) unsupported.push_back("chimeric detection");
    if (pf.u("peOverlap.NbasesMin") != 0) unsupported.push_back("peOverlap");
    if (pf.u("Var.yes") != 0) unsupported.push_back("variation");
    if (pf.i("pGe.transform.type") > 0) unsupported.push_back("genome transform");
    if (pf.u("pGe.gType") == 101) unsupported.push_back("splice graph");
    if (pf.i("outFilterBySJoutStage") == 2) unsupported.push_back("BySJout stage 2");
    if (P.multOrderRandom || P.outSAMmultNmax != (u64)-1 || P.primaryAllBest) unsupported.push_back("multimapper order / primary flag");
    if (P.gSAsparseD > 4) unsupported.push_back("gSAsparseD > 4");
    if (P.intronMotifsFilter < 0) unsupported.push_back("outFilterIntronMotifs");
    if (P.outFilterMultimapScoreRange < 0) unsupported.push_back("negative score range");
    if (P.alignWindowsPerReadNmax > (u64)Heavy::NW || P.seedPerWindowNmax > (u64)Heavy::NWA || P.alignTranscriptsPerWindowNmax > (u64)Heavy::NWTR
        || P.seedPerReadNmax > (u64)Heavy::NP || P.maxNsplit > (u64)Heavy::NSPLIT) unsupported.push_back("STAR limits above the heavy lane's");
    if (!unsupported.empty()) { for (auto& s : unsupported) fprintf(stderr, "unsupported: %s\n", s.c_str()); return 3; }
    // genomic length score breakpoints (the value is nonincreasing in x for scale < 0)
    if (P.scoreGenomicLengthLog2scale > 0) { fprintf(stderr, "positive scoreGenomicLengthLog2scale\n"); return 3; }
    {
        double s = P.scoreGenomicLengthLog2scale;
        P.glVal0 = glScore(s, 1); P.glN = 0;
        u64 x = 1; int v = P.glVal0; const u64 XMAX = 1ULL << 40;
        while (x < XMAX && s != 0) {
            u64 lo = x, hi = x + 1;
            while (hi < XMAX && glScore(s, hi) == v) { lo = hi; hi = std::min(XMAX, hi * 2); }
            if (glScore(s, hi) == v) break;
            while (hi - lo > 1) { u64 m = lo + (hi - lo) / 2; if (glScore(s, m) == v) lo = m; else hi = m; }
            if (P.glN >= 64) { fprintf(stderr, "too many genomic length breakpoints\n"); return 3; }
            v = glScore(s, hi); P.glBreak[P.glN] = hi; P.glVal[P.glN] = v; P.glN++; x = hi;
        }
        // check against the C library on a sweep
        for (u64 xx = 1; xx < (1ULL << 24); xx = xx + 1 + xx / 997)
            if (genomicLengthScore(P, xx) != glScore(s, xx)) { fprintf(stderr, "genomic length score breakpoints wrong at %llu\n", xx); return 3; }
        fprintf(stderr, "genomic length score: %d breakpoints\n", P.glN);
    }

    // ---- genome arrays and index
    double t0 = now();
    std::vector<char> gb = loadFile(truth + ".genome.bin");
    if (memcmp(gb.data(), "STARM5G1", 8)) { fprintf(stderr, "bad genome.bin\n"); return 2; }
    u64 o = 8;
    std::vector<u64> chrStart = takeArr<u64>(gb, o), chrLength = takeArr<u64>(gb, o), chrBin = takeArr<u64>(gb, o);
    std::vector<u64> sjdbStart = takeArr<u64>(gb, o), sjdbEnd = takeArr<u64>(gb, o);
    std::vector<uint8_t> sjdbMotif = takeArr<uint8_t>(gb, o), sjdbShiftLeft = takeArr<uint8_t>(gb, o), sjdbShiftRight = takeArr<uint8_t>(gb, o);
    std::vector<uint8_t> sjdbStrand = takeArr<uint8_t>(gb, o);
    std::vector<u64> sjDstart = takeArr<u64>(gb, o), sjAstart = takeArr<u64>(gb, o);

    GenomeView H{};
    H.nGenome = fileSize(gdir + "/Genome");
    u64 nSAbyte = fileSize(gdir + "/SA");
    H.GstrandBit = pf.u("GstrandBit");
    H.GstrandMask = ~(1ULL << H.GstrandBit);
    H.saBits = H.GstrandBit + 1; H.saiBits = H.GstrandBit + 3;
    H.nSA = nSAbyte * 8 / H.saBits;
    if (H.nSA != pf.u("nSA")) { fprintf(stderr, "nSA %llu != STAR's %llu\n", H.nSA, pf.u("nSA")); }
    H.nSA = pf.u("nSA");
    if (H.nGenome != pf.u("nGenome")) { fprintf(stderr, "Genome file %llu bytes, STAR nGenome %llu\n", H.nGenome, pf.u("nGenome")); }
    H.SAiMarkNmaskC = 1ULL << (H.GstrandBit + 1); H.SAiMarkNmask = ~H.SAiMarkNmaskC; H.SAiMarkAbsentMaskC = 1ULL << (H.GstrandBit + 2);
    std::string saiPath = gdir + "/SAindex";
    u64 nb; readInto(saiPath, 0, (char*)&nb, 8);
    if (nb != P.gSAindexNbases || nb > 30) { fprintf(stderr, "gSAindexNbases mismatch\n"); return 2; }
    readInto(saiPath, 8, (char*)H.saiStart, 8 * (nb + 1));
    u64 nSAi = H.saiStart[nb], saiHdr = 8 + 8 * (nb + 1), saiBytes = (nSAi - 1) * H.saiBits / 8 + 8;
    const u64 PAD = 200;
    u64 fileG = fileSize(gdir + "/Genome");
    std::vector<char> hG(fileG + 2 * PAD + 64, 5);
    readInto(gdir + "/Genome", 0, hG.data() + PAD, fileG);
    std::vector<u64> hSA(nSAbyte / 8 + 4, 0); readInto(gdir + "/SA", 0, (char*)hSA.data(), nSAbyte);
    std::vector<u64> hSAi(saiBytes / 8 + 4, 0); readInto(saiPath, saiHdr, (char*)hSAi.data(), saiBytes);
    H.nGenome = pf.u("nGenome");
    H.G = hG.data() + PAD; H.SA = hSA.data(); H.SAi = hSAi.data();
    H.chrStart = chrStart.data(); H.chrLength = chrLength.data(); H.chrBin = chrBin.data(); H.nChr = chrLength.size(); H.chrBinN = chrBin.size();
    H.sjGstart = pf.u("sjGstart"); H.sjdbOverhang = pf.u("sjdbOverhang"); H.sjdbLength = pf.u("sjdbLength"); H.sjdbN = pf.u("sjdbN");
    H.sjdbStart = sjdbStart.data(); H.sjdbEnd = sjdbEnd.data(); H.sjDstart = sjDstart.data(); H.sjAstart = sjAstart.data();
    H.sjdbMotif = sjdbMotif.data(); H.sjdbShiftLeft = sjdbShiftLeft.data(); H.sjdbShiftRight = sjdbShiftRight.data(); H.sjdbStrand = sjdbStrand.data();
    GenomeView D = H;
    char* dG = toDev(hG); D.G = dG + PAD;
    D.SA = toDev(hSA); D.SAi = toDev(hSAi);
    D.chrStart = toDev(chrStart); D.chrLength = toDev(chrLength); D.chrBin = toDev(chrBin);
    D.sjdbStart = toDev(sjdbStart); D.sjdbEnd = toDev(sjdbEnd); D.sjDstart = toDev(sjDstart); D.sjAstart = toDev(sjAstart);
    D.sjdbMotif = toDev(sjdbMotif); D.sjdbShiftLeft = toDev(sjdbShiftLeft); D.sjdbShiftRight = toDev(sjdbShiftRight); D.sjdbStrand = toDev(sjdbStrand);
    double tIndex = now() - t0;
    fprintf(stderr, "index + genome arrays loaded in %.1f s; sjdbN %llu, chr %llu\n", tIndex, H.sjdbN, H.nChr);

    // ---- truth
    t0 = now();
    Truth T;
    if (input.empty()) loadTruth(truth, T);
    else {   // no STAR truth: the sequences of a whole sample
        FILE* f = fopen(input.c_str(), "r"); if (!f) { perror(input.c_str()); return 2; }
        char line[4096];
        while (fgets(line, sizeof line, f)) {
            char* q = line; while (*q == ' ') q++;
            uint32_t c = (uint32_t)strtoul(q, &q, 10); while (*q == ' ') q++;
            int L = 0;
            for (; q[L] && q[L] != '\n' && q[L] != '\r'; L++) {
                char b = q[L]; T.rb.push_back(b == 'A' ? 0 : b == 'C' ? 1 : b == 'G' ? 2 : b == 'T' ? 3 : 4);
            }
            T.roff.push_back(T.rb.size()); T.rlen.push_back((uint16_t)L); T.count.push_back(c);
            T.boff.push_back(0); T.unmap.push_back(0); T.flags.push_back(0); T.marker.push_back(0);
        }
        fclose(f);
    }
    if (presort) {   // the engine's order (engine_gpu.cu k_prefix_key + stable sort): first 32 bases, 2 bits each
        u64 nT = T.rlen.size();
        std::vector<u64> key(nT); std::vector<uint32_t> ord(nT);
        for (u64 i = 0; i < nT; i++) {
            u64 kk = 0, L = T.rlen[i];
            for (u64 t = 0; t < 32; t++) kk = (kk << 2) | (t < L ? (u64)(uint8_t)T.rb[T.roff[i] + t] : 0);
            key[i] = kk;
        }
        std::iota(ord.begin(), ord.end(), 0u);
        std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return key[a] < key[b]; });
        Truth U;
        for (u64 j = 0; j < nT; j++) {
            u64 i = ord[j];
            U.rb.insert(U.rb.end(), T.rb.begin() + T.roff[i], T.rb.begin() + T.roff[i + 1]); U.roff.push_back(U.rb.size());
            U.blob.insert(U.blob.end(), T.blob.begin() + T.boff[i], T.blob.begin() + T.boff[i + 1]); U.boff.push_back(U.blob.size());
            U.rlen.push_back(T.rlen[i]); U.count.push_back(T.count[i]); U.unmap.push_back(T.unmap[i]); U.flags.push_back(T.flags[i]);
            U.marker.push_back(T.marker[i]);
        }
        T = std::move(U);
        fprintf(stderr, "presort: %llu reads in the order of their first 32 bases\n", nT);
    }
    u64 n = T.rlen.size(), nReadsW = 0; for (auto c : T.count) nReadsW += c;
    fprintf(stderr, "truth: %llu distinct reads (%llu reads), loaded in %.1f s\n", n, nReadsW, now() - t0);

    // ---- GPU lanes: A (small capacities, every read), B (wide capacities, A's overflow); then the host (STAR's limits)
    char* dRb = toDev(T.rb); u64* dRoff = toDev(T.roff); uint16_t* dRlen = toDev(T.rlen);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    CK(cudaDeviceSetLimit(cudaLimitStackSize, localWork ? (int)(sizeof(Work<LaneA>) + 4096) : 16 * 1024));
    LaneOut A = mono ? runLane<LaneA>(P, D, dRb, dRoff, dRlen, nullptr, n, prop.multiProcessorCount * bps, nthr, reps, localWork, e0, e1, "A")
                     : runStaged<LaneA>(P, D, dRb, dRoff, dRlen, nullptr, n, (u64)nbA, (u64)budgetA, reps, e0, e1, "A");
    std::vector<uint32_t> idxM, idxB;
    for (u64 i = 0; i < n; i++) if (A.status[i] != (uint8_t)Status::Ok) idxM.push_back((uint32_t)i);
    LaneOut M;
    if (!idxM.empty() && !noMid) {
        uint32_t* dIdx = toDev(idxM);
        M = runStaged<LaneM>(P, D, dRb, dRoff, dRlen, dIdx, idxM.size(), (u64)nbM, (u64)budgetM, reps, e0, e1, "M");
        CK(cudaFree(dIdx));
        for (u64 k = 0; k < idxM.size(); k++) if (M.status[k] != (uint8_t)Status::Ok) idxB.push_back(idxM[k]);
    } else idxB = idxM;
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
    LaneOut B;
    if (!idxB.empty()) {
        uint32_t* dIdx = toDev(idxB);
        B = mono ? runLane<LaneB>(P, D, dRb, dRoff, dRlen, dIdx, idxB.size(), prop.multiProcessorCount * bpsB, nthrB, reps, false, e0, e1, "B")
                 : runStaged<LaneB>(P, D, dRb, dRoff, dRlen, dIdx, idxB.size(), (u64)nbB, (u64)budgetB, reps, e0, e1, "B");
        CK(cudaFree(dIdx));
    }
    double gpuMs = A.ms + M.ms + B.ms;
    // where each read's result is: lane 0 = A (index i), 1 = B (index in idxB), 2 = host (index in heavy)
    std::vector<uint8_t> lane(n, 0); std::vector<uint32_t> lanePos(n);
    for (u64 i = 0; i < n; i++) lanePos[i] = (uint32_t)i;
    u64 whyA[16] = {0}, whyCount[16] = {0};
    std::vector<u64> heavy;
    u64 whyM[16] = {0};
    for (u64 k = 0; k < idxM.size(); k++) {
        u64 i = idxM[k]; whyA[A.why[i] & 15]++;
        if (!noMid) { lane[i] = 3; lanePos[i] = (uint32_t)k; if (M.status[k] != (uint8_t)Status::Ok) whyM[M.why[k] & 15]++; }
    }
    for (u64 k = 0; k < idxB.size(); k++) {
        u64 i = idxB[k]; lane[i] = 1; lanePos[i] = (uint32_t)k;
        if (B.status[k] != (uint8_t)Status::Ok) { lane[i] = 2; lanePos[i] = (uint32_t)heavy.size(); heavy.push_back(i); whyCount[B.why[k] & 15]++; }
    }
    std::vector<std::vector<uint8_t>> hout(heavy.size()); std::vector<int8_t> hunmap(heavy.size()); std::vector<uint8_t> hstatus(heavy.size());
    t0 = now();
    int nHostThreads = omp_get_max_threads();
    #pragma omp parallel
    {
        Work<Heavy>* w = new Work<Heavy>;
        w->stepBudget = 0; w->wbEpoch = 0; w->nScan = 0;
        ReadBuf<Heavy::LREAD>* b = new ReadBuf<Heavy::LREAD>;
        std::vector<uint8_t> buf(Heavy::NBLOB);
        #pragma omp for schedule(dynamic, 1)
        for (long long k = 0; k < (long long)heavy.size(); k++) {
            u64 i = heavy[k];
            Blob bl{buf.data(), 0, Heavy::NBLOB, false};
            Status s = mapRead(P, H, *w, *b, T.rb.data() + T.roff[i], T.rlen[i], bl);
            hstatus[k] = (uint8_t)s; hunmap[k] = (int8_t)w->unmapType; hout[k].assign(buf.data(), buf.data() + bl.n);
        }
        delete w; delete b;
    }
    double heavySec = now() - t0;
    auto result = [&](u64 i, const uint8_t*& p, int& len, int8_t& un, bool& ok) {
        uint32_t k = lanePos[i];
        if (lane[i] == 0) { p = A.out.data() + (u64)i * LaneA::NBLOB; len = A.len[i]; un = A.unmap[i]; ok = true; }
        else if (lane[i] == 1) { p = B.out.data() + (u64)k * LaneB::NBLOB; len = B.len[k]; un = B.unmap[k]; ok = true; }
        else if (lane[i] == 3) { p = M.out.data() + (u64)k * LaneM::NBLOB; len = M.len[k]; un = M.unmap[k]; ok = true; }
        else { p = hout[k].data(); len = (int)hout[k].size(); un = hunmap[k]; ok = hstatus[k] == (uint8_t)Status::Ok; }
    };

    // ---- compare
    u64 same = 0, diff = 0, sameW = 0, diffW = 0, heavyFail = 0, nPrinted = 0, sameUnmapOnly = 0;
    u64 classN[5] = {0}, classSame[5] = {0};   // by STAR result: unmapped(0..3 -> idx) / mapped unique / mapped multi
    for (u64 i = 0; i < n; i++) {
        const uint8_t* ours; int olen; int8_t oun; bool okr;
        result(i, ours, olen, oun, okr);
        if (!okr) heavyFail++;
        const uint8_t* tb = T.blob.data() + T.boff[i]; int tlen = (int)(T.boff[i + 1] - T.boff[i]);
        bool eq = olen == tlen && memcmp(ours, tb, tlen) == 0 && oun == T.unmap[i];
        int cls = T.unmap[i] >= 0 ? 0 : 1;
        classN[cls]++; if (eq) classSame[cls]++;
        if (eq) { same++; sameW += T.count[i]; }
        else {
            diff++; diffW += T.count[i];
            if (olen == tlen && memcmp(ours, tb, tlen) == 0) sameUnmapOnly++;
            if (nPrinted < 40) {
                nPrinted++;
                std::string seq; for (int k = 0; k < T.rlen[i]; k++) seq += "ACGTN"[std::min(4, (int)T.rb[T.roff[i] + k])];
                fprintf(stderr, "DIFF read %llu L=%u count=%u flags=%u marker=%u lane=%s unmap star=%d ours=%d\n  seq  %s\n  star %s\n  ours %s\n",
                        i, T.rlen[i], T.count[i], T.flags[i], T.marker[i], lane[i] == 0 ? "A" : lane[i] == 1 ? "B" : lane[i] == 3 ? "M" : "host", T.unmap[i], oun,
                        seq.c_str(), blobStr(tb, tlen).c_str(), blobStr(ours, olen).c_str());
            }
        }
    }
    if (!emit.empty()) {   // header, then per sequence without N: u16 L, L bases, u32 blobLen, blob
        FILE* f = fopen(emit.c_str(), "wb"); fwrite("M5BLOBS1", 8, 1, f); u64 nEmit = 0;
        for (u64 i = 0; i < n; i++) {
            bool hasN = false; for (int k = 0; k < T.rlen[i]; k++) if (T.rb[T.roff[i] + k] > 3) hasN = true;
            if (hasN) continue;
            const uint8_t* ours; int olenI; int8_t oun; bool okr;
            result(i, ours, olenI, oun, okr);
            if (!okr) continue;
            uint32_t olen = (uint32_t)olenI;
            uint16_t L = T.rlen[i]; fwrite(&L, 2, 1, f); fwrite(T.rb.data() + T.roff[i], 1, L, f); fwrite(&olen, 4, 1, f); fwrite(ours, 1, olen, f); nEmit++;
        }
        fclose(f); fprintf(stderr, "emitted %llu sequence results to %s\n", nEmit, emit.c_str());
    }
    double rate = n / (gpuMs / 1e3);
    fprintf(stderr, "\nM5: %llu / %llu distinct reads identical (%.6f%%), reads weighted %llu / %llu (%.6f%%); differ %llu (unmapType only %llu)\n",
            same, n, 100.0 * same / n, sameW, nReadsW, 100.0 * sameW / nReadsW, diff, sameUnmapOnly);
    fprintf(stderr, "    unmapped by STAR: %llu (%llu identical), mapped: %llu (%llu identical)\n", classN[0], classSame[0], classN[1], classSame[1]);
    fprintf(stderr, "    GPU lane A %.1f ms (%llu reads), lane M %.1f ms (%zu reads), lane B %.1f ms (%zu reads) = %.2f M distinct reads/s on the GPU\n",
            A.ms, n, M.ms, noMid ? (size_t)0 : idxM.size(), B.ms, idxB.size(), rate / 1e6);
    { const char* wn[] = {"none", "lread", "split", "pc", "win", "wa", "winbin", "exons", "wtr", "keep", "blob", "steps"};
      fprintf(stderr, "    lane M overflow why:"); for (int k = 0; k < 12; k++) if (whyM[k]) fprintf(stderr, " %s=%llu", wn[k], whyM[k]); fprintf(stderr, "\n"); }
    { const char* wn[] = {"none", "lread", "split", "pc", "win", "wa", "winbin", "exons", "wtr", "keep", "blob", "steps"};
      fprintf(stderr, "    lane A overflow why:"); for (int k = 0; k < 12; k++) if (whyA[k]) fprintf(stderr, " %s=%llu", wn[k], whyA[k]); fprintf(stderr, "\n"); }
    fprintf(stderr, "    heavy lane: %zu reads (%.4f%%), %.2f s on %d host threads, failures %llu; why:", heavy.size(), 100.0 * heavy.size() / n,
            heavySec, nHostThreads, heavyFail);
    const char* whyName[] = {"none", "lread", "split", "pc", "win", "wa", "winbin", "exons", "wtr", "keep", "blob", "steps"};
    for (int k = 0; k < 12; k++) if (whyCount[k]) fprintf(stderr, " %s=%llu", whyName[k], whyCount[k]);
    fprintf(stderr, "\n");
    if (!json.empty()) {
        FILE* f = fopen(json.c_str(), "w");
        fprintf(f, "{\n  \"device\": \"%s\",\n  \"distinct_reads\": %llu,\n  \"reads\": %llu,\n  \"identical_distinct\": %llu,\n  \"identical_reads\": %llu,\n"
                   "  \"differ_distinct\": %llu,\n  \"differ_reads\": %llu,\n  \"star_unmapped\": %llu,\n  \"star_unmapped_identical\": %llu,\n"
                   "  \"star_mapped\": %llu,\n  \"star_mapped_identical\": %llu,\n  \"gpu_ms_median\": %.3f,\n  \"gpu_ms_runs\": [",
                prop.name, n, nReadsW, same, sameW, diff, diffW, classN[0], classSame[0], classN[1], classSame[1], gpuMs);
        for (size_t k = 0; k < A.runs.size(); k++) fprintf(f, "%s%.3f", k ? ", " : "", A.runs[k] + (k < M.runs.size() ? M.runs[k] : 0) + (k < B.runs.size() ? B.runs[k] : 0));
        fprintf(f, "],\n  \"gpu_distinct_reads_per_s\": %.0f,\n  \"heavy_reads\": %zu,\n  \"heavy_host_s\": %.3f,\n  \"heavy_host_threads\": %d,\n  \"heavy_failures\": %llu,\n  \"heavy_why\": {",
                rate, heavy.size(), heavySec, nHostThreads, heavyFail);
        bool first = true; for (int k = 0; k < 12; k++) if (whyCount[k]) { fprintf(f, "%s\"%s\": %llu", first ? "" : ", ", whyName[k], whyCount[k]); first = false; }
        fprintf(f, "},\n  \"lane_a_ms\": %.3f,\n  \"lane_m_ms\": %.3f,\n  \"lane_m_reads\": %zu,\n  \"lane_b_ms\": %.3f,\n  \"lane_b_reads\": %zu,\n  \"lane_a_phase_ms\": [%.3f, %.3f, %.3f],\n  \"lane_b_phase_ms\": [%.3f, %.3f, %.3f],\n  \"lane_a_local\": %s,\n  \"work_bytes\": [%zu, %zu],\n  \"index_load_s\": %.1f,\n",
                A.ms, M.ms, noMid ? (size_t)0 : idxM.size(), B.ms, idxB.size(), A.phaseMs[0], A.phaseMs[1], A.phaseMs[2], B.phaseMs[0], B.phaseMs[1], B.phaseMs[2], localWork ? "true" : "false", sizeof(Work<LaneA>), sizeof(Work<LaneB>), tIndex);
        fprintf(f, "  \"e4\": {\"build\": \"%s\", \"label\": \"%s\", \"presort\": %s, \"p3order\": \"%s\", \"queue\": %s, \"queue_bps\": %d, \"fused\": \"%s%s%s\", "
                   "\"fused_bps\": [%d, %d, %d], \"fused_bits\": %llu, \"tail\": %s, \"batches\": [%lld, %lld, %lld], \"budgets\": [%lld, %lld, %lld],\n    \"lanes\": {",
                M5_GIT, label.c_str(), presort ? "true" : "false", g_bin ? "cost-asc(--bin)" : p3orderName[g_p3order], g_queue ? "true" : "false", g_queueBps,
                (g_fused & 1) ? "A" : "", (g_fused & 2) ? "M" : "", (g_fused & 4) ? "B" : "", g_fusedBps[0], g_fusedBps[1], g_fusedBps[2], g_fusedBits,
                g_tail ? "true" : "false", nbA, nbM, nbB, budgetA, budgetM, budgetB);
        const LaneOut* LL[3] = {&A, &M, &B}; const char* LN[3] = {"A", "M", "B"};
        for (int k = 0; k < 3; k++) {
            const LaneOut& X = *LL[k];
            fprintf(f, "%s\n      \"%s\": {\"mode\": \"%s\", \"ms\": %.3f, \"runs\": [", k ? "," : "", LN[k], X.mode.c_str(), X.ms);
            for (size_t r = 0; r < X.runs.size(); r++) fprintf(f, "%s%.3f", r ? ", " : "", X.runs[r]);
            fprintf(f, "], \"phase_ms\": [%.3f, %.3f, %.3f], \"work_gb\": %.3f, \"work_threads\": %llu, \"device_used_gb\": %.3f, \"occ_blocks_per_sm\": %d, "
                       "\"fused_fallback_reads\": %llu,\n        \"tail\": {\"span_ms\": %.3f, \"slot_fill\": %.4f, \"warp_balance\": %.4f, \"lane_balance\": %.4f, "
                       "\"drain_share\": %.4f, \"thread_slot_util\": %.4f, \"resident_warps\": %.0f, \"launches\": %.0f}}",
                    X.phaseMs[0], X.phaseMs[1], X.phaseMs[2], X.workGB, X.workThreads, X.usedGB, X.occBlocks, X.fallback,
                    X.tail[0], X.tail[1], X.tail[2], X.tail[3], X.tail[4], X.tail[5], X.tail[6], X.tail[7]);
        }
        fprintf(f, "}}\n}\n");
        fclose(f);
    }
    return 0;
}
