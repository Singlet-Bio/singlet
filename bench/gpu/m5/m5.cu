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
#include "stitch.cuh"
#ifdef M5_CENSUS
#include "census.cuh"   // E2: per-read / per-window records (--census PREFIX)
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
__device__ unsigned long long g_monoBudget;   // --mono-budget: stitching step budget of the one-kernel lanes (default 0 = none, as before)
#ifdef M5_TAIL
#include "../../fable/profile/tail.cuh"      // E1: per-thread start/end of k_p1/k_p3 (batch tails); not in the default build
#else
#define TAIL_BEGIN(which, i) do { } while (0)
#define TAIL_END(which, i) do { } while (0)
#define TAIL_INFO(i, ne, nw, st, sc) do { } while (0)
#endif
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
    wp->wbEpoch = 0; wp->stepBudget = g_monoBudget;
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

struct LaneOut { std::vector<uint8_t> out, status, why; std::vector<uint16_t> len; std::vector<int8_t> unmap; double ms = 0; std::vector<double> runs; double phaseMs[4] = {0, 0, 0, 0};
#ifdef M5_CENSUS
                 std::vector<CenRead> cen; std::vector<CenWin> win; unsigned long long winN = 0;
#endif
};

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
//
// E7 (seeds, docs/gpu-engine/fable/E7-seeds.md), compile-time, each exact:
//  M5_CMPU     compareSeqToGenome's four direction cases as one strided loop (stitch.cuh)
//  M5_SEEDFLAT phase 1 with flat loops: the seed-loop nest as an iterator, one maxMappableLength2strands call per loop
//              iteration at one call site; the window creation as one loop over the anchors' SA entries (implies M5_CMPU)
//  M5_P1SPLIT  (with M5_SEEDFLAT) phase 1 as two kernels: k_p1s, persistent warps that make one seed call per lane per
//              iteration and refill a lane with the batch's next read as soon as its read is done (warp-aggregated
//              atomic counter), then k_p1w, the windows (thread per read)
//  M5_P1WPERS  (with M5_P1SPLIT) the windows kernel persistent as well (k_p1wp: one anchor SA entry or one window's
//              flanks per lane per iteration, lanes refilled from the batch)
//  M5_G2       compareSeqToGenome on a 2-bit genome text (stitch.cuh; needs M5_SEEDFLAT)
//  M5_P1MINB=n __launch_bounds__(128, n) on the phase-1 kernels (register cap for occupancy)
//  M5_CMPNI    compareSeqToGenome out of line (one copy instead of five inlined ones; stitch.cuh)
//  M5_P1FUSE   (with M5_P1SPLIT) the windows in k_p1s, for each read as its seed search completes (no windows kernel)
//  M5_PREF     binary-search prefetch: the SA words of both next midpoints into L2 during each compare (stitch.cuh)
//  M5_PREF2    M5_PREF plus the two SA values loaded and the genome lines of both next compares prefetched
//  M5_PREF3    M5_PREF (or with M5_PREF2) plus the SAi words of the later iDist passes prefetched before the first
//  M5_P1WPB    (with M5_P1SPLIT) the persistent windows kernel for lane B only (lanes A and M: thread per read)
//  M5_SEEDSMEM (with M5_P1SPLIT) k_p1s keeps the seed iterator (SeedIt) in shared memory, 128 threads per block
//  M5_STEPSMEM (with M5_SEEDSTEP) the stepped call's state (MmIt) in shared memory, 128 threads per block
//  M5_SEEDSTEP (with M5_P1SPLIT) k_p1s steps maxMappableLength2strands one memory operation at a time (an SAi lookup or
//              one compareSeqToGenome per lane per iteration, at one call site; stitch.cuh MmIt / mmAdvance)
#ifdef M5_SEEDFLAT
#ifndef M5_CMPU
#define M5_CMPU
#endif
#if defined(M5_CENSUS) || defined(M5_TAIL)
#error "M5_SEEDFLAT: no census / tail probes"
#endif
#endif
#if defined(M5_P1FUSE) && (!defined(M5_P1SPLIT) || defined(M5_P1WPERS))
#error "M5_P1FUSE needs M5_P1SPLIT, without M5_P1WPERS"
#endif
#if defined(M5_SEEDSTEP) && !defined(M5_P1SPLIT)
#error "M5_SEEDSTEP needs M5_P1SPLIT"
#endif
#if defined(M5_P1SPLIT) && !defined(M5_SEEDFLAT)
#error "M5_P1SPLIT needs M5_SEEDFLAT"
#endif
#if defined(M5_P1WPERS) && !defined(M5_P1SPLIT)
#error "M5_P1WPERS needs M5_P1SPLIT"
#endif
#if defined(M5_P1WPB) && (!defined(M5_P1SPLIT) || defined(M5_P1FUSE))
#error "M5_P1WPB needs M5_P1SPLIT, without M5_P1FUSE"
#endif
// the windows kernel of lane C: persistent (k_p1wp) or thread per read (k_p1w)
template <class C> constexpr bool p1WinPers() {
#if defined(M5_P1WPERS)
    return true;
#elif defined(M5_P1WPB)
    return C::NW >= 128;   // lane B only
#else
    return false;
#endif
}
#ifdef M5_P1MINB
#define M5_P1_BOUNDS __launch_bounds__(128, M5_P1MINB)
#else
#define M5_P1_BOUNDS
#endif
#ifdef M5_P1SPLIT
// seeds: persistent warps; every iteration each lane with a read makes its read's next maxMappableLength2strands call
// (M5_SEEDSTEP: the next memory operation of that call). M5_P1FUSE: a read whose seed search is complete gets its windows
// in this kernel (p1Windows at one site, in the iteration its seeds finish) and no windows kernel runs.
template <class C>
__global__ void M5_P1_BOUNDS k_p1s(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx,
                                   u64 b0, u64 n, Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* status, unsigned long long* ctr, u64* nEnt) {
    const unsigned FULL = 0xffffffffu, lane = threadIdx.x & 31;
    u64 k = 0;
    int st = 0;            // 0: needs a read, 1: holds a read with a pending call, 2: the batch is exhausted, 3: seeds done
#ifdef M5_SEEDSMEM
    __shared__ SeedIt smIt[128];   // the seed iterator in shared memory (fewer registers, more resident warps)
    SeedIt& it = smIt[threadIdx.x];
#else
    SeedIt it;
#endif
#ifdef M5_SEEDSTEP
#ifdef M5_STEPSMEM
    __shared__ MmIt smM[128];   // the call's state in shared memory (registers: temporaries only)
    MmIt& m = smM[threadIdx.x];
#else
    MmIt m;
#endif
    MmRes rr;
#endif
    long long tc = 0;
    (void)nEnt;
    for (;;) {
        // refill: the lanes without a read take the batch's next reads (consecutive per warp); a read that overflows at
        // set-up is finished at once and its lane takes another; a read without any call goes to state 3
        for (;;) {
            const unsigned need = __ballot_sync(FULL, st == 0);
            if (!need) break;
            const int leader = __ffs((int)need) - 1;
            unsigned long long base = 0;
            if ((int)lane == leader) base = atomicAdd(ctr, (unsigned long long)__popc(need));
            base = __shfl_sync(FULL, base, leader);
            if (st == 0) {
                k = base + __popc(need & ((1u << lane) - 1));
                if (k >= n) st = 2;
                else {
                    tc = clock64();
                    const u64 i = idx ? idx[b0 + k] : b0 + k;
                    Work<C>& w = work[k];
                    const Status s = p1Setup(P, w, rbuf[k], rb + roff[i], rlen[i], it);
                    if (s != Status::Ok) {
                        status[k] = (uint8_t)s;
#ifdef M5_P1FUSE
                        nEnt[k] = 0;
#endif
                    } else if (seedNext(P, w, it)) {
                        st = 1;
#ifdef M5_SEEDSTEP
                        mmBegin(P, m, it);
#endif
                    } else {
#ifdef M5_P1FUSE
                        st = 3;
#else
                        status[k] = (uint8_t)Status::Ok; w.clk[0] += clock64() - tc;
#endif
                    }
                }
            }
        }
        if (__all_sync(FULL, st == 2)) break;
        if (st == 1) {
            Work<C>& w = work[k];
            M5_READ1(Read1, rbuf[k], w.Lread);
#ifdef M5_SEEDSTEP
            // one memory operation of the read's current maxMappableLength2strands call, then its control flow up to
            // the next operation; a complete call moves the seed iterator on (the next call starts with no operation)
            bool cr = false;
            const u64 v = mmOp(g, Read1, m, cr);
            const int r = mmAdvance(P, g, w, Read1, m, rr, v, cr);
            if (r < 0) {
                status[k] = (uint8_t)Status::Overflow; st = 0;
#ifdef M5_P1FUSE
                nEnt[k] = 0;
#endif
            } else if (r > 0) {
                seedAfter(it, (m5::uint)m.maxLbest);
                if (!seedNext(P, w, it)) {
#ifdef M5_P1FUSE
                    st = 3;
#else
                    status[k] = (uint8_t)Status::Ok; w.clk[0] += clock64() - tc; st = 0;
#endif
                } else mmBegin(P, m, it);
            }
#else
            m5::uint L;
            seedCall(P, g, w, Read1, it, L);
            if (w.ov) {
                status[k] = (uint8_t)Status::Overflow; st = 0;
#ifdef M5_P1FUSE
                nEnt[k] = 0;
#endif
            } else {
                seedAfter(it, L);
                if (!seedNext(P, w, it)) {
#ifdef M5_P1FUSE
                    st = 3;
#else
                    status[k] = (uint8_t)Status::Ok; w.clk[0] += clock64() - tc; st = 0;
#endif
                }
            }
#endif
        }
#ifdef M5_P1FUSE
        if (st == 3) {   // the read's seed search is complete (status Ok): its windows
            Work<C>& w = work[k];
            w.clk[0] += clock64() - tc;
            const Status s = p1Windows(P, g, w);
            status[k] = (uint8_t)s;
            nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
            st = 0;
        }
#endif
    }
}
#if defined(M5_P1WPERS) || defined(M5_P1WPB)
// windows, persistent warps (M5_P1WPERS): every iteration each lane with a read makes one step of its read's windows (one SA
// entry of an anchor, or one window's flanks); lanes refill from the batch as in k_p1s
template <class C>
__global__ void M5_P1_BOUNDS k_p1wp(Params P, GenomeView g, u64 n, Work<C>* work, uint8_t* status, u64* nEnt, unsigned long long* ctr) {
    const unsigned FULL = 0xffffffffu, lane = threadIdx.x & 31;
    u64 k = 0;
    int st = 0;
    WinIt it;
    long long tc = 0;
    for (;;) {
        for (;;) {
            const unsigned need = __ballot_sync(FULL, st == 0);
            if (!need) break;
            const int leader = __ffs((int)need) - 1;
            unsigned long long base = 0;
            if ((int)lane == leader) base = atomicAdd(ctr, (unsigned long long)__popc(need));
            base = __shfl_sync(FULL, base, leader);
            if (st == 0) {
                k = base + __popc(need & ((1u << lane) - 1));
                if (k >= n) st = 2;
                else if ((Status)status[k] != Status::Ok) nEnt[k] = 0;
                else {
                    tc = clock64();
                    if (p1WinBegin(P, work[k], it)) st = 1;
                    else nEnt[k] = 0;   // not stitched: status stays Ok
                }
            }
        }
        if (__all_sync(FULL, st == 2)) break;
        if (st == 1) {
            Work<C>& w = work[k];
            const int r = p1WinStep(P, g, w, it);
            if (r < 0) { status[k] = (uint8_t)Status::Overflow; nEnt[k] = 0; st = 0; }
            else if (r > 0) { nEnt[k] = w.nEntries; w.clk[1] += clock64() - tc; st = 0; }
        }
    }
}
#endif
// windows from the anchors and flanks, thread per read (reads that overflowed in k_p1s are skipped)
template <class C>
__global__ void M5_P1_BOUNDS k_p1w(Params P, GenomeView g, u64 n, Work<C>* work, uint8_t* status, u64* nEnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    Status s = (Status)status[k];
    if (s == Status::Ok) s = p1Windows(P, g, w);
    status[k] = (uint8_t)s;
    nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
}
#endif
template <class C>
__global__ void M5_P1_BOUNDS k_p1(Params P, GenomeView g, const char* rb, const u64* roff, const uint16_t* rlen, const uint32_t* idx, u64 b0, u64 n,
                     Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* status, u64* nEnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
#ifdef M5_SEEDFLAT
    {
        const u64 i = idx ? idx[b0 + k] : b0 + k;
        Work<C>& w = work[k];
        const Status s = mapPhase1Flat(P, g, w, rbuf[k], rb + roff[i], rlen[i]);
        status[k] = (uint8_t)s;
        nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
        return;
    }
#endif
    TAIL_BEGIN(0, k);
    u64 i = idx ? idx[b0 + k] : b0 + k;
    Work<C>& w = work[k];
#ifdef M5_CENSUS
    long long cnT = clock64();
#endif
    Status s = mapPhase1(P, g, w, rbuf[k], rb + roff[i], rlen[i]);
#ifdef M5_CENSUS
    w.cn[0] = clock64() - cnT;
#endif
    status[k] = (uint8_t)s;
    nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
    TAIL_END(0, k);
}
template <class C>
__global__ void k_p2(Params P, GenomeView g, u64 n, const Work<C>* work, const u64* nEnt, const u64* base, uint32_t* bits, long long* scanCyc) {
    u64 wid = (blockIdx.x * (u64)blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (wid >= n || nEnt[wid] == 0) return;
#ifdef M5_CENSUS
    long long cnT = clock64();
#else
    (void)scanCyc;
#endif
    const Work<C>& w = work[wid];
    u64 off = base[wid];
    for (m5::uint iP = 0; iP < w.nP; iP++) {
        const u64 s0 = w.PC[iP][PC_SAstart], cnt = w.PC[iP][PC_SAend] - s0 + 1;
        for (u64 j = lane; j < cnt; j += 32)
            if (entryHit(P, g, w, iP, (m5::uint)(s0 + j))) atomicOr(&bits[(off + j) >> 5], 1u << ((off + j) & 31));
        off += cnt;
    }
#ifdef M5_CENSUS
    __syncwarp();
    if (lane == 0 && scanCyc) scanCyc[wid] = clock64() - cnT;
#endif
}
template <class C>
__global__ void k_p3(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* nEnt, const u64* base,
                     const uint32_t* bits, uint8_t* status, uint8_t* out, uint16_t* outLen, uint8_t* why, int8_t* unmap, const uint32_t* perm,
                     const long long* scanCyc, void* cenP, void* cenWinP, unsigned long long* cenWinN, u64 cenWinCap, uint8_t cenLane) {
    u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (t >= n) return;
    u64 k = perm ? perm[t] : t;   // --bin: threads take the batch's reads in order of estimated cost (each read is independent)
    TAIL_BEGIN(1, k);             // E1 tail records are indexed by read k (= thread index t without --bin)
    Work<C>& w = work[k];
    w.stepBudget = budget;
    u64 o = b0 + k;
    Status s = (Status)status[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
#ifdef M5_CENSUS
    long long cnT0 = clock64();
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    w.cn[2] = clock64() - cnT0;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
    w.cn[3] = clock64() - cnT0;
    w.cn[1] = nEnt[k] ? scanCyc[k] : 0;
    {
        CenRead* cen = (CenRead*)cenP;
        cenFill(P, w, cenLane, (uint8_t)s, (uint16_t)bl.n, cen[o]);
        if (s == Status::Ok && w.cnFinish) {   // per-window records, in the lane that gives the read's result
            m5::uint nw = 0, nWall = mn<m5::uint>(w.nWall, (m5::uint)C::NW);
            for (m5::uint iW = 0; iW < nWall; iW++) if (w.cnWinSteps[iW]) nw++;
            if (nw) {
                unsigned long long p = atomicAdd(cenWinN, (unsigned long long)nw);
                CenWin* cw = (CenWin*)cenWinP;
                for (m5::uint iW = 0; iW < nWall; iW++) {
                    if (!w.cnWinSteps[iW]) continue;
                    if (p < cenWinCap) {
                        CenWin& r = cw[p];
                        r.read = (uint32_t)o; r.iW = (uint16_t)iW; r.lane = cenLane; r.nWA = (uint8_t)w.nWA[iW];
                        r.steps = w.cnWinSteps[iW]; r.cyc = w.cnWinCyc[iW]; r.nTr = w.cnWinTr[iW]; r.pad = 0;
                    }
                    p++;
                }
            }
        }
    }
#else
    (void)scanCyc; (void)cenP; (void)cenWinP; (void)cenWinN; (void)cenWinCap; (void)cenLane;
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
#endif
    status[k] = (uint8_t)s; why[o] = (uint8_t)w.ov; outLen[o] = (uint16_t)bl.n; unmap[o] = (int8_t)w.unmapType;
    TAIL_INFO(k, nEnt[k], w.nW, (uint32_t)w.nSteps, (unsigned long long)(w.clk[2] + w.clk[3]));
    for (int c = 0; c < 8; c++) { atomicAdd(&g_clk[c], (unsigned long long)w.clk[c]); w.clk[c] = 0; }
    atomicAdd(&g_scan, (unsigned long long)nEnt[k]);
    TAIL_END(1, k);
}
// cost key of each read after phase 1, for --bin: windows, then SA entries to assign (reads that need no phase 3 first)
template <class C>
__global__ void k_cost(u64 n, const Work<C>* work, const u64* nEnt, u64* key, uint32_t* perm) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    key[k] = nEnt[k] ? ((u64)work[k].nW << 40) | min(nEnt[k], (u64)0xFFFFFFFFFFULL) : 0;
    perm[k] = (uint32_t)k;
}
#ifdef M5_G2
// E7 M5_G2: pack the byte genome (Gb = STAR's G - pad, nb bytes; position p is Gb[p + pad], outside = 5) into G2 / NB
__device__ inline char g2Byte(const char* Gb, u64 nb, long long pad, long long p) {
    const long long idx = p + pad;
    return (idx >= 0 && (u64)idx < nb) ? Gb[idx] : (char)5;
}
__global__ void k_g2pack(const char* Gb, u64 nb, long long pad, u64 nw, u64* G2) {
    const u64 w = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (w >= nw) return;
    u64 v = 0;
    for (int j = 0; j < 32; j++) v |= (u64)(g2Byte(Gb, nb, pad, (long long)(w * 32 + j) - G2OFF) & 3) << (2 * j);
    G2[w] = v;
}
__global__ void k_nbpack(const char* Gb, u64 nb, long long pad, u64 nblk, uint32_t* NB) {
    const u64 b = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (b >= nblk) return;
    bool n = false;
    for (int j = 0; j < 64; j++) n |= (unsigned char)g2Byte(Gb, nb, pad, (long long)(b * 64 + j) - G2OFF) > 3;
    if (n) atomicOr(&NB[b >> 5], 1u << (b & 31));
}
#if !defined(M5_SEEDFLAT)
#error "M5_G2 needs M5_SEEDFLAT (the packed read is set up by p1Setup)"
#endif
#endif
static bool g_bin = false;
__global__ void k_copy_status(const uint8_t* s, uint8_t* dst, u64 n) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k < n) dst[k] = s[k];
}

template <class C>
static LaneOut runStaged(const Params& P, const GenomeView& D, const char* dRb, const u64* dRoff, const uint16_t* dRlen, const uint32_t* dIdx,
                         u64 n, u64 NB, u64 budget, int reps, cudaEvent_t e0, cudaEvent_t e1, const char* name, uint8_t cenLane = 0) {
    LaneOut L;
    NB = std::min<u64>(NB, std::max<u64>(n, 1));
    uint8_t* dOut; CK(cudaMalloc(&dOut, n * (u64)C::NBLOB));
    uint16_t* dLen; CK(cudaMalloc(&dLen, n * 2)); uint8_t* dSt; CK(cudaMalloc(&dSt, n)); uint8_t* dWhy; CK(cudaMalloc(&dWhy, n));
    int8_t* dUn; CK(cudaMalloc(&dUn, n));
    Work<C>* dWork; CK(cudaMalloc(&dWork, NB * sizeof(Work<C>))); CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>)));
    ReadBuf<C::LREAD>* dRbuf; CK(cudaMalloc(&dRbuf, NB * sizeof(ReadBuf<C::LREAD>)));
    uint8_t* dBst; CK(cudaMalloc(&dBst, NB)); u64* dNent; CK(cudaMalloc(&dNent, NB * 8)); u64* dBase; CK(cudaMalloc(&dBase, NB * 8));
    u64 bitsCap = 1 << 20; uint32_t* dBits; CK(cudaMalloc(&dBits, bitsCap * 4));
    u64* dKey; CK(cudaMalloc(&dKey, NB * 8)); uint32_t* dPerm; CK(cudaMalloc(&dPerm, NB * 4));
    long long* dScanCyc = nullptr; void* dCen = nullptr; void* dCenWin = nullptr; unsigned long long* dCenWinN = nullptr; u64 cenWinCap = 0;
#ifdef M5_CENSUS
    CK(cudaMalloc(&dScanCyc, NB * 8)); CK(cudaMalloc(&dCen, n * sizeof(CenRead))); CK(cudaMemset(dCen, 0, n * sizeof(CenRead)));
    cenWinCap = n * (u64)C::NW + 1024; CK(cudaMalloc(&dCenWin, cenWinCap * sizeof(CenWin))); CK(cudaMalloc(&dCenWinN, 8));
#endif
    fprintf(stderr, "lane %s (staged): %llu reads, batches of %llu, work %zu B/read (%.2f GB)\n", name, n, NB, sizeof(Work<C>),
            NB * (double)sizeof(Work<C>) / 1e9);
    cudaEvent_t q[5]; for (auto& e : q) cudaEventCreate(&e);
    unsigned long long clk[8] = {0}, scan = 0; double phMs[4] = {0, 0, 0, 0};
#ifdef M5_P1SPLIT
    int nSm = 0, bpsS = 0; cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, 0);
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsS, k_p1s<C>, 128, 0));
    unsigned long long* dCtr; CK(cudaMalloc(&dCtr, 16));
#if defined(M5_P1WPERS) || defined(M5_P1WPB)
    int bpsW = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsW, k_p1wp<C>, 128, 0));
    fprintf(stderr, "  lane %s: k_p1s persistent, %d blocks of 128 per SM (%d threads); %s, %d blocks per SM\n", name, bpsS, nSm * bpsS * 128,
            p1WinPers<C>() ? "k_p1wp persistent" : "k_p1w", bpsW);
#else
    { int b1 = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, k_p1w<C>, 128, 0));
#ifdef M5_P1FUSE
      (void)b1; fprintf(stderr, "  lane %s: k_p1s persistent with the windows fused, %d blocks of 128 per SM (%d threads)\n", name, bpsS, nSm * bpsS * 128); }
#else
      fprintf(stderr, "  lane %s: k_p1s persistent, %d blocks of 128 per SM (%d threads); k_p1w %d blocks per SM\n", name, bpsS, nSm * bpsS * 128, b1); }
#endif
#endif
#else
    { int b1 = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&b1, k_p1<C>, 128, 0)); fprintf(stderr, "  lane %s: k_p1 %d blocks of 128 per SM\n", name, b1); }
#endif
#ifdef M5_TAIL
    TailHost tail; { int sms = 0; cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0); tail.init(NB, name, sms); }
    int bps1 = 0, bps3 = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps1, k_p1<C>, 128, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bps3, k_p3<C>, 128, 0));
#endif
    for (int r = 0; r < reps + 1; r++) {
        unsigned long long z[8] = {0}; CK(cudaMemcpyToSymbol(g_clk, z, sizeof z)); CK(cudaMemcpyToSymbol(g_scan, z, 8));
#ifdef M5_TAIL
        tail.arm(r == reps);
#endif
        double ph[4] = {0, 0, 0, 0};
        if (dCenWinN) CK(cudaMemset(dCenWinN, 0, 8));
        cudaEventRecord(e0);
        for (u64 b0 = 0; b0 < n; b0 += NB) {
            u64 nb = std::min(NB, n - b0);
            int T = 128; unsigned G = (unsigned)((nb + T - 1) / T);
#ifdef M5_P1SPLIT
            CK(cudaMemsetAsync(dCtr, 0, 8));
            cudaEventRecord(q[0]);
            k_p1s<C><<<(unsigned)std::min<u64>((u64)nSm * bpsS, G), T>>>(P, D, dRb, dRoff, dRlen, dIdx, b0, nb, dWork, dRbuf, dBst, dCtr, dNent);
            cudaEventRecord(q[4]);
#if defined(M5_P1FUSE)
            // the windows ran in k_p1s
#elif defined(M5_P1WPERS) || defined(M5_P1WPB)
            if (p1WinPers<C>()) {
                CK(cudaMemsetAsync(dCtr + 1, 0, 8));
                k_p1wp<C><<<(unsigned)std::min<u64>((u64)nSm * bpsW, G), T>>>(P, D, nb, dWork, dBst, dNent, dCtr + 1);
            } else k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent);
#else
            k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent);
#endif
#else
            cudaEventRecord(q[0]);
            k_p1<C><<<G, T>>>(P, D, dRb, dRoff, dRlen, dIdx, b0, nb, dWork, dRbuf, dBst, dNent);
#endif
            cudaEventRecord(q[1]);
            thrust::exclusive_scan(thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dBase));
            u64 lastB, lastN; CK(cudaMemcpy(&lastB, dBase + nb - 1, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&lastN, dNent + nb - 1, 8, cudaMemcpyDeviceToHost));
            u64 words = (lastB + lastN) / 32 + 2;
            if (words > bitsCap) { CK(cudaFree(dBits)); bitsCap = words * 2; CK(cudaMalloc(&dBits, bitsCap * 4)); }
            CK(cudaMemsetAsync(dBits, 0, words * 4));
#ifdef M5_TAIL
            cudaEventRecord(tail.ev[0]);
#endif
            k_p2<C><<<(unsigned)((nb * 32 + 255) / 256), 256>>>(P, D, nb, dWork, dNent, dBase, dBits, dScanCyc);
            cudaEventRecord(q[2]);
            if (g_bin) {
                k_cost<C><<<G, T>>>(nb, dWork, dNent, dKey, dPerm);
                thrust::sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm));
            }
            k_p3<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dNent, dBase, dBits, dBst, dOut, dLen, dWhy, dUn, g_bin ? dPerm : nullptr,
                              dScanCyc, dCen, dCenWin, dCenWinN, cenWinCap, cenLane);
#ifdef M5_TAIL
            cudaEventRecord(tail.ev[1]);
#endif
            k_copy_status<<<G, T>>>(dBst, dSt + b0, nb);
            cudaEventRecord(q[3]);
            CK(cudaEventSynchronize(q[3])); CK(cudaGetLastError());
            float m; cudaEventElapsedTime(&m, q[0], q[1]); ph[0] += m; cudaEventElapsedTime(&m, q[1], q[2]); ph[1] += m;
            cudaEventElapsedTime(&m, q[2], q[3]); ph[2] += m;
#ifdef M5_P1SPLIT
            cudaEventElapsedTime(&m, q[0], q[4]); ph[3] += m;
#endif
#ifdef M5_TAIL
            if (r + 1 == reps) tail.events(e0, q, (int)(b0 / NB), nb);   // event times from the rep before the analysed one (no host work in between)
            if (r == reps) { tail.batch(0, nb, T, bps1, (int)(b0 / NB)); tail.batch(1, nb, T, bps3, (int)(b0 / NB)); }
#endif
        }
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
        float m; cudaEventElapsedTime(&m, e0, e1); if (r > 0) { L.runs.push_back(m); for (int k = 0; k < 4; k++) phMs[k] += ph[k] / reps; }
        fprintf(stderr, "  lane %s run %d: %.1f ms (phase 1 %.1f, scan %.1f, phase 3 %.1f; seeds kernel %.1f)\n", name, r, m, ph[0], ph[1], ph[2], ph[3]);
    }
#ifdef M5_TAIL
    tail.summary(); tail.release();
#endif
    CK(cudaMemcpyFromSymbol(clk, g_clk, sizeof clk)); CK(cudaMemcpyFromSymbol(&scan, g_scan, 8));
    { double t = (double)clk[0] + clk[1] + clk[2] + clk[3]; if (t > 0)
        fprintf(stderr, "  lane %s thread cycles: seeds %.1f%%, windows %.1f%%, stitching %.1f%%, select+encode %.1f%%; %.1f SA entries per read\n",
                name, 100 * clk[0] / t, 100 * clk[1] / t, 100 * clk[2] / t, 100 * clk[3] / t, (double)scan / n); }
    if (clk[2] > 0) fprintf(stderr, "  lane %s inside stitching: stitchAlignToTranscript %.1f%%, finalization %.1f%% (of which recording %.1f%%), copies %.1f%%\n",
                            name, 100.0 * clk[4] / clk[2], 100.0 * clk[5] / clk[2], 100.0 * clk[7] / clk[2], 100.0 * clk[6] / clk[2]);
    L.ms = medianv(L.runs);
    L.phaseMs[0] = phMs[0]; L.phaseMs[1] = phMs[1]; L.phaseMs[2] = phMs[2]; L.phaseMs[3] = phMs[3];
    fprintf(stderr, "  lane %s phases (mean of timed runs): phase 1 %.2f ms (seeds kernel %.2f), scan %.2f, phase 3 %.2f\n", name, phMs[0], phMs[3], phMs[1], phMs[2]);
#ifdef M5_P1SPLIT
    cudaFree(dCtr);
#endif
    L.out.resize(n * (u64)C::NBLOB); L.status.resize(n); L.why.resize(n); L.len.resize(n); L.unmap.resize(n);
    CK(cudaMemcpy(L.out.data(), dOut, L.out.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.len.data(), dLen, n * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.status.data(), dSt, n, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(L.why.data(), dWhy, n, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(L.unmap.data(), dUn, n, cudaMemcpyDeviceToHost));
#ifdef M5_CENSUS
    L.cen.resize(n); CK(cudaMemcpy(L.cen.data(), dCen, n * sizeof(CenRead), cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&L.winN, dCenWinN, 8, cudaMemcpyDeviceToHost));
    L.win.resize(std::min<u64>(L.winN, cenWinCap));
    if (!L.win.empty()) CK(cudaMemcpy(L.win.data(), dCenWin, L.win.size() * sizeof(CenWin), cudaMemcpyDeviceToHost));
    fprintf(stderr, "  lane %s census: %llu window records (capacity %llu)\n", name, L.winN, cenWinCap);
    cudaFree(dScanCyc); cudaFree(dCen); cudaFree(dCenWin); cudaFree(dCenWinN);
#endif
    cudaFree(dOut); cudaFree(dLen); cudaFree(dSt); cudaFree(dWhy); cudaFree(dUn); cudaFree(dWork); cudaFree(dRbuf);
    cudaFree(dBst); cudaFree(dNent); cudaFree(dBase); cudaFree(dBits); cudaFree(dKey); cudaFree(dPerm);
    return L;
}

// E7 (--p1probe): phase 1 alone over reads idx (or all n) with lane C's capacities, timing only (median of reps after a
// warm-up), plus how many reads would stay in lane A's capacities (status Ok, nW <= LaneA::NW): the cost of "phase 1 once
// with lane M's window capacity" against the cascade's phase-1 time of lanes A + M
template <class C>
__global__ void k_p1count(u64 n, const Work<C>* work, const uint8_t* status, unsigned long long* cnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    if ((Status)status[k] != Status::Ok) atomicAdd(&cnt[0], 1ULL);
    else if (work[k].stitched && work[k].nW > (m5::uint)LaneA::NW) atomicAdd(&cnt[1], 1ULL);
    else atomicAdd(&cnt[2], 1ULL);
}
template <class C>
static double probeP1(const Params& P, const GenomeView& D, const char* dRb, const u64* dRoff, const uint16_t* dRlen, u64 n, u64 NB, int reps,
                      cudaEvent_t e0, cudaEvent_t e1, const char* name) {
    NB = std::min<u64>(NB, std::max<u64>(n, 1));
    Work<C>* dWork; CK(cudaMalloc(&dWork, NB * sizeof(Work<C>))); CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>)));
    ReadBuf<C::LREAD>* dRbuf; CK(cudaMalloc(&dRbuf, NB * sizeof(ReadBuf<C::LREAD>)));
    uint8_t* dBst; CK(cudaMalloc(&dBst, NB)); u64* dNent; CK(cudaMalloc(&dNent, NB * 8));
    unsigned long long* dCnt; CK(cudaMalloc(&dCnt, 24));
#ifdef M5_P1SPLIT
    int nSm = 0, bpsS = 0; cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, 0);
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsS, k_p1s<C>, 128, 0));
    unsigned long long* dCtr; CK(cudaMalloc(&dCtr, 16));
#if defined(M5_P1WPERS) || defined(M5_P1WPB)
    int bpsW = 0; CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsW, k_p1wp<C>, 128, 0));
#endif
#endif
    std::vector<double> runs; unsigned long long cnt[3] = {0, 0, 0};
    CK(cudaMemset(dCnt, 0, 24));   // counted in the warm-up run only
    for (int r = 0; r < reps + 1; r++) {
        cudaEventRecord(e0);
        for (u64 b0 = 0; b0 < n; b0 += NB) {
            u64 nb = std::min(NB, n - b0);
            int T = 128; unsigned G = (unsigned)((nb + T - 1) / T);
#ifdef M5_P1SPLIT
            CK(cudaMemsetAsync(dCtr, 0, 16));
            k_p1s<C><<<(unsigned)std::min<u64>((u64)nSm * bpsS, G), T>>>(P, D, dRb, dRoff, dRlen, nullptr, b0, nb, dWork, dRbuf, dBst, dCtr, dNent);
#if defined(M5_P1FUSE)
#elif defined(M5_P1WPERS) || defined(M5_P1WPB)
            if (p1WinPers<C>()) k_p1wp<C><<<(unsigned)std::min<u64>((u64)nSm * bpsW, G), T>>>(P, D, nb, dWork, dBst, dNent, dCtr + 1);
            else k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent);
#else
            k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent);
#endif
#else
            k_p1<C><<<G, T>>>(P, D, dRb, dRoff, dRlen, nullptr, b0, nb, dWork, dRbuf, dBst, dNent);
#endif
            if (r == 0) k_p1count<C><<<G, T>>>(nb, dWork, dBst, dCnt);   // warm-up run only
        }
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float m; cudaEventElapsedTime(&m, e0, e1); if (r > 0) runs.push_back(m);
    }
    CK(cudaMemcpy(cnt, dCnt, 24, cudaMemcpyDeviceToHost));
    double med = medianv(runs);
    fprintf(stderr, "p1probe %s: phase 1 of %llu reads with this lane's capacities in batches of %llu: %.1f ms (median of %d after a warm-up); "
                    "overflow %llu, more than %d windows %llu, within lane A's windows %llu\n",
            name, n, NB, med, reps, cnt[0], LaneA::NW, cnt[1], cnt[2]);
    cudaFree(dWork); cudaFree(dRbuf); cudaFree(dBst); cudaFree(dNent); cudaFree(dCnt);
#ifdef M5_P1SPLIT
    cudaFree(dCtr);
#endif
    return med;
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
    std::string gdir, truth, json, input, emit, census; int reps = 3, bps = 4, nthr = 128, bpsB = 1, nthrB = 64; bool localWork = false, mono = false; long long nbA = 131072, nbB = 32768, nbM = 131072, budgetA = 0, budgetB = 0, budgetM = 0, monoBudget = 0; bool noMid = false, p1probe = false;
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
        else if (a == "--no-mid") noMid = true;          // lanes A, B only
        else if (a == "--p1probe") p1probe = true;      // E7: time phase 1 alone over every read with lane A's and lane M's capacities
        else if (a == "--mono-budget") monoBudget = atoll(argv[++i]);   // E1: step budget of the one-kernel lanes (--mono; default 0 = none)
        else if (a == "--input") input = argv[++i];     // "count sequence" lines (uniq -c): map these instead of the truth reads
        else if (a == "--emit") emit = argv[++i];       // write (sequence, result) for the sequence cache preload (inject/)
        else if (a == "--census") census = argv[++i];   // E2 (build with -DM5_CENSUS): PREFIX.reads.bin, PREFIX.wins.bin, PREFIX.meta.json
    }
    if (gdir.empty() || truth.empty()) { fprintf(stderr, "usage: m5 --genome-dir D --truth PREFIX [--json F]\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    { unsigned long long mb = (unsigned long long)monoBudget; CK(cudaMemcpyToSymbol(g_monoBudget, &mb, 8)); }
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
#ifdef M5_G2
    {   // E7: the genome text 2 bits per base and the N-block bits, built on the device from the byte genome
        const u64 npos = H.nGenome + 2 * (u64)G2OFF + 128, nw = npos / 32 + 2, nblk = npos / 64 + 2, nbw = nblk / 32 + 2;
        u64* dG2; uint32_t* dNB; CK(cudaMalloc(&dG2, nw * 8)); CK(cudaMalloc(&dNB, nbw * 4)); CK(cudaMemset(dNB, 0, nbw * 4));
        k_g2pack<<<(unsigned)((nw + 255) / 256), 256>>>(dG, (u64)hG.size(), (long long)PAD, nw, dG2);
        k_nbpack<<<(unsigned)((nblk + 255) / 256), 256>>>(dG, (u64)hG.size(), (long long)PAD, nblk, dNB);
        CK(cudaDeviceSynchronize()); CK(cudaGetLastError());
        D.G2 = dG2; D.NB = dNB;
        unsigned long long nN = 0; { std::vector<uint32_t> h(nbw); CK(cudaMemcpy(h.data(), dNB, nbw * 4, cudaMemcpyDeviceToHost)); for (auto x : h) nN += __builtin_popcount(x); }
        fprintf(stderr, "2-bit genome: %.2f GB, N blocks %.1f MB (%llu of %llu 64-base blocks hold a byte > 3)\n", nw * 8 / 1e9, nbw * 4 / 1e6, nN, nblk);
    }
#endif
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
    u64 n = T.rlen.size(), nReadsW = 0; for (auto c : T.count) nReadsW += c;
    fprintf(stderr, "truth: %llu distinct reads (%llu reads), loaded in %.1f s\n", n, nReadsW, now() - t0);

    // ---- GPU lanes: A (small capacities, every read), B (wide capacities, A's overflow); then the host (STAR's limits)
    char* dRb = toDev(T.rb); u64* dRoff = toDev(T.roff); uint16_t* dRlen = toDev(T.rlen);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    CK(cudaDeviceSetLimit(cudaLimitStackSize, localWork ? (int)(sizeof(Work<LaneA>) + 4096) : 16 * 1024));
    if (p1probe) {
        probeP1<LaneA>(P, D, dRb, dRoff, dRlen, n, (u64)nbA, reps, e0, e1, "A-all");
        probeP1<LaneM>(P, D, dRb, dRoff, dRlen, n, (u64)nbM, reps, e0, e1, "M-all");
    }
    LaneOut A = mono ? runLane<LaneA>(P, D, dRb, dRoff, dRlen, nullptr, n, prop.multiProcessorCount * bps, nthr, reps, localWork, e0, e1, "A")
                     : runStaged<LaneA>(P, D, dRb, dRoff, dRlen, nullptr, n, (u64)nbA, (u64)budgetA, reps, e0, e1, "A", 0);
    std::vector<uint32_t> idxM, idxB;
    for (u64 i = 0; i < n; i++) if (A.status[i] != (uint8_t)Status::Ok) idxM.push_back((uint32_t)i);
    LaneOut M;
    if (!idxM.empty() && !noMid) {
        uint32_t* dIdx = toDev(idxM);
        M = runStaged<LaneM>(P, D, dRb, dRoff, dRlen, dIdx, idxM.size(), (u64)nbM, (u64)budgetM, reps, e0, e1, "M", 1);
        CK(cudaFree(dIdx));
        for (u64 k = 0; k < idxM.size(); k++) if (M.status[k] != (uint8_t)Status::Ok) idxB.push_back(idxM[k]);
    } else idxB = idxM;
    CK(cudaDeviceSetLimit(cudaLimitStackSize, 16 * 1024));
    LaneOut B;
    if (!idxB.empty()) {
        uint32_t* dIdx = toDev(idxB);
        B = mono ? runLane<LaneB>(P, D, dRb, dRoff, dRlen, dIdx, idxB.size(), prop.multiProcessorCount * bpsB, nthrB, reps, false, e0, e1, "B")
                 : runStaged<LaneB>(P, D, dRb, dRoff, dRlen, dIdx, idxB.size(), (u64)nbB, (u64)budgetB, reps, e0, e1, "B", 2);
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
#ifdef M5_CENSUS
    std::vector<CenRead> hcen(heavy.size()); std::vector<std::vector<CenWin>> hwin(heavy.size());
#endif
    t0 = now();
    int nHostThreads = omp_get_max_threads();
    #pragma omp parallel
    {
        Work<Heavy>* w = new Work<Heavy>;
        w->stepBudget = 0; w->wbEpoch = 0; w->nScan = 0;
        for (int c = 0; c < 8; c++) w->clk[c] = 0;
        ReadBuf<Heavy::LREAD>* b = new ReadBuf<Heavy::LREAD>;
        std::vector<uint8_t> buf(Heavy::NBLOB);
        #pragma omp for schedule(dynamic, 1)
        for (long long k = 0; k < (long long)heavy.size(); k++) {
            u64 i = heavy[k];
            Blob bl{buf.data(), 0, Heavy::NBLOB, false};
#ifdef M5_CENSUS
            double cnT = now();
#endif
            Status s = mapRead(P, H, *w, *b, T.rb.data() + T.roff[i], T.rlen[i], bl);
            hstatus[k] = (uint8_t)s; hunmap[k] = (int8_t)w->unmapType; hout[k].assign(buf.data(), buf.data() + bl.n);
#ifdef M5_CENSUS
            w->cn[3] = (long long)((now() - cnT) * 1e9);
            cenFill(P, *w, 3, (uint8_t)s, (uint16_t)std::min(bl.n, 65535), hcen[k]);
            if (s == Status::Ok && w->cnFinish)
                for (m5::uint iW = 0; iW < w->nWall; iW++) {
                    if (!w->cnWinSteps[iW]) continue;
                    CenWin r; r.read = (uint32_t)i; r.iW = (uint16_t)std::min<m5::uint>(iW, 65535); r.lane = 3; r.nWA = (uint8_t)w->nWA[iW];
                    r.steps = w->cnWinSteps[iW]; r.cyc = 0; r.nTr = w->cnWinTr[iW]; r.pad = 0;
                    hwin[k].push_back(r);
                }
#endif
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
    std::vector<uint8_t> identV;
    for (u64 i = 0; i < n; i++) {
        const uint8_t* ours; int olen; int8_t oun; bool okr;
        result(i, ours, olen, oun, okr);
        if (!okr) heavyFail++;
        const uint8_t* tb = T.blob.data() + T.boff[i]; int tlen = (int)(T.boff[i + 1] - T.boff[i]);
        bool eq = olen == tlen && memcmp(ours, tb, tlen) == 0 && oun == T.unmap[i];
#ifdef M5_CENSUS
        if (identV.empty()) identV.assign(n, 0);
        identV[i] = eq;
#endif
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
#ifdef M5_CENSUS
    if (!census.empty()) {
        // read records: lane A (every read), M, B (A's / M's overflow), host; position in each lane -> truth index
        auto fixRead = [&](CenRead& r, u64 i, uint8_t laneId) {
            r.read = (uint32_t)i; r.count = T.count[i]; r.Lread = T.rlen[i];
            uint8_t fl = lane[i] == 0 ? 0 : lane[i] == 3 ? 1 : lane[i] == 1 ? 2 : 3;   // lane[]: 0 A, 3 M, 1 B, 2 host
            r.final_ = fl == laneId; r.ident = identV.empty() ? 0 : identV[i];
        };
        u64 nRec = 0, nWin = 0;
        FILE* f = fopen((census + ".reads.bin").c_str(), "wb");
        uint32_t hdr[4] = {(uint32_t)sizeof(CenRead), (uint32_t)sizeof(CenWin), 0, 0};
        fwrite("M5CEN001", 8, 1, f); fwrite(hdr, 4, 4, f);
        for (u64 i = 0; i < n; i++) { CenRead r = A.cen[i]; fixRead(r, i, 0); fwrite(&r, sizeof r, 1, f); nRec++; }
        if (!noMid) for (u64 k = 0; k < idxM.size(); k++) { CenRead r = M.cen[k]; fixRead(r, idxM[k], 1); fwrite(&r, sizeof r, 1, f); nRec++; }
        for (u64 k = 0; k < idxB.size(); k++) { CenRead r = B.cen[k]; fixRead(r, idxB[k], 2); fwrite(&r, sizeof r, 1, f); nRec++; }
        for (u64 k = 0; k < heavy.size(); k++) { CenRead r = hcen[k]; fixRead(r, heavy[k], 3); fwrite(&r, sizeof r, 1, f); nRec++; }
        fclose(f);
        f = fopen((census + ".wins.bin").c_str(), "wb");
        fwrite("M5CENW01", 8, 1, f); fwrite(hdr, 4, 4, f);
        auto putWins = [&](const std::vector<CenWin>& v, const uint32_t* idx) {
            for (CenWin r : v) { if (idx) r.read = idx[r.read]; fwrite(&r, sizeof r, 1, f); nWin++; }
        };
        putWins(A.win, nullptr);
        if (!noMid) putWins(M.win, idxM.data());
        putWins(B.win, idxB.data());
        for (auto& v : hwin) putWins(v, nullptr);
        fclose(f);
        int clkKHz = 0; cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);
        f = fopen((census + ".meta.json").c_str(), "w");
        fprintf(f, "{\n  \"distinct_reads\": %llu,\n  \"reads\": %llu,\n  \"identical_distinct\": %llu,\n  \"identical_reads\": %llu,\n"
                   "  \"read_records\": %llu,\n  \"window_records\": %llu,\n  \"window_records_dropped\": [%llu, %llu, %llu],\n"
                   "  \"lane_reads\": [%llu, %zu, %zu, %zu],\n  \"lane_ms\": [%.3f, %.3f, %.3f],\n  \"heavy_host_s\": %.3f,\n"
                   "  \"budget\": [%lld, %lld, %lld],\n  \"batch\": [%lld, %lld, %lld],\n  \"winAnchorMultimapNmax\": %llu,\n"
                   "  \"seedMultimapNmax\": %llu,\n  \"seedPerWindowNmax\": %llu,\n  \"alignTranscriptsPerWindowNmax\": %llu,\n"
                   "  \"caps\": {\"fields\": [\"LREAD\", \"NSPLIT\", \"NP\", \"NW\", \"NWA\", \"NWB\", \"NEX\", \"NWTR\", \"NKEEP\", \"NBLOB\"],\n"
                   "    \"A\": [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d], \"M\": [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d],\n"
                   "    \"B\": [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d], \"host\": [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d]},\n"
                   "  \"work_bytes\": [%zu, %zu, %zu, %zu],\n  \"clock_khz\": %d\n}\n",
                n, nReadsW, same, sameW, nRec, nWin,
                A.winN > A.win.size() ? A.winN - A.win.size() : 0ULL, M.winN > M.win.size() ? M.winN - M.win.size() : 0ULL,
                B.winN > B.win.size() ? B.winN - B.win.size() : 0ULL,
                n, noMid ? (size_t)0 : idxM.size(), idxB.size(), heavy.size(), A.ms, M.ms, B.ms, heavySec,
                budgetA, budgetM, budgetB, nbA, nbM, nbB, P.winAnchorMultimapNmax, P.seedMultimapNmax, P.seedPerWindowNmax,
                P.alignTranscriptsPerWindowNmax,
                LaneA::LREAD, LaneA::NSPLIT, LaneA::NP, LaneA::NW, LaneA::NWA, LaneA::NWB, LaneA::NEX, LaneA::NWTR, LaneA::NKEEP, LaneA::NBLOB,
                LaneM::LREAD, LaneM::NSPLIT, LaneM::NP, LaneM::NW, LaneM::NWA, LaneM::NWB, LaneM::NEX, LaneM::NWTR, LaneM::NKEEP, LaneM::NBLOB,
                LaneB::LREAD, LaneB::NSPLIT, LaneB::NP, LaneB::NW, LaneB::NWA, LaneB::NWB, LaneB::NEX, LaneB::NWTR, LaneB::NKEEP, LaneB::NBLOB,
                Heavy::LREAD, Heavy::NSPLIT, Heavy::NP, Heavy::NW, Heavy::NWA, Heavy::NWB, Heavy::NEX, Heavy::NWTR, Heavy::NKEEP, Heavy::NBLOB,
                sizeof(Work<LaneA>), sizeof(Work<LaneM>), sizeof(Work<LaneB>), sizeof(Work<Heavy>), clkKHz);
        fclose(f);
        fprintf(stderr, "census: %llu read records, %llu window records -> %s.{reads,wins}.bin\n", nRec, nWin, census.c_str());
    }
#else
    if (!census.empty()) { fprintf(stderr, "--census needs a build with -DM5_CENSUS\n"); return 2; }
#endif
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
        fprintf(f, "},\n  \"lane_a_ms\": %.3f,\n  \"lane_m_ms\": %.3f,\n  \"lane_m_reads\": %zu,\n  \"lane_b_ms\": %.3f,\n  \"lane_b_reads\": %zu,\n  \"lane_a_phase_ms\": [%.3f, %.3f, %.3f, %.3f],\n  \"lane_m_phase_ms\": [%.3f, %.3f, %.3f, %.3f],\n  \"lane_b_phase_ms\": [%.3f, %.3f, %.3f, %.3f],\n  \"lane_a_local\": %s,\n  \"work_bytes\": [%zu, %zu],\n  \"index_load_s\": %.1f\n}\n",
                A.ms, M.ms, noMid ? (size_t)0 : idxM.size(), B.ms, idxB.size(), A.phaseMs[0], A.phaseMs[1], A.phaseMs[2], A.phaseMs[3], M.phaseMs[0], M.phaseMs[1], M.phaseMs[2], M.phaseMs[3], B.phaseMs[0], B.phaseMs[1], B.phaseMs[2], B.phaseMs[3], localWork ? "true" : "false", sizeof(Work<LaneA>), sizeof(Work<LaneB>), tIndex);
        fclose(f);
    }
    return 0;
}
