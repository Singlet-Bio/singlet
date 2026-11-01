// singlet GPU engine: distinct-sequence dedup and STAR's mapping (seeds → windows → stitching → selection) on the GPU.
// API: include/singlet/engine_gpu.h. The port: stitch.cuh (line-by-line from the vendored STAR 2.7.11b). Measured identity and
// speed: docs/gpu-engine/exploration.md §12.12–12.14 (W11 round 2).
//
// Flow per sample: upload the reads, hash every read on the GPU (two independent 64-bit hashes), sort by hash, keep one read per
// distinct sequence (every member of a group is compared byte for byte with its representative: a hash collision aborts the GPU
// path, so the result never depends on the hash), drop sequences with N (STAR's cache does not hold them), then map the distinct
// sequences through the lane cascade A → M → B (GPU, staged) → host lane (STAR's limits, same code), and return
// (sequence, result) records.
#include "singlet/engine_gpu.h"
#include "stitch.cuh"
#include "records_gpu.h"   // E8: the pileup's records from the device-resident results (SINGLET_ENGINE_GPU_E2E=1)
#include "hostloop_gpu.h"  // E10: per-read entries and the per-read products of STAR's loop (SINGLET_ENGINE_GPU_E2E >= 2)
#include "solo_gpu.h"      // E11: STARsolo's inputs kept on the device for soloRun after STAR's loop (SINGLET_ENGINE_GPU_E2E=5)
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/reduce.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>

using namespace m5;

namespace {

#define EG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

//                 LREAD NSPLIT    NP     NW  NWA      NWB NEX NWTR NKEEP    NBLOB
typedef Caps<      160,     4,   32,    16,  16,     256,  8,   8,    8,     640> LaneA;
typedef Caps<      160,     4,   32,    64,   8,    1024,  8,   8,   32,    2048> LaneM;
typedef Caps<      160,     8,  128,   128,  50,    4096, 20,  32,   64,    4096> LaneB;
typedef Caps<      650,    50, 1000, 10000,  50, 1 << 20, 20, 100, 4096, 1 << 20> Heavy;

// ---- dedup
__global__ void k_hash(const char* seq, const u64* off, u64 n, u64* h1, u64* h2, uint8_t* hasN) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 a = 0x9E3779B97F4A7C15ULL ^ (off[i + 1] - off[i]), b = 0xC2B2AE3D27D4EB4FULL + (off[i + 1] - off[i]);
    uint8_t N = 0;
    for (u64 k = off[i]; k < off[i + 1]; k++) {
        u64 c = (u64)(uint8_t)seq[k];
        N |= (c > 3);
        a = (a ^ c) * 0x100000001B3ULL; a ^= a >> 29;
        b = (b + c + 1) * 0xFF51AFD7ED558CCDULL; b ^= b >> 31;
    }
    h1[i] = a; h2[i] = b; hasN[i] = N | (off[i + 1] == off[i]);   // an empty (fully clipped) read is not cached either
}
// group boundaries after the sort, and the byte-for-byte check of every member against the group's first read
__global__ void k_groups(const char* seq, const u64* off, const u64* h1, const u64* h2, const uint32_t* order, u64 n,
                         uint32_t* head, unsigned long long* collisions) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    bool first = k == 0 || h1[k] != h1[k - 1] || h2[k] != h2[k - 1];
    head[k] = first ? 1 : 0;
    if (!first) {   // find the group's first read (groups are short) and compare
        u64 j = k; while (j > 0 && h1[j - 1] == h1[k] && h2[j - 1] == h2[k]) j--;
        uint32_t a = order[j], b = order[k];
        u64 la = off[a + 1] - off[a], lb = off[b + 1] - off[b];
        bool same = la == lb;
        for (u64 t = 0; same && t < la; t++) same = seq[off[a] + t] == seq[off[b] + t];
        if (!same) atomicAdd(collisions, 1ULL);
    }
}

// ---- clipping (STAR's ClipMate::clip for mate 1, Hamming adapter type; SequenceFuns.cpp localSearch), per read: the clipped
// sequence starts at cs[i] and has cl[i] bases
struct ClipDev { uint32_t n5, after5, n3, after3, nad; double adMMp; const char* ad; const uint16_t* p5; const uint16_t* p3; };
__global__ void k_clip(const char* seq, const u64* off, u64 n, ClipDev c, u64* cs, u64* cl) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    u64 s = off[i], L = off[i + 1] - off[i];
    if (c.p5) {   // clip lengths computed on the host (ClipMate::clip with the CellRanger4 clip, opal)
        u64 a = c.p5[i], b = c.p3[i];
        if (a > L) a = L;
        if (b > L - a) b = L - a;
        cs[i] = s + a; cl[i] = L - a - b;
        return;
    }
    // 5': N bases, then NafterAd (each moves the start; a read not longer than the clip becomes empty)
    if (c.n5 > 0) { if (L > c.n5) { L -= c.n5; s += c.n5; } else L = 0; }
    if (c.after5 > 0) { if (L > c.after5) { L -= c.after5; s += c.after5; } else L = 0; }
    // 3': N bases, then the adapter, then NafterAd
    if (c.n3 > 0) { if (L > c.n3) L -= c.n3; else L = 0; }
    if (c.nad > 0) {
        const char* x = seq + s;
        u64 nMatchBest = 0, nMMbest = 0, ixBest = L;
        for (u64 ix = 0; ix < L; ix++) {
            u64 nMatch = 0, nMM = 0, m = (u64)c.nad < L - ix ? (u64)c.nad : L - ix;
            for (u64 iy = 0; iy < m; iy++) {
                if (x[ix + iy] > 3) continue;
                if (x[ix + iy] == c.ad[iy]) nMatch++; else nMM++;
            }
            if ((nMatch > nMatchBest || (nMatch == nMatchBest && nMM < nMMbest)) && double(nMM) / double(nMatch) <= c.adMMp) {
                ixBest = ix; nMatchBest = nMatch; nMMbest = nMM;
            }
        }
        L = ixBest;   // clippedAdN = L - ixBest
    }
    if (c.after3 > 0) { if (L > c.after3) L -= c.after3; else L = 0; }
    cs[i] = s; cl[i] = L;
}
__global__ void k_gather_clip(const char* seq, const u64* cs, const u64* offC, u64 n, char* out) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    for (u64 k = 0, L = offC[i + 1] - offC[i]; k < L; k++) out[offC[i] + k] = seq[cs[i] + k];
}

// E12: the group boundaries without k_groups' walk back to the group's first read (quadratic in a group's size; a sample's most frequent
// sequences have many copies). k_heads: the boundaries and per sorted position k or 0 (an inclusive max-scan then gives each
// position its group's first position); k_collide: the byte-for-byte check of every member against its group's first read, as k_groups
__global__ void k_heads(const u64* h1, const u64* h2, u64 n, uint32_t* head, uint32_t* hp) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const bool first = k == 0 || h1[k] != h1[k - 1] || h2[k] != h2[k - 1];
    head[k] = first ? 1 : 0; hp[k] = first ? (uint32_t)k : 0u;
}
__global__ void k_collide(const char* seq, const u64* off, const uint32_t* order, const uint32_t* hp, u64 n, unsigned long long* collisions) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n || hp[k] == k) return;
    const uint32_t a = order[hp[k]], b = order[k];
    const u64 la = off[a + 1] - off[a], lb = off[b + 1] - off[b];
    bool same = la == lb;
    for (u64 t = 0; same && t < la; t++) same = seq[off[a] + t] == seq[off[b] + t];
    if (!same) atomicAdd(collisions, 1ULL);
}
// E12: the dedup's products on the device (before: three arrays to the host and two serial passes over the reads there). Per sorted
// position its group's first position (k_heads, then an inclusive max-scan), per read its representative (the group's first read; NONE
// for a sequence with N), and the representatives without N in hash order (stream compaction)
__global__ void k_readrep(const uint32_t* order, const uint32_t* hp, const uint8_t* hasN, u64 n, uint32_t* readRep) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const uint32_t f = order[hp[k]];
    readRep[order[k]] = hasN[f] ? 0xFFFFFFFFu : f;
}
struct RepPred {
    const uint32_t* head; const uint8_t* hasN; const uint32_t* order;
    __host__ __device__ bool operator()(u64 k) const { return head[k] && !hasN[order[k]]; }
};
__global__ void k_prefix_key(const char* seq, const u64* off, const uint32_t* rep, u64 n, u64* key) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    u64 i = rep[k], L = off[i + 1] - off[i], kk = 0;
    for (u64 t = 0; t < 32; t++) kk = (kk << 2) | (t < L ? (u64)(uint8_t)seq[off[i] + t] : 0);
    key[k] = kk;
}

// ---- staged lanes (bench/gpu/m5/m5.cu)
template <class C>
__global__ void k_p1(Params P, GenomeView g, const char* rb, const u64* roff, const uint32_t* idx, u64 b0, u64 n,
                     Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* status, u64* nEnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    u64 i = idx[b0 + k];
    Work<C>& w = work[k];
    Status s = mapPhase1(P, g, w, rbuf[k], rb + roff[i], (m5::uint)(roff[i + 1] - roff[i]));
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
template <class C>
__global__ void k_p3(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* base,
                     const uint32_t* bits, uint8_t* bstatus, uint8_t* status, uint8_t* out, uint16_t* outLen) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    w.stepBudget = budget;
    u64 o = b0 + k;
    Status s = (Status)bstatus[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
    status[o] = (uint8_t)s; outLen[o] = (uint16_t)bl.n;
}

// ---- E12 (docs/gpu-engine/fable/E12-prefill.md): the m5 harness's best exact configuration (bench/gpu/m5/m5.cu, gate job of
// bench/fable/engine/fable_gate.sbatch: --best-order --p3-maxreg-a 64 --batch-b 131072 --warp-m --wsort --wthreads 128), ported
// kernel by kernel from m5.cu without its probes (census, tail, clocks summed into globals, why / unmapType outputs). The read
// length comes from the offsets (m5: a uint16 array). Phase 1 is E7's "flatqs8b" (stitch.cuh's defaults): k_p1s, persistent
// warps with the seed iterator in shared memory and lane refill, then the windows thread per read (k_p1w) or, in lane B,
// persistent (k_p1wp). Phase 3: lane A thread per read under a 64-register cap (k_p3lb), lane M warp per read (k_p3a, then
// k_p3w over the batch sorted by window count, most first), lane B thread per read (k_p3e). The old kernels above (k_p1:
// STAR's nest; k_p3) stay for SINGLET_ENGINE_GPU_CFG=old.
#if !defined(M5_P1SPLIT) || !defined(M5_SEEDSMEM) || !defined(M5_P1WPB) || !defined(M5_P1MINB) || defined(M5_P1FUSE) || defined(M5_SEEDSTEP) || defined(M5_G2)
#error "E12 lanes: stitch.cuh's E7 default phase 1 (flatqs8b) expected"
#endif
#define E12_P1_BOUNDS __launch_bounds__(128, M5_P1MINB)
#ifdef M5_NOCLK
#define E12_P1T() 0LL
#define E12_P1ADD(w, k, t) do { (void)(t); } while (0)
#else
#define E12_P1T() clock64()
#define E12_P1ADD(w, k, t) do { (w).clk[k] += clock64() - (t); } while (0)
#endif
template <class C> constexpr bool p1WinPers() { return C::NW >= 128; }   // M5_P1WPB: the persistent windows kernel in lane B only
// seeds: persistent warps; every iteration each lane with a read makes its read's next maxMappableLength2strands call; lanes
// without a read take the batch's next reads (warp-aggregated atomic counter ctr)
template <class C>
__global__ void E12_P1_BOUNDS k_p1s(Params P, GenomeView g, const char* rb, const u64* roff, const uint32_t* idx, u64 b0, u64 n,
                                    Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* status, unsigned long long* ctr) {
    const unsigned FULL = 0xffffffffu, lane = threadIdx.x & 31;
    u64 k = 0;
    int st = 0;            // 0: needs a read, 1: holds a read with a pending call, 2: the batch is exhausted
    __shared__ SeedIt smIt[128];
    SeedIt& it = smIt[threadIdx.x];
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
                else {
                    tc = E12_P1T();
                    const u64 i = idx[b0 + k];
                    Work<C>& w = work[k];
                    const Status s = p1Setup(P, w, rbuf[k], rb + roff[i], (m5::uint)(roff[i + 1] - roff[i]), it);
                    if (s != Status::Ok) status[k] = (uint8_t)s;
                    else if (seedNext(P, w, it)) st = 1;
                    else { status[k] = (uint8_t)Status::Ok; E12_P1ADD(w, 0, tc); }
                }
            }
        }
        if (__all_sync(FULL, st == 2)) break;
        if (st == 1) {
            Work<C>& w = work[k];
            M5_READ1(Read1, rbuf[k], w.Lread);
            m5::uint L;
            seedCall(P, g, w, Read1, it, L);
            if (w.ov) { status[k] = (uint8_t)Status::Overflow; st = 0; }
            else {
                seedAfter(it, L);
                if (!seedNext(P, w, it)) { status[k] = (uint8_t)Status::Ok; E12_P1ADD(w, 0, tc); st = 0; }
            }
        }
    }
}
// windows, persistent warps: every iteration each lane with a read makes one step of its read's windows
template <class C>
__global__ void E12_P1_BOUNDS k_p1wp(Params P, GenomeView g, u64 n, Work<C>* work, uint8_t* status, u64* nEnt, unsigned long long* ctr) {
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
                    tc = E12_P1T();
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
            else if (r > 0) { nEnt[k] = w.nEntries; E12_P1ADD(w, 1, tc); st = 0; }
        }
    }
}
// windows from the anchors and flanks, thread per read (reads that overflowed in k_p1s are skipped)
template <class C>
__global__ void E12_P1_BOUNDS k_p1w(Params P, GenomeView g, u64 n, Work<C>* work, uint8_t* status, u64* nEnt) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    Status s = (Status)status[k];
    if (s == Status::Ok) s = p1Windows(P, g, w);
    status[k] = (uint8_t)s;
    nEnt[k] = (s == Status::Ok && w.stitched) ? w.nEntries : 0;
}
// phase 3 of read k of the batch, thread per read; its status stays in the batch's bst (k_copy_status moves it)
template <class C>
__device__ __forceinline__ void p3one(const Params& P, const GenomeView& g, u64 b0, u64 k, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf,
                                      const u64* base, const uint32_t* bits, uint8_t* bst, uint8_t* out, uint16_t* outLen) {
    Work<C>& w = work[k];
    w.stepBudget = budget;
    const u64 o = b0 + k;
    Status s = (Status)bst[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
    bst[k] = (uint8_t)s; outLen[o] = (uint16_t)bl.n;
}
template <class C>
__global__ void k_p3e(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* base,
                      const uint32_t* bits, uint8_t* bst, uint8_t* out, uint16_t* outLen) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    p3one<C>(P, g, b0, k, budget, work, rbuf, base, bits, bst, out, outLen);
}
// the same under __launch_bounds__(BT, MINB): at most 65536 / (BT x MINB) registers per thread (128 x 8: 64)
template <class C, int BT, int MINB>
__global__ void __launch_bounds__(BT, MINB) k_p3lb(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf,
                                                   const u64* base, const uint32_t* bits, uint8_t* bst, uint8_t* out, uint16_t* outLen) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    p3one<C>(P, g, b0, k, budget, work, rbuf, base, bits, bst, out, outLen);
}
// the warp lane's order: windows, then SA entries to assign, descending (reads that need no phase 3 last)
template <class C>
__global__ void k_cost(u64 n, const Work<C>* work, const u64* nEnt, u64* key, uint32_t* perm) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    key[k] = ~(nEnt[k] ? ((u64)work[k].nW << 40) | min(nEnt[k], (u64)0xFFFFFFFFFFULL) : 0);
    perm[k] = (uint32_t)k;
}
// warp lane (E3): the assignment thread per read, then k_p3w: warp per read, the read's windows stitched across the lanes
// (stitch.cuh warpStitch), selection and encoding on lane 0
template <class C>
__global__ void k_p3a(Params P, GenomeView g, u64 n, Work<C>* work, const u64* base, const uint32_t* bits, uint8_t* bst) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    if ((Status)bst[k] == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) bst[k] = (uint8_t)Status::Overflow;
}
template <class C, class LW>
__global__ void k_p3w(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, uint8_t* bst,
                      uint8_t* out, uint16_t* outLen, const uint32_t* perm) {
    // shared memory as m5's (one AssignEnt per thread for the group assignment, unused here; then C::NW window indices per warp)
    extern __shared__ unsigned long long shAll[];
    constexpr int G = 32;
    const Grp<G> gr;
    const int lane = gr.gl, gib = threadIdx.x / G;
    const u64 wid = (blockIdx.x * (u64)blockDim.x + threadIdx.x) / G;
    if (wid >= n) return;   // whole warps
    uint16_t* wl = reinterpret_cast<uint16_t*>(reinterpret_cast<AssignEnt*>(shAll) + blockDim.x) + gib * C::NW;
    const u64 k = perm ? perm[wid] : wid;
    Work<C>& W = work[k];
    const u64 o = b0 + k;
    Status s = (Status)bst[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
    if (s == Status::Ok) {
        W.stepBudget = budget;
        if (!W.stitched || W.nW == 0) {
            if (lane == 0) s = mapFinish(P, g, W, rbuf[k], bl);
        } else {
            LW L;
            L.Lread = W.Lread; L.outFilterMismatchNmaxTotal = W.outFilterMismatchNmaxTotal;
            for (int c = 0; c < 8; c++) L.clk[c] = 0;
            WarpMerge M;
            const int r = warpStitch<C, LW, G>(gr, P, g, W, rbuf[k], L, wl, budget, M, nullptr);
#ifndef M5_NOCLK
            for (int c = 4; c < 8; c++) { const u64 v = warpSumU64(gr, (u64)L.clk[c]); if (lane == 0) W.clk[c] += (long long)v; }
#endif
            if (lane == 0) {
                if (r == WR_OVERFLOW) { s = Status::Overflow; W.ov = M.ov; }
                else if (r == WR_FALLBACK) s = mapFinish(P, g, W, rbuf[k], bl);
                else s = warpSelect(P, g, W, bl, M);
            }
        }
    }
    if (lane == 0) { bst[k] = (uint8_t)s; outLen[o] = (uint16_t)bl.n; }
}
__global__ void k_copy_status(const uint8_t* s, uint8_t* dst, u64 n) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k < n) dst[k] = s[k];
}
typedef LaneWork<8, 8, 8> LWM;   // lane M's per-lane stitching context (m5's LWM: its own capacities, no fallback)
static int g_wthreads = 128;     // k_p3w block size (m5 --wthreads)

struct ToU64 { __host__ __device__ u64 operator()(uint16_t x) const { return (u64)x; } };
// out packed: read k at off[k], len[k] bytes. E8 (keepDev): the packed bytes and offsets also stay on the device (dOut, dOff; freed by the caller)
struct Lane { singlet_engine_gpu::NoInitVector<uint8_t> out; std::vector<uint8_t> status; std::vector<uint16_t> len; singlet_engine_gpu::NoInitVector<u64> off;
              uint8_t* dOut = nullptr; u64* dOff = nullptr; };

// E12: a memcpy on nt threads (each also first-touches its part of a fresh destination)
static void parMemcpy(uint8_t* d, const uint8_t* src, u64 m, int nt) {
    if (nt <= 1 || m < (4u << 20)) { std::memcpy(d, src, m); return; }
    std::vector<std::thread> th;
    const u64 per = (m + nt - 1) / nt;
    for (int t = 1; t < nt; t++) { const u64 a = std::min(m, t * per), e = std::min(m, a + per); if (a < e) th.emplace_back([=] { std::memcpy(d + a, src + a, e - a); }); }
    std::memcpy(d, src, std::min(m, per));
    for (auto& x : th) x.join();
}
// E12: device -> host copies through two page-locked buffers of 64 MB: the DMA of one chunk runs while nt threads copy the previous one
// out (and fault the destination's pages in, in parallel). Before: pageable cudaMemcpy into vectors zero-filled by one thread first.
// SINGLET_ENGINE_GPU_D2H=pageable (or CFG=old): as before
struct D2H {
    static constexpr u64 CH = 64ull << 20;
    uint8_t* buf[2] = {nullptr, nullptr}; cudaEvent_t ev[2] = {nullptr, nullptr}; cudaStream_t s = nullptr; int nt = 1;
    u64 bytes = 0; double sec = 0;
    bool init(int threads) {
        nt = std::max(1, threads);
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return false;
        for (int b = 0; b < 2; b++)
            if (cudaHostAlloc((void**)&buf[b], CH, cudaHostAllocDefault) != cudaSuccess || cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming) != cudaSuccess)
                return false;
        return true;
    }
    bool copy(void* dst, const void* src, u64 n) {   // src: written by work on the legacy stream (waited for first)
        if (n == 0) return true;
        const auto t0 = std::chrono::steady_clock::now();
        if (cudaStreamSynchronize(0) != cudaSuccess) return false;
        const u64 nc = (n + CH - 1) / CH;
        auto issue = [&](u64 c) {
            const u64 o = c * CH, m = std::min(CH, n - o);
            return cudaMemcpyAsync(buf[c & 1], (const uint8_t*)src + o, m, cudaMemcpyDeviceToHost, s) == cudaSuccess && cudaEventRecord(ev[c & 1], s) == cudaSuccess;
        };
        if (!issue(0)) return false;
        for (u64 c = 0; c < nc; c++) {
            if (c + 1 < nc && !issue(c + 1)) return false;   // into the buffer chunk c - 1 was copied out of
            if (cudaEventSynchronize(ev[c & 1]) != cudaSuccess) return false;
            const u64 o = c * CH;
            parMemcpy((uint8_t*)dst + o, buf[c & 1], std::min(CH, n - o), nt);
        }
        bytes += n; sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
    void release() {
        if (s) cudaStreamSynchronize(s);
        for (int b = 0; b < 2; b++) { if (ev[b]) cudaEventDestroy(ev[b]); if (buf[b]) cudaFreeHost(buf[b]); ev[b] = nullptr; buf[b] = nullptr; }
        if (s) cudaStreamDestroy(s);
        s = nullptr;
    }
    ~D2H() { release(); }
};
static D2H* g_d2h = nullptr;   // set by mapDistinct while it runs (runLane's copies)

// packs the lane's fixed-size result slots (slot k at k * slot) into consecutive bytes at off[k]
__global__ void k_pack(const uint8_t* slots, u64 slot, const uint16_t* len, const u64* off, u64 n, uint8_t* packed) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    for (u64 t = 0; t < len[k]; t++) packed[off[k] + t] = slots[k * slot + t];
}

static double g_devPeakGB = 0;   // largest device memory in use seen inside the lanes (GB)
static void devNote() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); g_devPeakGB = std::max(g_devPeakGB, (double) (t - f) / (1u << 30)); }
// E12 (docs/gpu-engine/fable/E12-prefill.md): wall time of each step of mapDistinct (s, host clock; a device step ends where the code
// synchronises), summed over chunks and logged as one line "engine_gpu: steps (s): ..." in the order the steps first ran
static std::vector<std::pair<std::string, double>> g_steps;
static void stepAdd(const std::string& k, double s) { for (auto& p : g_steps) if (p.first == k) { p.second += s; return; } g_steps.push_back({k, s}); }
static std::string stepLine() {
    std::string s; char b[96];
    for (auto& p : g_steps) { snprintf(b, sizeof b, "%s%s %.3f", s.empty() ? "" : ", ", p.first.c_str(), p.second); s += b; }
    return s;
}

// one lane over reads idx in batches of NB (one Work per read in flight). NEW: E12's kernels (above; WARP: phase 3 warp per read,
// MAXREG 64: phase 3 under the register cap); otherwise the engine's kernels before E12 (SINGLET_ENGINE_GPU_CFG=old)
static std::string g_laneLog;   // E12: per lane and chunk, its configuration (logged once by mapDistinct)
// E12: called after each batch with the batch's statuses (lane B: its overflows go to the host lane while the device runs on)
typedef std::function<void(u64 b0, u64 nb, const uint8_t* status)> BatchFn;
template <class C, bool WARP = false, int MAXREG = 0>
bool runLane(const Params& P, const GenomeView& D, const char* dSeq, const u64* dOff, const std::vector<uint32_t>& idx, u64 NB, u64 budget,
             Lane& L, std::string& err, bool keepDev, const char* nm, bool NEW, const BatchFn* onBatch = nullptr) {
    const double tA = now();
    const std::string sn = std::string("lane") + nm;
    u64 n = idx.size();
    L.status.assign(n, (uint8_t)Status::Overflow); L.len.assign(n, 0); L.off.resize(n + 1); L.off[0] = 0;   // off: every entry copied below
    if (n == 0) return true;
    NB = std::min<u64>(NB, n);
    uint32_t* dIdx; uint8_t *dOut, *dSt, *dBst; uint16_t* dLen; Work<C>* dWork; ReadBuf<C::LREAD>* dRbuf; u64 *dNent, *dBase; uint32_t* dBits;
    EG_CK(cudaMalloc(&dIdx, n * 4)); EG_CK(cudaMemcpy(dIdx, idx.data(), n * 4, cudaMemcpyHostToDevice));
    EG_CK(cudaMalloc(&dOut, n * (u64)C::NBLOB)); EG_CK(cudaMalloc(&dSt, n)); EG_CK(cudaMalloc(&dLen, n * 2));
    EG_CK(cudaMalloc(&dWork, NB * sizeof(Work<C>))); EG_CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>)));
    EG_CK(cudaMalloc(&dRbuf, NB * sizeof(ReadBuf<C::LREAD>))); EG_CK(cudaMalloc(&dBst, NB));
    EG_CK(cudaMalloc(&dNent, NB * 8)); EG_CK(cudaMalloc(&dBase, NB * 8));
    u64 bitsCap = 1 << 20; EG_CK(cudaMalloc(&dBits, bitsCap * 4));
    // E12: the seed / windows kernels' counters, the warp lane's sort keys and order, the persistent kernels' grids
    unsigned long long* dCtr = nullptr; u64* dKey = nullptr; uint32_t* dPerm = nullptr;
    int nSm = 0, bpsS = 0, bpsW = 0; size_t stack0 = 0;
    if (NEW) {
        EG_CK(cudaMalloc(&dCtr, 16));
        if (WARP) { EG_CK(cudaMalloc(&dKey, NB * 8)); EG_CK(cudaMalloc(&dPerm, NB * 4)); }
        int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, dev);
        EG_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsS, k_p1s<C>, 128, 0));
        if constexpr (p1WinPers<C>()) EG_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsW, k_p1wp<C>, 128, 0));
        if constexpr (WARP) {   // the lanes' stitching contexts are local memory: room for them in the per-thread stack (as m5)
            cudaDeviceGetLimit(&stack0, cudaLimitStackSize);
            EG_CK(cudaDeviceSetLimit(cudaLimitStackSize, std::max<size_t>(16 * 1024, sizeof(LWM) + 8192)));
        }
        int regs = 0;
        { cudaFuncAttributes fa{};
          if constexpr (WARP) cudaFuncGetAttributes(&fa, (const void*)k_p3w<C, LWM>);
          else if constexpr (MAXREG == 64) cudaFuncGetAttributes(&fa, (const void*)k_p3lb<C, 128, 8>);
          else cudaFuncGetAttributes(&fa, (const void*)k_p3e<C>);
          regs = fa.numRegs; }
        char b[320];
        snprintf(b, sizeof b, "%s %llu reads, batches of %llu (%.1f GB of Work), k_p1s %d blocks/SM, %s, phase 3 %s (%d registers); ", nm,
                 (unsigned long long)n, (unsigned long long)NB, NB * (double)(sizeof(Work<C>) + sizeof(ReadBuf<C::LREAD>)) / 1e9, bpsS,
                 p1WinPers<C>() ? "k_p1wp persistent" : "k_p1w", WARP ? "warp per read, by windows desc" : MAXREG == 64 ? "capped at 64 registers" : "thread per read", regs);
        g_laneLog += b;
    }
    devNote();
    const double tK = now(); stepAdd(sn + ".alloc", tK - tA);
    for (u64 b0 = 0; b0 < n; b0 += NB) {
        u64 nb = std::min(NB, n - b0);
        int T = 128; unsigned G = (unsigned)((nb + T - 1) / T);
        if (!NEW) k_p1<C><<<G, T>>>(P, D, dSeq, dOff, dIdx, b0, nb, dWork, dRbuf, dBst, dNent);
        else {
            EG_CK(cudaMemsetAsync(dCtr, 0, 16));
            k_p1s<C><<<(unsigned)std::min<u64>((u64)nSm * bpsS, G), T>>>(P, D, dSeq, dOff, dIdx, b0, nb, dWork, dRbuf, dBst, dCtr);
            if constexpr (p1WinPers<C>()) k_p1wp<C><<<(unsigned)std::min<u64>((u64)nSm * bpsW, G), T>>>(P, D, nb, dWork, dBst, dNent, dCtr + 1);
            else k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent);
        }
        thrust::exclusive_scan(thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dBase));
        u64 lastB, lastN; EG_CK(cudaMemcpy(&lastB, dBase + nb - 1, 8, cudaMemcpyDeviceToHost)); EG_CK(cudaMemcpy(&lastN, dNent + nb - 1, 8, cudaMemcpyDeviceToHost));
        u64 words = (lastB + lastN) / 32 + 2;
        if (words > bitsCap) { cudaFree(dBits); bitsCap = words * 2; EG_CK(cudaMalloc(&dBits, bitsCap * 4)); }
        EG_CK(cudaMemset(dBits, 0, words * 4));
        k_p2<C><<<(unsigned)((nb * 32 + 255) / 256), 256>>>(P, D, nb, dWork, dNent, dBase, dBits);
        if (!NEW) k_p3<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dSt, dOut, dLen);
        else {
            if constexpr (WARP) {
                k_cost<C><<<G, T>>>(nb, dWork, dNent, dKey, dPerm);
                thrust::sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm));
                k_p3a<C><<<G, T>>>(P, D, nb, dWork, dBase, dBits, dBst);
                const int wt = g_wthreads; const unsigned GW = (unsigned)((nb * 32 + wt - 1) / wt);
                const size_t sh = (size_t)wt * sizeof(AssignEnt) + (size_t)(wt / 32) * C::NW * sizeof(uint16_t);
                k_p3w<C, LWM><<<GW, wt, sh>>>(P, D, b0, nb, budget, dWork, dRbuf, dBst, dOut, dLen, dPerm);
            } else if constexpr (MAXREG == 64) k_p3lb<C, 128, 8><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dOut, dLen);
            else k_p3e<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dOut, dLen);
            k_copy_status<<<G, T>>>(dBst, dSt + b0, nb);
        }
        EG_CK(cudaGetLastError());
        if (onBatch && *onBatch) {   // the batch's statuses now (the next batch's kernels wait for the host for this copy's duration)
            std::vector<uint8_t> hs(nb);
            EG_CK(cudaMemcpy(hs.data(), dSt + b0, nb, cudaMemcpyDeviceToHost));
            (*onBatch)(b0, nb, hs.data());
        }
    }
    EG_CK(cudaDeviceSynchronize());
    if (dCtr) cudaFree(dCtr);
    if (dKey) cudaFree(dKey);
    if (dPerm) cudaFree(dPerm);
    if (stack0) cudaDeviceSetLimit(cudaLimitStackSize, stack0);
    const double tP = now(); stepAdd(sn + ".kernels", tP - tK);
    {   // copy only the bytes of the results (the slots are 640-4096 bytes, results ~70)
        u64* dPoff; EG_CK(cudaMalloc(&dPoff, (n + 1) * 8)); EG_CK(cudaMemset(dPoff, 0, (n + 1) * 8));
        thrust::transform_inclusive_scan(thrust::device_ptr<uint16_t>(dLen), thrust::device_ptr<uint16_t>(dLen + n), thrust::device_ptr<u64>(dPoff + 1),
                                         ToU64(), thrust::plus<u64>());
        u64 total; EG_CK(cudaMemcpy(&total, dPoff + n, 8, cudaMemcpyDeviceToHost));
        uint8_t* dPacked; EG_CK(cudaMalloc(&dPacked, total + 1));
        devNote();
        k_pack<<<(unsigned)((n + 255) / 256), 256>>>(dOut, C::NBLOB, dLen, dPoff, n, dPacked);
        EG_CK(cudaGetLastError());
        L.out.resize(total);
        if (g_d2h) { if (!g_d2h->copy(L.out.data(), dPacked, total) || !g_d2h->copy(L.off.data(), dPoff, (n + 1) * 8)) { err = "lane results copy"; return false; } }
        else {
        EG_CK(cudaMemcpy(L.out.data(), dPacked, total, cudaMemcpyDeviceToHost));
        EG_CK(cudaMemcpy(L.off.data(), dPoff, (n + 1) * 8, cudaMemcpyDeviceToHost));
        }
        if (keepDev) { L.dOut = dPacked; L.dOff = dPoff; }
        else { cudaFree(dPoff); cudaFree(dPacked); }
    }
    EG_CK(cudaMemcpy(L.status.data(), dSt, n, cudaMemcpyDeviceToHost)); EG_CK(cudaMemcpy(L.len.data(), dLen, n * 2, cudaMemcpyDeviceToHost));
    cudaFree(dIdx); cudaFree(dOut); cudaFree(dSt); cudaFree(dLen); cudaFree(dWork); cudaFree(dRbuf); cudaFree(dBst); cudaFree(dNent);
    cudaFree(dBase); cudaFree(dBits);
    stepAdd(sn + ".d2h", now() - tP);
    return true;
}

// E12: the host lane as a pool of threads that maps the reads lane B gives up as each of its batches ends, while the device lanes run on
// (before: after every chunk's lanes, with the device idle). A task is (chunk, index in the chunk's host lane, read); hostPart waits for its
// chunk. Each read's result does not depend on when or on which thread it is mapped (m5 maps every heavy read on a reused Work).
struct HostPool {
    struct Task { uint32_t c, q, i; };
    std::mutex mu; std::condition_variable cv, cvDone;
    std::deque<Task> qu; bool stop = false;
    std::vector<u64> pend;               // per chunk: tasks not yet done
    std::vector<std::thread> th;
    std::atomic<u64> busyNs{0}, nDone{0};
    std::function<void(int, const Task&)> fn;
    void start(int nt, u64 nChunks, std::function<void(int, const Task&)> f) {
        fn = std::move(f); pend.assign(nChunks, 0);
        for (int t = 0; t < nt; t++) th.emplace_back([this, t]() {
            for (;;) {
                Task tk;
                { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || !qu.empty(); }); if (stop) return; tk = qu.front(); qu.pop_front(); }
                const auto t0 = std::chrono::steady_clock::now();
                fn(t, tk);
                busyNs += (u64) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count(); nDone++;
                { std::lock_guard<std::mutex> lk(mu); if (--pend[tk.c] == 0) cvDone.notify_all(); }
            }
        });
    }
    void push(const std::vector<Task>& v) {
        if (v.empty()) return;
        { std::lock_guard<std::mutex> lk(mu); for (const Task& t : v) { qu.push_back(t); pend[t.c]++; } }
        cv.notify_all();
    }
    void wait(u64 c) { std::unique_lock<std::mutex> lk(mu); cvDone.wait(lk, [&] { return pend[c] == 0; }); }
    ~HostPool() { { std::lock_guard<std::mutex> lk(mu); stop = true; } cv.notify_all(); for (auto& x : th) x.join(); }
};
// E12: host -> device copies through two page-locked staging buffers of 64 MB: the CPU copies a chunk into one (on nt threads) while the
// other one's DMA runs, at most two chunks in flight on stream s. The copy runs at the page-locked rate instead of the driver's pageable
// path (~6 GB/s here), needs no page-locking of the source (cudaHostRegister of GBs costs seconds), and on a non-blocking stream of
// another host thread an implicit device synchronisation elsewhere (cudaFree, thrust's temporaries) waits for two chunks, not the copy.
struct Stager {
    static constexpr u64 CH = 64ull << 20;
    uint8_t* buf[2] = {nullptr, nullptr}; cudaEvent_t ev[2] = {nullptr, nullptr}; bool used[2] = {false, false};
    cudaStream_t s = nullptr; int nt = 1; u64 k = 0, bytes = 0; double sec = 0;
    bool init(int threads) {   // on the thread that owns the context's allocations (page-locked allocation synchronises)
        nt = std::max(1, threads);
        if (cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking) != cudaSuccess) return false;
        for (int b = 0; b < 2; b++)
            if (cudaHostAlloc((void**)&buf[b], CH, cudaHostAllocDefault) != cudaSuccess || cudaEventCreateWithFlags(&ev[b], cudaEventDisableTiming) != cudaSuccess)
                return false;
        return true;
    }
    bool copy(void* dst, const void* src, u64 n) {
        const auto t0 = std::chrono::steady_clock::now();
        for (u64 o = 0; o < n; o += CH, k++) {
            const int b = (int)(k & 1);
            const u64 m = std::min(CH, n - o);
            if (used[b] && cudaEventSynchronize(ev[b]) != cudaSuccess) return false;
            parMemcpy(buf[b], (const uint8_t*)src + o, m, nt);
            if (cudaMemcpyAsync((uint8_t*)dst + o, buf[b], m, cudaMemcpyHostToDevice, s) != cudaSuccess || cudaEventRecord(ev[b], s) != cudaSuccess) return false;
            used[b] = true;
        }
        bytes += n; sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return true;
    }
    bool finish() { const auto t0 = std::chrono::steady_clock::now(); const bool ok = cudaStreamSynchronize(s) == cudaSuccess;
                    sec += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); return ok; }
    void release() {
        if (s) cudaStreamSynchronize(s);
        for (int b = 0; b < 2; b++) { if (ev[b]) cudaEventDestroy(ev[b]); if (buf[b]) cudaFreeHost(buf[b]); ev[b] = nullptr; buf[b] = nullptr; }
        if (s) cudaStreamDestroy(s);
        s = nullptr;
    }
    ~Stager() { release(); }
};
// runs fn(lo, hi) over [0, n) cut into nt contiguous ranges, one thread each (the host assembly loops: they run on all the job's threads)
template <class F> static void parallelFor(u64 n, int nt, F fn) {
    nt = (int) std::max<u64>(1, std::min<u64>((u64) std::max(1, nt), n / 65536 + 1));
    if (nt == 1) { fn(0, n); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) th.emplace_back([&, t]() { fn(n * t / nt, n * (t + 1) / nt); });
    for (auto& x : th) x.join();
}

template <class T> bool upload(const T* h, u64 n, T*& d, std::string& err, u64 pad = 0) {
    EG_CK(cudaMalloc(&d, (n + pad) * sizeof(T) + 8));
    if (pad) EG_CK(cudaMemset(d, 0, (n + pad) * sizeof(T) + 8));
    if (n) EG_CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
    return true;
}

// ---- E8: the records' blobs and sequences gathered in record order on the device, from the lanes' device-resident packed results (and the
// host lane's, uploaded), so that buildRecords runs where the results are. Record r: source = chunk << 40 | lane << 32 | position in the
// lane's packed output; its sequence is read `rd` of the (clipped) device reads.
struct LaneSrc { const uint8_t* out; const u64* off; };
__global__ void k_rec_len(const u64* src, const uint32_t* rd, u64 n, const LaneSrc* tab, const u64* off, u64* blen, u64* slen) {
    const u64 r = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (r >= n) return;
    const u64 s = src[r]; const LaneSrc L = tab[(s >> 40) * 4 + ((s >> 32) & 0xff)]; const u64 q = s & 0xffffffffull;
    blen[r] = L.off[q + 1] - L.off[q];
    slen[r] = off[rd[r] + 1] - off[rd[r]];
}
__global__ void k_rec_copy(const u64* src, const uint32_t* rd, u64 n, const LaneSrc* tab, const char* seq, const u64* off, const u64* bOff,
                           const u64* sOff, uint8_t* blob, char* rseq) {
    const u64 r = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (r >= n) return;
    const u64 s = src[r]; const LaneSrc L = tab[(s >> 40) * 4 + ((s >> 32) & 0xff)]; const u64 q = s & 0xffffffffull;
    const uint8_t* b = L.out + L.off[q];
    for (u64 k = 0, m = bOff[r + 1] - bOff[r]; k < m; k++) blob[bOff[r] + k] = b[k];
    const char* x = seq + off[rd[r]];
    for (u64 k = 0, m = sOff[r + 1] - sOff[r]; k < m; k++) rseq[sOff[r] + k] = x[k];
}

// E6 (bench/fable/records, docs/gpu-engine/fable/E6-records.md): SINGLET_ENGINE_GPU_DUMP=<path> writes mapDistinct's output (the records'
// sequences and sequence-cache blobs, readRecord, the clip lengths), the filter parameters the hand-off records depend on (mappedFilter) and
// chrStart, so that the GPU record builder can be checked byte for byte against the hand-off of the same run. Debug only; unset (the
// default) nothing is written and nothing changes. Layout (little endian): "SGEDUMP1", u64 nRec, nReads, seqBytes, blobBytes, nChr, nClip;
// i64 outFilterScoreMin; f64 outFilterScoreMinOverLread; u64 outFilterMatchNmin; f64 outFilterMatchNminOverLread; u64 outFilterMismatchNmax;
// f64 outFilterMismatchNoverLmax, outFilterMismatchNoverReadLmax; u64 outFilterMultimapNmax, outSAMmultNmax; u64 chrStart[nChr + 1];
// u64 seqOff[nRec + 1]; seq[seqBytes]; u64 blobOff[nRec + 1]; blob[blobBytes]; u32 readRecord[nReads]; u16 clip5[nClip], clip3[nClip].
void dumpOutput(const char* path, const Params& P, const GenomeView& H, const singlet_engine_gpu::Output& out) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return;
    bool ok = true;
    auto put = [&](const void* p, u64 n) { if (ok && n) ok = std::fwrite(p, 1, n, f) == n; };
    const u64 hdr[6] = {(u64) out.seqOff.size() - 1, (u64) out.readRecord.size(), (u64) out.seq.size(), (u64) out.blob.size(), (u64) H.nChr,
                        (u64) out.clip5.size()};
    put("SGEDUMP1", 8); put(hdr, sizeof hdr);
    const long long sMin = P.outFilterScoreMin; const double sOver = P.outFilterScoreMinOverLread; const u64 mMin = P.outFilterMatchNmin;
    const double mOver = P.outFilterMatchNminOverLread; const u64 mmMax = P.outFilterMismatchNmax;
    const double mmL = P.outFilterMismatchNoverLmax, mmRL = P.outFilterMismatchNoverReadLmax; const u64 mult = P.outFilterMultimapNmax, outMult = P.outSAMmultNmax;
    put(&sMin, 8); put(&sOver, 8); put(&mMin, 8); put(&mOver, 8); put(&mmMax, 8); put(&mmL, 8); put(&mmRL, 8); put(&mult, 8); put(&outMult, 8);
    put(H.chrStart, 8 * (H.nChr + 1));
    put(out.seqOff.data(), 8 * out.seqOff.size()); put(out.seq.data(), out.seq.size());
    put(out.blobOff.data(), 8 * out.blobOff.size()); put(out.blob.data(), out.blob.size());
    put(out.readRecord.data(), 4 * out.readRecord.size());
    put(out.clip5.data(), 2 * out.clip5.size()); put(out.clip3.data(), 2 * out.clip3.size());
    ok = (std::fclose(f) == 0) && ok;
    if (!ok) std::remove(path);
}

} // namespace

namespace singlet_engine_gpu {

bool available(std::string* why) {
    int n = 0;
    cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess || n == 0) { if (why) *why = e != cudaSuccess ? cudaGetErrorString(e) : "no CUDA device"; return false; }
    cudaDeviceProp p; cudaGetDeviceProperties(&p, 0);
    if (p.major < 8) { if (why) *why = std::string("device ") + p.name + " below compute capability 8.0"; return false; }
    return true;
}

std::string deviceId(int device) {
    char b[64] = {0};
    if (cudaDeviceGetPCIBusId(b, sizeof b, device) != cudaSuccess) return "dev" + std::to_string(device);
    std::string s(b);
    for (auto& c : s) if (c == ':' || c == '.') c = '_';
    return s;
}
int deviceCount() { int n = 0; if (cudaGetDeviceCount(&n) != cudaSuccess) return 0; return n; }

bool mapDistinct(const Params& P, const GenomeView& H, uint64_t saBytes, uint64_t saiBytes, const Input& in, int device, int hostThreads,
                 Output& out) {
    std::string err;
    Stager SR, SI;   // E12: staged uploads of the reads (this thread) and of the index (idxTh); declared first: destroyed after the join
    D2H DR;          // E12: staged device -> host copies (g_d2h while mapDistinct runs)
    struct D2HReset { ~D2HReset() { g_d2h = nullptr; } } d2hReset;
    bool idxOk = true; double tIdxEnd = 0;
    std::thread idxTh;   // E12: the index upload (staged, on its own stream) while the reads are uploaded, clipped and deduplicated
    struct JoinOnExit { std::thread& t; ~JoinOnExit() { if (t.joinable()) t.join(); } } idxJoin{idxTh};
    // on failure, release everything this process holds on the device (STAR then maps every read on the CPU)
    auto fail = [&](const std::string& e) { if (idxTh.joinable()) idxTh.join(); out.log += "engine_gpu: " + e + "\n"; e9::soloKeepRelease(); out.e10.soloKept = false; cudaDeviceReset(); return false; };
    g_steps.clear();
    // E12: SINGLET_ENGINE_GPU_CFG=old: the configuration before E12 (lanes, dedup products on the host, uploads, host lane after the lanes)
    const char* cfgE = std::getenv("SINGLET_ENGINE_GPU_CFG");
    const bool NEW0 = !(cfgE && std::string(cfgE) == "old");
    const double tF0 = now();
    double tS = tF0;   // E12: start of the current step
    auto step = [&](const char* k) { const double t = now(); stepAdd(k, t - tS); tS = t; };
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);   // before the context exists: a waiting host thread sleeps, not spins
    if (cudaSetDevice(device) != cudaSuccess) return fail("cannot select device");
    double t0 = now();
    // device memory in use at the phases of the call (logged: what a sample needs when several share a GPU)
    std::string memLog;
    auto memMark = [&](const char* what) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); char b[64]; snprintf(b, sizeof b, "%s %.1f, ", what, (double) (t - f) / (1u << 30)); memLog += b; };
    memMark("start");
    step("init");
    // ---- index (G with 200 bytes of padding on each side, as STAR's G1)
    const u64 PAD = 200;
    GenomeView D = H;
    char* dG1; u64* dSA; u64* dSAi;
    // E12 (default): the reads and E10's aux through page-locked staging buffers (Stager) on this thread, the index the same way from a second
    // host thread while the reads are uploaded, clipped and deduplicated (joined before the lanes). SINGLET_ENGINE_GPU_UPLOAD=pageable (or
    // CFG=old): the driver's pageable copies, in sequence, as before
    const char* upE = std::getenv("SINGLET_ENGINE_GPU_UPLOAD");
    const bool staged = NEW0 && !(upE && std::string(upE) == "pageable");
    // SINGLET_ENGINE_GPU_PIN=1: page-lock the host index first, so the copies run at the PCIe rate instead of through the driver's staging buffers
    const bool pinIdx = !staged && std::getenv("SINGLET_ENGINE_GPU_PIN") != nullptr;
    const double tIdx0 = now(); double tPin = 0;
    const char* d2E = std::getenv("SINGLET_ENGINE_GPU_D2H");
    if (NEW0 && !(d2E && std::string(d2E) == "pageable")) {
        if (!DR.init(std::max(1, std::min(16, hostThreads)))) return fail("page-locked staging buffers (D2H)");
        g_d2h = &DR;
    }
    if (staged) {
        const int ni = std::max(1, std::min(4, hostThreads / 4));
        if (!SR.init(std::max(1, std::min(8, hostThreads))) || !SI.init(ni)) return fail("page-locked staging buffers");
        if (cudaMalloc(&dG1, H.nGenome + 2 * PAD + 8) != cudaSuccess || cudaMalloc(&dSA, saBytes + 64 + 8) != cudaSuccess
            || cudaMalloc(&dSAi, saiBytes + 64 + 8) != cudaSuccess) return fail("out of device memory (index)");
        cudaMemset(dSA, 0, saBytes + 64 + 8); cudaMemset(dSAi, 0, saiBytes + 64 + 8);
        if (cudaDeviceSynchronize() != cudaSuccess) return fail("index memset");
        idxTh = std::thread([&, dG1, dSA, dSAi] {
            cudaSetDevice(device);
            idxOk = SI.copy(dG1, H.G - PAD, H.nGenome + 2 * PAD) && SI.copy(dSA, H.SA, saBytes) && SI.copy(dSAi, H.SAi, saiBytes) && SI.finish();
            tIdxEnd = now();
        });
        D.G = dG1 + PAD;
    } else {
    if (pinIdx) {
        cudaHostRegister((void*)(H.G - PAD), H.nGenome + 2 * PAD, cudaHostRegisterReadOnly);
        cudaHostRegister((void*)H.SA, saBytes, cudaHostRegisterReadOnly);
        cudaHostRegister((void*)H.SAi, saiBytes, cudaHostRegisterReadOnly);
        tPin = now() - tIdx0;
    }
    if (!upload(H.G - PAD, H.nGenome + 2 * PAD, dG1, err)) return fail(err);
    D.G = dG1 + PAD;
    if (!upload((const char*)H.SA, saBytes, *(char**)&dSA, err, 64)) return fail(err);
    if (!upload((const char*)H.SAi, saiBytes, *(char**)&dSAi, err, 64)) return fail(err);
    cudaDeviceSynchronize();
    if (pinIdx) { cudaHostUnregister((void*)(H.G - PAD)); cudaHostUnregister((void*)H.SA); cudaHostUnregister((void*)H.SAi); }
    { char l[160]; snprintf(l, sizeof l, "engine_gpu: index upload %.2f s (%s%.2f s to pin)\n", now() - tIdx0, pinIdx ? "pinned, " : "pageable, ", tPin); out.log += l; }
    }
    D.SA = dSA; D.SAi = dSAi;
    memMark("index");
    step("index.h2d");
    u64 *dcs, *dcl, *dcb, *dss, *dse, *dsd, *dsa; uint8_t *dsm, *dsl, *dsr, *dst;
    if (!upload(H.chrStart, H.nChr + 1, dcs, err) || !upload(H.chrLength, H.nChr, dcl, err) || !upload(H.chrBin, H.chrBinN, dcb, err)
        || !upload(H.sjdbStart, H.sjdbN, dss, err) || !upload(H.sjdbEnd, H.sjdbN, dse, err) || !upload(H.sjDstart, H.sjdbN, dsd, err)
        || !upload(H.sjAstart, H.sjdbN, dsa, err) || !upload(H.sjdbMotif, H.sjdbN, dsm, err) || !upload(H.sjdbShiftLeft, H.sjdbN, dsl, err)
        || !upload(H.sjdbShiftRight, H.sjdbN, dsr, err) || !upload(H.sjdbStrand, H.sjdbN, dst, err)) return fail(err);
    D.chrStart = dcs; D.chrLength = dcl; D.chrBin = dcb; D.sjdbStart = dss; D.sjdbEnd = dse; D.sjDstart = dsd; D.sjAstart = dsa;
    D.sjdbMotif = dsm; D.sjdbShiftLeft = dsl; D.sjdbShiftRight = dsr; D.sjdbStrand = dst;
    step("tables.h2d");
    // ---- reads
    u64 n = in.off.size() - 1;
    out.nReads = n;
    char* dSeq; u64* dOff;
    if (staged) {
        if (cudaMalloc(&dSeq, in.seq.size() + 8) != cudaSuccess || cudaMalloc(&dOff, (n + 1) * 8 + 8) != cudaSuccess) return fail("out of device memory (reads)");
        if (!SR.copy(dSeq, in.seq.data(), in.seq.size()) || !SR.copy(dOff, in.off.data(), (n + 1) * 8) || !SR.finish()) return fail("reads upload");
    } else {
    if (!upload(in.seq.data(), in.seq.size(), dSeq, err) || !upload((const u64*)in.off.data(), n + 1, dOff, err)) return fail(err);
    }
    cudaStreamSynchronize(0);
    step("reads.h2d");
    // E10 (SINGLET_ENGINE_GPU_E2E >= 2): per read its hand-off entry and low-quality mask, from the original read (before the clip replaces it),
    // its clip and what the host decode packed (aux); kept on the device until the records exist (products() zeroes the reads without one)
    bool e10 = in.e10.on && in.e2e != nullptr && in.e2e->base != nullptr && n > 0 && in.e10.nw >= 1 && in.e10.aux.size() == n * (2 + (u64)in.e10.nw);
    e6::ReadEntry* dEnt = nullptr; uint64_t* dQm = nullptr; uint64_t* dAux = nullptr; bool e10Done = false;
    auto e10Off = [&](const std::string& why) {
        if (e10) out.e10.log += "engine_gpu: E10 not used (" + why + "); STAR's loop does the per-read work\n";
        if (!why.empty()) { e9::soloKeepRelease(); out.e10.soloKept = false; }   // E11: nothing to count on the device without E10
        e10 = false; cudaGetLastError();
        if (dEnt) cudaFree(dEnt); if (dQm) cudaFree(dQm); if (dAux) cudaFree(dAux); dEnt = nullptr; dQm = nullptr; dAux = nullptr;
    };
    if (in.e10.on && !e10) out.e10.log += "engine_gpu: E10 not used (no shared region or no host decode)\n";
    if (e10) {
        const double ta = now();
        if (cudaMalloc(&dEnt, n * sizeof(e6::ReadEntry)) != cudaSuccess || (in.e10.wantMask && cudaMalloc(&dQm, n * 16) != cudaSuccess)
            || cudaMalloc(&dAux, in.e10.aux.size() * 8) != cudaSuccess) e10Off("out of device memory");
        else if (staged ? !(SR.copy(dAux, in.e10.aux.data(), in.e10.aux.size() * 8) && SR.finish())
                        : cudaMemcpy(dAux, in.e10.aux.data(), in.e10.aux.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) e10Off("aux upload");
        out.e10.msUpAux = (now() - ta) * 1e3;
        step("e10.aux.h2d");
    }
    auto e10Entries = [&](const u64* cs, const u64* cl) {
        e10Done = true;
        if (!e10) return;
        float ms = 0; std::string w;
        if (!singlet_hostloop_gpu::entries(dSeq, (const uint64_t*) dOff, (const uint64_t*) cs, (const uint64_t*) cl, n, dAux, in.e10.nw, in.e10.hook, dEnt, dQm,
                                           ms, w)) { e10Off("entries: " + w); return; }
        out.e10.msEntries = ms;
        if (in.e10.wl != nullptr && in.e10.nWL > 0) {   // R11's exact CB histogram from the same keys (failure only loses the histogram)
            uint64_t* dWL = nullptr; uint32_t* dH = nullptr; float msr = 0; std::string wr;
            if (cudaMalloc(&dWL, in.e10.nWL * 8) == cudaSuccess && cudaMalloc(&dH, in.e10.nWL * 4) == cudaSuccess
                && cudaMemcpy(dWL, in.e10.wl, in.e10.nWL * 8, cudaMemcpyHostToDevice) == cudaSuccess
                && singlet_hostloop_gpu::r11(dAux, in.e10.nw, n, dWL, in.e10.nWL, in.e10.cbL, in.e10.umiL, dH, msr, wr)) {
                out.e10.r11.resize(in.e10.nWL);
                if (cudaMemcpy(out.e10.r11.data(), dH, in.e10.nWL * 4, cudaMemcpyDeviceToHost) != cudaSuccess) out.e10.r11.clear();
                for (uint32_t c : out.e10.r11) out.e10.r11Total += c;
                out.e10.msR11 = msr;
            } else { cudaGetLastError(); out.e10.log += "engine_gpu: E10: R11 histogram not computed (" + wr + ")\n"; }
            if (dWL) cudaFree(dWL); if (dH) cudaFree(dH);
        }
        if (in.e10.soloModel != nullptr) {   // E11: the CR/UR of every read, contiguous, kept for soloRun after STAR's loop
            std::string ws;
            if (!e9::soloKeepReads(dAux, 2 + (u64) in.e10.nw, n, ws)) out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (" + ws + ")\n";
        }
        cudaFree(dAux); dAux = nullptr;
    };
    // clipping: the reads become their clipped sequences (device copy compacted; on the host each read's span is (cs, length))
    singlet_engine_gpu::NoInitVector<u64> hcs, hoffC;   // every element copied from the device
    if (in.clip.on && n > 0) {
        double tc = now();
        char* dAd = nullptr;
        if (!in.clip.ad3.empty() && !upload(in.clip.ad3.data(), in.clip.ad3.size(), dAd, err)) return fail(err);
        uint16_t *dP5 = nullptr, *dP3 = nullptr;
        if (in.clip.pre) {
            if (in.pre5.size() != n || in.pre3.size() != n) return fail("host clip lengths do not match the reads");
            if (!upload(in.pre5.data(), n, dP5, err) || !upload(in.pre3.data(), n, dP3, err)) return fail(err);
        }
        ClipDev c{in.clip.n5, in.clip.after5, in.clip.n3, in.clip.after3, (uint32_t)in.clip.ad3.size(), in.clip.adMMp, dAd, dP5, dP3};
        u64 *dCs, *dCl, *dOffC; char* dSeqC;
        if (cudaMalloc(&dCs, n * 8) || cudaMalloc(&dCl, n * 8) || cudaMalloc(&dOffC, (n + 1) * 8)) return fail("out of device memory (clip)");
        k_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, c, dCs, dCl);
        cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
        step("clip.kernel");
        if (e10) e10Entries(dCs, dCl);   // E10: the original reads and their clip
        cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
        step("e10.entries");
        cudaMemset(dOffC, 0, 8);
        thrust::inclusive_scan(thrust::device_ptr<u64>(dCl), thrust::device_ptr<u64>(dCl + n), thrust::device_ptr<u64>(dOffC + 1));
        u64 tot = 0;
        if (cudaMemcpy(&tot, dOffC + n, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("clip copy");
        if (cudaMalloc(&dSeqC, tot + 8)) return fail("out of device memory (clipped reads)");
        k_gather_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dCs, dOffC, n, dSeqC);
        cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
        step("clip.gather");
        hcs.resize(n); hoffC.resize(n + 1);
        if (g_d2h ? !(g_d2h->copy(hcs.data(), dCs, n * 8) && g_d2h->copy(hoffC.data(), dOffC, (n + 1) * 8))
                  : (cudaMemcpy(hcs.data(), dCs, n * 8, cudaMemcpyDeviceToHost) != cudaSuccess
                     || cudaMemcpy(hoffC.data(), dOffC, (n + 1) * 8, cudaMemcpyDeviceToHost) != cudaSuccess)) return fail("clip copy");
        step("clip.d2h");
        cudaFree(dSeq); cudaFree(dOff); cudaFree(dCs); cudaFree(dCl); if (dAd) cudaFree(dAd); if (dP5) cudaFree(dP5); if (dP3) cudaFree(dP3);
        dSeq = dSeqC; dOff = dOffC;
        step("clip.free");
        u64 nClipped = 0;
        out.clip5.resize(n); out.clip3.resize(n);
        std::atomic<u64> nClip{0};
        parallelFor(n, hostThreads, [&](u64 lo, u64 hi) {
            u64 cnt = 0;
            for (u64 i = lo; i < hi; i++) {
                const u64 L0 = in.off[i + 1] - in.off[i], L = hoffC[i + 1] - hoffC[i], c5 = hcs[i] - in.off[i];
                out.clip5[i] = (uint16_t) std::min<u64>(c5, 65535); out.clip3[i] = (uint16_t) std::min<u64>(L0 - L - c5, 65535);
                cnt += L != L0;
            }
            nClip += cnt;
        });
        nClipped = nClip;
        step("clip.host");
        char line[200];
        snprintf(line, sizeof line, "engine_gpu: clipping: %llu of %llu reads clipped, %.2f s\n", (unsigned long long)nClipped, (unsigned long long)n, now() - tc);
        out.log += line;
    }
    if (e10 && !e10Done) e10Entries(nullptr, nullptr);   // E10: no clipping
    memMark("reads+clip");
    // a read's (clipped) sequence on the host
    auto rseq = [&](u64 i) { return hcs.empty() ? in.seq.data() + in.off[i] : in.seq.data() + hcs[i]; };
    auto rlen = [&](u64 i) { return hcs.empty() ? in.off[i + 1] - in.off[i] : hoffC[i + 1] - hoffC[i]; };
    cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
    out.secUpload = now() - t0;
    step("clip.rest");
    // ---- dedup
    t0 = now();
    u64 *dH1, *dH2; uint8_t* dN; uint32_t *dOrder, *dHead; unsigned long long* dColl;
    if (cudaMalloc(&dH1, n * 8) || cudaMalloc(&dH2, n * 8) || cudaMalloc(&dN, n) || cudaMalloc(&dOrder, n * 4) || cudaMalloc(&dHead, n * 4)
        || cudaMalloc(&dColl, 8)) return fail("out of device memory (dedup)");
    k_hash<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, dH1, dH2, dN);
    thrust::sequence(thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n));
    auto keys = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<u64>(dH1), thrust::device_ptr<u64>(dH2)));
    thrust::sort_by_key(keys, keys + n, thrust::device_ptr<uint32_t>(dOrder));
    cudaMemset(dColl, 0, 8);
    // E12: rep, readRep and the counts on the device (SINGLET_ENGINE_GPU_DEDUP=host, or CFG=old: on the host, as before)
    const char* ddE = std::getenv("SINGLET_ENGINE_GPU_DEDUP");
    const bool dedupDev = NEW0 && !(ddE && std::string(ddE) == "host");
    uint32_t* dHp = nullptr;   // E12: per sorted position its group's first position
    if (dedupDev) {
        if (cudaMalloc(&dHp, n * 4)) return fail("out of device memory (dedup)");
        k_heads<<<(unsigned)((n + 255) / 256), 256>>>(dH1, dH2, n, dHead, dHp);
        thrust::inclusive_scan(thrust::device_ptr<uint32_t>(dHp), thrust::device_ptr<uint32_t>(dHp + n), thrust::device_ptr<uint32_t>(dHp), thrust::maximum<uint32_t>());
        k_collide<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dOrder, dHp, n, dColl);
    } else
    k_groups<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dH1, dH2, dOrder, n, dHead, dColl);
    unsigned long long coll = 0;
    if (cudaMemcpy(&coll, dColl, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
    if (coll) return fail("hash collision between different sequences (" + std::to_string(coll) + "): GPU path not used");
    step("dedup.kernels");
    std::vector<uint32_t> rep;   // one read per distinct sequence without N
    // per read: the representative read of its distinct sequence (NONE for sequences with N, which STAR maps itself)
    const uint32_t NONE = 0xFFFFFFFFu;
    singlet_engine_gpu::NoInitVector<uint32_t> readRep(n);
    if (dedupDev) {
        uint32_t* dRR; uint32_t* dRep;
        if (cudaMalloc(&dRR, n * 4) || cudaMalloc(&dRep, n * 4)) return fail("out of device memory (dedup)");
        const unsigned G = (unsigned)((n + 255) / 256);
        k_readrep<<<G, 256>>>(dOrder, dHp, dN, n, dRR);
        const u64 nDist = thrust::reduce(thrust::device_ptr<uint32_t>(dHead), thrust::device_ptr<uint32_t>(dHead + n), (u64)0);
        const u64 nRep = (u64)(thrust::copy_if(thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n), thrust::counting_iterator<u64>(0),
                                               thrust::device_ptr<uint32_t>(dRep), RepPred{dHead, dN, dOrder}) - thrust::device_ptr<uint32_t>(dRep));
        out.nDistinct = nDist; out.nWithN = nDist - nRep;
        cudaFree(dHp);
        step("dedup.dev");
        if (g_d2h ? !g_d2h->copy(readRep.data(), dRR, n * 4) : cudaMemcpy(readRep.data(), dRR, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
        cudaFree(dRR);
        cudaFree(dH1); cudaFree(dH2); cudaFree(dN); cudaFree(dOrder); cudaFree(dHead); cudaFree(dColl);
        step("dedup.d2h");
        rep.resize(nRep);
        if (nRep) {   // map them in sequence order (first 32 bases), not hash order (below)
            u64* dKey;
            if (cudaMalloc(&dKey, nRep * 8)) return fail("out of device memory (order)");
            k_prefix_key<<<(unsigned)((nRep + 255) / 256), 256>>>(dSeq, dOff, dRep, nRep, dKey);
            thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nRep), thrust::device_ptr<uint32_t>(dRep));
            if (g_d2h ? !g_d2h->copy(rep.data(), dRep, nRep * 4) : cudaMemcpy(rep.data(), dRep, nRep * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
            cudaFree(dKey);
        }
        cudaFree(dRep);
    } else {
    std::vector<uint32_t> order(n), head(n); std::vector<uint8_t> hasN(n);
    step("dedup.hostalloc");
    cudaMemcpy(order.data(), dOrder, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(head.data(), dHead, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hasN.data(), dN, n, cudaMemcpyDeviceToHost);
    step("dedup.d2h");
    cudaFree(dH1); cudaFree(dH2); cudaFree(dN); cudaFree(dOrder); cudaFree(dHead); cudaFree(dColl);
    for (u64 k = 0; k < n; k++) if (head[k]) { out.nDistinct++; if (hasN[order[k]]) out.nWithN++; else rep.push_back(order[k]); }
    { uint32_t cur = NONE; for (u64 k = 0; k < n; k++) { if (head[k]) cur = hasN[order[k]] ? NONE : order[k]; readRep[order[k]] = cur; } }
    step("dedup.host");
    }
    // map them in sequence order (first 32 bases), not hash order: neighbouring threads then search the same SA ranges and loci
    // (2.7x faster on pbmc1k_v3). Each sequence's result does not depend on the order.
    if (!dedupDev && !rep.empty()) {   // keys (first 32 bases) and the stable sort on the device
        uint32_t* dRep; u64* dKey;
        if (cudaMalloc(&dRep, rep.size() * 4) || cudaMalloc(&dKey, rep.size() * 8)) return fail("out of device memory (order)");
        cudaMemcpy(dRep, rep.data(), rep.size() * 4, cudaMemcpyHostToDevice);
        k_prefix_key<<<(unsigned)((rep.size() + 255) / 256), 256>>>(dSeq, dOff, dRep, rep.size(), dKey);
        thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + rep.size()), thrust::device_ptr<uint32_t>(dRep));
        if (cudaMemcpy(rep.data(), dRep, rep.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
        cudaFree(dRep); cudaFree(dKey);
    }
    out.secDedup = now() - t0;
    step("dedup.order");
    if (idxTh.joinable()) {   // E12: the index, uploaded meanwhile from the other thread
        idxTh.join();
        if (!idxOk) return fail("index upload");
        char l[400];
        snprintf(l, sizeof l, "engine_gpu: index upload %.2f s (staged on a second thread during the reads, clip and dedup: %.1f GB, %.1f GB/s; waited %.2f s for it); "
                 "reads and aux staged: %.1f GB, %.1f GB/s\n", tIdxEnd - tIdx0, SI.bytes / 1e9, SI.bytes / 1e9 / std::max(1e-9, SI.sec), now() - tS,
                 SR.bytes / 1e9, SR.bytes / 1e9 / std::max(1e-9, SR.sec));
        out.log += l;
        SI.release(); SR.release();
    }
    step("index.wait");
    memMark("dedup");
    std::vector<uint32_t> recOfPos(rep.size(), NONE);   // rep position -> output record (NONE: not mapped, STAR maps it)
    // ---- map in chunks of distinct sequences (device memory for the lanes' outputs is per chunk)
    u64 nGpuMapped = 0;
    // a chunk of distinct sequences in two halves: the device lanes (gpuPart, under the caller's device lock) and the host lane and records
    // (hostPart, after in.release(): they hold the GPU for nothing)
    struct ChunkRes { std::vector<uint32_t> repc; Lane A, M, B; std::vector<uint8_t> lane; std::vector<uint32_t> lpos, repH, posH; u64 c0 = 0, ci = 0;
                      std::vector<uint8_t> hBlob; std::vector<u64> hOff;    // E8: the host lane's results, packed (k of repH at hOff[k])
                      std::vector<std::vector<uint8_t>> hout; std::vector<uint8_t> hst;   // the host lane's results and statuses (k of repH)
                      std::vector<uint8_t> okv; std::vector<u64> sOff, bOff; std::vector<uint32_t> rank; };   // E12: per sequence: kept, prefix sums
    std::vector<ChunkRes> chunks;
    // sequences per chunk: the lanes' device buffers scale with it (SINGLET_ENGINE_GPU_CHUNK, default 16M); samples sharing a GPU (SINGLET_ENGINE_GPU_SLOTS) want it smaller
    u64 CHUNK = 16u << 20;
    if (const char* ce = std::getenv("SINGLET_ENGINE_GPU_CHUNK")) CHUNK = std::max<u64>(1u << 18, std::strtoull(ce, nullptr, 10));
    const u64 nChunks = (rep.size() + CHUNK - 1) / CHUNK;
    // E12: the host lane during the device lanes (a pool fed by lane B's batches; SINGLET_ENGINE_GPU_HOSTLANE=after, or CFG=old: after
    // each chunk's lanes, as before). Per chunk its results by index in the chunk's host lane; per pool thread its Work
    struct HostRes { std::vector<std::vector<uint8_t>> out; std::vector<uint8_t> st; };
    std::vector<std::unique_ptr<HostRes>> hres(nChunks);
    struct HostCtx { std::unique_ptr<Work<Heavy>> w; std::unique_ptr<ReadBuf<Heavy::LREAD>> b; std::vector<uint8_t> buf; };
    const int poolThreads = std::max(1, hostThreads - 1);   // one core for the thread that drives the device
    std::vector<HostCtx> hctx(poolThreads);
    const char* hlE = std::getenv("SINGLET_ENGINE_GPU_HOSTLANE");
    std::unique_ptr<HostPool> pool;
    if (NEW0 && !(hlE && std::string(hlE) == "after") && nChunks > 0) {
        pool.reset(new HostPool);
        pool->start(poolThreads, nChunks, [&](int t, const HostPool::Task& tk) {
            HostCtx& X = hctx[t];
            if (!X.w) { X.w.reset(new Work<Heavy>); memset((void*)X.w.get(), 0, sizeof(Work<Heavy>)); X.b.reset(new ReadBuf<Heavy::LREAD>); X.buf.resize(Heavy::NBLOB); }
            Blob bl{X.buf.data(), 0, Heavy::NBLOB, false};
            const Status s = mapRead(P, H, *X.w, *X.b, rseq(tk.i), (m5::uint)rlen(tk.i), bl);
            HostRes& R = *hres[tk.c];
            R.st[tk.q] = (uint8_t)s; R.out[tk.q].assign(X.buf.data(), X.buf.data() + bl.n);
        });
    }
    // E8 (SINGLET_ENGINE_GPU_E2E=1): the lanes keep their packed results on the device; per record its source and read (k_rec_len)
    const bool e2e = in.e2e != nullptr && in.e2e->base != nullptr;
    singlet_engine_gpu::NoInitVector<u64> recSrc; singlet_engine_gpu::NoInitVector<uint32_t> recRead; u64 chunksDone = 0;   // every element written
    // E12: the lanes' configuration. Default: the m5 harness's best exact configuration (lane A: 524288 reads per batch, phase 3
    // under 64 registers; lane M: 393216, warp per read, the batch by window count; lane B: 131072 or as many as the free device
    // memory holds). SINGLET_ENGINE_GPU_CFG=old: the engine before E12 (k_p1 / k_p3 thread per read; 262144 / 262144 / 65536).
    // SINGLET_ENGINE_GPU_NB / _NBM / _NBB set the batches of lanes A / M / B; SINGLET_ENGINE_GPU_WARPM=0: lane M thread per read.
    const bool NEW = NEW0;
    const char* wmE = std::getenv("SINGLET_ENGINE_GPU_WARPM");
    // the warp lane's window loop must never reach STAR's trNtotal break (every window is then stitched, as in STAR)
    const bool warpM = NEW && !(wmE && std::string(wmE) == "0")
        && (u64)LaneM::NW * std::min<u64>((u64)LWM::kNWTR, P.alignTranscriptsPerWindowNmax) + P.alignTranscriptsPerWindowNmax < P.alignTranscriptsPerReadNmax;
    auto envU = [](const char* k, u64 d, u64 lo) { const char* v = std::getenv(k); return v ? std::max<u64>(lo, std::strtoull(v, nullptr, 10)) : d; };
    const u64 nbA = envU("SINGLET_ENGINE_GPU_NB", NEW ? 524288 : 262144, 4096), nbM = envU("SINGLET_ENGINE_GPU_NBM", NEW ? 393216 : 262144, 4096);
    const bool nbBset = std::getenv("SINGLET_ENGINE_GPU_NBB") != nullptr;
    const u64 nbB0 = envU("SINGLET_ENGINE_GPU_NBB", NEW ? 131072 : 65536, 1024);
    g_laneLog.clear();
    auto gpuPart = [&](const std::vector<uint32_t>& repc, u64 c0) -> bool {
        // ---- lanes
        double tc0 = now();
        Lane A, M, B;
        memMark("before-laneA");
        // reads in flight per lane (nbA, nbM, nbB): Work + ReadBuf per read are 36 KB (A), 68 KB (M), 550 KB (B)
        if (!runLane<LaneA, false, 64>(P, D, dSeq, dOff, repc, nbA, 1024, A, err, e2e, "A", NEW)) return false;
        double th = now();
        std::vector<uint32_t> repM, posM;
        for (u64 k = 0; k < repc.size(); k++) if (A.status[k] != (uint8_t)Status::Ok) { repM.push_back(repc[k]); posM.push_back((uint32_t)k); }
        stepAdd("lanes.host", now() - th);
        if (!(warpM ? runLane<LaneM, true>(P, D, dSeq, dOff, repM, nbM, 2048, M, err, e2e, "M", NEW)
                    : runLane<LaneM>(P, D, dSeq, dOff, repM, nbM, 2048, M, err, e2e, "M", NEW))) return false;
        th = now();
        std::vector<uint32_t> repB, posB;
        for (u64 k = 0; k < repM.size(); k++) if (M.status[k] != (uint8_t)Status::Ok) { repB.push_back(repM[k]); posB.push_back(posM[k]); }
        stepAdd("lanes.host", now() - th);
        u64 nbB = nbB0;   // the heavy lane's reads in flight: its Work is the largest device allocation of the prefill
        if (NEW && !nbBset && !repB.empty()) {   // E12: up to nbB0, as many as the free device memory holds (3 GB spare)
            size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
            const u64 fixed = repB.size() * (u64)(LaneB::NBLOB + 4 + 1 + 2 + 8 + 8) + (3ull << 30);
            const u64 per = sizeof(Work<LaneB>) + sizeof(ReadBuf<LaneB::LREAD>) + 1 + 8 + 8;
            const u64 fit = f > fixed ? (f - fixed) / per / 1024 * 1024 : 0;
            nbB = std::max<u64>(8192, std::min<u64>(nbB0, fit));
        }
        const u64 ci = chunks.size();
        u64 qn = 0;   // E12: host-lane tasks of this chunk so far (in lane B's order: the order of repH below)
        BatchFn onB;
        if (pool) {
            hres[ci].reset(new HostRes); hres[ci]->out.resize(repB.size()); hres[ci]->st.assign(repB.size(), 0);
            onB = [&](u64 b0, u64 nb, const uint8_t* st) {
                std::vector<HostPool::Task> v;
                for (u64 k = 0; k < nb; k++) if (st[k] != (uint8_t)Status::Ok) v.push_back({(uint32_t)ci, (uint32_t)qn++, repB[b0 + k]});
                pool->push(v);
            };
        }
        if (!runLane<LaneB>(P, D, dSeq, dOff, repB, nbB, 2048, B, err, e2e, "B", NEW, pool ? &onB : nullptr)) return false;
        out.secMapGpu += now() - tc0;
        th = now();
        // where each result is: 0 = A, 1 = M, 2 = B, 3 = host
        std::vector<uint8_t> lane(repc.size(), 0); std::vector<uint32_t> lpos(repc.size());
        for (u64 k = 0; k < repc.size(); k++) lpos[k] = (uint32_t)k;
        for (u64 k = 0; k < repM.size(); k++) { lane[posM[k]] = 1; lpos[posM[k]] = (uint32_t)k; }
        std::vector<uint32_t> repH, posH;
        for (u64 k = 0; k < repB.size(); k++) {
            lane[posB[k]] = 2; lpos[posB[k]] = (uint32_t)k;
            if (B.status[k] != (uint8_t)Status::Ok) { repH.push_back(repB[k]); posH.push_back(posB[k]); lane[posB[k]] = 3; lpos[posB[k]] = (uint32_t)(repH.size() - 1); }
        }
        { ChunkRes cr; cr.repc = repc; cr.A = std::move(A); cr.M = std::move(M); cr.B = std::move(B); cr.lane = std::move(lane); cr.lpos = std::move(lpos);
          if (pool && qn != repH.size()) { err = "host lane stream: " + std::to_string(qn) + " tasks for " + std::to_string(repH.size()) + " reads"; return false; }
          cr.repH = std::move(repH); cr.posH = std::move(posH); cr.c0 = c0; cr.ci = ci; chunks.push_back(std::move(cr)); }
        stepAdd("lanes.host", now() - th);
        return true;
    };
    // where sequence k of a chunk has its result (lane, pointer, length; ok: the host lane mapped it within STAR's limits)
    auto laneOf = [&](const ChunkRes& cr, u64 k, const uint8_t*& p, u64& len, bool& ok) {
        const uint32_t q = cr.lpos[k]; ok = true;
        switch (cr.lane[k]) {
            case 0: p = cr.A.out.data() + cr.A.off[q]; len = cr.A.len[q]; break;
            case 1: p = cr.M.out.data() + cr.M.off[q]; len = cr.M.len[q]; break;
            case 2: p = cr.B.out.data() + cr.B.off[q]; len = cr.B.len[q]; break;
            default: p = cr.hout[q].data(); len = cr.hout[q].size(); ok = cr.hst[q] == (uint8_t)Status::Ok; break;
        }
    };
    // a chunk's host lane, then per sequence its kept flag and lengths and their prefix sums (the records' order is the serial loop's)
    auto hostPrep = [&](ChunkRes& cr) -> bool {
        const std::vector<uint32_t>& repH = cr.repH;
        // ---- host lane (STAR's limits)
        double th0 = now();
        std::vector<std::vector<uint8_t>>& hout = cr.hout; std::vector<uint8_t>& hst = cr.hst;
        if (pool) {   // E12: mapped by the pool while the device ran; wait for what is left of this chunk
            pool->wait(cr.ci);
            hout = std::move(hres[cr.ci]->out); hst = std::move(hres[cr.ci]->st); hres[cr.ci].reset();
            hout.resize(repH.size()); hst.resize(repH.size());
        } else {
        hout.resize(repH.size()); hst.assign(repH.size(), 0);
            std::atomic<u64> next{0};
            int nt = std::max(1, hostThreads);
            std::vector<std::thread> th;
            for (int t = 0; t < nt; t++) th.emplace_back([&]() {
                Work<Heavy>* w = new Work<Heavy>; memset((void*)w, 0, sizeof(Work<Heavy>));
                ReadBuf<Heavy::LREAD>* b = new ReadBuf<Heavy::LREAD>;
                std::vector<uint8_t> buf(Heavy::NBLOB);
                for (u64 k; (k = next++) < repH.size();) {
                    u64 i = repH[k];
                    Blob bl{buf.data(), 0, Heavy::NBLOB, false};
                    Status s = mapRead(P, H, *w, *b, rseq(i), (m5::uint)rlen(i), bl);
                    hst[k] = (uint8_t)s; hout[k].assign(buf.data(), buf.data() + bl.n);
                }
                delete w; delete b;
            });
            for (auto& x : th) x.join();
        }
        out.secHostLane += now() - th0;
        stepAdd("hostlane", now() - th0);
        const double tR0 = now();
        out.nHostLane += repH.size();
        if (e2e) {   // E8: the host lane's results packed for the upload (records only of the reads it mapped)
            cr.hOff.assign(repH.size() + 1, 0);
            for (u64 q = 0; q < repH.size(); q++) cr.hOff[q + 1] = cr.hOff[q] + (hst[q] == (uint8_t)Status::Ok ? hout[q].size() : 0);
            cr.hBlob.resize(cr.hOff[repH.size()]);
            for (u64 q = 0; q < repH.size(); q++) if (hst[q] == (uint8_t)Status::Ok && !hout[q].empty()) std::memcpy(cr.hBlob.data() + cr.hOff[q], hout[q].data(), hout[q].size());
        }
        const std::vector<uint32_t>& repc = cr.repc;
        const u64 nK = repc.size();
        cr.okv.assign(nK, 1); cr.sOff.assign(nK + 1, 0); cr.bOff.assign(nK + 1, 0); cr.rank.assign(nK + 1, 0);
        std::vector<uint8_t>& okv = cr.okv; std::vector<u64>& sOff = cr.sOff; std::vector<u64>& bOff = cr.bOff; std::vector<uint32_t>& rank = cr.rank;
        parallelFor(nK, hostThreads, [&](u64 lo, u64 hi) {
            for (u64 k = lo; k < hi; k++) {
                const uint8_t* p; u64 len; bool ok;
                laneOf(cr, k, p, len, ok);
                okv[k] = ok; sOff[k + 1] = ok ? rlen(repc[k]) : 0; bOff[k + 1] = ok ? len : 0; rank[k + 1] = ok ? 1 : 0;
            }
        });
        for (u64 k = 0; k < nK; k++) { sOff[k + 1] += sOff[k]; bOff[k + 1] += bOff[k]; rank[k + 1] += rank[k]; if (!okv[k]) out.nFailed++; }   // not cached: STAR maps it
        stepAdd("records.prep", now() - tR0);
        return true;
    };
    // a chunk's records copied to their places (s0, b0, r0: the chunk's first sequence byte, blob byte and record)
    auto hostCopy = [&](ChunkRes& cr, u64 s0, u64 b0, u64 r0) {
        const double tC = now();
        const std::vector<uint32_t>& repc = cr.repc; const u64 c0 = cr.c0, nK = repc.size();
        const u64 chunkIx = chunksDone;
        parallelFor(nK, hostThreads, [&](u64 lo, u64 hi) {
            for (u64 k = lo; k < hi; k++) {
                if (!cr.okv[k]) continue;
                const uint8_t* p; u64 len; bool ok;
                laneOf(cr, k, p, len, ok);
                const u64 i = repc[k], L = rlen(i), rec = r0 + cr.rank[k];
                recOfPos[c0 + k] = (uint32_t)rec;
                if (e2e) { recSrc[rec] = chunkIx << 40 | (u64)cr.lane[k] << 32 | (u64)cr.lpos[k]; recRead[rec] = (uint32_t)i; }
                std::memcpy(out.seq.data() + s0 + cr.sOff[k], rseq(i), L); out.seqOff[rec + 1] = s0 + cr.sOff[k] + L;
                std::memcpy(out.blob.data() + b0 + cr.bOff[k], p, len); out.blobOff[rec + 1] = b0 + cr.bOff[k] + len;
            }
        });
        nGpuMapped += repc.size() - cr.repH.size();
        chunksDone++;
        std::vector<std::vector<uint8_t>>().swap(cr.hout); std::vector<uint8_t>().swap(cr.hst); std::vector<uint8_t>().swap(cr.okv);
        std::vector<u64>().swap(cr.sOff); std::vector<u64>().swap(cr.bOff); std::vector<uint32_t>().swap(cr.rank); std::vector<uint32_t>().swap(cr.repc);
        stepAdd("records.copy", now() - tC);
    };
    // E12: every chunk's sizes first, then the records' arrays allocated once at their final size without a zero-fill (before: grown per
    // chunk by resize, i.e. a serial zero-fill and a copy of the earlier chunks), filled by the parallel copies
    auto hostAll = [&]() -> bool {
        for (auto& cr : chunks) if (!hostPrep(cr)) return false;
        const double tA0 = now();
        u64 S = out.seq.size(), Bt = out.blob.size(), R = out.seqOff.size() - 1;
        std::vector<u64> s0(chunks.size()), b0(chunks.size()), r0(chunks.size());
        for (u64 c = 0; c < chunks.size(); c++) {
            const u64 nK = chunks[c].repc.size();
            s0[c] = S; b0[c] = Bt; r0[c] = R;
            S += chunks[c].sOff[nK]; Bt += chunks[c].bOff[nK]; R += chunks[c].rank[nK];
        }
        out.seq.resize(S); out.blob.resize(Bt); out.seqOff.resize(R + 1); out.blobOff.resize(R + 1);
        if (e2e) { recSrc.resize(R); recRead.resize(R); }
        stepAdd("records.alloc", now() - tA0);
        for (u64 c = 0; c < chunks.size(); c++) hostCopy(chunks[c], s0[c], b0[c], r0[c]);
        return true;
    };
    tS = now();
    chunks.reserve(nChunks);
    for (u64 c0 = 0; c0 < rep.size(); c0 += CHUNK) {
        std::vector<uint32_t> repc(rep.begin() + c0, rep.begin() + std::min<u64>(rep.size(), c0 + CHUNK));
        if (!gpuPart(repc, c0)) return fail(err);
    }
    tS = now();
    // E8: the pileup's records built on the device (E6 buildRecords) from the device-resident results and copied into the shared region
    // (in.e2e). Runs after hostPart (the host lane's results and the record order exist). false: the region is left without magic and the
    // hand-off file path runs as before.
    auto e2eBuild = [&](std::string& why) -> bool {
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        const u64 nRec = out.seqOff.size() - 1;
        if (nRec == 0 || recSrc.size() != nRec || recRead.size() != nRec) { why = "no records"; return false; }
        struct DevFree { std::vector<void*> p; ~DevFree() { for (void* x : p) cudaFree(x); } } df;
        auto dal = [&](u64 bytes) -> void* { void* p = nullptr; if (cudaMalloc(&p, bytes + 64) != cudaSuccess) { cudaGetLastError(); return nullptr; } df.p.push_back(p); return p; };
        const double tg = now();
        // the gather: per chunk the lanes' packed results (device-resident; the host lane's uploaded), per record its source and sequence
        std::vector<LaneSrc> tab(chunks.size() * 4, LaneSrc{nullptr, nullptr});
        u64 hostBytes = 0;
        for (u64 c = 0; c < chunks.size(); c++) {
            ChunkRes& cr = chunks[c];
            tab[c * 4 + 0] = LaneSrc{cr.A.dOut, cr.A.dOff}; tab[c * 4 + 1] = LaneSrc{cr.M.dOut, cr.M.dOff}; tab[c * 4 + 2] = LaneSrc{cr.B.dOut, cr.B.dOff};
            if (cr.hOff.size() > 1) {
                uint8_t* b = (uint8_t*) dal(cr.hBlob.size()); u64* o = (u64*) dal(cr.hOff.size() * 8);
                if (!b || !o) { why = "out of device memory (host lane)"; return false; }
                if ((!cr.hBlob.empty() && cudaMemcpy(b, cr.hBlob.data(), cr.hBlob.size(), cudaMemcpyHostToDevice) != cudaSuccess)
                    || cudaMemcpy(o, cr.hOff.data(), cr.hOff.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) { why = "host lane upload"; return false; }
                tab[c * 4 + 3] = LaneSrc{b, o};
                hostBytes += cr.hBlob.size() + cr.hOff.size() * 8;
            }
            std::vector<uint8_t>().swap(cr.hBlob); std::vector<u64>().swap(cr.hOff);
        }
        LaneSrc* dTab = (LaneSrc*) dal(tab.size() * sizeof(LaneSrc)); u64* dSrc = (u64*) dal(nRec * 8); uint32_t* dRd = (uint32_t*) dal(nRec * 4);
        u64* dBOff = (u64*) dal((nRec + 1) * 8); u64* dSOff = (u64*) dal((nRec + 1) * 8);
        if (!dTab || !dSrc || !dRd || !dBOff || !dSOff) { why = "out of device memory (gather)"; return false; }
        if (cudaMemcpy(dTab, tab.data(), tab.size() * sizeof(LaneSrc), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(dSrc, recSrc.data(), nRec * 8, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(dRd, recRead.data(), nRec * 4, cudaMemcpyHostToDevice) != cudaSuccess) { why = "gather upload"; return false; }
        cudaMemset(dBOff, 0, 8); cudaMemset(dSOff, 0, 8);
        const unsigned G = (unsigned) ((nRec + 255) / 256);
        k_rec_len<<<G, 256>>>(dSrc, dRd, nRec, dTab, dOff, dBOff + 1, dSOff + 1);
        thrust::inclusive_scan(thrust::device_ptr<u64>(dBOff + 1), thrust::device_ptr<u64>(dBOff + 1 + nRec), thrust::device_ptr<u64>(dBOff + 1));
        thrust::inclusive_scan(thrust::device_ptr<u64>(dSOff + 1), thrust::device_ptr<u64>(dSOff + 1 + nRec), thrust::device_ptr<u64>(dSOff + 1));
        u64 nB = 0, nS = 0;
        if (cudaMemcpy(&nB, dBOff + nRec, 8, cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(&nS, dSOff + nRec, 8, cudaMemcpyDeviceToHost) != cudaSuccess) {
            why = "gather copy"; return false;
        }
        if (nB != out.blob.size() || nS != out.seq.size()) {
            why = "gathered sizes differ from the host records (blob " + std::to_string(nB) + " vs " + std::to_string(out.blob.size()) + ", seq "
                  + std::to_string(nS) + " vs " + std::to_string(out.seq.size()) + ")";
            return false;
        }
        uint8_t* dBlob = (uint8_t*) dal(nB); char* dRS = (char*) dal(nS);
        if (!dBlob || !dRS) { why = "out of device memory (gathered records)"; return false; }
        k_rec_copy<<<G, 256>>>(dSrc, dRd, nRec, dTab, dSeq, dOff, dBOff, dSOff, dBlob, dRS);
        if (cudaDeviceSynchronize() != cudaSuccess) { why = std::string("gather: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        if (std::getenv("SINGLET_ENGINE_GPU_E2E_CHECK")) {   // debug: the gathered records against the host-assembled ones, byte for byte
            std::vector<uint8_t> hb(nB); std::vector<char> hs(nS); std::vector<u64> hbo(nRec + 1), hso(nRec + 1);
            bool same = cudaMemcpy(hb.data(), dBlob, nB, cudaMemcpyDeviceToHost) == cudaSuccess && cudaMemcpy(hs.data(), dRS, nS, cudaMemcpyDeviceToHost) == cudaSuccess
                        && cudaMemcpy(hbo.data(), dBOff, (nRec + 1) * 8, cudaMemcpyDeviceToHost) == cudaSuccess
                        && cudaMemcpy(hso.data(), dSOff, (nRec + 1) * 8, cudaMemcpyDeviceToHost) == cudaSuccess;
            same = same && std::memcmp(hb.data(), out.blob.data(), nB) == 0 && std::memcmp(hs.data(), out.seq.data(), nS) == 0
                   && std::memcmp(hbo.data(), out.blobOff.data(), (nRec + 1) * 8) == 0 && std::memcmp(hso.data(), out.seqOff.data(), (nRec + 1) * 8) == 0;
            out.log += same ? "engine_gpu: E8 check: the device-gathered records equal the host-assembled records byte for byte\n"
                            : "engine_gpu: E8 check: the device-gathered records DIFFER from the host-assembled records\n";
            if (!same) { why = "E8 check failed"; return false; }
        }
        const double tb = now();
        e6::RecParams RP;
        RP.outFilterScoreMin = (int) P.outFilterScoreMin; RP.outFilterScoreMinOverLread = P.outFilterScoreMinOverLread;
        RP.outFilterMatchNmin = P.outFilterMatchNmin; RP.outFilterMatchNminOverLread = P.outFilterMatchNminOverLread;
        RP.outFilterMismatchNmax = P.outFilterMismatchNmax; RP.outFilterMismatchNoverLmax = P.outFilterMismatchNoverLmax;
        RP.outFilterMismatchNoverReadLmax = P.outFilterMismatchNoverReadLmax; RP.outFilterMultimapNmax = P.outFilterMultimapNmax;
        RP.outSAMmultNmax = P.outSAMmultNmax; RP.outSAMmapqUnique = in.outSAMmapqUnique; RP.outSAMflagAND = in.outSAMflagAND; RP.outSAMflagOR = in.outSAMflagOR;
        singlet_records_gpu::DeviceRecords R6;
        struct R6Free { singlet_records_gpu::DeviceRecords& r; ~R6Free() { singlet_records_gpu::release(r); } } r6f{R6};
        std::string e6err;
        if (!singlet_records_gpu::buildRecords(RP, dBlob, (const uint64_t*) dBOff, dRS, (const uint64_t*) dSOff, nRec, (const uint64_t*) dcs, false, R6, e6err, true)) {
            why = "buildRecords: " + e6err; return false;
        }
        if (R6.nMateGap) { why = std::to_string(R6.nMateGap) + " records with a mate gap (paired-end; alignBAM's path)"; return false; }
        const double td = now();
        // into the region (page-locked for the copies when the driver allows it; else pageable)
        const u64 oS = RG.take(R6.nRec * sizeof(SigRec)), oA = RG.take(R6.nAln * sizeof(AlnRec)), oO = RG.take(R6.nOps * 4), oN = RG.take(R6.nAln * 2),
                  oQ = RG.take(R6.nSeq);
        if (!oS || !oA || !oO || !oN || !oQ) { why = "the shared region is full"; return false; }
        bool pinned = false, cp = false;
        if (g_d2h)   // E12: through the page-locked buffers (the region's pages faulted in by the copying threads)
            cp = g_d2h->copy(RG.base + oS, R6.sigs, R6.nRec * sizeof(SigRec)) && g_d2h->copy(RG.base + oA, R6.alns, R6.nAln * sizeof(AlnRec))
                 && g_d2h->copy(RG.base + oO, R6.ops, R6.nOps * 4) && g_d2h->copy(RG.base + oN, R6.nm, R6.nAln * 2) && g_d2h->copy(RG.base + oQ, R6.seq, R6.nSeq);
        else {
        const u64 lo = oS & ~4095ull, hi = std::min<u64>(RG.cap, (RG.hdr()->used + 4095) & ~4095ull);
        // fault the pages in on all host threads first (cudaHostRegister would fault the ~1 M tmpfs pages one by one); *q = *q keeps the bytes
        parallelFor((hi - lo) >> 12, hostThreads, [&](u64 a, u64 b) { for (u64 q = a; q < b; q++) { volatile uint8_t* x = RG.base + lo + (q << 12); *x = *x; } });
        pinned = cudaHostRegister(RG.base + lo, hi - lo, cudaHostRegisterDefault) == cudaSuccess;
        if (!pinned) cudaGetLastError();
        cp = cudaMemcpy(RG.base + oS, R6.sigs, R6.nRec * sizeof(SigRec), cudaMemcpyDeviceToHost) == cudaSuccess
                        && (!R6.nAln || cudaMemcpy(RG.base + oA, R6.alns, R6.nAln * sizeof(AlnRec), cudaMemcpyDeviceToHost) == cudaSuccess)
                        && (!R6.nOps || cudaMemcpy(RG.base + oO, R6.ops, R6.nOps * 4, cudaMemcpyDeviceToHost) == cudaSuccess)
                        && (!R6.nAln || cudaMemcpy(RG.base + oN, R6.nm, R6.nAln * 2, cudaMemcpyDeviceToHost) == cudaSuccess)
                        && (!R6.nSeq || cudaMemcpy(RG.base + oQ, R6.seq, R6.nSeq, cudaMemcpyDeviceToHost) == cudaSuccess);
        if (pinned) cudaHostUnregister(RG.base + lo);
        }
        if (!cp) { why = std::string("records copy: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        const double te = now();
        singlet_engine_gpu::E2EHeader& E2H = *RG.hdr();
        E2H.nRec = R6.nRec; E2H.nAln = R6.nAln; E2H.nOps = R6.nOps; E2H.nSeq = R6.nSeq;
        E2H.oSigs = oS; E2H.oAlns = oA; E2H.oOps = oO; E2H.oNm = oN; E2H.oSeq = oQ;
        E2H.nMateGap = R6.nMateGap; E2H.nNoExon = R6.nNoExon; E2H.nBadBest = R6.nBadBest; E2H.nMapped = R6.nMapped;
        E2H.msGather = (tb - tg) * 1e3; E2H.msBuild = (td - tb) * 1e3; E2H.msD2H = (te - td) * 1e3;
        if (e10) {   // E10: the per-read products of the reads with a record, then the entries and masks into the region
            uint32_t* dRR = (uint32_t*) dal(n * 4);
            std::string w;
            if (!dRR || cudaMemcpy(dRR, out.readRecord.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) e10Off("readRecord upload");
            else if (!singlet_hostloop_gpu::products(RP, dBlob, (const uint64_t*) dBOff, (const uint64_t*) dSOff, nRec, dRR, n, dEnt, dQm, in.e10.outSJ,
                                                     in.e10.sjAll, out.e10, w)) e10Off("products: " + w);
            else if (out.e10.nOddRec) e10Off(std::to_string(out.e10.nOddRec) + " records mapped without a transcript");
            else {
                const double t0e = now();
                const u64 oEn = RG.take(n * sizeof(e6::ReadEntry)), oQm = dQm ? RG.take(n * 16) : 0;
                if (!oEn || (dQm && !oQm)) e10Off("the shared region is full");
                else {
                    bool pin2 = false, cp2 = false;
                    if (g_d2h) cp2 = g_d2h->copy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry)) && (!dQm || g_d2h->copy(RG.base + oQm, dQm, n * 16));
                    else {
                    const u64 lo2 = oEn & ~4095ull, hi2 = std::min<u64>(RG.cap, (RG.hdr()->used + 4095) & ~4095ull);
                    parallelFor((hi2 - lo2) >> 12, hostThreads, [&](u64 a, u64 b) { for (u64 q = a; q < b; q++) { volatile uint8_t* x = RG.base + lo2 + (q << 12); *x = *x; } });
                    pin2 = cudaHostRegister(RG.base + lo2, hi2 - lo2, cudaHostRegisterDefault) == cudaSuccess;
                    if (!pin2) cudaGetLastError();
                    cp2 = cudaMemcpy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry), cudaMemcpyDeviceToHost) == cudaSuccess
                                     && (!dQm || cudaMemcpy(RG.base + oQm, dQm, n * 16, cudaMemcpyDeviceToHost) == cudaSuccess);
                    if (pin2) cudaHostUnregister(RG.base + lo2);
                    }
                    if (!cp2) e10Off("entries copy");
                    else {
                        out.e10.oEntries = oEn; out.e10.oQMask = oQm; out.e10.ok = true;
                        out.e10.msD2H = (now() - t0e) * 1e3;
                        char l2[400];
                        snprintf(l2, sizeof l2, "engine_gpu: E10 on the device: entries %.1f ms (aux upload %.0f ms), R11 histogram %.1f ms (%llu reads), "
                                 "products %.1f ms (%llu junction entries -> %llu junctions, "
                                 "%llu unmapped reads, %llu hook reads), entries/masks to the region %.0f ms (%s)\n", out.e10.msEntries, out.e10.msUpAux,
                                 out.e10.msR11, (unsigned long long) out.e10.r11Total,
                                 out.e10.msProducts, (unsigned long long) out.e10.nSJentries, (unsigned long long) out.e10.nSJ,
                                 (unsigned long long) out.e10.unmappedReads.size(), (unsigned long long) out.e10.hookReads.size(), out.e10.msD2H,
                                 g_d2h ? "staged" : pin2 ? "page-locked" : "pageable");
                        out.e10.log += l2;
                        // E11: the records' Gene annotation (Solo's stage (b), while the blobs are here) and the read -> record map, kept
                        if (in.e10.soloModel != nullptr && in.e10.soloParams != nullptr && e9::soloKeep().reads) {
                            std::string ws;
                            if (e9::soloKeepRecords(*in.e10.soloParams, *in.e10.soloModel, RP, dBlob, (const uint64_t*) dBOff, (const uint64_t*) dSOff, nRec,
                                                    dRR, n, ws)) {
                                const e9::SoloKeep& K = e9::soloKeep();
                                out.e10.soloKept = K.nOverflow == 0;
                                char l3[300];
                                snprintf(l3, sizeof l3, "engine_gpu: E11: Solo inputs kept on the device: CR/UR of %llu reads (%.1f ms), Gene annotation of %llu records "
                                         "and the read -> record map (%.1f ms, %llu exon overflows), %.2f GB\n", (unsigned long long) K.n, K.msReads,
                                         (unsigned long long) K.nRec, K.msRecs, (unsigned long long) K.nOverflow, (K.n * 20.0 + K.nRec * 8.0) / 1e9);
                                out.e10.log += l3;
                                if (!out.e10.soloKept) e9::soloKeepRelease();
                            } else out.e10.log += "engine_gpu: E11: the records' Gene annotation not kept (" + ws + ")\n";
                        } else if (in.e10.soloModel != nullptr) { e9::soloKeepRelease(); out.e10.log += "engine_gpu: E11: Solo inputs not kept\n"; }
                    }
                }
            }
        }
        char l[512];
        snprintf(l, sizeof l, "engine_gpu: E8 records on the device: %llu records, %llu alignments, %llu ops, %llu seq bytes (%llu mapped, no exon %llu, "
                 "trBest=trInit %llu); gather %.0f ms (host lane upload %.1f MB), buildRecords %.0f ms (kernels %.0f/%.0f/%.0f), copy to the region %.0f ms (%s)\n",
                 (unsigned long long) R6.nRec, (unsigned long long) R6.nAln, (unsigned long long) R6.nOps, (unsigned long long) R6.nSeq,
                 (unsigned long long) R6.nMapped, (unsigned long long) R6.nNoExon, (unsigned long long) R6.nBadBest, E2H.msGather, hostBytes / 1e6, E2H.msBuild,
                 R6.msCount, R6.msScan, R6.msFill, E2H.msD2H, g_d2h ? "staged" : pinned ? "page-locked" : "pageable");
        out.log += l;
        out.e2eOk = true;
        return true;
    };
    // per read: its record (sequence-cache result) or NONE; final once every hostPart ran
    auto fillReadRecord = [&]() {
        singlet_engine_gpu::NoInitVector<uint32_t> posOf(n);   // set for every representative, the only reads readRep names
        parallelFor(rep.size(), hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) posOf[rep[k]] = (uint32_t)k; });
        out.readRecord.assign(n, NONE);
        parallelFor(n, hostThreads, [&](u64 lo, u64 hi) { for (u64 i = lo; i < hi; i++) { const uint32_t r = readRep[i]; if (r != NONE) out.readRecord[i] = recOfPos[posOf[r]]; } });
    };
    // E8: the host lane and the host records first (the device still held), then the records on the device from the device-resident results
    if (e2e) {
        if (!hostAll()) return fail(err);
        tS = now();
        if (e10) fillReadRecord();   // E10: the products need it on the device
        step("readRecord");
        cudaFree(dG1); cudaFree(dSA); cudaFree(dSAi); dG1 = nullptr; dSA = nullptr; dSAi = nullptr;   // the index is no longer needed: room for the records
        step("index.free");
        std::string e2eWhy;
        if (!e2eBuild(e2eWhy)) { out.log += "engine_gpu: E8 records not built on the device (" + e2eWhy + "); the hand-off file path is used\n"; e10Off("no E8 records"); }
        step("e2e.build");
        { const bool ok10 = out.e10.ok; e10 = false; e10Off(""); out.e10.ok = ok10; }   // frees the device entries
        for (auto& cr : chunks) for (Lane* L : {&cr.A, &cr.M, &cr.B}) { cudaFree(L->dOut); cudaFree(L->dOff); L->dOut = nullptr; L->dOff = nullptr; }
        decltype(recSrc)().swap(recSrc); decltype(recRead)().swap(recRead);
        step("e2e.free");
    }
    // the last device work is done: free the device buffers and let the next sample have the GPU while this one assembles its records
    cudaDeviceSynchronize();
    cudaFree(dG1); cudaFree(dSA); cudaFree(dSAi); cudaFree(dSeq); cudaFree(dOff);
    cudaFree(dcs); cudaFree(dcl); cudaFree(dcb); cudaFree(dss); cudaFree(dse); cudaFree(dsd); cudaFree(dsa);
    cudaFree(dsm); cudaFree(dsl); cudaFree(dsr); cudaFree(dst);
    step("free");
    if (in.release) in.release();
    if (!e2e && !hostAll()) return fail(err);
    tS = now();
    if (out.readRecord.size() != n) fillReadRecord();
    step("readRecord");
    out.log += "engine_gpu: device memory (GB in use): " + memLog + "lanes peak " + std::to_string(g_devPeakGB) + "\n";
    char line[512];
    snprintf(line, sizeof line, "engine_gpu: %llu reads, %llu distinct (%llu with N), mapped on the GPU %llu, host lane %llu (%.2f s), not mapped %llu; "
             "upload %.2f s, dedup %.2f s, GPU mapping %.2f s\n", (unsigned long long)out.nReads, (unsigned long long)out.nDistinct,
             (unsigned long long)out.nWithN, (unsigned long long)nGpuMapped, (unsigned long long)out.nHostLane, out.secHostLane,
             (unsigned long long)out.nFailed, out.secUpload, out.secDedup, out.secMapGpu);
    out.log += line;
    out.log += std::string("engine_gpu: E12 lanes (") + (NEW ? "new" : "old: SINGLET_ENGINE_GPU_CFG=old") + "): batches " + std::to_string(nbA) + " / "
               + std::to_string(nbM) + " / " + std::to_string(nbB0) + (warpM ? ", lane M warp per read" : "") + "; " + g_laneLog + "\n";
    if (g_d2h) {
        char ld[160];
        snprintf(ld, sizeof ld, "engine_gpu: E12 device -> host copies staged: %.2f GB in %.2f s (%.1f GB/s)\n", DR.bytes / 1e9, DR.sec, DR.bytes / 1e9 / std::max(1e-9, DR.sec));
        out.log += ld;
    }
    if (pool) {
        char lp[240];
        snprintf(lp, sizeof lp, "engine_gpu: E12 host lane during the device lanes: %llu reads on %d threads, %.2f core-s; waited %.2f s for it after the lanes\n",
                 (unsigned long long)pool->nDone.load(), poolThreads, pool->busyNs.load() / 1e9, out.secHostLane);
        out.log += lp;
    }
    out.log += "engine_gpu: steps (s): " + stepLine() + "; total " + std::to_string(now() - tF0) + "\n";
    if (const char* dp = std::getenv("SINGLET_ENGINE_GPU_DUMP")) {   // E6 debug dump (off by default; see dumpOutput)
        const double td = now();
        dumpOutput(dp, P, H, out);
        snprintf(line, sizeof line, "engine_gpu: E6 dump of %llu records -> %s (%.1f s)\n", (unsigned long long)(out.seqOff.size() - 1), dp, now() - td);
        out.log += line;
    }
    return true;
}

} // namespace singlet_engine_gpu
