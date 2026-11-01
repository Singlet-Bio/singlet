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
#include "singlet/nvtx.h"
#include "stitch.cuh"
#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/transform_scan.h>
#include <thrust/functional.h>
#include <thrust/iterator/zip_iterator.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <map>
#include <string>
#include <thread>
#include <vector>

using namespace m5;

namespace {

#define EG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// SINGLET_ENGINE_GPU_TRACE=1: the transfer ledger and the kernel times of this file, in the log. The prefill runs in the forked STAR child, which nsys
// does not follow, so this is its profile: every cudaMemcpy (calls, bytes, host seconds, pinned or pageable host side) by source line, and the GPU time of
// each kernel launch below (cudaEvents) by name. Off: one branch per call.
bool egTraceOn() { static const bool on = std::getenv("SINGLET_ENGINE_GPU_TRACE") != nullptr; return on; }
struct EgRow { unsigned long long n = 0, bytes = 0; double sec = 0; bool pinned = false; int kind = 0; };
struct EgKRow { unsigned long long n = 0; double ms = 0; };
struct EgPending { const char* tag; cudaEvent_t a, b; };
std::map<int, EgRow>& egRows() { static std::map<int, EgRow> m; return m; }
std::vector<EgPending>& egPending() { static std::vector<EgPending> v; return v; }
cudaError_t egMemcpy(int line, void* dst, const void* src, size_t n, cudaMemcpyKind k) {
    if (!egTraceOn()) return ::cudaMemcpy(dst, src, n, k);
    const double t = now();
    const cudaError_t r = ::cudaMemcpy(dst, src, n, k);
    const double dt = now() - t;
    bool pinned = false;
    const void* h = k == cudaMemcpyHostToDevice ? src : (k == cudaMemcpyDeviceToHost ? dst : nullptr);
    if (h != nullptr) { cudaPointerAttributes a; if (cudaPointerGetAttributes(&a, h) == cudaSuccess) pinned = a.type == cudaMemoryTypeHost; cudaGetLastError(); }
    EgRow& w = egRows()[line];
    w.n++; w.bytes += n; w.sec += dt; w.pinned = pinned; w.kind = (int) k;
    return r;
}
#define cudaMemcpy(d, s, n, k) egMemcpy(__LINE__, (d), (s), (n), (k))
#define EG_KT(tag, ...) do { cudaEvent_t a_, b_; const bool on_ = egTraceOn(); if (on_) { cudaEventCreate(&a_); cudaEventCreate(&b_); cudaEventRecord(a_); } __VA_ARGS__; if (on_) { cudaEventRecord(b_); egPending().push_back({tag, a_, b_}); } } while (0)
// warp balance of stage 3 (SINGLET_ENGINE_GPU_TRACE): per read the stitching steps (nSteps); a warp takes as long as its slowest read, so the sum over warps of
// the largest steps is the time in step units. Efficiency = total / (32 x that sum), for the batch's reads in the order they run, in the best possible order
// (oracle: sorted by steps) and sorted by the stage-1 hit count (the bucketing key), ascending.
struct EgDiag { double clk[8] = {0, 0, 0, 0, 0, 0, 0, 0}; double total = 0, wOrig = 0, wOracle = 0, wNent = 0, wNW = 0; u64 reads = 0, zero = 0; double top1 = 0; };
EgDiag& egDiag() { static EgDiag d; return d; }
double egWarpMax(const std::vector<u64>& v) { double s = 0; for (size_t i = 0; i < v.size(); i += 32) { u64 m = 0; for (size_t j = i; j < std::min(v.size(), i + 32); j++) m = std::max(m, v[j]); s += (double) m; } return s; }
void egDiagBatch(const std::vector<u64>& steps, const std::vector<u64>& nent, const std::vector<u64>& nw) {
    EgDiag& d = egDiag();
    const size_t n = steps.size();
    for (u64 x : steps) { d.total += (double) x; d.zero += x == 0; }
    d.reads += n; d.wOrig += egWarpMax(steps);
    std::vector<u64> t = steps; std::sort(t.begin(), t.end()); d.wOracle += egWarpMax(t);
    for (size_t i = n - n / 100; i < n; i++) d.top1 += (double) t[i];
    std::vector<uint32_t> ix(n); for (size_t i = 0; i < n; i++) ix[i] = (uint32_t) i;
    std::stable_sort(ix.begin(), ix.end(), [&](uint32_t a, uint32_t b) { return nent[a] < nent[b]; });
    for (size_t i = 0; i < n; i++) t[i] = steps[ix[i]];
    d.wNent += egWarpMax(t);
    std::stable_sort(ix.begin(), ix.end(), [&](uint32_t a, uint32_t b) { return nw[a] < nw[b]; });
    for (size_t i = 0; i < n; i++) t[i] = steps[ix[i]];
    d.wNW += egWarpMax(t);
}
void egReport(std::string& log) {
    if (!egTraceOn()) return;
    char l[256];
    {
        const EgDiag& d = egDiag();
        if (d.reads) {
            char lb[512]; snprintf(lb, sizeof lb, "engine_gpu trace: stage-3 warp balance: %llu reads, %.4g steps total (%.1f%% of reads 0 steps, top 1%% of reads %.1f%% of steps); warp efficiency original order %.1f%%, "
                     "sorted by hits %.1f%%, by windows %.1f%%, oracle by steps %.1f%%\n", d.reads, d.total, 100.0 * d.zero / d.reads, 100.0 * d.top1 / std::max(1.0, d.total),
                     100.0 * d.total / (32 * std::max(1.0, d.wOrig)), 100.0 * d.total / (32 * std::max(1.0, d.wNent)), 100.0 * d.total / (32 * std::max(1.0, d.wNW)),
                     100.0 * d.total / (32 * std::max(1.0, d.wOracle)));
            log += lb;
            snprintf(lb, sizeof lb, "engine_gpu trace: thread cycles by phase (G): seeds %.2f, windows %.2f, stitching %.2f (stitchAlign %.2f, finalize %.2f, copies %.2f, record %.2f), selection+encoding %.2f; %.0f cycles per stitching step"
                     , d.clk[0] / 1e9, d.clk[1] / 1e9, d.clk[2] / 1e9, d.clk[4] / 1e9, d.clk[5] / 1e9, d.clk[6] / 1e9, d.clk[7] / 1e9, d.clk[3] / 1e9, d.clk[2] / std::max(1.0, d.total));
            log += lb; log += "\n";
        }
    }
    std::map<std::string, EgKRow> kr;
    for (auto& e : egPending()) { float ms = 0; cudaEventSynchronize(e.b); cudaEventElapsedTime(&ms, e.a, e.b); kr[e.tag].n++; kr[e.tag].ms += ms; cudaEventDestroy(e.a); cudaEventDestroy(e.b); }
    egPending().clear();
    for (auto& kv : kr) { snprintf(l, sizeof l, "engine_gpu trace: kernel %s: %llu launches, %.1f ms GPU time\n", kv.first.c_str(), kv.second.n, kv.second.ms); log += l; }
    std::vector<std::pair<int, EgRow>> v(egRows().begin(), egRows().end());
    std::sort(v.begin(), v.end(), [](const auto& x, const auto& y) { return x.second.sec > y.second.sec; });
    for (auto& kv : v) {
        const EgRow& w = kv.second;
        snprintf(l, sizeof l, "engine_gpu trace: cudaMemcpy line %d %s: %llu calls, %.3f GB, %.3f s host, %.2f GB/s, %s host memory\n", kv.first,
                 w.kind == cudaMemcpyHostToDevice ? "H2D" : (w.kind == cudaMemcpyDeviceToHost ? "D2H" : "other"), w.n, w.bytes / 1e9, w.sec,
                 w.sec > 0 ? w.bytes / 1e9 / w.sec : 0.0, w.pinned ? "pinned" : "pageable");
        log += l;
    }
    egRows().clear();
}

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

// the per-read pass of the dedup (after the sort and k_groups): per sorted position its read is order[k], head[k] marks the first of a group
__global__ void k_dd_flags(const uint32_t* order, const uint32_t* head, const uint8_t* hasN, u64 n, uint8_t* repFlag, uint8_t* nFlag) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const bool h = head[k] != 0, w = hasN[order[k]] != 0;
    repFlag[k] = h && !w; nFlag[k] = h && w;
}
__global__ void k_dd_repofgroup(const uint32_t* order, const uint32_t* head, const uint32_t* g, const uint8_t* hasN, u64 n, uint32_t* repOfGroup) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n || !head[k]) return;
    repOfGroup[g[k] - 1] = hasN[order[k]] ? 0xFFFFFFFFu : order[k];
}
__global__ void k_dd_readrep(const uint32_t* order, const uint32_t* g, const uint32_t* repOfGroup, u64 n, uint32_t* readRep) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    readRep[order[k]] = repOfGroup[g[k] - 1];
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
__global__ void k_diag(u64 n, const Work<C>* work, u64* steps, u64* nw, unsigned long long* clk) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    steps[k] = work[k].nSteps; nw[k] = work[k].nW;
    for (int i = 0; i < 8; i++) atomicAdd(&clk[i], (unsigned long long) work[k].clk[i]);
}
template <class C>
__global__ void k_p3(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* base,
                     const uint32_t* bits, uint8_t* bstatus, uint8_t* status, uint8_t* out, uint16_t* outLen, const uint32_t* perm) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    if (perm) k = perm[k];   // work bucketing: thread t finishes read perm[t] of the batch (reads with the same amount of work share a warp); results land at the read's own slot
    Work<C>& w = work[k];
    w.stepBudget = budget;
    u64 o = b0 + k;
    Status s = (Status)bstatus[k];
    Blob bl{out + o * C::NBLOB, 0, C::NBLOB, false};
    if (s == Status::Ok && w.stitched && !assignHits(P, g, w, bits, base[k])) s = Status::Overflow;
    if (s == Status::Ok) s = mapFinish(P, g, w, rbuf[k], bl);
    status[o] = (uint8_t)s; outLen[o] = (uint16_t)bl.n;
}

struct ToU64 { __host__ __device__ u64 operator()(uint16_t x) const { return (u64)x; } };
struct Lane { std::vector<uint8_t> out, status; std::vector<uint16_t> len; std::vector<u64> off; };   // out packed: read k at off[k], len[k] bytes

// packs the lane's fixed-size result slots (slot k at k * slot) into consecutive bytes at off[k]
__global__ void k_pack(const uint8_t* slots, u64 slot, const uint16_t* len, const u64* off, u64 n, uint8_t* packed) {
    u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    for (u64 t = 0; t < len[k]; t++) packed[off[k] + t] = slots[k * slot + t];
}

static double g_devPeakGB = 0;   // largest device memory in use seen inside the lanes (GB)
static void devNote() { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); g_devPeakGB = std::max(g_devPeakGB, (double) (t - f) / (1u << 30)); }

template <class C>
bool runLane(const Params& P, const GenomeView& D, const char* dSeq, const u64* dOff, const std::vector<uint32_t>& idx, u64 NB, u64 budget,
             Lane& L, std::string& err) {
    u64 n = idx.size();
    L.status.assign(n, (uint8_t)Status::Overflow); L.len.assign(n, 0); L.off.assign(n + 1, 0);
    if (n == 0) return true;
    NB = std::min<u64>(NB, n);
    uint32_t* dIdx; uint8_t *dOut, *dSt, *dBst; uint16_t* dLen; Work<C>* dWork; ReadBuf<C::LREAD>* dRbuf; u64 *dNent, *dBase; uint32_t* dBits;
    EG_CK(cudaMalloc(&dIdx, n * 4)); EG_CK(cudaMemcpy(dIdx, idx.data(), n * 4, cudaMemcpyHostToDevice));
    EG_CK(cudaMalloc(&dOut, n * (u64)C::NBLOB)); EG_CK(cudaMalloc(&dSt, n)); EG_CK(cudaMalloc(&dLen, n * 2));
    EG_CK(cudaMalloc(&dWork, NB * sizeof(Work<C>))); EG_CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>)));
    EG_CK(cudaMalloc(&dRbuf, NB * sizeof(ReadBuf<C::LREAD>))); EG_CK(cudaMalloc(&dBst, NB));
    EG_CK(cudaMalloc(&dNent, NB * 8)); EG_CK(cudaMalloc(&dBase, NB * 8));
    u64 bitsCap = 1 << 20; EG_CK(cudaMalloc(&dBits, bitsCap * 4));
    // work bucketing (SINGLET_ENGINE_GPU_BUCKET=1 ascending, 2 descending; off by default): stage 3 (the stitch) runs the batch's reads in the order of their stage-1 hit counts, so that
    // the reads that stop after the seed stage (no hits) fill whole warps and the others share a warp with reads of similar work
    const char* bk = std::getenv("SINGLET_ENGINE_GPU_BUCKET");
    const bool bucket = bk != nullptr && (bk[0] == '1' || bk[0] == '2');   // off by default: slower in the S8 measurements (k_p3 8.7 s -> 10.9 s descending on pbmc1k_v3)
    const bool bucketDesc = bk != nullptr && bk[0] == '2';   // 2: the heaviest reads first (no tail of heavy warps at the end of the grid); default: ascending
    uint32_t* dPerm = nullptr; u64* dKey = nullptr;
    if (bucket) { EG_CK(cudaMalloc(&dPerm, NB * 4)); EG_CK(cudaMalloc(&dKey, NB * 8)); }
    devNote();
    for (u64 b0 = 0; b0 < n; b0 += NB) {
        u64 nb = std::min(NB, n - b0);
        int T = 128; unsigned G = (unsigned)((nb + T - 1) / T);
        EG_KT("p1", k_p1<C><<<G, T>>>(P, D, dSeq, dOff, dIdx, b0, nb, dWork, dRbuf, dBst, dNent));
        thrust::exclusive_scan(thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dBase));
        u64 lastB, lastN; EG_CK(cudaMemcpy(&lastB, dBase + nb - 1, 8, cudaMemcpyDeviceToHost)); EG_CK(cudaMemcpy(&lastN, dNent + nb - 1, 8, cudaMemcpyDeviceToHost));
        u64 words = (lastB + lastN) / 32 + 2;
        if (words > bitsCap) { cudaFree(dBits); bitsCap = words * 2; EG_CK(cudaMalloc(&dBits, bitsCap * 4)); }
        EG_CK(cudaMemset(dBits, 0, words * 4));
        EG_KT("p2", k_p2<C><<<(unsigned)((nb * 32 + 255) / 256), 256>>>(P, D, nb, dWork, dNent, dBase, dBits));
        if (bucket) {
            EG_KT("bucket", thrust::copy(thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dKey));
                            thrust::sequence(thrust::device_ptr<uint32_t>(dPerm), thrust::device_ptr<uint32_t>(dPerm + nb));
                            if (bucketDesc) thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm), thrust::greater<u64>());
                            else thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm)));
        }
        EG_KT("p3", k_p3<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dSt, dOut, dLen, bucket ? dPerm : nullptr));
        EG_CK(cudaGetLastError());
        if (egTraceOn() && !bucket) {   // warp-balance diagnostic (reads run in batch order): cost of the extra copies is outside the kernel timings
            u64 *dSteps, *dNW; unsigned long long* dClk; EG_CK(cudaMalloc(&dSteps, nb * 8)); EG_CK(cudaMalloc(&dNW, nb * 8)); EG_CK(cudaMalloc(&dClk, 64)); EG_CK(cudaMemset(dClk, 0, 64));
            k_diag<C><<<G, T>>>(nb, dWork, dSteps, dNW, dClk);
            { unsigned long long hc[8]; EG_CK(cudaMemcpy(hc, dClk, 64, cudaMemcpyDeviceToHost)); for (int i = 0; i < 8; i++) egDiag().clk[i] += (double) hc[i]; }
            cudaFree(dClk);
            std::vector<u64> hs(nb), hn(nb), hw(nb);
            EG_CK(cudaMemcpy(hs.data(), dSteps, nb * 8, cudaMemcpyDeviceToHost)); EG_CK(cudaMemcpy(hw.data(), dNW, nb * 8, cudaMemcpyDeviceToHost));
            EG_CK(cudaMemcpy(hn.data(), dNent, nb * 8, cudaMemcpyDeviceToHost));
            egDiagBatch(hs, hn, hw);
            cudaFree(dSteps); cudaFree(dNW);
        }
    }
    EG_CK(cudaDeviceSynchronize());
    {   // copy only the bytes of the results (the slots are 640-4096 bytes, results ~70)
        u64* dPoff; EG_CK(cudaMalloc(&dPoff, (n + 1) * 8)); EG_CK(cudaMemset(dPoff, 0, (n + 1) * 8));
        thrust::transform_inclusive_scan(thrust::device_ptr<uint16_t>(dLen), thrust::device_ptr<uint16_t>(dLen + n), thrust::device_ptr<u64>(dPoff + 1),
                                         ToU64(), thrust::plus<u64>());
        u64 total; EG_CK(cudaMemcpy(&total, dPoff + n, 8, cudaMemcpyDeviceToHost));
        uint8_t* dPacked; EG_CK(cudaMalloc(&dPacked, total + 1));
        devNote();
        EG_KT("pack", k_pack<<<(unsigned)((n + 255) / 256), 256>>>(dOut, C::NBLOB, dLen, dPoff, n, dPacked));
        EG_CK(cudaGetLastError());
        L.out.resize(total);
        EG_CK(cudaMemcpy(L.out.data(), dPacked, total, cudaMemcpyDeviceToHost));
        EG_CK(cudaMemcpy(L.off.data(), dPoff, (n + 1) * 8, cudaMemcpyDeviceToHost));
        cudaFree(dPoff); cudaFree(dPacked);
    }
    EG_CK(cudaMemcpy(L.status.data(), dSt, n, cudaMemcpyDeviceToHost)); EG_CK(cudaMemcpy(L.len.data(), dLen, n * 2, cudaMemcpyDeviceToHost));
    cudaFree(dIdx); cudaFree(dOut); cudaFree(dSt); cudaFree(dLen); cudaFree(dWork); cudaFree(dRbuf); cudaFree(dBst); cudaFree(dNent);
    cudaFree(dBase); cudaFree(dBits);
    if (dPerm) cudaFree(dPerm);
    if (dKey) cudaFree(dKey);
    return true;
}

// runs fn(lo, hi) over [0, n) cut into nt contiguous ranges, one thread each (the host assembly loops: they run on all the job's threads)
template <class F> static void parallelFor(u64 n, int nt, F fn) {
    nt = (int) std::max<u64>(1, std::min<u64>((u64) std::max(1, nt), n / 65536 + 1));
    if (nt == 1) { fn(0, n); return; }
    std::vector<std::thread> th;
    for (int t = 0; t < nt; t++) th.emplace_back([&, t]() { fn(n * t / nt, n * (t + 1) / nt); });
    for (auto& x : th) x.join();
}

// Large host->device copies at the PCIe rate. cudaMemcpy from pageable memory runs through the driver's staging buffers at ~5.7 GB/s (S8 ledger, one thread of
// memcpy); here several threads fill pinned buffers (64 MB x 4) while cudaMemcpyAsync drains them on a stream, so the copy and the filling overlap.
// SINGLET_ENGINE_GPU_FASTUP=0 turns it off. Byte-for-byte the same data reaches the device.
struct PinStage {
    static constexpr int NB = 4;
    static constexpr size_t SZ = 64u << 20;
    char* buf[NB] = {nullptr, nullptr, nullptr, nullptr};
    cudaEvent_t ev[NB] = {nullptr, nullptr, nullptr, nullptr};
    cudaStream_t st = nullptr;
    bool ok = false;
    int nt = 8;
    double secInit = 0;
    bool init(int threads) {
        if (ok) return true;
        const char* off = std::getenv("SINGLET_ENGINE_GPU_FASTUP");
        if (off != nullptr && off[0] == '0') return false;
        const double t0 = now();
        nt = std::max(2, std::min(threads, 16));
        for (int i = 0; i < NB; i++) {
            if (cudaHostAlloc((void**) &buf[i], SZ, cudaHostAllocDefault) != cudaSuccess) { buf[i] = nullptr; release(); return false; }
            if (cudaEventCreateWithFlags(&ev[i], cudaEventDisableTiming) != cudaSuccess) { ev[i] = nullptr; release(); return false; }
        }
        if (cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking) != cudaSuccess) { st = nullptr; release(); return false; }
        secInit = now() - t0;
        ok = true;
        return true;
    }
    void release() {
        ok = false;
        for (int i = 0; i < NB; i++) {
            if (ev[i]) { cudaEventDestroy(ev[i]); ev[i] = nullptr; }
            if (buf[i]) { cudaFreeHost(buf[i]); buf[i] = nullptr; }
        }
        if (st) { cudaStreamDestroy(st); st = nullptr; }
    }
    // d[0..n) = h[0..n); returns after the data is on the device
    bool copy(void* d, const void* h, size_t n) {
        const double t0 = now();
        bool used[NB] = {false, false, false, false};
        size_t off = 0; int k = 0;
        while (off < n) {
            const size_t c = std::min(SZ, n - off);
            if (used[k] && cudaEventSynchronize(ev[k]) != cudaSuccess) return false;
            const char* src = (const char*) h + off; char* dst = buf[k];
            parallelFor(c, nt, [&](u64 lo, u64 hi) { std::memcpy(dst + lo, src + lo, hi - lo); });
            if (cudaMemcpyAsync((char*) d + off, buf[k], c, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
            if (cudaEventRecord(ev[k], st) != cudaSuccess) return false;
            used[k] = true; off += c; k = (k + 1) % NB;
        }
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        if (egTraceOn()) { EgRow& w = egRows()[-1]; w.n++; w.bytes += n; w.sec += now() - t0; w.pinned = true; w.kind = (int) cudaMemcpyHostToDevice; }
        return true;
    }
};
PinStage& pinStage() { static PinStage p; return p; }

template <class T> bool upload(const T* h, u64 n, T*& d, std::string& err, u64 pad = 0) {
    EG_CK(cudaMalloc(&d, (n + pad) * sizeof(T) + 8));
    if (pad) EG_CK(cudaMemset(d, 0, (n + pad) * sizeof(T) + 8));
    if (n) {
        if (n * sizeof(T) >= (32u << 20) && pinStage().ok) {
            if (!pinStage().copy(d, h, n * sizeof(T))) { err = "fast upload: " + std::string(cudaGetErrorString(cudaGetLastError())); return false; }
        } else {
            EG_CK(cudaMemcpy(d, h, n * sizeof(T), cudaMemcpyHostToDevice));
        }
    }
    return true;
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
    // on failure, release everything this process holds on the device (STAR then maps every read on the CPU)
    auto fail = [&](const std::string& e) { out.log += "engine_gpu: " + e + "\n"; pinStage().release(); cudaDeviceReset(); return false; };
    cudaInitDevice(device, cudaDeviceScheduleBlockingSync, 0);   // before the context exists: a waiting host thread sleeps, not spins
    if (cudaSetDevice(device) != cudaSuccess) return fail("cannot select device");
    double t0 = now();
    pinStage().init(hostThreads);
    if (pinStage().ok) { char l[96]; snprintf(l, sizeof l, "engine_gpu: pinned staging buffers ready in %.2f s\n", pinStage().secInit); out.log += l; }
    singlet::NvtxChain nvch; nvch.next("mapDistinct: setup");
    // device memory in use at the phases of the call (logged: what a sample needs when several share a GPU)
    std::string memLog;
    auto memMark = [&](const char* what) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); char b[64]; snprintf(b, sizeof b, "%s %.1f, ", what, (double) (t - f) / (1u << 30)); memLog += b; };
    memMark("start");
    // ---- index (G with 200 bytes of padding on each side, as STAR's G1)
    nvch.next("mapDistinct: index upload");
    const u64 PAD = 200;
    GenomeView D = H;
    char* dG1; u64* dSA; u64* dSAi;
    // SINGLET_ENGINE_GPU_PIN=1: page-lock the host index first, so the copies run at the PCIe rate instead of through the driver's staging buffers
    const bool pinIdx = std::getenv("SINGLET_ENGINE_GPU_PIN") != nullptr;
    const double tIdx0 = now(); double tPin = 0;
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
    D.SA = dSA; D.SAi = dSAi;
    memMark("index");
    u64 *dcs, *dcl, *dcb, *dss, *dse, *dsd, *dsa; uint8_t *dsm, *dsl, *dsr, *dst;
    if (!upload(H.chrStart, H.nChr + 1, dcs, err) || !upload(H.chrLength, H.nChr, dcl, err) || !upload(H.chrBin, H.chrBinN, dcb, err)
        || !upload(H.sjdbStart, H.sjdbN, dss, err) || !upload(H.sjdbEnd, H.sjdbN, dse, err) || !upload(H.sjDstart, H.sjdbN, dsd, err)
        || !upload(H.sjAstart, H.sjdbN, dsa, err) || !upload(H.sjdbMotif, H.sjdbN, dsm, err) || !upload(H.sjdbShiftLeft, H.sjdbN, dsl, err)
        || !upload(H.sjdbShiftRight, H.sjdbN, dsr, err) || !upload(H.sjdbStrand, H.sjdbN, dst, err)) return fail(err);
    D.chrStart = dcs; D.chrLength = dcl; D.chrBin = dcb; D.sjdbStart = dss; D.sjdbEnd = dse; D.sjDstart = dsd; D.sjAstart = dsa;
    D.sjdbMotif = dsm; D.sjdbShiftLeft = dsl; D.sjdbShiftRight = dsr; D.sjdbStrand = dst;
    // ---- reads
    nvch.next("mapDistinct: read upload + clip");
    u64 n = in.off.size() - 1;
    out.nReads = n;
    char* dSeq; u64* dOff;
    if (!upload(in.seq.data(), in.seq.size(), dSeq, err) || !upload((const u64*)in.off.data(), n + 1, dOff, err)) return fail(err);
    // clipping: the reads become their clipped sequences (device copy compacted; on the host each read's span is (cs, length))
    std::vector<u64> hcs, hoffC;
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
        EG_KT("clip", k_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, c, dCs, dCl));
        cudaMemset(dOffC, 0, 8);
        thrust::inclusive_scan(thrust::device_ptr<u64>(dCl), thrust::device_ptr<u64>(dCl + n), thrust::device_ptr<u64>(dOffC + 1));
        u64 tot = 0;
        if (cudaMemcpy(&tot, dOffC + n, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("clip copy");
        if (cudaMalloc(&dSeqC, tot + 8)) return fail("out of device memory (clipped reads)");
        EG_KT("gather_clip", k_gather_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dCs, dOffC, n, dSeqC));
        hcs.resize(n); hoffC.resize(n + 1);
        if (cudaMemcpy(hcs.data(), dCs, n * 8, cudaMemcpyDeviceToHost) != cudaSuccess
            || cudaMemcpy(hoffC.data(), dOffC, (n + 1) * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("clip copy");
        cudaFree(dSeq); cudaFree(dOff); cudaFree(dCs); cudaFree(dCl); if (dAd) cudaFree(dAd); if (dP5) cudaFree(dP5); if (dP3) cudaFree(dP3);
        dSeq = dSeqC; dOff = dOffC;
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
        char line[200];
        snprintf(line, sizeof line, "engine_gpu: clipping: %llu of %llu reads clipped, %.2f s\n", (unsigned long long)nClipped, (unsigned long long)n, now() - tc);
        out.log += line;
    }
    memMark("reads+clip");
    // a read's (clipped) sequence on the host
    auto rseq = [&](u64 i) { return hcs.empty() ? in.seq.data() + in.off[i] : in.seq.data() + hcs[i]; };
    auto rlen = [&](u64 i) { return hcs.empty() ? in.off[i + 1] - in.off[i] : hoffC[i + 1] - hoffC[i]; };
    cudaDeviceSynchronize();
    out.secUpload = now() - t0;
    // ---- dedup
    nvch.next("mapDistinct: dedup");
    t0 = now();
    u64 *dH1, *dH2; uint8_t* dN; uint32_t *dOrder, *dHead; unsigned long long* dColl;
    if (cudaMalloc(&dH1, n * 8) || cudaMalloc(&dH2, n * 8) || cudaMalloc(&dN, n) || cudaMalloc(&dOrder, n * 4) || cudaMalloc(&dHead, n * 4)
        || cudaMalloc(&dColl, 8)) return fail("out of device memory (dedup)");
    EG_KT("hash", k_hash<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, dH1, dH2, dN));
    thrust::sequence(thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n));
    auto keys = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<u64>(dH1), thrust::device_ptr<u64>(dH2)));
    thrust::sort_by_key(keys, keys + n, thrust::device_ptr<uint32_t>(dOrder));
    cudaMemset(dColl, 0, 8);
    EG_KT("groups", k_groups<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dH1, dH2, dOrder, n, dHead, dColl));
    unsigned long long coll = 0;
    if (cudaMemcpy(&coll, dColl, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
    if (coll) return fail("hash collision between different sequences (" + std::to_string(coll) + "): GPU path not used");
    // The per-read pass of the dedup on the device (the host loops over every read cost 3.8 s of 25 s of the lock on pbmc1k_v3 and 17 s of 96 s on a 263 M-read
    // sample, against 0.04 s / 0.16 s of kernels): the group heads, the sequences without N, the representative read of every group and of every read,
    // and the order the distinct sequences are mapped in (first 32 bases) are computed on the GPU; only rep, readRep and the two counts come back.
    // SINGLET_ENGINE_GPU_HOSTDEDUP=1 runs the former host loops (the identity reference).
    const uint32_t NONE = 0xFFFFFFFFu;
    std::vector<uint32_t> rep;   // one read per distinct sequence without N, in mapping order
    std::vector<uint32_t> readRep;   // per read: the representative read of its distinct sequence (NONE for sequences with N, which STAR maps itself)
    if (std::getenv("SINGLET_ENGINE_GPU_HOSTDEDUP") != nullptr) {
        std::vector<uint32_t> order(n), head(n); std::vector<uint8_t> hasN(n);
        cudaMemcpy(order.data(), dOrder, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(head.data(), dHead, n * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(hasN.data(), dN, n, cudaMemcpyDeviceToHost);
        for (u64 k = 0; k < n; k++) if (head[k]) { out.nDistinct++; if (hasN[order[k]]) out.nWithN++; else rep.push_back(order[k]); }
        readRep.assign(n, NONE);
        { uint32_t cur = NONE; for (u64 k = 0; k < n; k++) { if (head[k]) cur = hasN[order[k]] ? NONE : order[k]; readRep[order[k]] = cur; } }
        if (!rep.empty()) {
            uint32_t* dRep; u64* dKey;
            if (cudaMalloc(&dRep, rep.size() * 4) || cudaMalloc(&dKey, rep.size() * 8)) return fail("out of device memory (order)");
            cudaMemcpy(dRep, rep.data(), rep.size() * 4, cudaMemcpyHostToDevice);
            EG_KT("prefix_key", k_prefix_key<<<(unsigned)((rep.size() + 255) / 256), 256>>>(dSeq, dOff, dRep, rep.size(), dKey));
            thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + rep.size()), thrust::device_ptr<uint32_t>(dRep));
            if (cudaMemcpy(rep.data(), dRep, rep.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
            cudaFree(dRep); cudaFree(dKey);
        }
    } else if (n > 0) {
        uint8_t *dFlag; uint32_t *dG, *dRepOfGroup, *dReadRep, *dRep; u64* dKey;
        if (cudaMalloc(&dFlag, 2 * n) || cudaMalloc(&dG, n * 4) || cudaMalloc(&dReadRep, n * 4) || cudaMalloc(&dRep, n * 4)) return fail("out of device memory (dedup pass)");
        uint8_t *dRepFlag = dFlag, *dNFlag = dFlag + n;
        const unsigned nb = (unsigned)((n + 255) / 256);
        EG_KT("dd_flags", k_dd_flags<<<nb, 256>>>(dOrder, dHead, dN, n, dRepFlag, dNFlag));
        thrust::inclusive_scan(thrust::device_ptr<uint32_t>(dHead), thrust::device_ptr<uint32_t>(dHead + n), thrust::device_ptr<uint32_t>(dG));
        uint32_t nGroups = 0;
        if (cudaMemcpy(&nGroups, dG + n - 1, 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
        if (cudaMalloc(&dRepOfGroup, (u64) nGroups * 4 + 4)) return fail("out of device memory (dedup groups)");
        EG_KT("dd_groups", k_dd_repofgroup<<<nb, 256>>>(dOrder, dHead, dG, dN, n, dRepOfGroup));
        EG_KT("dd_readrep", k_dd_readrep<<<nb, 256>>>(dOrder, dG, dRepOfGroup, n, dReadRep));
        const u64 nRep = (u64) thrust::reduce(thrust::device_ptr<uint8_t>(dRepFlag), thrust::device_ptr<uint8_t>(dRepFlag + n), (u64) 0);
        out.nWithN = (u64) thrust::reduce(thrust::device_ptr<uint8_t>(dNFlag), thrust::device_ptr<uint8_t>(dNFlag + n), (u64) 0);
        out.nDistinct = (u64) nGroups;
        thrust::copy_if(thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n), thrust::device_ptr<uint8_t>(dRepFlag),
                        thrust::device_ptr<uint32_t>(dRep), thrust::identity<uint8_t>());
        readRep.resize(n);
        if (cudaMemcpy(readRep.data(), dReadRep, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("readRep copy");
        rep.resize(nRep);
        if (nRep > 0) {
            if (cudaMalloc(&dKey, nRep * 8)) return fail("out of device memory (order)");
            EG_KT("prefix_key", k_prefix_key<<<(unsigned)((nRep + 255) / 256), 256>>>(dSeq, dOff, dRep, nRep, dKey));
            thrust::stable_sort_by_key(thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nRep), thrust::device_ptr<uint32_t>(dRep));
            if (cudaMemcpy(rep.data(), dRep, nRep * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
            cudaFree(dKey);
        }
        cudaFree(dFlag); cudaFree(dG); cudaFree(dRepOfGroup); cudaFree(dReadRep); cudaFree(dRep);
    } else {
        readRep.assign(n, NONE);
    }
    cudaFree(dH1); cudaFree(dH2); cudaFree(dN); cudaFree(dOrder); cudaFree(dHead); cudaFree(dColl);
    out.secDedup = now() - t0;
    memMark("dedup");
    std::vector<uint32_t> recOfPos(rep.size(), NONE);   // rep position -> output record (NONE: not mapped, STAR maps it)
    // ---- map in chunks of distinct sequences (device memory for the lanes' outputs is per chunk)
    nvch.end();
    u64 nGpuMapped = 0;
    // a chunk of distinct sequences in two halves: the device lanes (gpuPart, under the caller's device lock) and the host lane and records
    // (hostPart, after in.release(): they hold the GPU for nothing)
    struct ChunkRes { std::vector<uint32_t> repc; Lane A, M, B; std::vector<uint8_t> lane; std::vector<uint32_t> lpos, repH, posH; u64 c0 = 0; };
    std::vector<ChunkRes> chunks;
    auto gpuPart = [&](const std::vector<uint32_t>& repc, u64 c0) -> bool {
        SINGLET_NVTX("mapDistinct: lanes of one chunk (GPU)");
        // ---- lanes
        double tc0 = now();
        Lane A, M, B;
        memMark("before-laneA");
        // reads in flight in lane A: its Work + ReadBuf are ~150 KB each (40 GB at 262144); SINGLET_ENGINE_GPU_NB trades device memory for occupancy
        u64 nbA = 262144;
        if (const char* ne = std::getenv("SINGLET_ENGINE_GPU_NB")) nbA = std::max<u64>(4096, std::strtoull(ne, nullptr, 10));
        if (!runLane<LaneA>(P, D, dSeq, dOff, repc, nbA, 1024, A, err)) return false;
        std::vector<uint32_t> repM, posM;
        for (u64 k = 0; k < repc.size(); k++) if (A.status[k] != (uint8_t)Status::Ok) { repM.push_back(repc[k]); posM.push_back((uint32_t)k); }
        if (!runLane<LaneM>(P, D, dSeq, dOff, repM, 262144, 2048, M, err)) return false;
        std::vector<uint32_t> repB, posB;
        for (u64 k = 0; k < repM.size(); k++) if (M.status[k] != (uint8_t)Status::Ok) { repB.push_back(repM[k]); posB.push_back(posM[k]); }
        u64 nbB = 65536;   // the heavy lane's reads in flight: its Work is the largest device allocation of the prefill (SINGLET_ENGINE_GPU_NBB)
        if (const char* nbe = std::getenv("SINGLET_ENGINE_GPU_NBB")) nbB = std::max<u64>(1024, std::strtoull(nbe, nullptr, 10));
        if (!runLane<LaneB>(P, D, dSeq, dOff, repB, nbB, 2048, B, err)) return false;
        out.secMapGpu += now() - tc0;
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
          cr.repH = std::move(repH); cr.posH = std::move(posH); cr.c0 = c0; chunks.push_back(std::move(cr)); }
        return true;
    };
    auto hostPart = [&](ChunkRes& cr) -> bool {
        SINGLET_NVTX("mapDistinct: host lane + records of one chunk");
        const std::vector<uint32_t>& repc = cr.repc; const u64 c0 = cr.c0; Lane &A = cr.A, &M = cr.M, &B = cr.B;
        const std::vector<uint8_t>& lane = cr.lane; const std::vector<uint32_t>& lpos = cr.lpos; const std::vector<uint32_t>& repH = cr.repH;
        // ---- host lane (STAR's limits)
        double th0 = now();
        std::vector<std::vector<uint8_t>> hout(repH.size()); std::vector<uint8_t> hst(repH.size(), 0);
        {
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
        out.nHostLane += repH.size();
        // ---- records: per sequence its kept flag and lengths, prefix sums, then the copies in parallel (the order is the serial loop's)
        const u64 nK = repc.size();
        std::vector<uint8_t> okv(nK, 1); std::vector<u64> sOff(nK + 1, 0), bOff(nK + 1, 0); std::vector<uint32_t> rank(nK + 1, 0);
        auto laneOf = [&](u64 k, const uint8_t*& p, u64& len, bool& ok) {
            uint32_t q = cr.lpos[k]; ok = true;
            switch (lane[k]) {
                case 0: p = A.out.data() + A.off[q]; len = A.len[q]; break;
                case 1: p = M.out.data() + M.off[q]; len = M.len[q]; break;
                case 2: p = B.out.data() + B.off[q]; len = B.len[q]; break;
                default: p = hout[q].data(); len = hout[q].size(); ok = hst[q] == (uint8_t)Status::Ok; break;
            }
        };
        parallelFor(nK, hostThreads, [&](u64 lo, u64 hi) {
            for (u64 k = lo; k < hi; k++) {
                const uint8_t* p; u64 len; bool ok;
                laneOf(k, p, len, ok);
                okv[k] = ok; sOff[k + 1] = ok ? rlen(repc[k]) : 0; bOff[k + 1] = ok ? len : 0; rank[k + 1] = ok ? 1 : 0;
            }
        });
        for (u64 k = 0; k < nK; k++) { sOff[k + 1] += sOff[k]; bOff[k + 1] += bOff[k]; rank[k + 1] += rank[k]; if (!okv[k]) out.nFailed++; }   // not cached: STAR maps it
        const u64 s0 = out.seq.size(), b0 = out.blob.size(), r0 = out.seqOff.size() - 1;   // out.seqOff[r0] == s0
        out.seq.resize(s0 + sOff[nK]); out.blob.resize(b0 + bOff[nK]);
        out.seqOff.resize(r0 + 1 + rank[nK]); out.blobOff.resize(r0 + 1 + rank[nK]);
        parallelFor(nK, hostThreads, [&](u64 lo, u64 hi) {
            for (u64 k = lo; k < hi; k++) {
                if (!okv[k]) continue;
                const uint8_t* p; u64 len; bool ok;
                laneOf(k, p, len, ok);
                const u64 i = repc[k], L = rlen(i), rec = r0 + rank[k];
                recOfPos[c0 + k] = (uint32_t)rec;
                std::memcpy(out.seq.data() + s0 + sOff[k], rseq(i), L); out.seqOff[rec + 1] = s0 + sOff[k] + L;
                std::memcpy(out.blob.data() + b0 + bOff[k], p, len); out.blobOff[rec + 1] = b0 + bOff[k] + len;
            }
        });
        nGpuMapped += repc.size() - repH.size();
        return true;
    };
    // sequences per chunk: the lanes' device buffers scale with it (SINGLET_ENGINE_GPU_CHUNK, default 16M); samples sharing a GPU (SINGLET_ENGINE_GPU_SLOTS) want it smaller
    u64 CHUNK = 16u << 20;
    if (const char* ce = std::getenv("SINGLET_ENGINE_GPU_CHUNK")) CHUNK = std::max<u64>(1u << 18, std::strtoull(ce, nullptr, 10));
    for (u64 c0 = 0; c0 < rep.size(); c0 += CHUNK) {
        std::vector<uint32_t> repc(rep.begin() + c0, rep.begin() + std::min<u64>(rep.size(), c0 + CHUNK));
        if (!gpuPart(repc, c0)) return fail(err);
    }
    // the last device work is done: free the device buffers and let the next sample have the GPU while this one assembles its records
    nvch.end();
    { SINGLET_NVTX("mapDistinct: final sync + free"); cudaDeviceSynchronize(); }
    cudaFree(dG1); cudaFree(dSA); cudaFree(dSAi); cudaFree(dSeq); cudaFree(dOff);
    cudaFree(dcs); cudaFree(dcl); cudaFree(dcb); cudaFree(dss); cudaFree(dse); cudaFree(dsd); cudaFree(dsa);
    cudaFree(dsm); cudaFree(dsl); cudaFree(dsr); cudaFree(dst);
    if (in.release) in.release();
    for (auto& cr : chunks) { if (!hostPart(cr)) return fail(err); std::vector<uint32_t>().swap(cr.repc); }
    {   // per read: its record (sequence-cache result) or NONE
        std::vector<uint32_t> posOf(n, NONE);
        parallelFor(rep.size(), hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) posOf[rep[k]] = (uint32_t)k; });
        out.readRecord.assign(n, NONE);
        parallelFor(n, hostThreads, [&](u64 lo, u64 hi) { for (u64 i = lo; i < hi; i++) { const uint32_t r = readRep[i]; if (r != NONE) out.readRecord[i] = recOfPos[posOf[r]]; } });
    }
    out.log += "engine_gpu: device memory (GB in use): " + memLog + "lanes peak " + std::to_string(g_devPeakGB) + "\n";
    char line[512];
    snprintf(line, sizeof line, "engine_gpu: %llu reads, %llu distinct (%llu with N), mapped on the GPU %llu, host lane %llu (%.2f s), not mapped %llu; "
             "upload %.2f s, dedup %.2f s, GPU mapping %.2f s\n", (unsigned long long)out.nReads, (unsigned long long)out.nDistinct,
             (unsigned long long)out.nWithN, (unsigned long long)nGpuMapped, (unsigned long long)out.nHostLane, out.secHostLane,
             (unsigned long long)out.nFailed, out.secUpload, out.secDedup, out.secMapGpu);
    out.log += line;
    pinStage().release();
    egReport(out.log);
    return true;
}

} // namespace singlet_engine_gpu
