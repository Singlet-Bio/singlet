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
#include "singlet/qmask_words.h"   // P3-ENG-14
#include "singlet/nvtx.h"
#include "singlet/thread_name.h"
#include "hostloop_gpu.h"  // E10: per-read entries and the per-read products of STAR's loop (SINGLET_ENGINE_GPU_E2E >= 2)
#include "solo_gpu.h"      // E11: STARsolo's inputs kept on the device for soloRun after STAR's loop (SINGLET_ENGINE_GPU_E2E=5)
#include <cuda_runtime.h>
#include <cuda.h>   // types of the driver's virtual memory functions (shared_index.cuh); the functions come from cudaGetDriverEntryPoint, no libcuda link
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
#include <thrust/system/cuda/execution_policy.h>
#include <algorithm>
#include <new>
#include <unordered_map>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>
#include <condition_variable>
#include <dirent.h>
#include <glob.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <cerrno>
extern char** environ;
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include "singlet/gpu_ledger.h"   // last: it wraps cudaMemcpy / cudaMemcpyAsync (SINGLET_ENGINE_GPU_TRACE=1)

using namespace m5;

namespace {

#define EG_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

// P3-ENG-14: bytes of one read's low-quality mask (16 up to 128 bases, the earlier layout)
inline u64 qmBytes() { return 8ull * singlet_qmask::words(); }

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

//                 LREAD NSPLIT    NP     NW  NWA      NWB NEX NWTR NKEEP    NBLOB
typedef Caps<      160,     4,   32,    16,  16,     256,  8,   8,    8,     640> LaneA;
typedef Caps<      160,     4,   32,    64,   8,    1024,  8,   8,   32,    2048> LaneM;
typedef Caps<      160,     8,  128,   128,  50,    4096, 20,  32,   64,    4096> LaneB;
typedef Caps<      650,    50, 1000, 10000,  50, 1 << 20, 20, 100, 4096, 1 << 20> Heavy;
// gpu/bold-kern2 (on by default; SINGLET_BOLD_WSHRINK=0 turns it off): lane B with the windows' aligns in a per-batch pool (Work::waP) and smaller NWA / NKEEP.
// A read that needs more than these capacities ends Overflow and goes to the host lane (STAR's limits), as lane B's own overflows do.
// Capacities raised from NW 128, NWA 32, NKEEP 32 (SINGLET_ENGINE_GPU_OVSTAT=1 on pbmc1k_v3: of lane B's 177,686 overflows at STEPS_B 8192,
// 70,154 were keep, 67,260 wa and 39,995 win; nuclei2k_v2 mostly keep). A read over a capacity is mapped by the host pool, so results do not change.
typedef Caps<      160,     8,  128,   192,  50,    4096, 20,  32,   64,    4096, true> LaneBc;

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
// ---- E20 (docs/gpu-engine/fable/E20-lockclip.md): STAR's CellRanger4 clip of mate 1 on the device, per read, as E13's cr4Clip runs it on the host
// (SeqCacheGpuPrefill.cpp: ClipMate::clipChunk's opal alignment of the 5' adapter, then ClipMate::clip of the 5' mate, type 10, and of the 3'
// mate, type 11, on the same buffer and length). A read's clip depends on its sequence alone, so every read is clipped (no grouping).
// opal (src/star/opal/opal.cpp, searchDatabase_ with OPAL_MODE_OV, OPAL_SEARCH_SCORE_END), as ClipCR4 calls it: the target is the read's
// first 91 bases (ClipCR4::readLen; numeric 0-3, anything else 4), padded with 4 when shorter; scores ClipCR4::scoreMatrix (match 1, N/N 0,
// otherwise -2), gap open and extension 2. Each database sequence runs in its own SIMD channel, so a read's result does not depend on the others
// of its batch. opal's 8-bit pass saturates, but with a query of at most 40 bases every H is >= -2 * 40 and every E, F >= -82 (the first row
// and column start at 0), and H <= 40: nothing saturates, its overflow test never fires, and int arithmetic gives the same values. Per column
// (target position) and row (query position), as the SIMD loop: E = max(H_left - 2, E_left - 2), F = max(H_up - 2, F_up - 2) (F_up starts
// at -126, H_up and H_upleft at 0 in every column; H_left 0 and E_left -126 before the first column), H = max(F, E, H_upleft + s); the best
// score is max(the last column's max H, the max of the last row); the end column is the last one (91 - 1) when the last column's max H
// exceeds the last row's max over the columns before it, else the first column at which the last row's running max rose to its maximum.
struct Cr4Dev { uint8_t q[40]; int nq; uint32_t n5, after5, n3, after3; };
__device__ __forceinline__ uint32_t cr4PolyTail3p(const char* seq, uint32_t L) {   // ClipCR4::polyTail3p
    if (L < 20) return 0;
    uint32_t ib1 = L - 1; int score = 0, score1 = 0;
    for (uint32_t ib = 1; ib <= L; ib++) {
        if (seq[L - ib] == 0) { score += 1; if (score * 10 >= (int) ib * 7) { ib1 = ib; score1 = score; } }
        else { score -= 2; if ((int) ib - score > 27) break; }
    }
    if (score1 < 20) ib1 = 0;
    return ib1;
}
template <int NQ>   // NQ > 0: the query length at compile time (the 10x TSO: 30), its rows in registers; 0: c.nq (<= 40)
__global__ void __launch_bounds__(128) k_cr4(const char* seq, const u64* off, u64 n, Cr4Dev c, uint16_t* p5, uint16_t* p3, uint8_t* info) {
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 s = off[i], L = off[i + 1] - s;
    constexpr int MQ = NQ > 0 ? NQ : 40;
    const int nq = NQ > 0 ? NQ : c.nq;
    int pH[MQ], pE[MQ];
#pragma unroll
    for (int r = 0; r < MQ; r++) { pH[r] = 0; pE[r] = -126; }
    int maxLast = -128, bestCol = 0, prevMaxAtLast = -128, maxHLast = -128;
    for (int col = 0; col < 91; col++) {
        int t = 4;
        if ((u64) col < L) { const int x = (uint8_t) seq[s + col] & 7; t = x < 4 ? x : 4; }
        int uF = -126, uH = 0, ulH = 0, maxH = -128, H = 0;
#pragma unroll
        for (int r = 0; r < MQ; r++) {
            if (NQ == 0 && r >= nq) break;
            const int q = c.q[r];
            const int E = max(pH[r] - 2, pE[r] - 2);
            const int F = max(uH - 2, uF - 2);
            const int sc = q == t ? (q == 4 ? 0 : 1) : -2;
            H = max(max(F, E), ulH + sc);
            maxH = max(maxH, H);
            uF = F; uH = H; ulH = pH[r];
            pE[r] = E; pH[r] = H;
        }
        const int prevMax = maxLast;
        maxLast = max(maxLast, H);   // H: the last row's
        if (maxLast > prevMax) bestCol = col;
        if (col == 90) { prevMaxAtLast = prevMax; maxHLast = maxH; }
    }
    const int score = max(maxHLast, maxLast);
    const int endT = maxHLast > prevMaxAtLast ? 90 : bestCol;
    const int Lm = endT + 1;
    const bool L0 = score < 20 || (score == 20 && Lm > 26) || (score == 21 && Lm > 30);
    const uint32_t inf = L0 ? 0u : (uint32_t) Lm;
    info[i] = (uint8_t) inf;
    uint32_t c5 = 0, c3 = 0;
    if (L <= 650) {   // DEF_readSeqLengthMax: cr4Clip clips only the reads STAR can load
        // ClipMate::clip, type 10 (5'): N bases (no move for this type), the adapter (clippedInfo; the buffer moves), NafterAd (no move)
        uint32_t Lr = (uint32_t) L, sh = 0, cn = 0, Lold = Lr;
        if (c.n5 > 0) { if (Lr > c.n5) { Lr -= c.n5; cn += c.n5; } else { Lr = 0; cn = Lold; } }
        { const uint32_t a = min((uint32_t) (char) inf, Lr); sh += a; Lr -= a; cn += a; }
        if (c.after5 > 0) { if (Lr > c.after5) { Lr -= c.after5; cn += c.after5; } else { Lr = 0; cn = Lold; } }
        c5 = cn;
        // type 11 (3'): N bases, the polyA tail of the (moved) buffer's first Lr bases, NafterAd
        cn = 0; Lold = Lr;
        if (c.n3 > 0) { if (Lr > c.n3) { Lr -= c.n3; cn += c.n3; } else { Lr = 0; cn = Lold; } }
        { const uint32_t a = cr4PolyTail3p(seq + s + sh, Lr); Lr -= a; cn += a; }
        if (c.after3 > 0) { if (Lr > c.after3) { Lr -= c.after3; cn += c.after3; } else { Lr = 0; cn = Lold; } }
        c3 = cn;
    }
    p5[i] = (uint16_t) min(c5, 65535u); p3[i] = (uint16_t) min(c3, 65535u);
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
// P3-MEM-03 (the reads phase by read chunks): k_hash of a read's clipped span (start cs[i], cl[i] bases) without the compacted copy, or of the
// whole read (cs null); the same bytes and length, so the same hashes as k_hash on the clipped copy
__global__ void k_hash_span(const char* seq, const u64* off, const u64* cs, const u64* cl, u64 n, u64* h1, u64* h2, uint8_t* hasN) {
    u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const u64 s = cs ? cs[i] : off[i], L = cs ? cl[i] : off[i + 1] - off[i];
    u64 a = 0x9E3779B97F4A7C15ULL ^ L, b = 0xC2B2AE3D27D4EB4FULL + L;
    uint8_t N = 0;
    for (u64 k = s; k < s + L; k++) {
        u64 c = (u64)(uint8_t)seq[k];
        N |= (c > 3);
        a = (a ^ c) * 0x100000001B3ULL; a ^= a >> 29;
        b = (b + c + 1) * 0xFF51AFD7ED558CCDULL; b ^= b >> 31;
    }
    h1[i] = a; h2[i] = b; hasN[i] = N | (L == 0);
}
// P3-MEM-03: per read (local index in its hash partition) its group's first read, from the sort's order and the max-scanned heads
__global__ void k_group_head(const uint32_t* order, const uint32_t* hp, u64 n, uint32_t* gh) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    gh[order[k]] = order[hp[k]];
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
// COMPACT lanes (gpu/bold-kern2): per read the aligns its windows need (planWA, once per batch); then rounds: the reads whose aligns fit what is
// left of the pool take their rows from it in index order (an exclusive scan of the pending reads' needs) and run phase 3; the rest wait for the
// next round. A read's result does not depend on where its rows are, so the order of the rounds changes nothing.
template <class C>
__global__ void k_wplan(Params P, GenomeView g, u64 n, Work<C>* work, const u64* base, const uint32_t* bits, const uint8_t* bst, u64* need) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    need[k] = ((Status)bst[k] == Status::Ok && w.stitched) ? planWA<C>(P, g, w, bits, base[k]) : 0;
}
template <class C>
__global__ void k_wkey(u64 n, const u64* need, const uint8_t* pend, u64* key, unsigned long long* cnt) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    key[k] = pend[k] ? need[k] : 0;
    if (pend[k]) atomicAdd(cnt, 1ULL);
}
// lane B's phase 3 thread order (SINGLET_B_SORT): thread t takes read perm[t] (largest expected work first), so the 32 reads of a warp cost about the same
// and the long reads start first. perm == nullptr: thread t takes read t. A read's result does not depend on its thread or on when it runs.
template <class C>
__global__ void k_p3c(Params P, GenomeView g, u64 b0, u64 n, u64 budget, Work<C>* work, ReadBuf<C::LREAD>* rbuf, const u64* base,
                      const uint32_t* bits, uint8_t* bst, uint8_t* out, uint16_t* outLen, uint8_t* pend, const u64* need, const u64* wb,
                      m5::uint* pool, u64 poolAl, const uint32_t* perm) {
    const u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (t >= n) return;
    const u64 k = perm ? perm[t] : t;
    if (!pend[k]) return;
    const u64 nd = need[k];
    if (nd && wb[k] + nd > poolAl) return;   // the next round
    work[k].waP = pool + (nd ? wb[k] * WA_SIZE : 0);
    p3one<C>(P, g, b0, k, budget, work, rbuf, base, bits, bst, out, outLen);
    pend[k] = 0;
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
__global__ void k_cost_need(u64 n, const u64* need, u64* key, uint32_t* perm) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    key[k] = ~need[k];
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
// SINGLET_LANE_PROF: the reads' phase cycle counters (Work::clk, filled by the mapping code) summed over a batch, then cleared for the next one
template <class C>
__global__ void k_clksum(u64 n, Work<C>* work, unsigned long long* acc) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    Work<C>& w = work[k];
    for (int c = 0; c < 8; c++) { if (w.clk[c]) atomicAdd(&acc[c], (unsigned long long)w.clk[c]); w.clk[c] = 0; }
}
// SINGLET_LANE_PROF=2: per read the work features of a COMPACT lane after phase 3 (stitching steps, windows, aligns, SA hits, sum of squared window rows, largest row)
template <class C>
__global__ void k_feat(u64 n, const Work<C>* work, const u64* need, const u64* nEnt, uint32_t* f) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= n) return;
    const Work<C>& w = work[k];
    u64 sq = 0, mxr = 0;
    for (uint i = 0; i < w.nWall && i < (uint)C::NW; i++) { const u64 r = w.waOff[i + 1] - w.waOff[i]; sq += r * r; if (r > mxr) mxr = r; }
    f[6 * k + 0] = (uint32_t)min(w.nSteps, (u64)0xFFFFFFFFull); f[6 * k + 1] = w.nW; f[6 * k + 2] = (uint32_t)min(need[k], (u64)0xFFFFFFFFull);
    f[6 * k + 3] = (uint32_t)min(nEnt[k], (u64)0xFFFFFFFFull); f[6 * k + 4] = (uint32_t)min(sq, (u64)0xFFFFFFFFull); f[6 * k + 5] = (uint32_t)mxr;
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

// ---- E17 (docs/gpu-engine/fable/E17-budget.md): the prefill's device memory budget.
// Every device allocation of mapDistinct and runLane goes through dmal / dfree, and thrust's temporaries through TAlloc (the TP policy), so
// Budget::used is this process's own device memory in the engine's buffers (plus its context), whatever other processes hold on the device;
// the modules mapDistinct calls (E8's buildRecords, E10's products, E11's keep) are checked against the budget by estimate before they run.
// GB in this file: 2^30 bytes (as cudaMemGetInfo's marks and nvidia-smi's MiB).
struct Budget {
    u64 cap = 0;              // bytes; 0: no budget (E12: lane B takes what the free device memory holds)
    u64 used = 0, peak = 0;   // the tally (context included)
    u64 nOver = 0, overMax = 0;
    std::string peakAt, overAt, at = "init";
    std::unordered_map<void*, u64> sz;
    void reset(u64 c, u64 ctx) { cap = c; used = peak = ctx; nOver = overMax = 0; peakAt = overAt = ""; at = "init"; sz.clear(); }
    u64 room() const { return cap > used ? cap - used : 0; }
    void note() {
        if (used > peak) { peak = used; peakAt = at; }
        if (cap && used > cap) { nOver++; if (used - cap > overMax) { overMax = used - cap; overAt = at; } }
    }
};
static Budget g_bud;
static std::vector<void*> g_ipcPtrs;   // shared index: mappings imported from the holder (shared_index.cuh); cudaFree is invalid on them, so dfree leaves them alone
static int g_bSort = 0;   // SINGLET_B_SORT: lane B's phase 3 thread order (0: batch order; 1: aligns of the read, most first)
static int g_waSlice = 160;   // aligns per read of lane B's pool (SINGLET_BOLD_WASLICE)
static inline u64 budRound(u64 b) { return b < (1u << 20) ? (b + 511) & ~511ull : (b + (2u << 20) - 1) & ~((2ull << 20) - 1); }   // the driver's granularity
static cudaError_t dmalv(void** p, u64 b) {
    const cudaError_t e = cudaMalloc(p, b);
    if (e != cudaSuccess) { *p = nullptr; return e; }
    const u64 r = budRound(b);
    g_bud.sz[*p] = r; g_bud.used += r; g_bud.note();
    return e;
}
template <class T> static inline cudaError_t dmal(T** p, u64 b) { return dmalv((void**)p, b); }
static void dfree(void* p) {
    if (p == nullptr) return;
    if (std::find(g_ipcPtrs.begin(), g_ipcPtrs.end(), p) != g_ipcPtrs.end()) return;
    auto it = g_bud.sz.find(p);
    if (it != g_bud.sz.end()) { g_bud.used -= std::min(g_bud.used, it->second); g_bud.sz.erase(it); }
    cudaFree(p);
}
// thrust's temporaries through the tally (thrust::cuda::par(alloc): same stream and synchronisation as thrust::device)
struct TAlloc {
    typedef char value_type;
    char* allocate(std::ptrdiff_t n) { void* p = nullptr; if (dmalv(&p, (u64)n) != cudaSuccess) throw std::bad_alloc(); return (char*)p; }
    void deallocate(char* p, size_t) { dfree(p); }
};
static TAlloc g_talloc;
#define TP thrust::cuda::par(g_talloc)
static inline u64 gb2b(double g) { return (u64)(g * (double)(1ull << 30)); }
static inline double b2gb(u64 b) { return (double)b / (double)(1ull << 30); }
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
static std::atomic<u64>* g_autoBudget = nullptr;   // SINGLET_ENGINE_GPU_STEPS_B=auto: lane B's step budget as the controller sets it, read after every batch
static std::string g_laneLog;   // E12: per lane and chunk, its configuration (logged once by mapDistinct)
static inline cudaError_t d2h(void* h, const void* d, size_t nbytes) { return cudaMemcpy(h, d, nbytes, cudaMemcpyDeviceToHost); }
// E12: called after each batch with the batch's statuses (lane B: its overflows go to the host lane while the device runs on)
typedef std::function<void(u64 b0, u64 nb, const uint8_t* status)> BatchFn;
// SINGLET_LANE_PROF=1: per-kernel device time inside the lanes (cudaEvent pairs around each launch; no change to results). The
// step names are in kProfName; the totals per lane go into the lane log line.
enum { K_P1S, K_P1W, K_SCAN, K_P2, K_WPLAN, K_WKEY, K_WSCAN, K_P3, K_COST, K_SORT, K_P3A, K_P3W, K_CSTAT, K_N };
static const char* const kProfName[K_N] = {"p1s", "p1w", "scan", "p2", "wplan", "wkey", "wscan", "p3", "cost", "sort", "p3a", "p3w", "cstat"};
struct LaneProf {
    bool on = false; int lvl = 0;
    struct Ev { cudaEvent_t a, b; int id; };
    std::vector<Ev> ev;
    double sec[K_N] = {}; u64 cnt[K_N] = {};
    double hostCb = 0, hostSync = 0; u64 nBatch = 0;
    void init() { const char* e = std::getenv("SINGLET_LANE_PROF"); on = e && (e[0] == '1' || e[0] == '2'); lvl = on ? e[0] - '0' : 0; }
    // lvl 2: SIMT model of a COMPACT lane's phase 3. Cost of a read = its stitching steps (+ a fixed part); a warp costs its largest read; efficiency = sum of costs / (32 x sum of warp maxima)
    // for the batch order and for the reads sorted by a feature (descending): f[6k+0] steps, 1 windows, 2 aligns, 3 SA hits, 4 sum of squared rows, 5 largest row.
    static constexpr int NK = 7;   // identity, then features 2, 3, 4, 1, 5, and the oracle (steps)
    double mSum[2] = {}, mMax[2][NK] = {}; u64 mReads = 0; double mStepsMax = 0, mStepsMean = 0;
    void model(const std::vector<uint32_t>& f, u64 n) {
        static const int feat[NK] = {-1, 2, 3, 4, 1, 5, 0};
        std::vector<uint32_t> ord(n);
        for (int kk = 0; kk < NK; kk++) {
            for (u64 i = 0; i < n; i++) ord[i] = (uint32_t)i;
            if (feat[kk] >= 0) { const int ft = feat[kk]; std::stable_sort(ord.begin(), ord.end(), [&](uint32_t a, uint32_t b) { return f[6 * a + ft] > f[6 * b + ft]; }); }
            for (int m = 0; m < 2; m++) {
                const double c0 = m ? 50.0 : 0.0; double sum = 0, mx = 0;
                for (u64 i = 0; i < n; i += 32) { double wm = 0; for (u64 j = i; j < std::min(n, i + 32); j++) { const double c = f[6 * ord[j]] + c0; sum += c; wm = std::max(wm, c); } mx += wm; }
                if (kk == 0) mSum[m] += sum;
                mMax[m][kk] += mx;
            }
        }
        mReads += n;
        double mxs = 0, sm = 0; for (u64 i = 0; i < n; i++) { mxs = std::max<double>(mxs, f[6 * i]); sm += f[6 * i]; }
        mStepsMax = std::max(mStepsMax, mxs); mStepsMean += sm;
    }
    void begin(int id) { if (!on) return; Ev e; e.id = id; cudaEventCreate(&e.a); cudaEventCreate(&e.b); cudaEventRecord(e.a); ev.push_back(e); }
    void end() { if (!on) return; cudaEventRecord(ev.back().b); }
    void collect() {
        for (auto& e : ev) { float ms = 0; cudaEventSynchronize(e.b); cudaEventElapsedTime(&ms, e.a, e.b); sec[e.id] += ms * 1e-3; cnt[e.id]++; cudaEventDestroy(e.a); cudaEventDestroy(e.b); }
        ev.clear();
    }
};
#define PK(PR, ID, STMT) do { (PR).begin(ID); STMT; (PR).end(); } while (0)

template <class C, bool WARP = false, int MAXREG = 0>
bool runLane(const Params& P, const GenomeView& D, const char* dSeq, const u64* dOff, const std::vector<uint32_t>& idx, u64 NB, u64 budget,
             Lane& L, std::string& err, bool keepDev, const char* nm, bool NEW, const BatchFn* onBatch = nullptr) {
    const double tA = now();
    const std::string sn = std::string("lane") + nm;
    const int bSort = g_bSort;
    g_bud.at = sn;
    u64 n = idx.size();
    L.status.assign(n, (uint8_t)Status::Overflow); L.len.assign(n, 0); L.off.resize(n + 1); L.off[0] = 0;   // off: every entry copied below
    if (n == 0) return true;
    NB = std::min<u64>(NB, n);
    // E17: every buffer through the budget's tally (dmal / dfree); Work and the batch buffers are freed before the results are packed
    uint32_t* dIdx; uint8_t *dOut, *dSt, *dBst; uint16_t* dLen; Work<C>* dWork; ReadBuf<C::LREAD>* dRbuf; u64 *dNent, *dBase; uint32_t* dBits;
    EG_CK(dmal(&dIdx, n * 4)); EG_CK(cudaMemcpy(dIdx, idx.data(), n * 4, cudaMemcpyHostToDevice));
    EG_CK(dmal(&dOut, n * (u64)C::NBLOB)); EG_CK(dmal(&dSt, n)); EG_CK(dmal(&dLen, n * 2));
    EG_CK(dmal(&dWork, NB * sizeof(Work<C>))); EG_CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>)));
    EG_CK(dmal(&dRbuf, NB * sizeof(ReadBuf<C::LREAD>))); EG_CK(dmal(&dBst, NB));
    EG_CK(dmal(&dNent, NB * 8)); EG_CK(dmal(&dBase, NB * 8));
    u64 bitsCap = 1 << 20; EG_CK(dmal(&dBits, bitsCap * 4));
    // COMPACT (lane B, SINGLET_BOLD_WSHRINK): the aligns' pool (room for the largest single read, else g_waSlice per read in flight), the reads' needs,
    // their rows' starts, the pending flags, the scan keys and a counter
    m5::uint* dPool = nullptr; u64 *dNeed = nullptr, *dWb = nullptr, *dWkey = nullptr; uint8_t* dPend = nullptr; unsigned long long* dWcnt = nullptr; u64 poolAl = 0;
    u64 nRounds = 0, nDeferred = 0;
    if constexpr (C::COMPACT) {
        poolAl = std::max<u64>(NB * (u64)g_waSlice, (u64)C::NW * C::NWA);
        EG_CK(dmal(&dPool, poolAl * WA_SIZE * sizeof(m5::uint)));
        EG_CK(dmal(&dNeed, NB * 8)); EG_CK(dmal(&dWb, NB * 8)); EG_CK(dmal(&dWkey, NB * 8)); EG_CK(dmal(&dPend, NB)); EG_CK(dmal(&dWcnt, 8));
    }
    // E12: the seed / windows kernels' counters, the warp lane's sort keys and order, the persistent kernels' grids
    unsigned long long* dCtr = nullptr; u64* dKey = nullptr; uint32_t* dPerm = nullptr;
    int nSm = 0, bpsS = 0, bpsW = 0; size_t stack0 = 0; u64 stackRes = 0;
    if (NEW) {
        EG_CK(dmal(&dCtr, 16));
        if (WARP || (C::COMPACT && bSort)) { EG_CK(dmal(&dKey, NB * 8)); EG_CK(dmal(&dPerm, NB * 4)); }
        int dev = 0; cudaGetDevice(&dev); cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, dev);
        EG_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsS, k_p1s<C>, 128, 0));
        if constexpr (p1WinPers<C>()) EG_CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpsW, k_p1wp<C>, 128, 0));
        if constexpr (WARP) {   // the lanes' stitching contexts are local memory: room for them in the per-thread stack (as m5)
            cudaDeviceGetLimit(&stack0, cudaLimitStackSize);
            size_t fs0 = 0, fs1 = 0, ts = 0; cudaMemGetInfo(&fs0, &ts);
            EG_CK(cudaDeviceSetLimit(cudaLimitStackSize, std::max<size_t>(16 * 1024, sizeof(LWM) + 8192)));
            cudaMemGetInfo(&fs1, &ts);
            stackRes = fs0 > fs1 ? fs0 - fs1 : 0;   // E17: the driver's reservation for the larger stack (device-wide measure), in the tally while it is set
            g_bud.used += stackRes; g_bud.note();
        }
        int regs = 0;
        { cudaFuncAttributes fa{};
          if constexpr (C::COMPACT) cudaFuncGetAttributes(&fa, (const void*)k_p3c<C>);
          else if constexpr (WARP) cudaFuncGetAttributes(&fa, (const void*)k_p3w<C, LWM>);
          else if constexpr (MAXREG == 64) cudaFuncGetAttributes(&fa, (const void*)k_p3lb<C, 128, 8>);
          else cudaFuncGetAttributes(&fa, (const void*)k_p3e<C>);
          regs = fa.numRegs; }
        char b[400];
        snprintf(b, sizeof b, "%s %llu reads, batches of %llu (%.1f GB of Work), k_p1s %d blocks/SM, %s, phase 3 %s (%d registers%s); ", nm,
                 (unsigned long long)n, (unsigned long long)NB, NB * (double)(sizeof(Work<C>) + sizeof(ReadBuf<C::LREAD>)) / 1e9, bpsS,
                 p1WinPers<C>() ? "k_p1wp persistent" : "k_p1w", WARP ? "warp per read, by windows desc" : MAXREG == 64 ? "capped at 64 registers" : "thread per read", regs,
                 WARP ? (", stack " + std::to_string(stackRes >> 20) + " MiB").c_str() : "");
        g_laneLog += b;
        if constexpr (C::COMPACT) {
            snprintf(b, sizeof b, "[wshrink: per read %llu bytes = Work %llu + ReadBuf %llu + pool %llu + 25, batch %llu, pool %llu aligns (%.2f GB)] ", (unsigned long long)(sizeof(Work<C>) + sizeof(ReadBuf<C::LREAD>) + (u64)g_waSlice * WA_SIZE * sizeof(m5::uint) + 25),
                     (unsigned long long)sizeof(Work<C>), (unsigned long long)sizeof(ReadBuf<C::LREAD>), (unsigned long long)((u64)g_waSlice * WA_SIZE * sizeof(m5::uint)), (unsigned long long)NB,
                     (unsigned long long)poolAl, poolAl * WA_SIZE * sizeof(m5::uint) / 1e9);
            g_laneLog += b;
        }
    }
    devNote();
    const double tK = now(); stepAdd(sn + ".alloc", tK - tA);
    LaneProf pr; pr.init();
    unsigned long long* dClk = nullptr; u64 nReadsDone = 0;
    if (pr.on) { EG_CK(dmal(&dClk, 64)); EG_CK(cudaMemset(dClk, 0, 64)); EG_CK(cudaMemset(dWork, 0, NB * sizeof(Work<C>))); }
    for (u64 b0 = 0; b0 < n; b0 += NB) {
        pr.nBatch++;
        u64 nb = std::min(NB, n - b0);
        int T = 128; unsigned G = (unsigned)((nb + T - 1) / T);
        if (!NEW) k_p1<C><<<G, T>>>(P, D, dSeq, dOff, dIdx, b0, nb, dWork, dRbuf, dBst, dNent);
        else {
            EG_CK(cudaMemset(dCtr, 0, 16));
            PK(pr, K_P1S, (k_p1s<C><<<(unsigned)std::min<u64>((u64)nSm * bpsS, G), T>>>(P, D, dSeq, dOff, dIdx, b0, nb, dWork, dRbuf, dBst, dCtr)));
            if constexpr (p1WinPers<C>()) PK(pr, K_P1W, (k_p1wp<C><<<(unsigned)std::min<u64>((u64)nSm * bpsW, G), T>>>(P, D, nb, dWork, dBst, dNent, dCtr + 1)));
            else PK(pr, K_P1W, (k_p1w<C><<<G, T>>>(P, D, nb, dWork, dBst, dNent)));
        }
        PK(pr, K_SCAN, thrust::exclusive_scan(TP, thrust::device_ptr<u64>(dNent), thrust::device_ptr<u64>(dNent + nb), thrust::device_ptr<u64>(dBase)));
        u64 lastB, lastN; EG_CK(d2h(&lastB, dBase + nb - 1, 8)); EG_CK(d2h(&lastN, dNent + nb - 1, 8));
        u64 words = (lastB + lastN) / 32 + 2;
        if (words > bitsCap) { dfree(dBits); bitsCap = words * 2; EG_CK(dmal(&dBits, bitsCap * 4)); }
        EG_CK(cudaMemset(dBits, 0, words * 4));
        PK(pr, K_P2, (k_p2<C><<<(unsigned)((nb * 32 + 255) / 256), 256>>>(P, D, nb, dWork, dNent, dBase, dBits)));
        if (!NEW) k_p3<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dSt, dOut, dLen);
        else {
            if constexpr (C::COMPACT) {
                PK(pr, K_WPLAN, (k_wplan<C><<<G, T>>>(P, D, nb, dWork, dBase, dBits, dBst, dNeed)));
                EG_CK(cudaMemset(dPend, 1, nb));
                const uint32_t* dPm = nullptr;
                if (bSort) {
                    PK(pr, K_COST, (k_cost_need<<<G, T>>>(nb, dNeed, dKey, dPerm)));
                    PK(pr, K_SORT, thrust::sort_by_key(TP, thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm)));
                    dPm = dPerm;
                }
                for (int rd = 0;; rd++) {
                    EG_CK(cudaMemset(dWcnt, 0, 8));
                    PK(pr, K_WKEY, (k_wkey<C><<<G, T>>>(nb, dNeed, dPend, dWkey, dWcnt)));
                    unsigned long long pendN = 0; EG_CK(d2h(&pendN, dWcnt, 8));
                    if (!pendN) break;
                    if (rd > 0) nDeferred += pendN;
                    nRounds++;
                    PK(pr, K_WSCAN, thrust::exclusive_scan(TP, thrust::device_ptr<u64>(dWkey), thrust::device_ptr<u64>(dWkey + nb), thrust::device_ptr<u64>(dWb)));
                    PK(pr, K_P3, (k_p3c<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dOut, dLen, dPend, dNeed, dWb, dPool, poolAl, dPm)));
                    EG_CK(cudaGetLastError());
                }
                if (pr.lvl >= 2) {   // SINGLET_LANE_PROF=2: the work features of this batch's reads, and the SIMT model
                    uint32_t* dF; EG_CK(dmal(&dF, nb * 24));
                    k_feat<C><<<G, T>>>(nb, dWork, dNeed, dNent, dF);
                    std::vector<uint32_t> hf(nb * 6); EG_CK(d2h(hf.data(), dF, nb * 24)); dfree(dF);
                    pr.model(hf, nb);
                }
            } else if constexpr (WARP) {
                PK(pr, K_COST, (k_cost<C><<<G, T>>>(nb, dWork, dNent, dKey, dPerm)));
                PK(pr, K_SORT, thrust::sort_by_key(TP, thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nb), thrust::device_ptr<uint32_t>(dPerm)));
                PK(pr, K_P3A, (k_p3a<C><<<G, T>>>(P, D, nb, dWork, dBase, dBits, dBst)));
                const int wt = g_wthreads; const unsigned GW = (unsigned)((nb * 32 + wt - 1) / wt);
                const size_t sh = (size_t)wt * sizeof(AssignEnt) + (size_t)(wt / 32) * C::NW * sizeof(uint16_t);
                PK(pr, K_P3W, (k_p3w<C, LWM><<<GW, wt, sh>>>(P, D, b0, nb, budget, dWork, dRbuf, dBst, dOut, dLen, dPerm)));
            } else if constexpr (MAXREG == 64) PK(pr, K_P3, (k_p3lb<C, 128, 8><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dOut, dLen)));
            else PK(pr, K_P3, (k_p3e<C><<<G, T>>>(P, D, b0, nb, budget, dWork, dRbuf, dBase, dBits, dBst, dOut, dLen)));
            PK(pr, K_CSTAT, (k_copy_status<<<G, T>>>(dBst, dSt + b0, nb)));
            if (pr.on) { k_clksum<C><<<G, T>>>(nb, dWork, dClk); nReadsDone += nb; }
        }
        EG_CK(cudaGetLastError());
        if (onBatch && *onBatch) {   // the batch's statuses now (the next batch's kernels wait for the host for this copy's duration)
            std::vector<uint8_t> hs(nb);
            const double tc0 = now();
            EG_CK(d2h(hs.data(), dSt + b0, nb));
            const double tc1 = now();
            (*onBatch)(b0, nb, hs.data());
            pr.hostSync += tc1 - tc0; pr.hostCb += now() - tc1;
            if (g_autoBudget && nm[0] == 'B') budget = g_autoBudget->load();
        }
    }
    EG_CK(cudaDeviceSynchronize());
    devNote();
    {   // E17: the device memory in use after the kernels (device-wide; the driver's local memory for the kernels shows here) against the tally
        size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
        char b[120]; snprintf(b, sizeof b, "(device %.1f GB in use, tally %.1f GB) ", (double)(t - f) / (1ull << 30), b2gb(g_bud.used)); g_laneLog += b;
    }
    if constexpr (C::COMPACT) {
        char b[160]; snprintf(b, sizeof b, "[wshrink: %llu pool rounds, %llu read-rounds deferred] ", (unsigned long long)nRounds, (unsigned long long)nDeferred); g_laneLog += b;
        dfree(dPool); dfree(dNeed); dfree(dWb); dfree(dWkey); dfree(dPend); dfree(dWcnt);
    }
    dfree(dCtr); dfree(dKey); dfree(dPerm);
    dfree(dIdx); dfree(dWork); dfree(dRbuf); dfree(dBst); dfree(dNent); dfree(dBase); dfree(dBits);   // E17: before the packed results
    if (stack0) { cudaDeviceSetLimit(cudaLimitStackSize, stack0); g_bud.used -= std::min(g_bud.used, stackRes); }
    const double tP = now(); stepAdd(sn + ".kernels", tP - tK);
    if (pr.on) {   // SINGLET_LANE_PROF: the lane's device seconds per step, launches, the rest of the kernels' wall time (gaps: launch, host syncs, callback)
        pr.collect();
        double sum = 0; for (int i = 0; i < K_N; i++) sum += pr.sec[i];
        char b[700]; int o = snprintf(b, sizeof b, "[prof %s: wall %.2f s, in kernels %.2f s, gaps %.2f s (status copy wait %.2f, callback %.2f), %llu batches:", nm, tP - tK, sum, tP - tK - sum, pr.hostSync, pr.hostCb, (unsigned long long)pr.nBatch);
        for (int i = 0; i < K_N; i++) if (pr.cnt[i]) o += snprintf(b + o, sizeof b - o, " %s %.2f s x%llu;", kProfName[i], pr.sec[i], (unsigned long long)pr.cnt[i]);
        snprintf(b + o, sizeof b - o, "] "); g_laneLog += b;
        if (dClk && nReadsDone) {   // the reads' own cycle counters (thread cycles, summed over the lane), per read and as a share of seeds + windows/assign + stitch + select
            unsigned long long h[8]; EG_CK(cudaMemcpy(h, dClk, 64, cudaMemcpyDeviceToHost));
            int clkKHz = 0; cudaDeviceGetAttribute(&clkKHz, cudaDevAttrClockRate, 0);
            const double tot = (double)(h[0] + h[1] + h[2] + h[3]);
            char c2[600]; snprintf(c2, sizeof c2, "[clk %s per read, kcycles: seeds %.1f, windows+assign %.1f, stitch %.1f (stitchAlign %.1f, finalize %.1f, copies %.1f, record %.1f), select+encode %.1f; thread-seconds %.1f; shares %.0f/%.0f/%.0f/%.0f %%] ", nm,
                     (double)h[0] / nReadsDone / 1e3, (double)h[1] / nReadsDone / 1e3, (double)h[2] / nReadsDone / 1e3, (double)h[4] / nReadsDone / 1e3, (double)h[5] / nReadsDone / 1e3,
                     (double)h[6] / nReadsDone / 1e3, (double)h[7] / nReadsDone / 1e3, (double)h[3] / nReadsDone / 1e3, tot / (clkKHz * 1e3),
                     100.0 * h[0] / tot, 100.0 * h[1] / tot, 100.0 * h[2] / tot, 100.0 * h[3] / tot);
            g_laneLog += c2;
        }
        if (dClk) dfree(dClk);
        if (pr.lvl >= 2 && pr.mReads) {
            static const char* const nmk[LaneProf::NK] = {"batch order", "aligns", "SA hits", "sum row^2", "windows", "max row", "oracle"};
            char m2[900]; int o2 = snprintf(m2, sizeof m2, "[model %s: %llu reads, mean steps %.0f, largest %.0f; SIMT efficiency by order (steps | steps+50):", nm, (unsigned long long)pr.mReads, pr.mStepsMean / pr.mReads, pr.mStepsMax);
            for (int kk = 0; kk < LaneProf::NK; kk++) o2 += snprintf(m2 + o2, sizeof m2 - o2, " %s %.2f|%.2f;", nmk[kk], pr.mSum[0] / (32.0 * pr.mMax[0][kk]), pr.mSum[1] / (32.0 * pr.mMax[1][kk]));
            snprintf(m2 + o2, sizeof m2 - o2, "] "); g_laneLog += m2;
        }
    }
    {   // copy only the bytes of the results (the slots are 640-4096 bytes, results ~70)
        u64* dPoff; EG_CK(dmal(&dPoff, (n + 1) * 8)); EG_CK(cudaMemset(dPoff, 0, (n + 1) * 8));
        thrust::transform_inclusive_scan(TP, thrust::device_ptr<uint16_t>(dLen), thrust::device_ptr<uint16_t>(dLen + n), thrust::device_ptr<u64>(dPoff + 1),
                                         ToU64(), thrust::plus<u64>());
        u64 total; EG_CK(d2h(&total, dPoff + n, 8));
        uint8_t* dPacked; EG_CK(dmal(&dPacked, total + 1));
        devNote();
        k_pack<<<(unsigned)((n + 255) / 256), 256>>>(dOut, C::NBLOB, dLen, dPoff, n, dPacked);
        EG_CK(cudaGetLastError());
        L.out.resize(total);
        if (g_d2h) { if (!g_d2h->copy(L.out.data(), dPacked, total) || !g_d2h->copy(L.off.data(), dPoff, (n + 1) * 8)) { err = "lane results copy"; return false; } }
        else {
        EG_CK(d2h(L.out.data(), dPacked, total));
        EG_CK(d2h(L.off.data(), dPoff, (n + 1) * 8));
        }
        if (keepDev) { L.dOut = dPacked; L.dOff = dPoff; }
        else { dfree(dPoff); dfree(dPacked); }
    }
    EG_CK(d2h(L.status.data(), dSt, n)); EG_CK(d2h(L.len.data(), dLen, n * 2));
    dfree(dOut); dfree(dSt); dfree(dLen);
    stepAdd(sn + ".d2h", now() - tP);
    return true;
}

// E17: a lane's per-read device bytes (index, result slot, status, length) and per read in flight (Work, ReadBuf, batch status, counts, the
// warp lane's sort key and order), which the budget sizes the chunk and the batches from
template <class C> constexpr u64 laneSlotB() { return 4 + (u64)C::NBLOB + 1 + 2; }
template <class C, bool WARP> constexpr u64 lanePerB() { return sizeof(Work<C>) + sizeof(ReadBuf<C::LREAD>) + 1 + 8 + 8 + (WARP ? 12 + 24 : 0); }
// E17: a lane's device bytes for m reads in batches of nb: the result slots, then the larger of the batch's buffers (with ~256 bytes per read of
// the windows' bit set and thrust's scratch) and the packed results (~112 bytes per read with their offsets: Work is freed before they are made)
static inline u64 laneNeed(u64 m, u64 nb, u64 slot, u64 per) { return m * slot + std::max(std::min(nb, m) * (per + 256), m * 112) + (16ull << 20); }
// E17: a lane's batch nb and part (reads per runLane; 0: all at once) for m reads within avail bytes: the batch nb0 if it fits with all the slots,
// else the largest multiple of gran (at least nbMin) that does; else parts, with a batch of up to nb0 in half the room and the slots of a part in
// the rest. false: not even a batch of nbMin and its own slots fit.
static bool lanePlan(u64 m, u64 avail, u64 slot, u64 per, u64 nb0, u64 nbMin, u64 gran, u64& nb, u64& part) {
    nb = nb0; part = 0;
    if (m == 0 || laneNeed(m, nb0, slot, per) <= avail) return true;
    const u64 fixed = m * slot + (16ull << 20);
    if (avail > fixed) {
        const u64 fit = (avail - fixed) / (per + 256) / gran * gran;
        if (fit >= nbMin && laneNeed(m, fit, slot, per) <= avail) { nb = std::min(nb0, fit); return true; }
    }
    nb = std::max(nbMin, std::min(nb0, avail / 2 / (per + 256) / gran * gran));
    if (avail < nb * (per + 256) + nb * (slot + 112) + (16ull << 20)) {
        nb = nbMin;
        if (avail < nb * (per + 256) + nb * (slot + 112) + (16ull << 20)) return false;
    }
    part = (avail - nb * (per + 256) - (16ull << 20)) / (slot + 112);
    if (part >= m) part = 0;
    return true;
}
// E17: a lane over idx in parts of at most `part` reads when its result slots for all of idx do not fit the budget (each part a runLane; the
// results concatenated on the host: statuses, lengths, packed bytes and offsets are what one runLane over idx gives). With more than one part
// nothing stays on the device (the records' gather uploads the packed results again). onBatch sees batch offsets within idx.
template <class C, bool WARP = false, int MAXREG = 0>
bool runLaneParts(const Params& P, const GenomeView& D, const char* dSeq, const u64* dOff, const std::vector<uint32_t>& idx, u64 NB, u64 part,
                  u64 budget, Lane& L, std::string& err, bool keepDev, const char* nm, bool NEW, const BatchFn* onBatch, u64& nParts) {
    const u64 n = idx.size();
    if (part == 0 || part >= n) { nParts = 1; return runLane<C, WARP, MAXREG>(P, D, dSeq, dOff, idx, NB, budget, L, err, keepDev, nm, NEW, onBatch); }
    nParts = 0;
    L.status.clear(); L.len.clear(); L.out.clear(); L.off.assign(1, 0); L.dOut = nullptr; L.dOff = nullptr;
    L.status.reserve(n); L.len.reserve(n); L.off.reserve(n + 1);
    for (u64 p0 = 0; p0 < n; p0 += part) {
        const u64 m = std::min(part, n - p0);
        std::vector<uint32_t> sub(idx.begin() + p0, idx.begin() + p0 + m);
        BatchFn ob;
        if (onBatch && *onBatch) ob = [&, p0](u64 b0, u64 nb, const uint8_t* st) { (*onBatch)(p0 + b0, nb, st); };
        Lane S;
        if (!runLane<C, WARP, MAXREG>(P, D, dSeq, dOff, sub, NB, budget, S, err, false, nm, NEW, (onBatch && *onBatch) ? &ob : nullptr)) return false;
        const u64 base = L.out.size();
        L.status.insert(L.status.end(), S.status.begin(), S.status.end()); L.len.insert(L.len.end(), S.len.begin(), S.len.end());
        L.out.resize(base + S.out.size()); if (!S.out.empty()) std::memcpy(L.out.data() + base, S.out.data(), S.out.size());
        for (u64 k = 1; k <= m; k++) L.off.push_back(base + S.off[k]);
        nParts++;
    }
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
    std::atomic<int> active{0};          // threads inside fn now (the STEPS_B=auto controller)
    int nThreads = 0;
    std::function<void(int, const Task&)> fn;
    u64 qsize() { std::lock_guard<std::mutex> lk(mu); return qu.size(); }
    void start(int nt, u64 nChunks, std::function<void(int, const Task&)> f) {
        fn = std::move(f); pend.assign(nChunks, 0); nThreads = nt;
        for (int t = 0; t < nt; t++) th.emplace_back([this, t]() {
            for (;;) {
                Task tk;
                { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return stop || !qu.empty(); }); if (stop) return; tk = qu.front(); qu.pop_front(); }
                const auto t0 = std::chrono::steady_clock::now();
                active++;
                fn(t, tk);
                active--;
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
    EG_CK(dmal(&d, (n + pad) * sizeof(T) + 8));
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
// E17: the representatives' (clipped) sequences compacted in rep order after the dedup: position k holds read rep[k]'s bytes. The lanes and the
// records' gather address the sequences by position from then on; the reads that are not a representative leave the device.
__global__ void k_rep_len(const u64* off, const uint32_t* rep, u64 nRep, u64* len) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k < nRep) len[k] = off[rep[k] + 1] - off[rep[k]];
}
__global__ void k_rep_copy(const char* seq, const u64* off, const uint32_t* rep, u64 nRep, const u64* offR, char* seqR) {
    const u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (k >= nRep) return;
    const char* x = seq + off[rep[k]];
    for (u64 t = 0, m = offR[k + 1] - offR[k]; t < m; t++) seqR[offR[k] + t] = x[t];
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
    // SINGLET_ENGINE_GPU_MIN_CC (default 80): the lowest compute capability accepted, as major*10+minor. The engine's kernels compile for
    // sm_70 and sm_75 too (CMAKE_CUDA_ARCHITECTURES="70;75;80;90"); 70 admits the V100S and RTX 8000 for identity checks
    const char* mc = std::getenv("SINGLET_ENGINE_GPU_MIN_CC");
    const int minCC = mc != nullptr && *mc ? std::atoi(mc) : 80;
    if (p.major * 10 + p.minor < minCC) { if (why) *why = std::string("device ") + p.name + " below compute capability " + std::to_string(minCC / 10) + "." + std::to_string(minCC % 10); return false; }
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
#include "shared_index.cuh"

// P3-MEM-03: a record chunk's offsets shifted by the earlier chunks' counts (alignments, sequence bytes, operations); a read's record local
// to the chunk [r0, r1) or NONE
__global__ void k_rebase_recs(SigRec* s, u64 nS, AlnRec* a, u64 nA, uint32_t alnBase, u64 seqBase, uint32_t opsBase) {
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i < nS) { s[i].aln_off += alnBase; s[i].seq_off += seqBase; }
    if (i < nA) a[i].ops_off += opsBase;
}
__global__ void k_local_rr(const uint32_t* g, u64 n, uint32_t r0, uint32_t r1, uint32_t* l) {
    const u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t v = g[i];
    l[i] = (v != 0xFFFFFFFFu && v >= r0 && v < r1) ? v - r0 : 0xFFFFFFFFu;
}
static bool mapDistinctBody(const Params& P, const GenomeView& H, uint64_t saBytes, uint64_t saiBytes, const Input& in, int device, int hostThreads,
                            Output& out);
// An exception out of the body (thrust's temporaries throw std::bad_alloc when the device is full; a host allocation can too) aborted STAR's
// process with "terminate called after throwing an instance of 'std::bad_alloc'" and the sample failed (pilot item 86, run c10: device OOM on
// attempts 1 and 2, the throw on attempt 3). It is now a failed attempt like fail()'s: the body's staging buffers, shared-index client and
// index thread are released by their destructors during the unwinding, then the device is reset, and the caller retries or lets STAR map
// every read on the CPU (same outputs).
bool mapDistinct(const Params& P, const GenomeView& H, uint64_t saBytes, uint64_t saiBytes, const Input& in, int device, int hostThreads,
                 Output& out) {
    std::string why;
    try {
        return mapDistinctBody(P, H, saBytes, saiBytes, in, device, hostThreads, out);
    } catch (const std::bad_alloc& e) {
        why = std::string("an allocation failed (") + e.what() + ")";
    } catch (const std::exception& e) {
        why = std::string("an exception (") + e.what() + ")";
    }
    out.log += "engine_gpu: " + why + "; the device is reset\n";
    e9::soloKeepRelease(); out.e10.soloKept = false;
    g_d2h = nullptr;
    cudaDeviceReset(); g_bud.reset(0, 0);
    return false;
}
static bool mapDistinctBody(const Params& P, const GenomeView& H, uint64_t saBytes, uint64_t saiBytes, const Input& in, int device, int hostThreads,
                            Output& out) {
    ::singlet::gpu_ledger::Phase ledger_phase("prefill");
    singlet::ThreadNameScope tnMap("gpu-map");   // mapDistinct's host threads inherit it (per-thread CPU by role)
    singlet::NvtxChain nvch; nvch.next("mapDistinct: setup");   // NVTX ranges of the phases (SINGLET_NVTX=1; the d164b34d ranges, redeveloped)
    std::string err;
    Stager SR, SI;   // E12: staged uploads of the reads (this thread) and of the index (idxTh); declared first: destroyed after the join
    D2H DR;          // E12: staged device -> host copies (g_d2h while mapDistinct runs)
    struct D2HReset { ~D2HReset() { g_d2h = nullptr; } } d2hReset;
    bool idxOk = true; double tIdxEnd = 0;
    std::thread idxTh;   // E12: the index upload (staged, on its own stream) while the reads are uploaded, clipped and deduplicated
    struct JoinOnExit { std::thread& t; ~JoinOnExit() { if (t.joinable()) t.join(); } } idxJoin{idxTh};
    // on failure, release everything this process holds on the device (STAR then maps every read on the CPU)
    // E17: the staging buffers' streams, events and page-locked memory are released before the reset (their handles are stale after it, and
    // their destructors would use them: a budget failure right after their init ended in a segmentation fault)
    ShClient SHC;   // shared index: the imports from the holder (SINGLET_GPU_SHARED_INDEX=1), closed at the end or by fail
    auto fail = [&](const std::string& e) {
        shClose(SHC);
        if (idxTh.joinable()) idxTh.join();
        out.log += "engine_gpu: " + e + "\n"; e9::soloKeepRelease(); out.e10.soloKept = false;
        g_d2h = nullptr; DR.release(); SR.release(); SI.release();
        cudaDeviceReset(); g_bud.reset(0, 0); return false;
    };
    // E17: the sample does not fit the device memory budget (decided before the allocation that would exceed it); the caller stops the run
    // with this message unless SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu lets STAR map every read
    auto failBudget = [&](const std::string& e) { out.budgetFail = true; out.budgetWhy = e; return fail("E17 budget: " + e); };
    g_steps.clear();
    // E12: SINGLET_ENGINE_GPU_CFG=old: the configuration before E12 (lanes, dedup products on the host, uploads, host lane after the lanes)
    const char* cfgE = std::getenv("SINGLET_ENGINE_GPU_CFG");
    const bool NEW0 = !(cfgE && std::string(cfgE) == "old");
    const double tF0 = now();
    double tS = tF0;   // E12: start of the current step
    auto step = [&](const char* k) { const double t = now(); stepAdd(k, t - tS); tS = t; };
    const cudaError_t initRc = cudaInitDevice(device, cudaDeviceScheduleBlockingSync, singlet_engine_gpu::gpuInitFlags());   // before the context exists: a waiting host thread sleeps, not spins
    if (initRc != cudaSuccess) cudaGetLastError();
    if (cudaSetDevice(device) != cudaSuccess) return fail("cannot select device");
    double t0 = now();
    // device memory in use at the phases of the call (logged: what a sample needs when several share a GPU)
    std::string memLog;
    auto memMark = [&](const char* what) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); char b[64]; snprintf(b, sizeof b, "%s %.1f, ", what, (double) (t - f) / (1u << 30)); memLog += b; };
    memMark("start");
    step("init");
    // ---- index (G with 200 bytes of padding on each side, as STAR's G1)
    nvch.next("mapDistinct: index");
    const u64 PAD = 200;
    GenomeView D = H;
    char* dG1 = nullptr; u64* dSA = nullptr; u64* dSAi = nullptr;
    // E12 (default): the reads and E10's aux through page-locked staging buffers (Stager) on this thread, the index the same way from a second
    // host thread while the reads are uploaded, clipped and deduplicated (joined before the lanes). SINGLET_ENGINE_GPU_UPLOAD=pageable (or
    // CFG=old): the driver's pageable copies, in sequence, as before
    const char* upE = std::getenv("SINGLET_ENGINE_GPU_UPLOAD");
    const bool staged = NEW0 && !(upE && std::string(upE) == "pageable");
    // SINGLET_ENGINE_GPU_PIN=1: page-lock the host index first, so the copies run at the PCIe rate instead of through the driver's staging buffers
    const bool pinIdx = !staged && std::getenv("SINGLET_ENGINE_GPU_PIN") != nullptr;
    double tIdx0 = now(), tPin = 0;
    const char* d2E = std::getenv("SINGLET_ENGINE_GPU_D2H");
    if (NEW0 && !(d2E && std::string(d2E) == "pageable")) {
        if (!DR.init(std::max(1, std::min(16, hostThreads)))) return fail("page-locked staging buffers (D2H)");
        g_d2h = &DR;
    }
    if (staged && (!SR.init(std::max(1, std::min(8, hostThreads))) || !SI.init(std::max(1, std::min(4, hostThreads / 4)))))
        return fail("page-locked staging buffers");
    // ---- E17 (docs/gpu-engine/fable/E17-budget.md): the device memory budget (bytes; 0: none, E12's sizes) and the tally, which starts at this
    // process's context (at most 1 GB of what the device holds now: the rest is other processes'). Mode "normal" (as E12: the index uploaded during
    // the reads, clip and dedup) when the estimates below fit the budget; else "lean": the reads' phase without the index, E10's entries and
    // masks into the shared region as soon as they exist, the E11 keep of the reads' CR/UR at the records, the index after the dedup (only the
    // distinct sequences left on the device), the lanes' results not kept on the device (E8's records uploaded from the host assembly)
    // shared index: before the budget is decided, because a mapped index is not this process's memory (IDX below is 0 and memBudget leaves the holder's bytes out)
    { std::string shLog; shAcquire(in.genomeDir, H, saBytes, saiBytes, device, SHC, shLog); out.log += shLog; }
    size_t fB0 = 0, tB0 = 0; cudaMemGetInfo(&fB0, &tB0);
    const u64 cap = memBudget(tB0);
    const bool budDefault = std::getenv("SINGLET_ENGINE_GPU_MEM_GB") == nullptr;
    g_bud.reset(cap, std::min<u64>(tB0 - fB0, 1ull << 30));
    const u64 ctx = g_bud.used;
    const u64 SLACK = 1ull << 30;   // what the tally does not see: the modules' temporaries (E8, E10, E11), the driver's local memory for the kernels
    const u64 n0 = in.off.size() - 1, S0 = in.seq.size();
    const bool e10W = in.e10.on && in.e2e != nullptr && in.e2e->base != nullptr && n0 > 0 && in.e10.nw >= 1 && in.e10.aux.size() == n0 * (2 + (u64)in.e10.nw);
    const u64 IDX = SHC.on && !SHC.priv ? 0 : budRound(H.nGenome + 2 * PAD + 8) + budRound(saBytes + 72) + budRound(saiBytes + 72);
    const u64 TAB = ((u64)H.nChr * 2 + 1 + H.chrBinN + (u64)H.sjdbN * 4) * 8 + (u64)H.sjdbN * 4 + (24ull << 20);
    const u64 AUX = e10W ? in.e10.aux.size() * 8 : 0, ENT = e10W ? n0 * 24 : 0, QM = e10W && in.e10.wantMask ? n0 * qmBytes() : 0;
    const u64 KEEP = e10W && in.e10.soloModel != nullptr ? n0 * 16 : 0;
    const u64 CLIP = in.clip.on ? n0 * (in.clip.dev ? 29 : 28) + S0 : 0;   // per read its clipped span, length, offset and host clip lengths (E20: and info); the clipped copy
    const u64 RD = S0 + 8 * n0;
    // the reads' phase without the index (upload and E10 entries, the clipped copy, the dedup, the compaction to the distinct sequences)
    auto readsPhase = [&](bool lean) {
        const u64 ek = lean ? 0 : ENT + QM + KEEP;
        return std::max({RD + n0 * 28 + AUX + ENT + QM + (lean ? 0 : KEEP), RD + ek + CLIP, RD + ek + 53 * n0, lean ? 0 : RD + ek + RD + 8 * n0});   // lean: no compaction
    };
    const u64 minLanes = std::max({laneNeed(4096, 4096, laneSlotB<LaneA>(), lanePerB<LaneA, false>()),
                                   laneNeed(4096, 4096, laneSlotB<LaneM>(), lanePerB<LaneM, false>()), laneNeed(1024, 1024, laneSlotB<LaneB>(), lanePerB<LaneB, false>())});
    const u64 needNormal = ctx + TAB + SLACK + IDX + std::max(readsPhase(false), RD + ENT + QM + KEEP + minLanes);
    const u64 needLean = ctx + TAB + SLACK + std::max(readsPhase(true), IDX + minLanes + (64ull << 20));   // lean: a chunk's sequences on the device
    const char* lnE = std::getenv("SINGLET_ENGINE_GPU_LEAN");   // E17 debug: 1 forces the lean mode, 0 the normal one (with a budget)
    const bool lean0 = cap && (lnE ? std::string(lnE) == "1" : needNormal > cap);
    // P3-MEM-03: the reads phase by read chunks when even the lean reads phase does not fit (or SINGLET_ENGINE_GPU_RCHUNK=<reads per chunk>);
    // it implies the lean mode for the rest
    const char* rcE = std::getenv("SINGLET_ENGINE_GPU_RCHUNK");
    u64 RCN = rcE ? std::strtoull(rcE, nullptr, 10) : 0;
    const bool RC = n0 > 0 && (RCN > 0 || (cap && lean0 && ctx + TAB + SLACK + readsPhase(true) > cap));
    const bool lean = lean0 || RC;
    // E20 (docs/gpu-engine/fable/E20-lockclip.md) "split": with E8's records (e2e) and a caller that can take the device again (Input::reacquire),
    // the device is released once the lanes are done and taken again for the records' device phase (E8's buildRecords, E10's products, E11's keep):
    // the host lane's wait, the host assembly of the records and readRecord run without it. The lanes do not keep their results on the device
    // (E8's records are uploaded from the host assembly, as E17's lean mode does), E10's entries and masks go to the region at the clip, and
    // nothing but E11's keep and the context stays on the device between the two holds. SINGLET_ENGINE_GPU_SPLIT=0: one hold, as before
    const char* spE = std::getenv("SINGLET_ENGINE_GPU_SPLIT");
    const bool split = in.e2e != nullptr && in.e2e->base != nullptr && in.release && in.reacquire && !(spE && spE[0] == '0');
    {
        char bl[900];
        if (!cap) snprintf(bl, sizeof bl, "engine_gpu: E17 budget: none (SINGLET_ENGINE_GPU_MEM_GB=0): E12's sizes (lane B takes the free device memory less 3 GB)\n");
        else snprintf(bl, sizeof bl, "engine_gpu: E17 budget %.1f GB (%s; device %.1f GB, %.1f GB in use at the start): mode %s; context %.2f GB, index %.2f GB, "
                      "tables %.2f GB, reads %.2f GB, E10 aux %.2f GB, entries+masks %.2f GB, E11 keep %.2f GB, clip %.2f GB; estimates: normal %.1f GB, lean %.1f GB "
                      "(the smallest lanes %.2f GB, slack %.1f GB)\n", b2gb(cap), budDefault ? "(device memory - 2 GB) / SINGLET_ENGINE_GPU_SLOTS" : "SINGLET_ENGINE_GPU_MEM_GB",
                      b2gb(tB0), b2gb(tB0 - fB0), lean ? "lean (index after the dedup, entries to the region at once, keep at the records)" : "normal",
                      b2gb(ctx), b2gb(IDX), b2gb(TAB), b2gb(RD), b2gb(AUX), b2gb(ENT + QM), b2gb(KEEP), b2gb(CLIP), b2gb(needNormal), b2gb(needLean), b2gb(minLanes), b2gb(SLACK));
        out.log += bl;
    }
    auto gbs = [](u64 b) { char x[32]; snprintf(x, sizeof x, "%.2f GB", b2gb(b)); return std::string(x); };
    const std::string budName = "the budget of " + gbs(cap) + (budDefault ? " (the default: (device memory - 2 GB) / SINGLET_ENGINE_GPU_SLOTS)" : " (SINGLET_ENGINE_GPU_MEM_GB)");
    const std::string budHint = "; raise SINGLET_ENGINE_GPU_MEM_GB or lower SINGLET_ENGINE_GPU_SLOTS (SINGLET_ENGINE_GPU_MEM_FALLBACK=cpu: STAR maps every read instead)";
    if (cap && lean && !RC && ctx + TAB + SLACK + readsPhase(true) > cap)
        return failBudget("the sample's reads phase (upload, clip, dedup) needs " + gbs(ctx + TAB + SLACK + readsPhase(true)) + " of device memory, more than "
                          + budName + budHint);
    if (cap && ctx + TAB + SLACK + IDX + minLanes > cap)
        return failBudget("the index (" + gbs(IDX) + ") and the smallest lanes need " + gbs(ctx + TAB + SLACK + IDX + minLanes) + " of device memory, more than "
                          + budName + budHint);
    // the index: allocated here and copied from a second thread while the reads are uploaded, clipped and deduplicated (staged); E17 lean: after the dedup
    auto startIndex = [&]() -> bool {
        if (SHC.on) {   // mapped from the holder: nothing to allocate or copy
            dG1 = (char*) SHC.p[0]; dSA = (u64*) SHC.p[1]; dSAi = (u64*) SHC.p[2];
            D.G = dG1 + PAD; D.SA = dSA; D.SAi = dSAi;
            memMark("index(shared)");
            return true;
        }
        g_bud.at = "index";
        tIdx0 = now();
        if (staged) {
            if (dmal(&dG1, H.nGenome + 2 * PAD + 8) != cudaSuccess || dmal(&dSA, saBytes + 64 + 8) != cudaSuccess
                || dmal(&dSAi, saiBytes + 64 + 8) != cudaSuccess) return fail("out of device memory (index)");
            cudaMemset(dSA, 0, saBytes + 64 + 8); cudaMemset(dSAi, 0, saiBytes + 64 + 8);
            if (cudaDeviceSynchronize() != cudaSuccess) return fail("index memset");
            idxTh = std::thread([&, dG1, dSA, dSAi] {
                cudaSetDevice(device);
                idxOk = SI.copy(dG1, H.G - PAD, H.nGenome + 2 * PAD) && SI.copy(dSA, H.SA, saBytes) && SI.copy(dSAi, H.SAi, saiBytes) && SI.finish();
                tIdxEnd = now();
            });
        } else {
            if (pinIdx) {
                cudaHostRegister((void*)(H.G - PAD), H.nGenome + 2 * PAD, cudaHostRegisterReadOnly);
                cudaHostRegister((void*)H.SA, saBytes, cudaHostRegisterReadOnly);
                cudaHostRegister((void*)H.SAi, saiBytes, cudaHostRegisterReadOnly);
                tPin = now() - tIdx0;
            }
            if (!upload(H.G - PAD, H.nGenome + 2 * PAD, dG1, err)) return fail(err);
            if (!upload((const char*)H.SA, saBytes, *(char**)&dSA, err, 64)) return fail(err);
            if (!upload((const char*)H.SAi, saiBytes, *(char**)&dSAi, err, 64)) return fail(err);
            cudaDeviceSynchronize();
            if (pinIdx) { cudaHostUnregister((void*)(H.G - PAD)); cudaHostUnregister((void*)H.SA); cudaHostUnregister((void*)H.SAi); }
            char l[160]; snprintf(l, sizeof l, "engine_gpu: index upload %.2f s (%s%.2f s to pin)\n", now() - tIdx0, pinIdx ? "pinned, " : "pageable, ", tPin); out.log += l;
        }
        D.G = dG1 + PAD; D.SA = dSA; D.SAi = dSAi;
        memMark("index");
        return true;
    };
    if (!lean && !startIndex()) return false;
    step("index.h2d");
    g_bud.at = "tables";
    u64 *dcs, *dcl, *dcb, *dss, *dse, *dsd, *dsa; uint8_t *dsm, *dsl, *dsr, *dst;
    if (!upload(H.chrStart, H.nChr + 1, dcs, err) || !upload(H.chrLength, H.nChr, dcl, err) || !upload(H.chrBin, H.chrBinN, dcb, err)
        || !upload(H.sjdbStart, H.sjdbN, dss, err) || !upload(H.sjdbEnd, H.sjdbN, dse, err) || !upload(H.sjDstart, H.sjdbN, dsd, err)
        || !upload(H.sjAstart, H.sjdbN, dsa, err) || !upload(H.sjdbMotif, H.sjdbN, dsm, err) || !upload(H.sjdbShiftLeft, H.sjdbN, dsl, err)
        || !upload(H.sjdbShiftRight, H.sjdbN, dsr, err) || !upload(H.sjdbStrand, H.sjdbN, dst, err)) return fail(err);
    D.chrStart = dcs; D.chrLength = dcl; D.chrBin = dcb; D.sjdbStart = dss; D.sjdbEnd = dse; D.sjDstart = dsd; D.sjAstart = dsa;
    D.sjdbMotif = dsm; D.sjdbShiftLeft = dsl; D.sjdbShiftRight = dsr; D.sjdbStrand = dst;
    step("tables.h2d");
    // ---- reads
    nvch.next("mapDistinct: reads");
    u64 n = in.off.size() - 1;
    out.nReads = n;
    char* dSeq = nullptr; u64* dOff = nullptr;
    g_bud.at = "reads";
    if (RC) {}   // P3-MEM-03: by chunks, at the dedup below
    else if (staged) {
        if (dmal(&dSeq, in.seq.size() + 8) != cudaSuccess || dmal(&dOff, (n + 1) * 8 + 8) != cudaSuccess) return fail("out of device memory (reads)");
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
    // E17 lean: the entries and masks are in the region from the clip on (entHost; zeroed for the reads without a record by the host at the
    // records), and the reads' CR/UR are kept for E11 at the records (keepLate)
    bool entHost = false; u64 entOff = 0, qmOff = 0, keepTally = 0;
    const bool keepLate = lean;
    std::string e10Oom;   // IDEAS.md item 11: why E10 turned off for lack of device memory (with in.oomFatal, mapDistinct fails at its end)
    auto e10Off = [&](const std::string& why) {
        if (e10 && e10Oom.empty() && (why.find("out of memory") != std::string::npos || why.find("out of device memory") != std::string::npos)) e10Oom = why;
        if (e10) out.e10.log += "engine_gpu: E10 not used (" + why + "); STAR's loop does the per-read work\n";
        if (!why.empty()) { e9::soloKeepRelease(); out.e10.soloKept = false; g_bud.used -= std::min(g_bud.used, keepTally); keepTally = 0; }   // E11: nothing to count on the device without E10
        e10 = false; cudaGetLastError();
        dfree(dEnt); dfree(dQm); dfree(dAux); dEnt = nullptr; dQm = nullptr; dAux = nullptr;
    };
    // with in.oomFatal an E10 turn-off for lack of device memory fails the call: fail at once, before the lanes, so the retry does not wait
    // for work that is thrown away (item 88: about 30 s of device time per failed attempt)
    auto e10OomNow = [&]() { return in.oomFatal && !e10Oom.empty(); };
    auto e10OomWhy = [&]() { return "E10 ran out of device memory (" + e10Oom + "); the caller runs mapDistinct again"; };
    // a kernel launch's own error (out of resources, or out of memory when the driver cannot grow the local memory the kernel needs) is
    // reported only by cudaGetLastError: name the kernel here, before a later check reads the error as its own (item 88: "entries: out of memory")
    auto launched = [&](const char* what) -> bool {
        const cudaError_t le = cudaGetLastError();
        if (le == cudaSuccess) return true;
        err = std::string(what) + " launch: " + cudaGetErrorString(le);
        return false;
    };
    if (in.e10.on && !e10) out.e10.log += "engine_gpu: E10 not used (no shared region or no host decode)\n";
    if (e10 && !RC) {
        const double ta = now();
        if (dmal(&dEnt, n * sizeof(e6::ReadEntry)) != cudaSuccess || (in.e10.wantMask && dmal(&dQm, n * qmBytes()) != cudaSuccess)
            || dmal(&dAux, in.e10.aux.size() * 8) != cudaSuccess) e10Off("out of device memory");
        else if (staged ? !(SR.copy(dAux, in.e10.aux.data(), in.e10.aux.size() * 8) && SR.finish())
                        : cudaMemcpy(dAux, in.e10.aux.data(), in.e10.aux.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) e10Off("aux upload");
        out.e10.msUpAux = (now() - ta) * 1e3;
        step("e10.aux.h2d");
        if (e10OomNow()) return fail(e10OomWhy());
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
            if (dmal(&dWL, in.e10.nWL * 8) == cudaSuccess && dmal(&dH, in.e10.nWL * 4) == cudaSuccess
                && cudaMemcpy(dWL, in.e10.wl, in.e10.nWL * 8, cudaMemcpyHostToDevice) == cudaSuccess
                && singlet_hostloop_gpu::r11(dAux, in.e10.nw, n, dWL, in.e10.nWL, in.e10.cbL, in.e10.umiL, dH, msr, wr)) {
                out.e10.r11.resize(in.e10.nWL);
                if (cudaMemcpy(out.e10.r11.data(), dH, in.e10.nWL * 4, cudaMemcpyDeviceToHost) != cudaSuccess) out.e10.r11.clear();
                for (uint32_t c : out.e10.r11) out.e10.r11Total += c;
                out.e10.msR11 = msr;
            } else { cudaGetLastError(); out.e10.log += "engine_gpu: E10: R11 histogram not computed (" + wr + ")\n"; }
            dfree(dWL); dfree(dH);
        }
        if (in.e10.soloModel != nullptr && !keepLate) {   // E11: the CR/UR of every read, contiguous, kept for soloRun after STAR's loop
            std::string ws;
            if (!e9::soloKeepReads(dAux, 2 + (u64) in.e10.nw, n, ws)) out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (" + ws + ")\n";
            else { keepTally = budRound(std::max<u64>(1, n) * 8) * 2; g_bud.used += keepTally; g_bud.note(); }   // E17: e9's buffers in the tally
        }
        dfree(dAux); dAux = nullptr;
    };
    // E17 lean: the entries and masks into the region now (as mapDistinct's E10 copy does after the products), off the device
    auto entToRegion = [&]() {
        if (!e10 || entHost) return;
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        const double t0e = now();
        const u64 oEn = RG.take(n * sizeof(e6::ReadEntry)), oQm = dQm ? RG.take(n * qmBytes()) : 0;
        if (!oEn || (dQm && !oQm)) { e10Off("the shared region is full"); return; }
        if (cudaStreamSynchronize(0) != cudaSuccess) { e10Off("entries"); return; }
        const bool cp = g_d2h ? g_d2h->copy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry)) && (!dQm || g_d2h->copy(RG.base + oQm, dQm, n * qmBytes()))
                              : cudaMemcpy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry), cudaMemcpyDeviceToHost) == cudaSuccess
                                && (!dQm || cudaMemcpy(RG.base + oQm, dQm, n * qmBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
        if (!cp) { e10Off("entries copy"); return; }
        dfree(dEnt); dfree(dQm); dEnt = nullptr; dQm = nullptr;
        entHost = true; entOff = oEn; qmOff = oQm;
        out.e10.msD2H = (now() - t0e) * 1e3;
    };
    // clipping: the reads become their clipped sequences (device copy compacted; on the host each read's span is (cs, length))
    singlet_engine_gpu::NoInitVector<u64> hcs, hoffC;   // every element copied from the device
    std::function<void()> clipHostFill;   // E20: set by the clip below and run there, or after the release with the split
    if (in.clip.on && n > 0 && !RC) {
        double tc = now();
        char* dAd = nullptr;
        if (!in.clip.ad3.empty() && !upload(in.clip.ad3.data(), in.clip.ad3.size(), dAd, err)) return fail(err);
        uint16_t *dP5 = nullptr, *dP3 = nullptr;
        if (in.clip.pre && in.clip.dev) {   // E20: STAR's CellRanger4 clip on the device (k_cr4); the 5' clip's info to the host for STAR's own reads
            const double t4 = now();
            const int nq = (int) in.clip.cr4q.size();
            if (nq < 1 || nq > 40) return fail("CellRanger4 clip on the device: a 5' adapter of " + std::to_string(nq) + " bases (1-40)");
            Cr4Dev c4{};
            for (int r = 0; r < nq; r++) c4.q[r] = in.clip.cr4q[r];
            c4.nq = nq; c4.n5 = in.clip.cr4n5; c4.after5 = in.clip.cr4after5; c4.n3 = in.clip.cr4n3; c4.after3 = in.clip.cr4after3;
            uint8_t* dInfo = nullptr;
            g_bud.at = "clip";
            if (dmal(&dP5, n * 2) || dmal(&dP3, n * 2) || dmal(&dInfo, n + 8)) return fail("out of device memory (CellRanger4 clip)");
            const unsigned g4 = (unsigned) ((n + 127) / 128);
            if (nq == 30) k_cr4<30><<<g4, 128>>>(dSeq, dOff, n, c4, dP5, dP3, dInfo);
            else k_cr4<0><<<g4, 128>>>(dSeq, dOff, n, c4, dP5, dP3, dInfo);
            if (!launched("CellRanger4 clip kernel")) return fail(err);
            if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("CellRanger4 clip kernel: ") + cudaGetErrorString(cudaGetLastError()));
            const double tk4 = now() - t4;
            out.cr4info.resize(n);
            if (g_d2h ? !g_d2h->copy(out.cr4info.data(), dInfo, n) : cudaMemcpy(out.cr4info.data(), dInfo, n, cudaMemcpyDeviceToHost) != cudaSuccess)
                return fail("CellRanger4 clip: info copy");
            dfree(dInfo);
            char l4[400];
            if (in.pre5.size() == n && in.pre3.size() == n && in.cr4info.size() == n) {   // SINGLET_ENGINE_GPU_CR4_CHECK=1: the host's clip as well
                std::vector<uint16_t> h5(n), h3(n);
                if (cudaMemcpy(h5.data(), dP5, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(h3.data(), dP3, n * 2, cudaMemcpyDeviceToHost) != cudaSuccess)
                    return fail("CellRanger4 clip: check copy");
                u64 d5 = 0, d3 = 0, dI = 0, first = n;
                for (u64 i = 0; i < n; i++) {
                    const bool a = h5[i] != in.pre5[i], b = h3[i] != in.pre3[i], c = out.cr4info[i] != in.cr4info[i];
                    d5 += a; d3 += b; dI += c;
                    if ((a || b || c) && first == n) first = i;
                }
                snprintf(l4, sizeof l4, "engine_gpu: E20 CellRanger4 clip check: the device's clip of %llu reads against the host's (cr4Clip): %s (5' %llu, 3' %llu, "
                         "info %llu reads differ%s); the host's are used\n", (unsigned long long) n, d5 + d3 + dI == 0 ? "IDENTICAL" : "DIFFERENT",
                         (unsigned long long) d5, (unsigned long long) d3, (unsigned long long) dI,
                         first < n ? (", first read " + std::to_string(first) + ": device " + std::to_string(h5[first]) + "/" + std::to_string(h3[first]) + "/"
                                      + std::to_string(out.cr4info[first]) + ", host " + std::to_string(in.pre5[first]) + "/" + std::to_string(in.pre3[first]) + "/"
                                      + std::to_string(in.cr4info[first])).c_str() : "");
                out.log += l4;
                if (cudaMemcpy(dP5, in.pre5.data(), n * 2, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(dP3, in.pre3.data(), n * 2, cudaMemcpyHostToDevice) != cudaSuccess)
                    return fail("CellRanger4 clip: check upload");
                std::memcpy(out.cr4info.data(), in.cr4info.data(), n);
            }
            snprintf(l4, sizeof l4, "engine_gpu: E20 CellRanger4 clip on the device: %llu reads, adapter %d bases, kernel %.3f s, info to the host %.3f s\n",
                     (unsigned long long) n, nq, tk4, now() - t4 - tk4);
            out.log += l4;
            step("clip.cr4");
        } else if (in.clip.pre) {
            if (in.pre5.size() != n || in.pre3.size() != n) return fail("host clip lengths do not match the reads");
            if (!upload(in.pre5.data(), n, dP5, err) || !upload(in.pre3.data(), n, dP3, err)) return fail(err);
        }
        ClipDev c{in.clip.n5, in.clip.after5, in.clip.n3, in.clip.after3, (uint32_t)in.clip.ad3.size(), in.clip.adMMp, dAd, dP5, dP3};
        u64 *dCs, *dCl, *dOffC; char* dSeqC;
        g_bud.at = "clip";
        if (dmal(&dCs, n * 8) || dmal(&dCl, n * 8) || dmal(&dOffC, (n + 1) * 8)) return fail("out of device memory (clip)");
        k_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, c, dCs, dCl);
        if (!launched("clip kernel")) return fail(err);
        if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("clip kernel: ") + cudaGetErrorString(cudaGetLastError()));   // E12: not the device (the index copy stream runs on)
        step("clip.kernel");
        if (e10) e10Entries(dCs, dCl);   // E10: the original reads and their clip
        if (e10OomNow()) return fail(e10OomWhy());
        cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
        step("e10.entries");
        if (lean || split) { entToRegion(); step("e10.entries.d2h"); }
        g_bud.at = "clip";
        cudaMemset(dOffC, 0, 8);
        thrust::inclusive_scan(TP, thrust::device_ptr<u64>(dCl), thrust::device_ptr<u64>(dCl + n), thrust::device_ptr<u64>(dOffC + 1));
        u64 tot = 0;
        if (cudaMemcpy(&tot, dOffC + n, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("clip copy");
        if (dmal(&dSeqC, tot + 8)) return fail("out of device memory (clipped reads)");
        k_gather_clip<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dCs, dOffC, n, dSeqC);
        if (!launched("clip gather kernel")) return fail(err);
        if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("clip gather kernel: ") + cudaGetErrorString(cudaGetLastError()));   // E12: not the device
        step("clip.gather");
        hcs.resize(n); hoffC.resize(n + 1);
        if (g_d2h ? !(g_d2h->copy(hcs.data(), dCs, n * 8) && g_d2h->copy(hoffC.data(), dOffC, (n + 1) * 8))
                  : (cudaMemcpy(hcs.data(), dCs, n * 8, cudaMemcpyDeviceToHost) != cudaSuccess
                     || cudaMemcpy(hoffC.data(), dOffC, (n + 1) * 8, cudaMemcpyDeviceToHost) != cudaSuccess)) return fail("clip copy");
        step("clip.d2h");
        dfree(dSeq); dfree(dOff); dfree(dCs); dfree(dCl); dfree(dAd); dfree(dP5); dfree(dP3);
        dSeq = dSeqC; dOff = dOffC;
        step("clip.free");
        // the host's per-read clip lengths (read after mapDistinct only); E20 split: after the device is released (the split block below)
        clipHostFill = [&, tc]() {
            const double th = now();
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
            const u64 nClipped = nClip;
            char line[200];
            snprintf(line, sizeof line, "engine_gpu: clipping: %llu of %llu reads clipped, %.2f s%s\n", (unsigned long long)nClipped, (unsigned long long)n,
                     split ? now() - th : now() - tc, split ? " (the host's lengths; E20: after the device was released)" : "");
            out.log += line;
        };
        if (!split) { clipHostFill(); step("clip.host"); }
    }
    if (e10 && !e10Done && !RC) e10Entries(nullptr, nullptr);   // E10: no clipping
    if (e10OomNow()) return fail(e10OomWhy());
    if ((lean || split) && !RC) entToRegion();
    memMark("reads+clip");
    // a read's (clipped) sequence on the host
    auto rseq = [&](u64 i) { return hcs.empty() ? in.seq.data() + in.off[i] : in.seq.data() + hcs[i]; };
    auto rlen = [&](u64 i) { return hcs.empty() ? in.off[i + 1] - in.off[i] : hoffC[i + 1] - hoffC[i]; };
    cudaStreamSynchronize(0);   // E12: not the device (the index copy stream runs on)
    out.secUpload = now() - t0;
    step("clip.rest");
    // per read: the representative read of its distinct sequence (NONE for sequences with N, which STAR maps itself); rep: one read per distinct sequence without N
    std::vector<uint32_t> rep;   // one read per distinct sequence without N
    const uint32_t NONE = 0xFFFFFFFFu;
    singlet_engine_gpu::NoInitVector<uint32_t> readRep(n);
    if (!RC) {
    // ---- dedup
    nvch.next("mapDistinct: dedup");
    t0 = now();
    u64 *dH1, *dH2; uint8_t* dN; uint32_t *dOrder, *dHead; unsigned long long* dColl;
    g_bud.at = "dedup";
    if (dmal(&dH1, n * 8) || dmal(&dH2, n * 8) || dmal(&dN, n) || dmal(&dOrder, n * 4) || dmal(&dHead, n * 4)
        || dmal(&dColl, 8)) return fail("out of device memory (dedup)");
    k_hash<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, n, dH1, dH2, dN);
    thrust::sequence(TP, thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n));
    auto keys = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<u64>(dH1), thrust::device_ptr<u64>(dH2)));
    thrust::sort_by_key(TP, keys, keys + n, thrust::device_ptr<uint32_t>(dOrder));
    cudaMemset(dColl, 0, 8);
    // E12: rep, readRep and the counts on the device (SINGLET_ENGINE_GPU_DEDUP=host, or CFG=old: on the host, as before)
    const char* ddE = std::getenv("SINGLET_ENGINE_GPU_DEDUP");
    const bool dedupDev = NEW0 && !(ddE && std::string(ddE) == "host");
    uint32_t* dHp = nullptr;   // E12: per sorted position its group's first position
    if (dedupDev) {
        if (dmal(&dHp, n * 4)) return fail("out of device memory (dedup)");
        k_heads<<<(unsigned)((n + 255) / 256), 256>>>(dH1, dH2, n, dHead, dHp);
        thrust::inclusive_scan(TP, thrust::device_ptr<uint32_t>(dHp), thrust::device_ptr<uint32_t>(dHp + n), thrust::device_ptr<uint32_t>(dHp), thrust::maximum<uint32_t>());
        k_collide<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dOrder, dHp, n, dColl);
    } else
    k_groups<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dH1, dH2, dOrder, n, dHead, dColl);
    unsigned long long coll = 0;
    if (cudaMemcpy(&coll, dColl, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
    if (coll) return fail("hash collision between different sequences (" + std::to_string(coll) + "): GPU path not used");
    step("dedup.kernels");
    // per read: the representative read of its distinct sequence (NONE for sequences with N, which STAR maps itself)
    // E17: only the representatives' (clipped) sequences stay on the device, at their positions in rep (the lanes and E8's gather address them
    // by position from here on); the copy of every read leaves the device
    auto compact = [&](const uint32_t* dR, u64 nR) -> bool {
        g_bud.at = "compact";
        u64* dOffR = nullptr; char* dSeqR = nullptr;
        if (dmal(&dOffR, (nR + 1) * 8 + 8) != cudaSuccess) return false;
        cudaMemset(dOffR, 0, 8);
        if (nR) {
            k_rep_len<<<(unsigned)((nR + 255) / 256), 256>>>(dOff, dR, nR, dOffR + 1);
            thrust::inclusive_scan(TP, thrust::device_ptr<u64>(dOffR + 1), thrust::device_ptr<u64>(dOffR + 1 + nR), thrust::device_ptr<u64>(dOffR + 1));
        }
        u64 tot = 0;
        if (cudaMemcpy(&tot, dOffR + nR, 8, cudaMemcpyDeviceToHost) != cudaSuccess) return false;
        if (dmal(&dSeqR, tot + 8) != cudaSuccess) return false;
        if (nR) k_rep_copy<<<(unsigned)((nR + 255) / 256), 256>>>(dSeq, dOff, dR, nR, dOffR, dSeqR);
        if (cudaStreamSynchronize(0) != cudaSuccess) return false;   // not the device (the index copy stream runs on)
        dfree(dSeq); dfree(dOff); dSeq = dSeqR; dOff = dOffR;
        return true;
    };
    if (dedupDev) {
        uint32_t* dRR; uint32_t* dRep;
        if (dmal(&dRR, n * 4) || dmal(&dRep, n * 4)) return fail("out of device memory (dedup)");
        const unsigned G = (unsigned)((n + 255) / 256);
        k_readrep<<<G, 256>>>(dOrder, dHp, dN, n, dRR);
        const u64 nDist = thrust::reduce(TP, thrust::device_ptr<uint32_t>(dHead), thrust::device_ptr<uint32_t>(dHead + n), (u64)0);
        const u64 nRep = (u64)(thrust::copy_if(TP, thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + n), thrust::counting_iterator<u64>(0),
                                               thrust::device_ptr<uint32_t>(dRep), RepPred{dHead, dN, dOrder}) - thrust::device_ptr<uint32_t>(dRep));
        out.nDistinct = nDist; out.nWithN = nDist - nRep;
        dfree(dHp);
        step("dedup.dev");
        if (g_d2h ? !g_d2h->copy(readRep.data(), dRR, n * 4) : cudaMemcpy(readRep.data(), dRR, n * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("dedup copy");
        dfree(dRR);
        dfree(dH1); dfree(dH2); dfree(dN); dfree(dOrder); dfree(dHead); dfree(dColl);
        step("dedup.d2h");
        rep.resize(nRep);
        if (nRep) {   // map them in sequence order (first 32 bases), not hash order (below)
            u64* dKey;
            g_bud.at = "order";
            if (dmal(&dKey, nRep * 8)) return fail("out of device memory (order)");
            k_prefix_key<<<(unsigned)((nRep + 255) / 256), 256>>>(dSeq, dOff, dRep, nRep, dKey);
            thrust::stable_sort_by_key(TP, thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + nRep), thrust::device_ptr<uint32_t>(dRep));
            if (g_d2h ? !g_d2h->copy(rep.data(), dRep, nRep * 4) : cudaMemcpy(rep.data(), dRep, nRep * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
            dfree(dKey);
        }
        if (!lean && !compact(dRep, nRep)) return fail("out of device memory (the distinct sequences)");
        dfree(dRep);
    } else {
    std::vector<uint32_t> order(n), head(n); std::vector<uint8_t> hasN(n);
    step("dedup.hostalloc");
    cudaMemcpy(order.data(), dOrder, n * 4, cudaMemcpyDeviceToHost); cudaMemcpy(head.data(), dHead, n * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(hasN.data(), dN, n, cudaMemcpyDeviceToHost);
    step("dedup.d2h");
    dfree(dH1); dfree(dH2); dfree(dN); dfree(dOrder); dfree(dHead); dfree(dColl);
    for (u64 k = 0; k < n; k++) if (head[k]) { out.nDistinct++; if (hasN[order[k]]) out.nWithN++; else rep.push_back(order[k]); }
    { uint32_t cur = NONE; for (u64 k = 0; k < n; k++) { if (head[k]) cur = hasN[order[k]] ? NONE : order[k]; readRep[order[k]] = cur; } }
    step("dedup.host");
    }
    // map them in sequence order (first 32 bases), not hash order: neighbouring threads then search the same SA ranges and loci
    // (2.7x faster on pbmc1k_v3). Each sequence's result does not depend on the order.
    if (!dedupDev) {   // keys (first 32 bases) and the stable sort on the device
        uint32_t* dRep; u64* dKey;
        g_bud.at = "order";
        if (dmal(&dRep, rep.size() * 4 + 4) || dmal(&dKey, rep.size() * 8 + 8)) return fail("out of device memory (order)");
        if (!rep.empty()) {
            cudaMemcpy(dRep, rep.data(), rep.size() * 4, cudaMemcpyHostToDevice);
            k_prefix_key<<<(unsigned)((rep.size() + 255) / 256), 256>>>(dSeq, dOff, dRep, rep.size(), dKey);
            thrust::stable_sort_by_key(TP, thrust::device_ptr<u64>(dKey), thrust::device_ptr<u64>(dKey + rep.size()), thrust::device_ptr<uint32_t>(dRep));
            if (cudaMemcpy(rep.data(), dRep, rep.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) return fail("order copy");
        }
        dfree(dKey);
        if (!lean && !compact(dRep, rep.size())) return fail("out of device memory (the distinct sequences)");
        dfree(dRep);
    }
    } else {
    // ---- P3-MEM-03: the reads phase by read chunks (upload, CellRanger4 and STAR clip, E10's entries and masks into the region, R11, the
    nvch.next("mapDistinct: reads by chunk (P3-MEM-03)");
    // hashes of the clipped reads to the host), then the dedup by hash partition (the sort and the group heads on the device, the byte-for-byte
    // check on the host). Only a chunk of reads, or a partition's keys, is on the device at a time; what follows is E17's lean mode, which
    // needs on the host what this leaves there: hcs / hoffC (the clipped spans), readRep, rep in sequence order, the entries in the region.
    // The records' order can differ from the whole-sample dedup's (ties of the 32-base keys); each record's content does not.
    const double tR0 = now();
    const u64 strideAux = e10 ? 2 + (u64) in.e10.nw : 0;
    const u64 perR = (S0 + n - 1) / std::max<u64>(1, n) + 16 + strideAux * 8 + (e10 ? sizeof(e6::ReadEntry) + (in.e10.wantMask ? qmBytes() : 0) : 0)
                     + (in.clip.on ? 8 * 2 + 5 : 0) + 17;
    auto roomRC = [&]() -> u64 {
        u64 r = cap ? g_bud.room() : ~0ull;
        size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
        r = std::min<u64>(r, f > (2ull << 30) ? f - (2ull << 30) : 0);
        return r > SLACK ? r - SLACK : 0;
    };
    if (RCN == 0) RCN = std::max<u64>(1u << 20, roomRC() / 4 / perR / (1u << 20) * (1u << 20));   // about a quarter of the room per chunk
    RCN = std::min<u64>(RCN, n);
    const u64 nRC = (n + RCN - 1) / RCN;
    u64 maxB = 0;
    for (u64 r0 = 0; r0 < n; r0 += RCN) maxB = std::max<u64>(maxB, in.off[std::min(n, r0 + RCN)] - in.off[r0]);
    g_bud.at = "read chunk";
    struct RcFree { std::vector<void*> p; void all() { for (void* x : p) dfree(x); p.clear(); } ~RcFree() { all(); } } rcf;
    auto al = [&](auto*& p, u64 b) -> bool { if (dmal(&p, b + 8) != cudaSuccess) { cudaGetLastError(); p = nullptr; return false; } rcf.p.push_back((void*) p); return true; };
    auto d2h = [&](void* dst, const void* src, u64 b) -> bool { return g_d2h ? g_d2h->copy(dst, src, b) : cudaMemcpy(dst, src, b, cudaMemcpyDeviceToHost) == cudaSuccess; };
    char* cSeq = nullptr; u64 *cOff = nullptr, *cCs = nullptr, *cCl = nullptr, *cH1 = nullptr, *cH2 = nullptr; uint64_t* cAux = nullptr;
    uint8_t *cN = nullptr, *cInfo = nullptr; uint16_t *cP5 = nullptr, *cP3 = nullptr; char* cAd = nullptr;
    e6::ReadEntry* cEnt = nullptr; uint64_t* cQm = nullptr; uint64_t* cWL = nullptr; uint32_t* cHist = nullptr;
    const bool ccr4 = in.clip.on && in.clip.pre && in.clip.dev, cpre = in.clip.on && in.clip.pre && !in.clip.dev;
    if (!al(cSeq, maxB) || !al(cOff, (RCN + 1) * 8) || !al(cH1, RCN * 8) || !al(cH2, RCN * 8) || !al(cN, RCN)
        || (in.clip.on && (!al(cCs, RCN * 8) || !al(cCl, RCN * 8)))
        || ((ccr4 || cpre) && (!al(cP5, RCN * 2) || !al(cP3, RCN * 2))) || (ccr4 && !al(cInfo, RCN)))
        return failBudget("the reads phase's chunk of " + std::to_string(RCN) + " reads does not fit (" + gbs(g_bud.used) + " in use), within " + budName + budHint);
    if (in.clip.on && !in.clip.ad3.empty()) {
        if (!al(cAd, in.clip.ad3.size()) || cudaMemcpy(cAd, in.clip.ad3.data(), in.clip.ad3.size(), cudaMemcpyHostToDevice) != cudaSuccess) return fail("adapter upload");
    }
    Cr4Dev c4{};
    if (ccr4) {
        const int nq = (int) in.clip.cr4q.size();
        if (nq < 1 || nq > 40) return fail("CellRanger4 clip on the device: a 5' adapter of " + std::to_string(nq) + " bases (1-40)");
        for (int r = 0; r < nq; r++) c4.q[r] = in.clip.cr4q[r];
        c4.nq = nq; c4.n5 = in.clip.cr4n5; c4.after5 = in.clip.cr4after5; c4.n3 = in.clip.cr4n3; c4.after3 = in.clip.cr4after3;
        out.cr4info.resize(n);
    }
    if (cpre && (in.pre5.size() != n || in.pre3.size() != n)) return fail("host clip lengths do not match the reads");
    if (e10) {
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        entOff = RG.take(n * sizeof(e6::ReadEntry)); qmOff = in.e10.wantMask ? RG.take(n * qmBytes()) : 0;
        if (!entOff || (in.e10.wantMask && !qmOff)) e10Off("the shared region is full");
        else if (!al(cAux, RCN * strideAux * 8) || !al(cEnt, RCN * sizeof(e6::ReadEntry)) || (in.e10.wantMask && !al(cQm, RCN * qmBytes())))
            e10Off("out of device memory (read chunk)");
        if (e10OomNow()) return fail(e10OomWhy());
    }
    bool r11On = e10 && in.e10.wl != nullptr && in.e10.nWL > 0;
    std::vector<uint32_t> r11t;
    if (r11On) {
        if (!al(cWL, in.e10.nWL * 8) || !al(cHist, in.e10.nWL * 4) || cudaMemcpy(cWL, in.e10.wl, in.e10.nWL * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaGetLastError(); r11On = false; out.e10.log += "engine_gpu: E10: R11 histogram not computed (read chunks: device memory)\n";
        } else { out.e10.r11.assign(in.e10.nWL, 0); r11t.resize(in.e10.nWL); }
    }
    singlet_engine_gpu::NoInitVector<u64> hH1(n), hH2(n); singlet_engine_gpu::NoInitVector<uint8_t> hN(n);
    if (in.clip.on) { hcs.resize(n); hoffC.resize(n + 1); hoffC[0] = 0; }
    singlet_engine_gpu::NoInitVector<u64> ho(RCN + 1), hcl(in.clip.on ? RCN : 0);
    u64 nClipped = 0; float msEnt = 0, msR = 0;
    double tUp = 0, tKer = 0, tD2H = 0;
    for (u64 r0 = 0; r0 < n; r0 += RCN) {
        const u64 m = std::min(RCN, n - r0), b0 = in.off[r0], nb = in.off[r0 + m] - b0;
        const unsigned G = (unsigned) ((m + 255) / 256);
        double ta = now();
        parallelFor(m + 1, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) ho[k] = in.off[r0 + k] - b0; });
        const bool up = staged ? SR.copy(cSeq, in.seq.data() + b0, nb) && SR.copy(cOff, ho.data(), (m + 1) * 8)
                                 && (!e10 || SR.copy(cAux, in.e10.aux.data() + r0 * strideAux, m * strideAux * 8)) && SR.finish()
                               : cudaMemcpy(cSeq, in.seq.data() + b0, nb, cudaMemcpyHostToDevice) == cudaSuccess
                                 && cudaMemcpy(cOff, ho.data(), (m + 1) * 8, cudaMemcpyHostToDevice) == cudaSuccess
                                 && (!e10 || cudaMemcpy(cAux, in.e10.aux.data() + r0 * strideAux, m * strideAux * 8, cudaMemcpyHostToDevice) == cudaSuccess);
        if (!up) return fail("reads upload (read chunk at " + std::to_string(r0) + ")");
        if (cpre && (cudaMemcpy(cP5, in.pre5.data() + r0, m * 2, cudaMemcpyHostToDevice) != cudaSuccess
                     || cudaMemcpy(cP3, in.pre3.data() + r0, m * 2, cudaMemcpyHostToDevice) != cudaSuccess)) return fail("host clip lengths upload");
        tUp += now() - ta; ta = now();
        if (ccr4) {
            const unsigned g4 = (unsigned) ((m + 127) / 128);
            if (c4.nq == 30) k_cr4<30><<<g4, 128>>>(cSeq, cOff, m, c4, cP5, cP3, cInfo);
            else k_cr4<0><<<g4, 128>>>(cSeq, cOff, m, c4, cP5, cP3, cInfo);
            if (!launched("CellRanger4 clip kernel")) return fail(err);
            if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("CellRanger4 clip kernel: ") + cudaGetErrorString(cudaGetLastError()));
            if (!d2h(out.cr4info.data() + r0, cInfo, m)) return fail("CellRanger4 clip: info copy");
        }
        if (in.clip.on) {
            ClipDev c{in.clip.n5, in.clip.after5, in.clip.n3, in.clip.after3, (uint32_t) in.clip.ad3.size(), in.clip.adMMp, cAd, cP5, cP3};
            k_clip<<<G, 256>>>(cSeq, cOff, m, c, cCs, cCl);
            if (!launched("clip kernel")) return fail(err);
            if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("clip kernel: ") + cudaGetErrorString(cudaGetLastError()));
        }
        if (e10) {   // E10: the original reads and their clip; i0 = r0 keeps the hook's sample of the whole-sample run
            float ms = 0; std::string w;
            if (!singlet_hostloop_gpu::entries(cSeq, (const uint64_t*) cOff, (const uint64_t*) cCs, (const uint64_t*) cCl, m, cAux, in.e10.nw, in.e10.hook,
                                               cEnt, cQm, ms, w, r0)) { e10Off("entries: " + w); if (e10OomNow()) return fail(e10OomWhy()); }
            else {
                msEnt += ms;
                singlet_engine_gpu::E2ERegion& RG = *in.e2e;
                if (!d2h(RG.base + entOff + r0 * sizeof(e6::ReadEntry), cEnt, m * sizeof(e6::ReadEntry)) || (cQm && !d2h(RG.base + qmOff + r0 * qmBytes(), cQm, m * qmBytes())))
                    e10Off("entries copy");
            }
            if (e10 && r11On) {
                float mr = 0; std::string wr;
                if (singlet_hostloop_gpu::r11(cAux, in.e10.nw, m, cWL, in.e10.nWL, in.e10.cbL, in.e10.umiL, cHist, mr, wr) && d2h(r11t.data(), cHist, in.e10.nWL * 4)) {
                    msR += mr;
                    parallelFor(in.e10.nWL, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) out.e10.r11[k] += r11t[k]; });
                } else { cudaGetLastError(); r11On = false; out.e10.r11.clear(); out.e10.log += "engine_gpu: E10: R11 histogram not computed (" + wr + ")\n"; }
            }
        }
        k_hash_span<<<G, 256>>>(cSeq, cOff, in.clip.on ? cCs : nullptr, cCl, m, cH1, cH2, cN);
        if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("hash kernel: ") + cudaGetErrorString(cudaGetLastError()));
        tKer += now() - ta; ta = now();
        if (!d2h(hH1.data() + r0, cH1, m * 8) || !d2h(hH2.data() + r0, cH2, m * 8) || !d2h(hN.data() + r0, cN, m)
            || (in.clip.on && (!d2h(hcs.data() + r0, cCs, m * 8) || !d2h(hcl.data(), cCl, m * 8)))) return fail("read chunk copy");
        if (in.clip.on) {
            parallelFor(m, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) hcs[r0 + k] += b0; });
            for (u64 k = 0; k < m; k++) { hoffC[r0 + k + 1] = hoffC[r0 + k] + hcl[k]; nClipped += hcl[k] != ho[k + 1] - ho[k]; }
        }
        tD2H += now() - ta;
    }
    if (e10) {
        entHost = true; e10Done = true; out.e10.msEntries = msEnt;
        if (r11On) { for (uint32_t c : out.e10.r11) out.e10.r11Total += c; out.e10.msR11 = msR; }
    }
    rcf.all();
    decltype(ho)().swap(ho); decltype(hcl)().swap(hcl);
    if (in.clip.on) {
        clipHostFill = [&, tR0, nClipped]() {
            const double th = now();
            out.clip5.resize(n); out.clip3.resize(n);
            parallelFor(n, hostThreads, [&](u64 lo, u64 hi) {
                for (u64 i = lo; i < hi; i++) {
                    const u64 L0 = in.off[i + 1] - in.off[i], L = hoffC[i + 1] - hoffC[i], c5 = hcs[i] - in.off[i];
                    out.clip5[i] = (uint16_t) std::min<u64>(c5, 65535); out.clip3[i] = (uint16_t) std::min<u64>(L0 - L - c5, 65535);
                }
            });
            char line[200];
            snprintf(line, sizeof line, "engine_gpu: clipping: %llu of %llu reads clipped, %.2f s%s\n", (unsigned long long) nClipped, (unsigned long long) n,
                     split ? now() - th : now() - tR0, split ? " (the host's lengths; E20: after the device was released)" : "");
            out.log += line;
        };
        if (!split) clipHostFill();
    }
    out.secUpload = now() - tR0;
    step("rc.reads");
    // ---- the dedup by hash partition: partition p holds the reads whose h1 falls in [p, p+1) * 2^64 / NP, in read order
    nvch.next("mapDistinct: dedup by partition");
    const double tD0 = now();
    t0 = tD0;
    const char* rpE = std::getenv("SINGLET_ENGINE_GPU_RPARTS");
    const u64 roomD = std::max<u64>(1, roomRC());
    const u64 NP = rpE ? std::max<u64>(1, std::strtoull(rpE, nullptr, 10)) : std::max<u64>(1, (n * 96 + roomD - 1) / roomD);   // ~96 bytes per read of a partition
    auto partOf = [NP](u64 h) { return (u64) (((unsigned __int128) h * NP) >> 64); };
    const u64 nT = (u64) std::max(1, std::min(hostThreads, 64));
    std::vector<u64> pcnt(nT * NP, 0), pBeg(NP + 1, 0), pPos(nT * NP);
    auto thr = [&](auto fn) {   // fn(t, lo, hi) on nT fixed contiguous ranges of the reads (the same ranges in every pass)
        std::vector<std::thread> th;
        for (u64 t = 0; t < nT; t++) th.emplace_back([&, t]() { fn(t, n * t / nT, n * (t + 1) / nT); });
        for (auto& x : th) x.join();
    };
    thr([&](u64 t, u64 lo, u64 hi) { u64* c = pcnt.data() + t * NP; for (u64 i = lo; i < hi; i++) c[partOf(hH1[i])]++; });
    for (u64 p = 0, o = 0; p < NP; p++) {
        pBeg[p] = o;
        for (u64 t = 0; t < nT; t++) { pPos[t * NP + p] = o; o += pcnt[t * NP + p]; }
        pBeg[p + 1] = o;
    }
    singlet_engine_gpu::NoInitVector<uint32_t> Ip(n);
    thr([&](u64 t, u64 lo, u64 hi) { u64* q = pPos.data() + t * NP; for (u64 i = lo; i < hi; i++) Ip[q[partOf(hH1[i])]++] = (uint32_t) i; });
    u64 maxP = 0;
    for (u64 p = 0; p < NP; p++) maxP = std::max<u64>(maxP, pBeg[p + 1] - pBeg[p]);
    g_bud.at = "dedup";
    u64 *pk1 = nullptr, *pk2 = nullptr; uint32_t *pOrd = nullptr, *pHead = nullptr, *pHp = nullptr, *pGh = nullptr;
    if (!al(pk1, maxP * 8) || !al(pk2, maxP * 8) || !al(pOrd, maxP * 4) || !al(pHead, maxP * 4) || !al(pHp, maxP * 4) || !al(pGh, maxP * 4))
        return failBudget("the dedup's largest hash partition (" + std::to_string(maxP) + " reads of " + std::to_string(NP) + " partitions) does not fit ("
                          + gbs(g_bud.used) + " in use), within " + budName + budHint);
    singlet_engine_gpu::NoInitVector<u64> s1(maxP), s2(maxP); singlet_engine_gpu::NoInitVector<uint32_t> ghH(maxP);
    std::atomic<u64> coll{0};
    u64 nDist = 0, nWN = 0;
    std::vector<std::vector<uint32_t>> repT(nT);
    double tSort = 0, tHostD = 0;
    for (u64 p = 0; p < NP; p++) {
        const u64 m = pBeg[p + 1] - pBeg[p];
        if (m == 0) continue;
        const uint32_t* I = Ip.data() + pBeg[p];
        double ta = now();
        parallelFor(m, hostThreads, [&](u64 lo, u64 hi) { for (u64 j = lo; j < hi; j++) { s1[j] = hH1[I[j]]; s2[j] = hH2[I[j]]; } });
        const bool up = staged ? SR.copy(pk1, s1.data(), m * 8) && SR.copy(pk2, s2.data(), m * 8) && SR.finish()
                               : cudaMemcpy(pk1, s1.data(), m * 8, cudaMemcpyHostToDevice) == cudaSuccess && cudaMemcpy(pk2, s2.data(), m * 8, cudaMemcpyHostToDevice) == cudaSuccess;
        if (!up) return fail("dedup upload");
        try {
            thrust::sequence(TP, thrust::device_ptr<uint32_t>(pOrd), thrust::device_ptr<uint32_t>(pOrd + m));
            auto keys = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<u64>(pk1), thrust::device_ptr<u64>(pk2)));
            thrust::sort_by_key(TP, keys, keys + m, thrust::device_ptr<uint32_t>(pOrd));
            const unsigned G = (unsigned) ((m + 255) / 256);
            k_heads<<<G, 256>>>(pk1, pk2, m, pHead, pHp);
            thrust::inclusive_scan(TP, thrust::device_ptr<uint32_t>(pHp), thrust::device_ptr<uint32_t>(pHp + m), thrust::device_ptr<uint32_t>(pHp), thrust::maximum<uint32_t>());
            k_group_head<<<G, 256>>>(pOrd, pHp, m, pGh);
        } catch (const std::exception& e) {
            return failBudget(std::string("the dedup's sort of a hash partition (") + std::to_string(m) + " reads): " + e.what() + ", within " + budName + budHint);
        }
        if (cudaStreamSynchronize(0) != cudaSuccess) return fail(std::string("dedup kernels: ") + cudaGetErrorString(cudaGetLastError()));
        if (!d2h(ghH.data(), pGh, m * 4)) return fail("dedup copy");
        tSort += now() - ta; ta = now();
        std::vector<u64> dT(nT, 0), nT1(nT, 0), cT(nT, 0);
        std::vector<std::thread> th;
        for (u64 t = 0; t < nT; t++) th.emplace_back([&, t]() {
            for (u64 j = m * t / nT, e = m * (t + 1) / nT; j < e; j++) {
                const uint32_t h = ghH[j], gj = I[j], gH = I[h];
                if (h != j) {
                    const u64 L = rlen(gj);
                    if (L != rlen(gH) || std::memcmp(rseq(gj), rseq(gH), L) != 0) cT[t]++;
                    readRep[gj] = hN[gH] ? NONE : gH;
                } else {
                    dT[t]++;
                    if (hN[gj]) { nT1[t]++; readRep[gj] = NONE; } else { readRep[gj] = gj; repT[t].push_back(gj); }
                }
            }
        });
        for (auto& x : th) x.join();
        for (u64 t = 0; t < nT; t++) { nDist += dT[t]; nWN += nT1[t]; coll += cT[t]; rep.insert(rep.end(), repT[t].begin(), repT[t].end()); repT[t].clear(); }
        tHostD += now() - ta;
    }
    rcf.all();
    decltype(hH1)().swap(hH1); decltype(hH2)().swap(hH2); decltype(hN)().swap(hN); decltype(Ip)().swap(Ip);
    decltype(s1)().swap(s1); decltype(s2)().swap(s2); decltype(ghH)().swap(ghH);
    if (coll) return fail("hash collision between different sequences (" + std::to_string(coll.load()) + "): GPU path not used");
    out.nDistinct = nDist; out.nWithN = nWN;
    // map them in sequence order (first 32 bases, as k_prefix_key), stable
    const double tO0 = now();
    const u64 nRep = rep.size();
    singlet_engine_gpu::NoInitVector<u64> pkey(nRep);
    parallelFor(nRep, hostThreads, [&](u64 lo, u64 hi) {
        for (u64 k = lo; k < hi; k++) {
            const u64 i = rep[k], L = rlen(i); const char* s = rseq(i); u64 kk = 0;
            for (u64 t = 0; t < 32; t++) kk = (kk << 2) | (t < L ? (u64) (uint8_t) s[t] : 0);
            pkey[k] = kk;
        }
    });
    bool sorted = false;
    if (nRep) {
        u64* oKey = nullptr; uint32_t* oRep = nullptr;
        g_bud.at = "order";
        if (al(oKey, nRep * 8) && al(oRep, nRep * 4) && cudaMemcpy(oKey, pkey.data(), nRep * 8, cudaMemcpyHostToDevice) == cudaSuccess
            && cudaMemcpy(oRep, rep.data(), nRep * 4, cudaMemcpyHostToDevice) == cudaSuccess) {
            try {
                thrust::stable_sort_by_key(TP, thrust::device_ptr<u64>(oKey), thrust::device_ptr<u64>(oKey + nRep), thrust::device_ptr<uint32_t>(oRep));
                sorted = cudaStreamSynchronize(0) == cudaSuccess && d2h(rep.data(), oRep, nRep * 4);
            } catch (const std::exception&) { cudaGetLastError(); }
        }
        rcf.all();
        if (!sorted) {   // the host's stable sort (the device had no room)
            std::vector<std::pair<u64, uint32_t>> kv(nRep);
            for (u64 k = 0; k < nRep; k++) kv[k] = {pkey[k], rep[k]};
            std::stable_sort(kv.begin(), kv.end(), [](const std::pair<u64, uint32_t>& a, const std::pair<u64, uint32_t>& b) { return a.first < b.first; });
            for (u64 k = 0; k < nRep; k++) rep[k] = kv[k].second;
        }
    }
    decltype(pkey)().swap(pkey);
    {
        char l[600];
        snprintf(l, sizeof l, "engine_gpu: P3-MEM-03 reads phase in %llu read chunks of %llu: upload %.2f s, kernels %.2f s, to the host %.2f s; dedup in %llu hash "
                 "partitions (largest %llu reads): sort %.2f s, groups on the host %.2f s, %llu distinct (%llu with N), 0 collisions; order %.2f s (%s)\n",
                 (unsigned long long) nRC, (unsigned long long) RCN, tUp, tKer, tD2H, (unsigned long long) NP, (unsigned long long) maxP, tSort, tHostD,
                 (unsigned long long) nDist, (unsigned long long) nWN, now() - tO0, sorted || nRep == 0 ? "device" : "host");
        out.log += l;
    }
    step("rc.dedup");
    }
    // E17 lean: each chunk's sequences are uploaded with it (gpuPart), from the representatives' (clipped) sequences compacted here on the host in
    // rep order (a chunk is a contiguous range of rep: one slice each)
    singlet_engine_gpu::NoInitVector<char> leanSeq; singlet_engine_gpu::NoInitVector<u64> leanOff;
    if (lean) {
        dfree(dSeq); dfree(dOff); dSeq = nullptr; dOff = nullptr;
        const double tu = now();
        const u64 nR = rep.size();
        leanOff.resize(nR + 1); leanOff[0] = 0;
        parallelFor(nR, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) leanOff[k + 1] = rlen(rep[k]); });
        for (u64 k = 0; k < nR; k++) leanOff[k + 1] += leanOff[k];
        leanSeq.resize(leanOff[nR] + 8);
        std::memset(leanSeq.data() + leanOff[nR], 0, 8);
        parallelFor(nR, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) std::memcpy(leanSeq.data() + leanOff[k], rseq(rep[k]), leanOff[k + 1] - leanOff[k]); });
        stepAdd("lean.seq.host", now() - tu);
    }
    out.secDedup = now() - t0;
    step("dedup.order");
    memMark("dedup");
    if (lean) {   // E17: the index now, next to the distinct sequences only
        if (cap && g_bud.used + IDX + minLanes + SLACK > cap)
            return failBudget("the index (" + gbs(IDX) + "), the " + std::to_string(rep.size()) + " distinct sequences (" + gbs(g_bud.used - ctx) + " with what stays on the device) "
                              "and the smallest lanes need " + gbs(g_bud.used + IDX + minLanes + SLACK) + " of device memory, more than " + budName + budHint);
        if (!startIndex()) return false;
    }
    if (idxTh.joinable()) {   // E12: the index, uploaded meanwhile from the other thread
        idxTh.join();
        if (!idxOk) return fail("index upload");
        char l[400];
        snprintf(l, sizeof l, "engine_gpu: index upload %.2f s (staged %s: %.1f GB, %.1f GB/s; waited %.2f s for it); "
                 "reads and aux staged: %.1f GB, %.1f GB/s\n", tIdxEnd - tIdx0, lean ? "after the dedup (E17 lean)" : "on a second thread during the reads, clip and dedup",
                 SI.bytes / 1e9, SI.bytes / 1e9 / std::max(1e-9, SI.sec), now() - tS, SR.bytes / 1e9, SR.bytes / 1e9 / std::max(1e-9, SR.sec));
        out.log += l;
        SI.release();
        if (!lean) SR.release();   // E17 lean: the chunks' sequences are staged through it (gpuPart)
    }
    step("index.wait");
    memMark("index+distinct");
    std::vector<uint32_t> recOfPos(rep.size(), NONE);   // rep position -> output record (NONE: not mapped, STAR maps it)
    // ---- map in chunks of distinct sequences (device memory for the lanes' outputs is per chunk)
    nvch.next("mapDistinct: map chunks");
    u64 nGpuMapped = 0;
    // a chunk of distinct sequences in two halves: the device lanes (gpuPart, under the caller's device lock) and the host lane and records
    // (hostPart, after in.release(): they hold the GPU for nothing)
    struct ChunkRes { std::vector<uint32_t> repc; Lane A, M, B; std::vector<uint8_t> lane; std::vector<uint32_t> lpos, repH, posH; u64 c0 = 0, ci = 0;
                      std::vector<uint8_t> hBlob; std::vector<u64> hOff;    // E8: the host lane's results, packed (k of repH at hOff[k])
                      std::vector<std::vector<uint8_t>> hout; std::vector<uint8_t> hst;   // the host lane's results and statuses (k of repH)
                      std::vector<uint8_t> okv; std::vector<u64> sOff, bOff; std::vector<uint32_t> rank; };   // E12: per sequence: kept, prefix sums
    std::vector<ChunkRes> chunks;
    // E12: the lanes' configuration. Default: the m5 harness's best exact configuration (lane A: 524288 reads per batch, phase 3
    // under 64 registers; lane M: 393216, warp per read, the batch by window count; lane B: 131072 or as many as the free device
    // memory holds). SINGLET_ENGINE_GPU_CFG=old: the engine before E12 (k_p1 / k_p3 thread per read; 262144 / 262144 / 65536).
    // SINGLET_ENGINE_GPU_NB / _NBM / _NBB set the batches of lanes A / M / B; SINGLET_ENGINE_GPU_WARPM=0: lane M thread per read.
    // E17: with a budget, the chunk and lane A's batch come from the room the budget leaves after the index and the distinct sequences, lanes M
    // and B's batches from the room when they start (their read counts known), a lane runs in parts when even its smallest batch leaves no room
    // for all its result slots, and lane M runs thread per read when its warp lane's stack does not fit; SINGLET_ENGINE_GPU_CHUNK and the
    // switches above still set their values
    const bool NEW = NEW0;
    const char* wmE = std::getenv("SINGLET_ENGINE_GPU_WARPM");
    // the warp lane's window loop must never reach STAR's trNtotal break (every window is then stitched, as in STAR)
    const bool warpM = NEW && !(wmE && std::string(wmE) == "0")
        && (u64)LaneM::NW * std::min<u64>((u64)LWM::kNWTR, P.alignTranscriptsPerWindowNmax) + P.alignTranscriptsPerWindowNmax < P.alignTranscriptsPerReadNmax;
    auto envU = [](const char* k, u64 d, u64 lo) { const char* v = std::getenv(k); return v ? std::max<u64>(lo, std::strtoull(v, nullptr, 10)) : d; };
    const bool nbAset = std::getenv("SINGLET_ENGINE_GPU_NB") != nullptr, nbMset = std::getenv("SINGLET_ENGINE_GPU_NBM") != nullptr;
    u64 nbA = envU("SINGLET_ENGINE_GPU_NB", NEW ? 524288 : 262144, 4096);
    const u64 nbM = envU("SINGLET_ENGINE_GPU_NBM", NEW ? 393216 : 262144, 4096);
    const bool nbBset = std::getenv("SINGLET_ENGINE_GPU_NBB") != nullptr;
    const u64 nbB0 = envU("SINGLET_ENGINE_GPU_NBB", NEW ? 131072 : 65536, 1024);
    // heavy-read split (gpu/bold-split, redeveloped): SINGLET_ENGINE_GPU_STEPS_B sets lane B's stitching step budget (default 8192; a read over it
    // ends Overflow and the host pool maps it with STAR's limits, so results do not change). "auto": a controller moves the budget between
    // batches (start 2048, bounds 64..8192): idle pool threads with an empty queue halve it, a queue the pool cannot drain before lane B ends doubles it
    const char* sbE = std::getenv("SINGLET_ENGINE_GPU_STEPS_B");
    const bool autoReq = sbE && std::string(sbE) == "auto";
    // default 8192 (was 2048): on pbmc1k_v3 the host pool then maps 177,686 reads instead of 198,643 at 45 instead of 65 core-s per run,
    // and lane B's kernels take about the same time (most of the rest overflow a capacity, not the budget; 65536 changes nothing more)
    const u64 stepsB = autoReq ? 2048 : envU("SINGLET_ENGINE_GPU_STEPS_B", 8192, 1);
    struct AutoB { std::atomic<u64> budget{2048}; double t = 0, batch = 0; std::string log; } autoB;
    // sequences per chunk: the lanes' device buffers scale with it (SINGLET_ENGINE_GPU_CHUNK, default 16M; E17: smaller when the budget needs it)
    const char* ceE = std::getenv("SINGLET_ENGINE_GPU_CHUNK");
    u64 CHUNK = ceE ? std::max<u64>(1u << 18, std::strtoull(ceE, nullptr, 10)) : 16u << 20;
    const u64 nRep = rep.size();
    constexpr u64 slotA = laneSlotB<LaneA>(), slotM = laneSlotB<LaneM>(), slotB = laneSlotB<LaneB>();
    constexpr u64 perA = lanePerB<LaneA, false>(), perMw = lanePerB<LaneM, true>(), perMt = lanePerB<LaneM, false>(), perB0 = lanePerB<LaneB, false>();
    // gpu/bold-kern2 (IDEAS item 8): lane B on LaneBc (aligns in a pool, 25 bytes per read of its bookkeeping, g_waSlice aligns of pool per read).
    // On by default since item 22 (A/B on pbmc1k_v3, +9 % pairs/s at flat core-s/M); SINGLET_BOLD_WSHRINK=0 turns it off.
    const char* wsV = std::getenv("SINGLET_BOLD_WSHRINK");
    const bool wshrink = NEW && !(wsV && wsV[0] == '0');
    if (const char* sbs = std::getenv("SINGLET_B_SORT")) g_bSort = std::atoi(sbs);
    if (const char* wsE = std::getenv("SINGLET_BOLD_WASLICE")) g_waSlice = (int)std::max<u64>(8, std::strtoull(wsE, nullptr, 10));
    const u64 perB = wshrink ? lanePerB<LaneBc, false>() + 25 + (u64)g_waSlice * WA_SIZE * sizeof(m5::uint) : perB0;
    const u64 RES = 112;   // E8: per sequence its packed result and offsets kept on the device for the records' gather (~70 bytes of result on pbmc1k_v3)
    bool resident = in.e2e != nullptr && in.e2e->base != nullptr && !split && !lean;   // P3-MEM-03: lean (also without a budget) keeps no results   // E8: the lanes keep their packed results on the device (E17 lean: not)
    bool allRes = true;     // E17: every lane's results stayed on the device (none ran in parts)
    bool budFail = false;   // E17: a lane did not fit the budget (err says which)
    auto roomNow = [&]() -> u64 {   // E17: the budget's room now (with the default budget, also no more than the free device memory less 2 GB)
        u64 r = g_bud.room();
        if (budDefault) { size_t f = 0, t = 0; cudaMemGetInfo(&f, &t); r = std::min<u64>(r, f > (2ull << 30) ? f - (2ull << 30) : 0); }
        return r;
    };
    // E17: lane M's warp per read keeps its stitching context in the stack of every thread the device can hold (the driver reserves it at once)
    u64 stackNeed = 0;
    if (warpM) {
        int dv = 0, nSm = 0, mt = 0; size_t st0 = 0;
        cudaGetDevice(&dv); cudaDeviceGetAttribute(&nSm, cudaDevAttrMultiProcessorCount, dv); cudaDeviceGetAttribute(&mt, cudaDevAttrMaxThreadsPerMultiProcessor, dv);
        cudaDeviceGetLimit(&st0, cudaLimitStackSize);
        const u64 st1 = std::max<size_t>(16 * 1024, sizeof(LWM) + 8192);
        stackNeed = st1 > st0 ? (st1 - st0) * (u64)nSm * (u64)mt : 0;
    }
    std::string budLog, planLog;
    u64 nbMlog = nbM, nbBlog = nbB0; bool wMlog = warpM;
    if (cap) {
        if (lean) resident = false;
        const u64 room = roomNow(), a = room > SLACK ? room - SLACK : 0;
        if (resident && a < nRep * RES + 4 * minLanes) resident = false;
        const u64 avail0 = resident ? a - nRep * RES : a;
        if (avail0 < minLanes)
            return failBudget("the lanes have " + gbs(avail0) + " of device memory left after the index and the distinct sequences (" + gbs(g_bud.used) + " in use), less than the "
                              + gbs(minLanes) + " of their smallest batches, within " + budName + budHint);
        const u64 m0 = std::min<u64>(CHUNK, std::max<u64>(1, nRep));
        if (!ceE && laneNeed(m0, nbA, slotA, perA) > avail0) {   // a smaller chunk: about 15% of the room for lane A's result slots, then balanced
            const u64 sl = slotA + (lean ? S0 / std::max<u64>(1, n0) + 9 : 0);   // lean: the chunk's sequences too
            CHUNK = std::min<u64>(CHUNK, std::max<u64>(1u << 18, (u64)((double)avail0 * 0.15) / sl / 65536 * 65536));
            const u64 k = std::max<u64>(1, (nRep + CHUNK - 1) / CHUNK);
            CHUNK = std::max<u64>(1u << 18, ((nRep + k - 1) / k + 4095) / 4096 * 4096);
        }
        u64 nb = nbA, part = 0;
        if (!nbAset && lanePlan(std::min<u64>(CHUNK, std::max<u64>(1, nRep)), avail0, slotA, perA, nbA, 4096, 4096, nb, part)) nbA = nb;
        char b[400];
        snprintf(b, sizeof b, "room %s after the index and the distinct sequences (%s in use), %s for the lanes (%s), lane M's warp stack %s",
                 gbs(room).c_str(), gbs(g_bud.used).c_str(), gbs(avail0).c_str(), resident ? "less the results kept for E8's gather" : "results not kept on the device",
                 gbs(stackNeed).c_str());
        budLog = b;
    }
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
    g_laneLog.clear();
    // gpu/bold-kern2: the chunk whose lane B is (or just was) on the device: what its entry in chunks needs besides B's result
    struct PendB { Lane A, M, B; std::vector<uint32_t> repc, repB, posB, posM, ixB; u64 c0 = 0, ci = 0, nbB = 0, nPB = 1, qn = 0, nbAc = 0, nPA = 1, nbMc = 0, nPM = 1, nRepM = 0, nRepc = 0;
                   double tc0 = 0; bool ok = true, wM = false; std::string err; BatchFn onB; };
    std::unique_ptr<PendB> pendB;
    std::function<bool()> drainB;
    auto gpuPart = [&](const std::vector<uint32_t>& repc, u64 c0) -> bool {
        // ---- lanes
        SINGLET_NVTX("mapDistinct: lanes of one chunk (GPU)");
        singlet::ThreadNameScope tnLanes("gpu-lanes");
        double tc0 = now();
        Lane A, M, B;
        memMark("before-laneA");
        // E17: the room for this chunk's lanes (less, while the lanes keep their results on the device, what this and the later chunks' results take)
        auto availNow = [&]() -> u64 { const u64 resv = resident ? (nRep - c0) * RES : 0, r = roomNow(); return r > SLACK + resv ? r - SLACK - resv : 0; };
        auto noFit = [&](const char* ln, u64 m, u64 av) {
            err = "lane " + std::string(ln) + " (" + std::to_string(m) + " reads of chunk " + std::to_string(c0 / CHUNK) + ") does not fit: " + gbs(av)
                  + " of device memory left for it (" + gbs(g_bud.used) + " in use), within " + budName + budHint;
            budFail = true;
            return false;
        };
        // E17: the device holds the distinct sequences by position (compacted after the dedup): ixA / ixM / ixB, the reads repc / repM / repB (host)
        // E17 lean: only this chunk's sequences on the device (from the host's clipped reads, the bytes the device clipped), by position in the chunk
        const char* dS = dSeq; const u64* dO = dOff; char* dSc = nullptr; u64* dOc = nullptr;
        struct ChunkFree { char*& s; u64*& o; ~ChunkFree() { dfree(s); dfree(o); } } chunkFree{dSc, dOc};
        const u64 base = lean ? 0 : c0;
        if (lean) {
            const double tu = now();
            const u64 m = repc.size(), o0 = leanOff[c0], nb = leanOff[c0 + m] - o0 + 8;   // the bytes after the chunk's last sequence too, as in rep's compaction
            singlet_engine_gpu::NoInitVector<u64> ho(m + 1);
            parallelFor(m + 1, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) ho[k] = leanOff[c0 + k] - o0; });
            g_bud.at = "chunk sequences";
            if (dmal(&dSc, nb) != cudaSuccess || dmal(&dOc, ho.size() * 8 + 8) != cudaSuccess) { err = "out of device memory (the chunk's sequences)"; return false; }
            if (staged ? !(SR.copy(dSc, leanSeq.data() + o0, nb) && SR.copy(dOc, ho.data(), ho.size() * 8) && SR.finish())
                       : (cudaMemcpy(dSc, leanSeq.data() + o0, nb, cudaMemcpyHostToDevice) != cudaSuccess || cudaMemcpy(dOc, ho.data(), ho.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess)) {
                err = "the chunk's sequences upload"; return false;
            }
            dS = dSc; dO = dOc;
            stepAdd("lanes.seq.h2d", now() - tu);
        }
        std::vector<uint32_t> ixA(repc.size());
        for (u64 k = 0; k < repc.size(); k++) ixA[k] = (uint32_t)(base + k);
        // reads in flight per lane (nbA, nbM, nbB): Work + ReadBuf per read are 36 KB (A), 68 KB (M), 550 KB (B)
        u64 nbAc = nbA, partA = 0, nPA = 1, nPM = 1, nPB = 1;
        if (cap && !lanePlan(repc.size(), availNow(), slotA, perA, nbA, nbAset ? nbA : 4096, 4096, nbAc, partA)) return noFit("A", repc.size(), availNow());
        if (!runLaneParts<LaneA, false, 64>(P, D, dS, dO, ixA, nbAc, partA, 1024, A, err, resident, "A", NEW, nullptr, nPA)) return false;
        if (nPA > 1) { allRes = false; resident = false; }
        double th = now();
        std::vector<uint32_t> repM, posM, ixM;
        for (u64 k = 0; k < repc.size(); k++) if (A.status[k] != (uint8_t)Status::Ok) { repM.push_back(repc[k]); posM.push_back((uint32_t)k); ixM.push_back((uint32_t)(base + k)); }
        stepAdd("lanes.host", now() - th);
        bool wM = warpM; u64 nbMc = nbM, partM = 0;
        if (cap && !repM.empty()) {   // E17: the warp lane when its stack and a batch of 32768 (or all of lane M) fit next to all its slots; else thread per read
            const u64 av = availNow();
            u64 nb = 0, part = 0;
            if (wM && !(av > stackNeed && lanePlan(repM.size(), av - stackNeed, slotM, perMw, nbM, nbMset ? nbM : 4096, 4096, nb, part) && part == 0
                        && (nbMset || nb >= std::min<u64>(32768, repM.size())))) wM = false;
            if (wM) nbMc = nb;
            else if (!lanePlan(repM.size(), av, slotM, perMt, nbM, nbMset ? nbM : 4096, 4096, nbMc, partM)) return noFit("M", repM.size(), av);
        }
        if (!(wM ? runLaneParts<LaneM, true>(P, D, dS, dO, ixM, nbMc, partM, 2048, M, err, resident, "M", NEW, nullptr, nPM)
                 : runLaneParts<LaneM>(P, D, dS, dO, ixM, nbMc, partM, 2048, M, err, resident, "M", NEW, nullptr, nPM))) return false;
        if (nPM > 1) { allRes = false; resident = false; }
        th = now();
        std::vector<uint32_t> repB, posB, ixB;
        for (u64 k = 0; k < repM.size(); k++) if (M.status[k] != (uint8_t)Status::Ok) { repB.push_back(repM[k]); posB.push_back(posM[k]); ixB.push_back(ixM[k]); }
        stepAdd("lanes.host", now() - th);
        u64 nbB = nbB0, partB = 0;   // the heavy lane's reads in flight: its Work is the largest device allocation of the prefill
        if (cap && !repB.empty()) {   // E17: up to nbB0, as many as the budget's room holds next to all its slots; else parts
            const u64 av = availNow();
            if (!lanePlan(repB.size(), av, slotB, perB, nbB0, nbBset ? nbB0 : 1024, 1024, nbB, partB)) return noFit("B", repB.size(), av);
        } else if (NEW && !nbBset && !repB.empty()) {   // E12 (no budget): up to nbB0, as many as the free device memory holds (3 GB spare)
            size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
            const u64 fixed = repB.size() * (u64)(LaneB::NBLOB + 4 + 1 + 2 + 8 + 8) + (3ull << 30);
            const u64 per = perB;
            const u64 fit = f > fixed ? (f - fixed) / per / 1024 * 1024 : 0;
            nbB = std::max<u64>(8192, std::min<u64>(nbB0, fit));
        }
        pendB.reset(new PendB);
        PendB& X = *pendB;
        X.A = std::move(A); X.M = std::move(M); X.repc = repc; X.repB = repB; X.posB = posB; X.posM = posM; X.ixB = std::move(ixB); X.c0 = c0; X.ci = chunks.size(); X.nbB = nbB; X.tc0 = tc0;
        X.nbAc = nbAc; X.nPA = nPA; X.nbMc = nbMc; X.nPM = nPM; X.wM = wM; X.nRepM = repM.size(); X.nRepc = repc.size();
        if (pool) {
            hres[X.ci].reset(new HostRes); hres[X.ci]->out.resize(X.repB.size()); hres[X.ci]->st.assign(X.repB.size(), 0);
            PendB* xp = &X;
            AutoB* ab = autoReq ? &autoB : nullptr;
            X.onB = [this_pool = pool.get(), xp, ab](u64 b0, u64 nb, const uint8_t* st) {
                std::vector<HostPool::Task> v;
                for (u64 k = 0; k < nb; k++) if (st[k] != (uint8_t)Status::Ok) v.push_back({(uint32_t)xp->ci, (uint32_t)xp->qn++, xp->repB[b0 + k]});
                this_pool->push(v);
                if (ab) {   // STEPS_B=auto controller
                    const double t = now();
                    if (ab->t > 0) ab->batch = ab->batch > 0 ? 0.5 * ab->batch + 0.5 * (t - ab->t) : t - ab->t;
                    ab->t = t;
                    const u64 left = xp->repB.size() > b0 + nb ? xp->repB.size() - (b0 + nb) : 0, q = this_pool->qsize();
                    const int idle = this_pool->nThreads - this_pool->active.load();
                    const u64 nd = this_pool->nDone.load();
                    const double avg = nd ? this_pool->busyNs.load() / 1e9 / (double)nd : 0;
                    const double devLeft = ab->batch * (double)((left + nb - 1) / nb), drain = this_pool->nThreads ? (double)q * avg / this_pool->nThreads : 0;
                    const u64 cur = ab->budget.load(); u64 nw = cur; const char* why = "";
                    if (ab->batch > 0 && nd > 0 && drain > devLeft) { nw = std::min<u64>(8192, cur * 2); why = "queue backs up"; }
                    else if (q == 0 && idle * 4 >= this_pool->nThreads) { nw = std::max<u64>(64, cur / 2); why = "pool idle"; }
                    if (nw != cur) {
                        ab->budget = nw;
                        char cb[200]; snprintf(cb, sizeof cb, "B budget %llu -> %llu (%s; chunk %llu at %llu, queue %llu, idle %d of %d, drain %.2f s, device left %.2f s); ",
                                               (unsigned long long)cur, (unsigned long long)nw, why, (unsigned long long)xp->ci, (unsigned long long)(b0 + nb), (unsigned long long)q, idle, this_pool->nThreads, drain, devLeft);
                        ab->log += cb;
                    }
                }
            };
        }
        const bool keepB = resident;
        const bool autoOn = autoReq && pool;   // the controller needs the host pool (not with HOSTLANE=after or CFG=old)
        auto runB = [&]() -> bool {
            if (autoOn) g_autoBudget = &autoB.budget;
            const u64 bud = autoOn ? autoB.budget.load() : stepsB;
            const bool ok = wshrink ? runLaneParts<LaneBc>(P, D, dS, dO, X.ixB, X.nbB, partB, bud, X.B, X.err, keepB, "B", NEW, pool ? &X.onB : nullptr, X.nPB)
                                    : runLaneParts<LaneB>(P, D, dS, dO, X.ixB, X.nbB, partB, bud, X.B, X.err, keepB, "B", NEW, pool ? &X.onB : nullptr, X.nPB);
            g_autoBudget = nullptr;
            return ok;
        };
        if (!runB()) return false;
        if (!drainB()) return false;
        return true;
    };
    // lane B's result of the pending chunk joined: the chunk's plan line, its lane map and its entry in chunks
    drainB = [&]() -> bool {
        if (!pendB) return true;
        PendB& X = *pendB;
        if (!X.ok) { err = X.err; pendB.reset(); return false; }
        const std::vector<uint32_t>& repc = X.repc; const std::vector<uint32_t>& repB = X.repB; const std::vector<uint32_t>& posB = X.posB; const std::vector<uint32_t>& posM = X.posM;
        Lane A = std::move(X.A), M = std::move(X.M), B = std::move(X.B);
        const u64 c0 = X.c0, ci = X.ci, nbB = X.nbB, nPB = X.nPB, qn = X.qn;
        if (nPB > 1) { allRes = false; resident = false; }
        {
            char pl[320];
            snprintf(pl, sizeof pl, "chunk %llu: A %llu x %llu%s, M %llu x %llu %s%s, B %llu x %llu%s; ", (unsigned long long)(c0 / CHUNK),
                     (unsigned long long)X.nRepc, (unsigned long long)X.nbAc, X.nPA > 1 ? (" in " + std::to_string(X.nPA) + " parts").c_str() : "",
                     (unsigned long long)X.nRepM, (unsigned long long)X.nbMc, X.wM ? "warp" : "thread", X.nPM > 1 ? (" in " + std::to_string(X.nPM) + " parts").c_str() : "",
                     (unsigned long long)repB.size(), (unsigned long long)nbB, nPB > 1 ? (" in " + std::to_string(nPB) + " parts").c_str() : "");
            planLog += pl;
            if (c0 == 0) { nbMlog = X.nbMc; nbBlog = nbB; wMlog = X.wM; }
        }
        out.secMapGpu += now() - X.tc0;
        double th = now();
        // where each result is: 0 = A, 1 = M, 2 = B, 3 = host
        std::vector<uint8_t> lane(repc.size(), 0); std::vector<uint32_t> lpos(repc.size());
        for (u64 k = 0; k < repc.size(); k++) lpos[k] = (uint32_t)k;
        for (u64 k = 0; k < posM.size(); k++) { lane[posM[k]] = 1; lpos[posM[k]] = (uint32_t)k; }
        std::vector<uint32_t> repH, posH;
        for (u64 k = 0; k < repB.size(); k++) {
            lane[posB[k]] = 2; lpos[posB[k]] = (uint32_t)k;
            if (B.status[k] != (uint8_t)Status::Ok) { repH.push_back(repB[k]); posH.push_back(posB[k]); lane[posB[k]] = 3; lpos[posB[k]] = (uint32_t)(repH.size() - 1); }
        }
        planLog += "[B to host " + std::to_string(repH.size()) + " of " + std::to_string(repB.size()) + "] ";
        { ChunkRes cr; cr.repc = repc; cr.A = std::move(A); cr.M = std::move(M); cr.B = std::move(B); cr.lane = std::move(lane); cr.lpos = std::move(lpos);
          if (pool && qn != repH.size()) { err = "host lane stream: " + std::to_string(qn) + " tasks for " + std::to_string(repH.size()) + " reads"; pendB.reset(); return false; }
          cr.repH = std::move(repH); cr.posH = std::move(posH); cr.c0 = c0; cr.ci = ci; chunks.push_back(std::move(cr)); }
        stepAdd("lanes.host", now() - th);
        pendB.reset();
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
        SINGLET_NVTX("mapDistinct: host lane of one chunk");
        singlet::ThreadNameScope tnHost("gpu-hostlane");
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
                if (e2e) { recSrc[rec] = chunkIx << 40 | (u64)cr.lane[k] << 32 | (u64)cr.lpos[k]; recRead[rec] = (uint32_t)(c0 + k); }   // E17: by position
                std::memcpy(out.seq.data() + s0 + cr.sOff[k], rseq(i), L); out.seqOff[rec + 1] = s0 + cr.sOff[k] + L;
                std::memcpy(out.blob.data() + b0 + cr.bOff[k], p, len); out.blobOff[rec + 1] = b0 + cr.bOff[k] + len;
            }
        });
        nGpuMapped += repc.size() - cr.repH.size();
        chunksDone++;
        if (lean) for (Lane* L : {&cr.A, &cr.M, &cr.B}) {   // P3-MEM-03: lean keeps no result on the device, and out.blob now holds the chunk's results
            decltype(L->out)().swap(L->out); decltype(L->off)().swap(L->off); std::vector<uint16_t>().swap(L->len); std::vector<uint8_t>().swap(L->status);
        }
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
        if (!gpuPart(repc, c0)) { return budFail ? failBudget(err) : fail(err); }
    }
    if (!drainB()) return fail(err);
    if (lean) { decltype(leanSeq)().swap(leanSeq); decltype(leanOff)().swap(leanOff); }   // P3-MEM-03: the lanes are done; the records copy from the reads
    tS = now();
    // E8: the pileup's records built on the device (E6 buildRecords) from the device-resident results and copied into the shared region
    // (in.e2e). Runs after hostPart (the host lane's results and the record order exist). false: the region is left without magic and the
    // hand-off file path runs as before.
    bool gatherDev = false;   // E17: E8's gather from the device-resident results (else the host-assembled records are uploaded)
    // E17 lean (E20 split: before the second hold): E10's entries and masks in the region: zero those of the reads without a record, list the
    // hook's reads (the products' rules); needs readRecord
    std::vector<uint64_t> hkPre; bool entPre = false;
    auto entHostPass = [&](std::vector<uint64_t>& hookList) {
        const double t0e = now();
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        e6::ReadEntry* E = (e6::ReadEntry*) (RG.base + entOff); uint64_t* Mq = qmOff ? (uint64_t*) (RG.base + qmOff) : nullptr;
        const int nt = std::max(1, std::min(hostThreads, 16));
        std::vector<std::vector<uint64_t>> hk(nt);
        std::vector<std::thread> th;
        for (int t = 0; t < nt; t++) th.emplace_back([&, t]() {
            for (u64 i = n * t / nt, e = n * (t + 1) / nt; i < e; i++) {
                if (out.readRecord[i] == NONE) { E[i] = e6::ReadEntry{0, 0, 0, 0, 0}; if (Mq) { const u64 qw = qmBytes() / 8; for (u64 w = 0; w < qw; w++) Mq[qw * i + w] = 0; } }
                else if (E[i].flags & 4u) hk[t].push_back(i);
            }
        });
        for (auto& x : th) x.join();
        hookList.clear();
        for (auto& v : hk) hookList.insert(hookList.end(), v.begin(), v.end());
        stepAdd("e2e.entries.host", now() - t0e);
    };
    // E17: the records' phase need (the tally, the records' arrays, buildRecords' output, E10's products, E11's keep and the slack), and its
    // cap: one slot's share, or (SL step 2, admitOn) the slots the relock took, within what the device holds free then
    auto recNeed = [&]() -> u64 {
        const u64 nRec = out.seqOff.size() - 1, nB0 = out.blob.size(), nS0 = out.seq.size();
        const u64 eIn = nRec * 28 + nB0 + nS0, eBuild = nRec * 101 + nS0 / 2, eProd = e10 ? nRec * 24 + n * 4 + (256ull << 20) : 0;
        const u64 eKeep = e10 && in.e10.soloModel != nullptr ? (keepLate ? AUX + n * 16 : 0) + n * 4 + nRec * 8 : 0;
        return g_bud.used + eIn + eBuild + eProd + eKeep + SLACK;
    };
    u64 capRec = cap; int recSlots = 1;
    auto e2eBuild = [&](std::string& why) -> bool {
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        const u64 nRec = out.seqOff.size() - 1;
        if (nRec == 0 || recSrc.size() != nRec || recRead.size() != nRec) { why = "no records"; return false; }
        g_bud.at = "records";
        struct DevFree { std::vector<void*> p; ~DevFree() { for (void* x : p) dfree(x); } } df;
        auto dal = [&](u64 bytes) -> void* { void* p = nullptr; if (dmalv(&p, bytes + 64) != cudaSuccess) { cudaGetLastError(); return nullptr; } df.p.push_back(p); return p; };
        // E17: what this phase adds (estimates: the records' arrays, buildRecords' output, E10's products, E11's keep); recNeed below
        const u64 nB0 = out.blob.size(), nS0 = out.seq.size();
        if (cap && recNeed() > capRec) {
            why = "E17 budget: the records' phase needs about " + gbs(recNeed()) + " of device memory, more than " + (capRec == cap ? budName : gbs(capRec) + " (" + std::to_string(recSlots) + " slots of " + budName + ")");
            return false;
        }
        if (keepLate && e10 && in.e10.soloModel != nullptr) {   // E17 lean: the reads' CR/UR kept for E11 now (their aux uploaded again)
            const double tk = now();
            uint64_t* dA2 = nullptr; std::string ws;
            if (dmal(&dA2, in.e10.aux.size() * 8 + 8) != cudaSuccess || cudaMemcpy(dA2, in.e10.aux.data(), in.e10.aux.size() * 8, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaGetLastError(); out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (aux upload)\n";
            } else if (!e9::soloKeepReads(dA2, 2 + (u64) in.e10.nw, n, ws)) out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (" + ws + ")\n";
            else { keepTally = budRound(std::max<u64>(1, n) * 8) * 2; g_bud.used += keepTally; g_bud.note(); }
            dfree(dA2);
            stepAdd("e2e.keep", now() - tk);
        }
        const double tg = now();
        LaneSrc* dTab = nullptr; u64 *dSrc = nullptr, *dBOff = nullptr, *dSOff = nullptr; uint32_t* dRd = nullptr; uint8_t* dBlob = nullptr; char* dRS = nullptr;
        u64 nB = 0, nS = 0, hostBytes = 0;
        const unsigned G = (unsigned) ((nRec + 255) / 256);
        if (!gatherDev) {   // E17: the lanes' results are not all on the device: the records the host assembled (the same bytes: SINGLET_ENGINE_GPU_E2E_CHECK)
            for (auto& cr : chunks) { std::vector<uint8_t>().swap(cr.hBlob); std::vector<u64>().swap(cr.hOff); }
            nB = nB0; nS = nS0;
            dBOff = (u64*) dal((nRec + 1) * 8); dSOff = (u64*) dal((nRec + 1) * 8); dBlob = (uint8_t*) dal(nB); dRS = (char*) dal(nS);
            if (!dBOff || !dSOff || !dBlob || !dRS) { why = "out of device memory (records)"; return false; }
            Stager SU;
            if (!SU.init(std::max(1, std::min(8, hostThreads))) || !SU.copy(dBOff, out.blobOff.data(), (nRec + 1) * 8) || !SU.copy(dSOff, out.seqOff.data(), (nRec + 1) * 8)
                || !SU.copy(dBlob, out.blob.data(), nB) || !SU.copy(dRS, out.seq.data(), nS) || !SU.finish()) { why = "records upload"; return false; }
            hostBytes = SU.bytes;
        } else {
        // the gather: per chunk the lanes' packed results (device-resident; the host lane's uploaded), per record its source and sequence
        std::vector<LaneSrc> tab(chunks.size() * 4, LaneSrc{nullptr, nullptr});
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
        dTab = (LaneSrc*) dal(tab.size() * sizeof(LaneSrc)); dSrc = (u64*) dal(nRec * 8); dRd = (uint32_t*) dal(nRec * 4);
        dBOff = (u64*) dal((nRec + 1) * 8); dSOff = (u64*) dal((nRec + 1) * 8);
        if (!dTab || !dSrc || !dRd || !dBOff || !dSOff) { why = "out of device memory (gather)"; return false; }
        if (cudaMemcpy(dTab, tab.data(), tab.size() * sizeof(LaneSrc), cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(dSrc, recSrc.data(), nRec * 8, cudaMemcpyHostToDevice) != cudaSuccess
            || cudaMemcpy(dRd, recRead.data(), nRec * 4, cudaMemcpyHostToDevice) != cudaSuccess) { why = "gather upload"; return false; }
        cudaMemset(dBOff, 0, 8); cudaMemset(dSOff, 0, 8);
        k_rec_len<<<G, 256>>>(dSrc, dRd, nRec, dTab, dOff, dBOff + 1, dSOff + 1);
        thrust::inclusive_scan(TP, thrust::device_ptr<u64>(dBOff + 1), thrust::device_ptr<u64>(dBOff + 1 + nRec), thrust::device_ptr<u64>(dBOff + 1));
        thrust::inclusive_scan(TP, thrust::device_ptr<u64>(dSOff + 1), thrust::device_ptr<u64>(dSOff + 1 + nRec), thrust::device_ptr<u64>(dSOff + 1));
        if (cudaMemcpy(&nB, dBOff + nRec, 8, cudaMemcpyDeviceToHost) != cudaSuccess || cudaMemcpy(&nS, dSOff + nRec, 8, cudaMemcpyDeviceToHost) != cudaSuccess) {
            why = "gather copy"; return false;
        }
        if (nB != out.blob.size() || nS != out.seq.size()) {
            why = "gathered sizes differ from the host records (blob " + std::to_string(nB) + " vs " + std::to_string(out.blob.size()) + ", seq "
                  + std::to_string(nS) + " vs " + std::to_string(out.seq.size()) + ")";
            return false;
        }
        dBlob = (uint8_t*) dal(nB); dRS = (char*) dal(nS);
        if (!dBlob || !dRS) { why = "out of device memory (gathered records)"; return false; }
        k_rec_copy<<<G, 256>>>(dSrc, dRd, nRec, dTab, dSeq, dOff, dBOff, dSOff, dBlob, dRS);
        if (cudaDeviceSynchronize() != cudaSuccess) { why = std::string("gather: ") + cudaGetErrorString(cudaGetLastError()); return false; }
        }
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
            else if (!singlet_hostloop_gpu::products(RP, dBlob, (const uint64_t*) dBOff, (const uint64_t*) dSOff, nRec, dRR, n, entHost ? nullptr : dEnt,
                                                     entHost ? nullptr : dQm, in.e10.outSJ, in.e10.sjAll, out.e10, w, H.chrStart[H.nChr])) e10Off("products: " + w);
            else if (out.e10.nOddRec) e10Off(std::to_string(out.e10.nOddRec) + " records mapped without a transcript");
            else {
                devNote();
                const double t0e = now();
                if (entHost) {   // E17 lean: the entries and masks are in the region: zero those of the reads without a record, list the hook's reads (products' rules)
                    if (entPre) out.e10.hookReads.swap(hkPre);   // E20 split: done before the second hold (products, without the entries, left the list empty)
                    else entHostPass(out.e10.hookReads);
                }
                const u64 oEn = entHost ? entOff : RG.take(n * sizeof(e6::ReadEntry)), oQm = entHost ? qmOff : dQm ? RG.take(n * qmBytes()) : 0;
                if (!oEn || (!entHost && dQm && !oQm)) e10Off("the shared region is full");
                else {
                    bool pin2 = false, cp2 = entHost;
                    if (entHost) {}
                    else if (g_d2h) cp2 = g_d2h->copy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry)) && (!dQm || g_d2h->copy(RG.base + oQm, dQm, n * qmBytes()));
                    else {
                    const u64 lo2 = oEn & ~4095ull, hi2 = std::min<u64>(RG.cap, (RG.hdr()->used + 4095) & ~4095ull);
                    parallelFor((hi2 - lo2) >> 12, hostThreads, [&](u64 a, u64 b) { for (u64 q = a; q < b; q++) { volatile uint8_t* x = RG.base + lo2 + (q << 12); *x = *x; } });
                    pin2 = cudaHostRegister(RG.base + lo2, hi2 - lo2, cudaHostRegisterDefault) == cudaSuccess;
                    if (!pin2) cudaGetLastError();
                    cp2 = cudaMemcpy(RG.base + oEn, dEnt, n * sizeof(e6::ReadEntry), cudaMemcpyDeviceToHost) == cudaSuccess
                                     && (!dQm || cudaMemcpy(RG.base + oQm, dQm, n * qmBytes(), cudaMemcpyDeviceToHost) == cudaSuccess);
                    if (pin2) cudaHostUnregister(RG.base + lo2);
                    }
                    if (!cp2) e10Off("entries copy");
                    else {
                        out.e10.oEntries = oEn; out.e10.oQMask = oQm; out.e10.ok = true;
                        if (!entHost) out.e10.msD2H = (now() - t0e) * 1e3;
                        char l2[400];
                        snprintf(l2, sizeof l2, "engine_gpu: E10 on the device: entries %.1f ms (aux upload %.0f ms), R11 histogram %.1f ms (%llu reads), "
                                 "products %.1f ms (%llu junction entries -> %llu junctions, "
                                 "%llu unmapped reads, %llu hook reads), entries/masks to the region %.0f ms (%s)\n", out.e10.msEntries, out.e10.msUpAux,
                                 out.e10.msR11, (unsigned long long) out.e10.r11Total,
                                 out.e10.msProducts, (unsigned long long) out.e10.nSJentries, (unsigned long long) out.e10.nSJ,
                                 (unsigned long long) out.e10.unmappedReads.size(), (unsigned long long) out.e10.hookReads.size(), out.e10.msD2H,
                                 entHost ? "E17 lean: at the clip" : g_d2h ? "staged" : pin2 ? "page-locked" : "pageable");
                        out.e10.log += l2;
                        // E11: the records' Gene annotation (Solo's stage (b), while the blobs are here) and the read -> record map, kept
                        if (in.e10.soloModel != nullptr && in.e10.soloParams != nullptr && e9::soloKeep().reads) {
                            std::string ws;
                            if (e9::soloKeepRecords(*in.e10.soloParams, *in.e10.soloModel, RP, dBlob, (const uint64_t*) dBOff, (const uint64_t*) dSOff, nRec,
                                                    dRR, n, ws)) {
                                const e9::SoloKeep& K = e9::soloKeep();
                                out.e10.soloKept = K.nOverflow == 0;
                                { const u64 kr = budRound(std::max<u64>(1, n) * 4) + budRound(std::max<u64>(1, nRec) * 8); keepTally += kr; g_bud.used += kr; g_bud.note(); devNote(); }
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
                 "trBest=trInit %llu); %s %.0f ms (%s %.1f MB), buildRecords %.0f ms (kernels %.0f/%.0f/%.0f), copy to the region %.0f ms (%s)\n",
                 (unsigned long long) R6.nRec, (unsigned long long) R6.nAln, (unsigned long long) R6.nOps, (unsigned long long) R6.nSeq,
                 (unsigned long long) R6.nMapped, (unsigned long long) R6.nNoExon, (unsigned long long) R6.nBadBest,
                 gatherDev ? "gather" : "E17: the host-assembled records uploaded", E2H.msGather, gatherDev ? "host lane upload" : "staged", hostBytes / 1e6, E2H.msBuild,
                 R6.msCount, R6.msScan, R6.msFill, E2H.msD2H, g_d2h ? "staged" : pinned ? "page-locked" : "pageable");
        out.log += l;
        out.e2eOk = true;
        return true;
    };
    // P3-MEM-03: e2eBuild by record chunks (with the read chunks). The host-assembled records go to the device one chunk at a time: a first
    // pass counts each chunk's alignments, operations and sequence bytes (buildRecords), the region's arrays are then taken at their final
    // sizes, and a second pass builds each chunk again, shifts its offsets by the earlier chunks' counts and copies it into place, so the
    // region holds the same bytes as the whole-sample build. E10's products run per chunk (a read counts in the chunk of its record) and are
    // summed; the junctions are merged by key (counts summed, overhang maxed, as products' reduction); the unmapped reads come from
    // readRecord and the records' mapped flags. E11's keep is filled in parts (the reads' CR/UR by read chunks of their aux).
    auto e2eBuildRC = [&](std::string& why) -> bool {
        singlet_engine_gpu::E2ERegion& RG = *in.e2e;
        const u64 nRec = out.seqOff.size() - 1;
        if (nRec == 0 || out.readRecord.size() != n) { why = "no records"; return false; }
        if (nRec >= 0xFFFFFFFFull) { why = "more records than 32-bit indices"; return false; }
        g_bud.at = "records";
        e6::RecParams RP;
        RP.outFilterScoreMin = (int) P.outFilterScoreMin; RP.outFilterScoreMinOverLread = P.outFilterScoreMinOverLread;
        RP.outFilterMatchNmin = P.outFilterMatchNmin; RP.outFilterMatchNminOverLread = P.outFilterMatchNminOverLread;
        RP.outFilterMismatchNmax = P.outFilterMismatchNmax; RP.outFilterMismatchNoverLmax = P.outFilterMismatchNoverLmax;
        RP.outFilterMismatchNoverReadLmax = P.outFilterMismatchNoverReadLmax; RP.outFilterMultimapNmax = P.outFilterMultimapNmax;
        RP.outSAMmultNmax = P.outSAMmultNmax; RP.outSAMmapqUnique = in.outSAMmapqUnique; RP.outSAMflagAND = in.outSAMflagAND; RP.outSAMflagOR = in.outSAMflagOR;
        struct DB { void* p = nullptr; ~DB() { dfree(p); } };
        auto dalc = [&](DB& b, u64 bytes) -> bool { if (dmalv(&b.p, bytes + 64) != cudaSuccess) { cudaGetLastError(); b.p = nullptr; return false; } return true; };
        auto d2hc = [&](void* dst, const void* src, u64 b) -> bool { return g_d2h ? g_d2h->copy(dst, src, b) : cudaMemcpy(dst, src, b, cudaMemcpyDeviceToHost) == cudaSuccess; };
        Stager SU;
        if (!SU.init(std::max(1, std::min(8, hostThreads)))) { why = "page-locked staging buffers (records)"; return false; }
        const double t0r = now();
        // E11: the reads' CR/UR from their aux, by read chunks
        bool keep = e10 && in.e10.soloModel != nullptr && in.e10.soloParams != nullptr;
        if (keep) {
            const u64 st = 2 + (u64) in.e10.nw, m0 = std::max<u64>(1, std::min<u64>(n, RCN ? RCN : n));
            DB dA; std::string ws;
            if (!dalc(dA, m0 * st * 8)) { keep = false; out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (device memory)\n"; }
            for (u64 i0 = 0; keep && i0 < n; i0 += m0) {
                const u64 m = std::min(m0, n - i0);
                if (!SU.copy(dA.p, in.e10.aux.data() + i0 * st, m * st * 8) || !SU.finish()) { keep = false; out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (aux upload)\n"; }
                else if (!e9::soloKeepReadsPart((const uint64_t*) dA.p, st, i0, m, n, ws)) { keep = false; out.e10.log += "engine_gpu: E11: the reads' CR/UR not kept (" + ws + ")\n"; }
            }
            if (keep) { keepTally = budRound(std::max<u64>(1, n) * 8) * 2; g_bud.used += keepTally; g_bud.note(); }
            stepAdd("e2e.keep", now() - t0r);
        }
        // readRecord on the device (products' and E11's read -> record map); per chunk its local copy
        DB dRRg, dRRl;
        if (!dalc(dRRg, n * 4) || !dalc(dRRl, n * 4) || cudaMemcpy(dRRg.p, out.readRecord.data(), n * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
            why = "out of device memory (readRecord)"; return false;
        }
        if (keep) {
            std::string ws;
            if (!e9::soloKeepRecordsBegin(nRec, (const uint32_t*) dRRg.p, n, ws)) { keep = false; out.e10.log += "engine_gpu: E11: the records' Gene annotation not kept (" + ws + ")\n"; }
        }
        // record chunks: SINGLET_ENGINE_GPU_RECCHUNK, else about a quarter of the room
        const u64 avgB = out.blob.size() / nRec + 1, avgS = out.seq.size() / nRec + 1;
        const u64 perRec = avgB + avgS + 16 + 101 + avgS / 2 + 64;
        u64 RRN = 0;
        if (const char* e = std::getenv("SINGLET_ENGINE_GPU_RECCHUNK")) RRN = std::strtoull(e, nullptr, 10);
        if (RRN == 0) {
            u64 room = cap ? g_bud.room() : ~0ull; size_t f = 0, t = 0; cudaMemGetInfo(&f, &t);
            room = std::min<u64>(room, f > (2ull << 30) ? f - (2ull << 30) : 0);
            room = room > SLACK ? room - SLACK : 0;
            RRN = std::max<u64>(1u << 20, room / 4 / perRec / (1u << 20) * (1u << 20));
        }
        RRN = std::min<u64>(RRN, nRec);
        const u64 nCh = (nRec + RRN - 1) / RRN;
        u64 maxNB = 0, maxNS = 0;
        for (u64 q0 = 0; q0 < nRec; q0 += RRN) {
            const u64 q1 = std::min(nRec, q0 + RRN);
            maxNB = std::max<u64>(maxNB, out.blobOff[q1] - out.blobOff[q0]); maxNS = std::max<u64>(maxNS, out.seqOff[q1] - out.seqOff[q0]);
        }
        DB dB, dBO, dS, dSO;
        if (!dalc(dB, maxNB) || !dalc(dBO, (RRN + 1) * 8) || !dalc(dS, maxNS) || !dalc(dSO, (RRN + 1) * 8)) {
            why = "E17 budget: the records' chunk of " + std::to_string(RRN) + " does not fit (" + gbs(g_bud.used) + " in use), within " + budName; return false;
        }
        singlet_engine_gpu::NoInitVector<u64> hbo(RRN + 1), hso(RRN + 1);
        double tUp = 0, tBuild = 0, tCopy = 0, tProd = 0;
        auto upChunk = [&](u64 q0, u64 m) -> bool {
            const double ta = now();
            const u64 b0 = out.blobOff[q0], s0 = out.seqOff[q0];
            parallelFor(m + 1, hostThreads, [&](u64 lo, u64 hi) { for (u64 k = lo; k < hi; k++) { hbo[k] = out.blobOff[q0 + k] - b0; hso[k] = out.seqOff[q0 + k] - s0; } });
            const bool ok = SU.copy(dB.p, out.blob.data() + b0, out.blobOff[q0 + m] - b0) && SU.copy(dS.p, out.seq.data() + s0, out.seqOff[q0 + m] - s0)
                            && SU.copy(dBO.p, hbo.data(), (m + 1) * 8) && SU.copy(dSO.p, hso.data(), (m + 1) * 8) && SU.finish();
            tUp += now() - ta;
            return ok;
        };
        // pass 1: the counts
        std::vector<u64> cAln(nCh), cOps(nCh), cSeq(nCh);
        u64 nAln = 0, nOps = 0, nSeqT = 0, nNoExon = 0, nBadBest = 0, nMapped = 0;
        for (u64 c = 0, q0 = 0; q0 < nRec; q0 += RRN, c++) {
            const u64 m = std::min(RRN, nRec - q0);
            if (!upChunk(q0, m)) { why = "records upload"; return false; }
            const double ta = now();
            singlet_records_gpu::DeviceRecords R6; std::string e6err;
            struct R6Free { singlet_records_gpu::DeviceRecords& r; ~R6Free() { singlet_records_gpu::release(r); } } r6f{R6};
            if (!singlet_records_gpu::buildRecords(RP, (const uint8_t*) dB.p, (const uint64_t*) dBO.p, (const char*) dS.p, (const uint64_t*) dSO.p, m,
                                                   (const uint64_t*) dcs, false, R6, e6err, true)) { why = "buildRecords: " + e6err; return false; }
            if (R6.nMateGap) { why = std::to_string(R6.nMateGap) + " records with a mate gap (paired-end; alignBAM's path)"; return false; }
            cAln[c] = R6.nAln; cOps[c] = R6.nOps; cSeq[c] = R6.nSeq;
            nAln += R6.nAln; nOps += R6.nOps; nSeqT += R6.nSeq; nNoExon += R6.nNoExon; nBadBest += R6.nBadBest; nMapped += R6.nMapped;
            tBuild += now() - ta;
        }
        if (nAln >= 0xFFFFFFFFull || nOps >= 0xFFFFFFFFull) { why = "the records' alignments or operations exceed 32-bit offsets"; return false; }
        const u64 oS = RG.take(nRec * sizeof(SigRec)), oA = RG.take(nAln * sizeof(AlnRec)), oO = RG.take(nOps * 4), oN = RG.take(nAln * 2), oQ = RG.take(nSeqT);
        if (!oS || !oA || !oO || !oN || !oQ) { why = "the shared region is full"; return false; }
        // pass 2: build, shift, copy; E10's products and E11's annotation per chunk
        decltype(out.e10) acc;   // the products, summed
        struct SJ { u64 start; uint32_t gap, cu, cm; uint8_t strand, motif, annot; uint16_t ov; };
        std::vector<SJ> sjList;
        bool prodOk = e10;
        if (e10) out.e10.recMapped.assign(nRec, 0);
        u64 aB = 0, oB = 0, sB = 0;
        for (u64 c = 0, q0 = 0; q0 < nRec; q0 += RRN, c++) {
            const u64 m = std::min(RRN, nRec - q0);
            if (!upChunk(q0, m)) { why = "records upload"; return false; }
            double ta = now();
            singlet_records_gpu::DeviceRecords R6; std::string e6err;
            struct R6Free { singlet_records_gpu::DeviceRecords& r; ~R6Free() { singlet_records_gpu::release(r); } } r6f{R6};
            if (!singlet_records_gpu::buildRecords(RP, (const uint8_t*) dB.p, (const uint64_t*) dBO.p, (const char*) dS.p, (const uint64_t*) dSO.p, m,
                                                   (const uint64_t*) dcs, false, R6, e6err, true)) { why = "buildRecords: " + e6err; return false; }
            if (R6.nAln != cAln[c] || R6.nOps != cOps[c] || R6.nSeq != cSeq[c]) { why = "a record chunk's counts differ between its two builds"; return false; }
            k_rebase_recs<<<(unsigned) ((std::max<u64>(m, R6.nAln) + 255) / 256), 256>>>(R6.sigs, m, R6.alns, R6.nAln, (uint32_t) aB, sB, (uint32_t) oB);
            if (cudaDeviceSynchronize() != cudaSuccess) { why = std::string("records rebase: ") + cudaGetErrorString(cudaGetLastError()); return false; }
            tBuild += now() - ta; ta = now();
            if (!d2hc(RG.base + oS + q0 * sizeof(SigRec), R6.sigs, m * sizeof(SigRec)) || (R6.nAln && !d2hc(RG.base + oA + aB * sizeof(AlnRec), R6.alns, R6.nAln * sizeof(AlnRec)))
                || (R6.nOps && !d2hc(RG.base + oO + oB * 4, R6.ops, R6.nOps * 4)) || (R6.nAln && !d2hc(RG.base + oN + aB * 2, R6.nm, R6.nAln * 2))
                || (R6.nSeq && !d2hc(RG.base + oQ + sB, R6.seq, R6.nSeq))) { why = std::string("records copy: ") + cudaGetErrorString(cudaGetLastError()); return false; }
            aB += R6.nAln; oB += R6.nOps; sB += R6.nSeq;
            tCopy += now() - ta; ta = now();
            if (prodOk || keep) {
                k_local_rr<<<(unsigned) ((n + 255) / 256), 256>>>((const uint32_t*) dRRg.p, n, (uint32_t) q0, (uint32_t) (q0 + m), (uint32_t*) dRRl.p);
                if (cudaDeviceSynchronize() != cudaSuccess) { why = "local readRecord"; return false; }
            }
            if (prodOk) {
                decltype(out.e10) pc; std::string w;
                if (!singlet_hostloop_gpu::products(RP, (const uint8_t*) dB.p, (const uint64_t*) dBO.p, (const uint64_t*) dSO.p, m, (const uint32_t*) dRRl.p, n,
                                                    nullptr, nullptr, in.e10.outSJ, in.e10.sjAll, pc, w, H.chrStart[H.nChr])) { prodOk = false; e10Off("products: " + w); }
                else {
                    for (int k = 0; k < decltype(out.e10)::S_N; k++) acc.stats[k] += pc.stats[k];
                    acc.mappedPortion += pc.mappedPortion; acc.nOddRec += pc.nOddRec; acc.nSJentries += pc.nSJentries; acc.msProducts += pc.msProducts;
                    acc.sjGapBits = pc.sjGapBits;
                    std::memcpy(out.e10.recMapped.data() + q0, pc.recMapped.data(), m);
                    for (u64 k = 0; k < pc.nSJ; k++) {
                        const uint8_t* d = pc.sj.data() + k * 27; SJ j;
                        std::memcpy(&j.start, d, 8); std::memcpy(&j.gap, d + 8, 4); j.strand = d[12]; j.motif = d[13]; j.annot = d[14];
                        std::memcpy(&j.cu, d + 15, 4); std::memcpy(&j.cm, d + 19, 4); std::memcpy(&j.ov, d + 23, 2);
                        sjList.push_back(j);
                    }
                }
            }
            if (keep) {
                std::string ws;
                if (!e9::soloKeepRecordsPart(*in.e10.soloParams, *in.e10.soloModel, RP, (const uint8_t*) dB.p, (const uint64_t*) dBO.p, (const uint64_t*) dSO.p, q0, m, ws)) {
                    keep = false; out.e10.log += "engine_gpu: E11: the records' Gene annotation not kept (" + ws + ")\n";
                }
            }
            tProd += now() - ta;
            devNote();
        }
        singlet_engine_gpu::E2EHeader& E2H = *RG.hdr();
        E2H.nRec = nRec; E2H.nAln = nAln; E2H.nOps = nOps; E2H.nSeq = nSeqT;
        E2H.oSigs = oS; E2H.oAlns = oA; E2H.oOps = oO; E2H.oNm = oN; E2H.oSeq = oQ;
        E2H.nMateGap = 0; E2H.nNoExon = nNoExon; E2H.nBadBest = nBadBest; E2H.nMapped = nMapped;
        E2H.msGather = tUp * 1e3; E2H.msBuild = tBuild * 1e3; E2H.msD2H = tCopy * 1e3;
        if (prodOk && acc.nOddRec) { prodOk = false; e10Off(std::to_string(acc.nOddRec) + " records mapped without a transcript"); }
        if (prodOk) {
            for (int k = 0; k < decltype(out.e10)::S_N; k++) out.e10.stats[k] = acc.stats[k];
            out.e10.mappedPortion = acc.mappedPortion; out.e10.nOddRec = 0; out.e10.nSJentries = acc.nSJentries; out.e10.msProducts = acc.msProducts;
            out.e10.sjGapBits = acc.sjGapBits;
            // the junctions merged by key (start, gap): counts summed, overhang maxed
            std::stable_sort(sjList.begin(), sjList.end(), [](const SJ& a, const SJ& b) { return a.start != b.start ? a.start < b.start : a.gap < b.gap; });
            std::vector<SJ> u;
            for (const SJ& j : sjList) {
                if (!u.empty() && u.back().start == j.start && u.back().gap == j.gap) { u.back().cu += j.cu; u.back().cm += j.cm; u.back().ov = std::max(u.back().ov, j.ov); }
                else u.push_back(j);
            }
            out.e10.sj.assign((u.size() + 1) * 27, 0);
            for (u64 k = 0; k < u.size(); k++) {
                uint8_t* d = out.e10.sj.data() + k * 27; const SJ& j = u[k];
                std::memcpy(d, &j.start, 8); std::memcpy(d + 8, &j.gap, 4); d[12] = j.strand; d[13] = j.motif; d[14] = j.annot;
                std::memcpy(d + 15, &j.cu, 4); std::memcpy(d + 19, &j.cm, 4); std::memcpy(d + 23, &j.ov, 2); std::memcpy(d + 25, &j.ov, 2);
            }
            std::memset(out.e10.sj.data() + u.size() * 27, 255, 27);
            out.e10.nSJ = u.size();
            // the reads whose record is unmapped, in read order
            const u64 nT = (u64) std::max(1, std::min(hostThreads, 64));
            std::vector<std::vector<uint32_t>> um(nT);
            std::vector<std::thread> th;
            for (u64 t = 0; t < nT; t++) th.emplace_back([&, t]() {
                for (u64 i = n * t / nT, e = n * (t + 1) / nT; i < e; i++) { const uint32_t r = out.readRecord[i]; if (r != NONE && !out.e10.recMapped[r]) um[t].push_back((uint32_t) i); }
            });
            for (auto& x : th) x.join();
            out.e10.unmappedReads.clear();
            for (auto& v : um) out.e10.unmappedReads.insert(out.e10.unmappedReads.end(), v.begin(), v.end());
            if (entHost) {   // the entries and masks are in the region since the read chunks: zero those of the reads without a record, list the hook's reads
                if (entPre) out.e10.hookReads.swap(hkPre);
                else entHostPass(out.e10.hookReads);
                out.e10.oEntries = entOff; out.e10.oQMask = qmOff; out.e10.ok = true;
                char l2[400];
                snprintf(l2, sizeof l2, "engine_gpu: E10 on the device: entries %.1f ms, R11 histogram %.1f ms (%llu reads), products %.1f ms in %llu record chunks "
                         "(%llu junction entries -> %llu junctions, %llu unmapped reads, %llu hook reads), entries/masks to the region at the read chunks\n",
                         out.e10.msEntries, out.e10.msR11, (unsigned long long) out.e10.r11Total, out.e10.msProducts, (unsigned long long) nCh,
                         (unsigned long long) out.e10.nSJentries, (unsigned long long) out.e10.nSJ, (unsigned long long) out.e10.unmappedReads.size(),
                         (unsigned long long) out.e10.hookReads.size());
                out.e10.log += l2;
            } else e10Off("the entries are not in the region");
        }
        if (keep && out.e10.ok) {
            e9::soloKeepRecordsEnd(tProd * 1e3);
            const e9::SoloKeep& K = e9::soloKeep();
            out.e10.soloKept = K.nOverflow == 0;
            { const u64 kr = budRound(std::max<u64>(1, n) * 4) + budRound(std::max<u64>(1, nRec) * 8); keepTally += kr; g_bud.used += kr; g_bud.note(); devNote(); }
            char l3[300];
            snprintf(l3, sizeof l3, "engine_gpu: E11: Solo inputs kept on the device: CR/UR of %llu reads (%.1f ms), Gene annotation of %llu records in %llu chunks "
                     "and the read -> record map (%llu exon overflows), %.2f GB\n", (unsigned long long) K.n, K.msReads, (unsigned long long) K.nRec,
                     (unsigned long long) nCh, (unsigned long long) K.nOverflow, (K.n * 20.0 + K.nRec * 8.0) / 1e9);
            out.e10.log += l3;
            if (!out.e10.soloKept) e9::soloKeepRelease();
        } else if (in.e10.soloModel != nullptr) { e9::soloKeepRelease(); out.e10.soloKept = false; if (e10) out.e10.log += "engine_gpu: E11: Solo inputs not kept\n"; }
        char l[512];
        snprintf(l, sizeof l, "engine_gpu: E8 records on the device in %llu record chunks of %llu (P3-MEM-03): %llu records, %llu alignments, %llu ops, %llu seq bytes "
                 "(%llu mapped, no exon %llu, trBest=trInit %llu); upload %.0f ms (two passes), buildRecords %.0f ms, copy to the region %.0f ms, products and keep %.0f ms\n",
                 (unsigned long long) nCh, (unsigned long long) RRN, (unsigned long long) nRec, (unsigned long long) nAln, (unsigned long long) nOps,
                 (unsigned long long) nSeqT, (unsigned long long) nMapped, (unsigned long long) nNoExon, (unsigned long long) nBadBest, tUp * 1e3, tBuild * 1e3,
                 tCopy * 1e3, tProd * 1e3);
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
    if (e2e && split) {   // E20 split: the first hold ends with the lanes; the host lane's wait, the records and readRecord without the device
        dfree(dG1); dfree(dSA); dfree(dSAi); dG1 = nullptr; dSA = nullptr; dSAi = nullptr;
        for (auto& cr : chunks) for (Lane* L : {&cr.A, &cr.M, &cr.B}) { dfree(L->dOut); dfree(L->dOff); L->dOut = nullptr; L->dOff = nullptr; }
        dfree(dSeq); dfree(dOff); dSeq = nullptr; dOff = nullptr;
        dfree(dcl); dfree(dcb); dfree(dss); dfree(dse); dfree(dsd); dfree(dsa); dfree(dsm); dfree(dsl); dfree(dsr); dfree(dst);   // dcs: buildRecords' chromosome starts
        dcl = dcb = dss = dse = dsd = dsa = nullptr; dsm = dsl = dsr = dst = nullptr;
        if (cudaDeviceSynchronize() != cudaSuccess) return fail(std::string("before the release: ") + cudaGetErrorString(cudaGetLastError()));
        step("split.free");
        memMark("released");
        in.release(); out.split = true;
        if (clipHostFill) { const double th = now(); clipHostFill(); stepAdd("clip.host", now() - th); tS = now(); }
        if (!hostAll()) return fail(err);
        tS = now();
        if (e10) fillReadRecord();   // E10: the products need it on the device
        step("readRecord");
        if (e10 && entHost) { entHostPass(hkPre); entPre = true; tS = now(); }
        int want = 1;
        if (cap && budDefault && !RC && singlet_engine_gpu::admitOn() && out.seqOff.size() > 1) {
            const u64 need = recNeed();
            if (need > cap) want = (int) std::min<u64>((u64) singlet_engine_gpu::gpuSlots(), (need + cap - 1) / cap);
        }
        recSlots = in.reacquire(want);
        if (cudaSetDevice(device) != cudaSuccess) return fail("cannot select the device again");
        if (recSlots > 1) {   // SL step 2: the slots' shares, within the device's free memory now (less 512 MB)
            size_t fR = 0, tR = 0; cudaMemGetInfo(&fR, &tR);
            const u64 room = g_bud.used + (fR > (512ull << 20) ? fR - (512ull << 20) : 0);
            capRec = std::max(cap, std::min(cap * (u64) recSlots, room)); g_bud.cap = capRec;
            char ab[200]; snprintf(ab, sizeof ab, "engine_gpu: SL admission: the records' phase needs %.1f GB; %d of %d slots taken, cap %.1f GB (free %.1f GB)\n",
                                   b2gb(recNeed()), recSlots, want, b2gb(capRec), b2gb(fR));
            out.log += ab;
        } else if (want > 1) out.log += "engine_gpu: SL admission: " + std::to_string(want) + " slots asked, 1 taken\n";
        step("split.relock");
        memMark("relocked");
        gatherDev = false;
    } else if (e2e) {
        if (!hostAll()) return fail(err);
        tS = now();
        if (e10) fillReadRecord();   // E10: the products need it on the device
        step("readRecord");
        dfree(dG1); dfree(dSA); dfree(dSAi); dG1 = nullptr; dSA = nullptr; dSAi = nullptr;   // the index is no longer needed: room for the records
        gatherDev = resident && allRes;
        if (!gatherDev) {   // E17: the records come from the host assembly: neither the sequences nor any kept result is needed on the device
            for (auto& cr : chunks) for (Lane* L : {&cr.A, &cr.M, &cr.B}) { dfree(L->dOut); dfree(L->dOff); L->dOut = nullptr; L->dOff = nullptr; }
            dfree(dSeq); dfree(dOff); dSeq = nullptr; dOff = nullptr;
        }
        step("index.free");
    }
    if (e2e) {
        std::string e2eWhy;
        if (!(RC ? e2eBuildRC(e2eWhy) : e2eBuild(e2eWhy))) { out.log += "engine_gpu: E8 records not built on the device (" + e2eWhy + "); the hand-off file path is used\n"; e10Off("no E8 records"); }
        step("e2e.build");
        { const bool ok10 = out.e10.ok; e10 = false; e10Off(""); out.e10.ok = ok10; }   // frees the device entries
        for (auto& cr : chunks) for (Lane* L : {&cr.A, &cr.M, &cr.B}) { dfree(L->dOut); dfree(L->dOff); L->dOut = nullptr; L->dOff = nullptr; }
        decltype(recSrc)().swap(recSrc); decltype(recRead)().swap(recRead);
        step("e2e.free");
    }
    if (clipHostFill && out.clip5.size() != n) clipHostFill();   // E20: never left out
    // the last device work is done: free the device buffers and let the next sample have the GPU while this one assembles its records
    cudaDeviceSynchronize();
    if (!e2e) for (auto& cr : chunks) for (Lane* L : {&cr.A, &cr.M, &cr.B}) { dfree(L->dOut); dfree(L->dOff); L->dOut = nullptr; L->dOff = nullptr; }
    dfree(dG1); dfree(dSA); dfree(dSAi); dfree(dSeq); dfree(dOff);
    shClose(SHC);   // the imports end with the sample's device phase
    dfree(dcs); dfree(dcl); dfree(dcb); dfree(dss); dfree(dse); dfree(dsd); dfree(dsa);
    dfree(dsm); dfree(dsl); dfree(dsr); dfree(dst);
    step("free");
    if (in.release) in.release();
    if (!e2e && !hostAll()) return fail(err);
    tS = now();
    if (out.readRecord.size() != n) fillReadRecord();
    step("readRecord");
    out.log += "engine_gpu: device memory (GB in use): " + memLog + "peak (lanes, records) " + std::to_string(g_devPeakGB) + "\n";
    {   // E17: the budget's derivation and what the tally saw
        char bl[900];
        if (cap) snprintf(bl, sizeof bl, "engine_gpu: budget %.1f GB: chunk %llu (%llu chunks), NB %llu / %llu / %llu (lane M %s per read); mode %s, E8 records %s; %s; "
                          "tally peak %.2f GB at %s (context, own buffers, e9's keep, lane M's stack), %s\n", b2gb(cap), (unsigned long long)CHUNK, (unsigned long long)nChunks,
                          (unsigned long long)nbA, (unsigned long long)nbMlog, (unsigned long long)nbBlog, wMlog ? "warp" : "thread", lean ? "lean" : "normal",
                          gatherDev ? "gathered from the device-resident results" : "uploaded from the host assembly", budLog.c_str(), b2gb(g_bud.peak), g_bud.peakAt.c_str(),
                          g_bud.nOver ? ("OVER the budget " + std::to_string(g_bud.nOver) + " times, by up to " + gbs(g_bud.overMax) + " at " + g_bud.overAt).c_str() : "within the budget");
        else snprintf(bl, sizeof bl, "engine_gpu: budget none: chunk %llu (%llu chunks), NB %llu / %llu / %llu; tally peak %.2f GB at %s\n", (unsigned long long)CHUNK,
                      (unsigned long long)nChunks, (unsigned long long)nbA, (unsigned long long)nbMlog, (unsigned long long)nbBlog, b2gb(g_bud.peak), g_bud.peakAt.c_str());
        out.log += bl;
        out.log += "engine_gpu: E17 lanes: " + planLog + "\n";
        if (autoReq || stepsB != 8192)
            out.log += "engine_gpu: STEPS_B " + (autoReq ? "auto, changes: " + (autoB.log.empty() ? std::string("none") : autoB.log) : std::to_string(stepsB)) + "\n";
    }
    char line[512];
    snprintf(line, sizeof line, "engine_gpu: %llu reads, %llu distinct (%llu with N), mapped on the GPU %llu, host lane %llu (%.2f s), not mapped %llu; "
             "upload %.2f s, dedup %.2f s, GPU mapping %.2f s\n", (unsigned long long)out.nReads, (unsigned long long)out.nDistinct,
             (unsigned long long)out.nWithN, (unsigned long long)nGpuMapped, (unsigned long long)out.nHostLane, out.secHostLane,
             (unsigned long long)out.nFailed, out.secUpload, out.secDedup, out.secMapGpu);
    out.log += line;
    out.log += std::string("engine_gpu: E12 lanes (") + (NEW ? "new" : "old: SINGLET_ENGINE_GPU_CFG=old") + "): batches " + std::to_string(nbA) + " / "
               + std::to_string(nbM) + " / " + std::to_string(nbB0) + (warpM ? ", lane M warp per read" : "") + "; kern2 wshrink=" + (wshrink ? "1" : "0") + " waslice=" + std::to_string(g_waSlice) + "; " + g_laneLog + "\n";
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
    if (in.oomFatal && !e10Oom.empty()) return fail("E10 ran out of device memory (" + e10Oom + "); the caller runs mapDistinct again");
    return true;
}

} // namespace singlet_engine_gpu
