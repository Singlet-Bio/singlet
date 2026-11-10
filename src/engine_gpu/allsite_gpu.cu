// SPDX-License-Identifier: MIT
// singlet GPU engine: the all-site pass on the device (D-101; API and rules: include/singlet/pileup/allsite_device.h, the host pass:
// include/singlet/pileup/allsite.h run()).
//
// Per chunk of contigs (the host gathers their reads into flat arrays, sized to the memory free on the device):
//  1. discovery: each read's mismatch observations against the reference -> (contig, pos, alt, barcode | UMI), radix-sorted; distinct
//     molecules and barcodes per (pos, alt) by reduce-by-key; per position the best passing alternate base.
//  2. counting, in blocks of sites that fit: each read's hits at the sites -> ((site, barcode), (UMI, class)), radix-sorted (two stable
//     passes); class counts per (barcode, site, UMI) saturated at 255; R9 components per (barcode, site) by a lock-free union-find over
//     the pairs that umi3_directional_roots joins; the vote per component; per (barcode, site) the DP and AD molecules.
// The pass runs in its own driver context (a cudaDeviceReset of the primary context by another stage of the process leaves it alone),
// under the engine's device lock, and frees everything before it returns.
#include <cuda.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "singlet/engine_gpu.h"
#include "singlet/pileup/allsite_device.h"

namespace singlet_engine_gpu {
namespace {
namespace AD = singlet::allsite::dev;
using u64 = unsigned long long;
using u32 = uint32_t;
using u8 = uint8_t;

#define AS_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { cudaGetLastError(); \
    throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(e_) + " at " #x); } } while (0)

double secSince(std::chrono::steady_clock::time_point t) { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count(); }
int bitsFor(u64 v) { return v == 0 ? 1 : 64 - __builtin_clzll(v); }   // bits that hold 0..v

// ---------------------------------------------------------------------------------------------------------------- driver context
struct DrvCtx {
    typedef CUresult (*FnDevGet)(CUdevice*, int);
    typedef CUresult (*FnCreate)(CUcontext*, unsigned int, CUdevice);
    typedef CUresult (*FnCtx)(CUcontext);
    typedef CUresult (*FnPop)(CUcontext*);
    CUcontext ctx = nullptr; FnCtx destroy = nullptr; FnPop pop = nullptr;
    bool make(int device, std::string& why) {
        // the versioned symbols from the driver library cudart already loaded (no link-time dependency on libcuda)
        void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_NOLOAD);
        if (!h) h = dlopen("libcuda.so.1", RTLD_NOW);
        if (!h) { why = "no libcuda.so.1"; return false; }
        const char* names[4] = {"cuDeviceGet", "cuCtxCreate_v2", "cuCtxDestroy_v2", "cuCtxPopCurrent_v2"};
        void* f[4] = {nullptr, nullptr, nullptr, nullptr};
        for (int i = 0; i < 4; ++i) if (!(f[i] = dlsym(h, names[i]))) { why = std::string("no driver symbol ") + names[i]; return false; }
        CUdevice d;
        CUresult r = ((FnDevGet) f[0])(&d, device);
        if (r != CUDA_SUCCESS) { why = "cuDeviceGet " + std::to_string((int) r); return false; }
        r = ((FnCreate) f[1])(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, d);
        if (r != CUDA_SUCCESS) { ctx = nullptr; why = "cuCtxCreate " + std::to_string((int) r); return false; }
        destroy = (FnCtx) f[2]; pop = (FnPop) f[3];   // cuCtxCreate made the context current on this thread
        return true;
    }
    ~DrvCtx() { if (ctx) { CUcontext c; pop(&c); destroy(ctx); } }
};

// ---------------------------------------------------------------------------------------------------------------- arena
struct Arena {
    char* base = nullptr; size_t cap = 0, top = 0;
    template <class T> T* get(size_t n) {
        const size_t b = (n * sizeof(T) + 255) & ~size_t(255);
        if (top + b > cap) throw std::runtime_error("arena: " + std::to_string((top + b) >> 20) + " MiB of " + std::to_string(cap >> 20));
        T* p = reinterpret_cast<T*>(base + top); top += b; return p;
    }
    size_t left() const { return cap - top; }
};

// ---------------------------------------------------------------------------------------------------------------- device helpers
struct ChunkV {
    const AD::Read* rd; const u32* ops; const u8* seq; const u64* lq; const u8* ref; const u64* refOff; const u64* refLen;
    const int32_t* wlo; const int32_t* whi;   // per entry: the positions whose sites it owns (Contig::lo, hi)
    int skipN, useQual;
};
__constant__ u8 kComp16[16] = {0, 8, 4, 12, 2, 10, 6, 14, 1, 9, 5, 13, 3, 11, 7, 15};
__device__ __forceinline__ int baseIdx(u8 c) { return c == 1 ? 0 : c == 2 ? 1 : c == 4 ? 2 : c == 8 ? 3 : -1; }
__device__ __forceinline__ u8 refBase(const ChunkV& C, u32 ci, int32_t pos) {
    if (pos < 0 || (u64) pos >= C.refLen[ci]) return 4;
    const u8 v = C.ref[C.refOff[ci] + (u64) pos];
    return v < 4 ? v : 4;
}
__device__ __forceinline__ bool passQ(const ChunkV& C, const AD::Read& x, u32 ri, u8 code) {   // allsite.h pass_q
    if (C.skipN && code == 15) return false;
    if (C.useQual && (ri >> 6) < x.nlq && ((C.lq[x.lq + (ri >> 6)] >> (ri & 63)) & 1ull)) return false;
    return true;
}
// f(pos, nt16 code, record index) for every aligned base in order (allsite.h walk); f returns false to stop
template <class F> __device__ __forceinline__ void walkRead(const ChunkV& C, const AD::Read& x, F&& f) {
    const u32* o = C.ops + x.ops; const u8* s = C.seq + x.seq;
    int32_t rpos = x.pos; int qpos = 0;
    for (int i = 0; i < x.nops; ++i) {
        const u32 c = o[i]; const int op = c & 15; const int32_t len = (int32_t) (c >> 4);
        if (op == 0 || op == 7 || op == 8) {
            for (int32_t j = 0; j < len; ++j) {
                const int qp = qpos + j; const int idx = x.rev ? (int) x.rlen - 1 - qp : qp;
                u8 code = 15;
                if (idx >= 0 && idx < (int) x.rlen) { code = (s[idx >> 1] >> ((idx & 1) * 4)) & 15; if (x.rev) code = kComp16[code]; }
                if (!f(rpos + j, code, (u32) idx)) return;
            }
            rpos += len; qpos += len;
        } else if (op == 1 || op == 4) qpos += len;
        else if (op == 2 || op == 3) rpos += len;
    }
}

// ---- discovery
__global__ void kDiscCount(ChunkV C, u32 n, u32* cnt) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const AD::Read x = C.rd[i]; u32 k = 0;
    walkRead(C, x, [&](int32_t pos, u8 code, u32 ri) {
        if (pos < C.wlo[x.ci] || pos >= C.whi[x.ci]) return true;
        const int bi = baseIdx(code); if (bi < 0) return true;
        const u8 rf = refBase(C, x.ci, pos); if (rf > 3 || bi == rf || !passQ(C, x, ri, code)) return true;
        ++k; return true;
    });
    cnt[i] = k;
}
__global__ void kDiscWrite(ChunkV C, u32 n, const u64* off, u64* keyB, u64* umi, int shPos, int shCi, int bcBits) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const AD::Read x = C.rd[i]; u64 o = off[i];
    const u64 hiPart = ((u64) x.ci << shCi) | (u64) (u32) x.bc;
    walkRead(C, x, [&](int32_t pos, u8 code, u32 ri) {
        if (pos < C.wlo[x.ci] || pos >= C.whi[x.ci]) return true;
        const int bi = baseIdx(code); if (bi < 0) return true;
        const u8 rf = refBase(C, x.ci, pos); if (rf > 3 || bi == rf || !passQ(C, x, ri, code)) return true;
        keyB[o] = hiPart | ((u64) (u32) pos << shPos) | ((u64) bi << bcBits); umi[o] = x.umi3; ++o; return true;
    });
}
// per sorted observation: a new barcode (new key or barcode) in the low half... new molecule in bit 0, new barcode in bit 32
__global__ void kDiscVals(const u64* B, const u64* U, u32 n, u64* val) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const bool cell = i == 0 || B[i] != B[i - 1];
    const bool mol = cell || U[i] != U[i - 1];
    val[i] = (u64) mol | ((u64) cell << 32);
}
struct DSite { u64 cp; u32 mol, cells, alt, pad; };
// one thread per (contig, pos, alt) key: the first key of each position picks the passing alternate base with the most molecules
// (ties: the lower base code, as the host's scan in key order)
__global__ void kSites(const u64* K, const u64* agg, const int* nU_, int minM, int minC, u8* flag, DSite* sit) {
    const int nU = *nU_;
    const int u = blockIdx.x * blockDim.x + threadIdx.x; if (u >= nU) return;
    const u64 posK = K[u] >> 2;
    flag[u] = 0;
    if (u > 0 && (K[u - 1] >> 2) == posK) return;
    u32 bm = 0, bc = 0; int bb = -1;
    for (int v = u; v < nU && (K[v] >> 2) == posK; ++v) {
        const u32 mol = (u32) (agg[v] & 0xffffffffull), cells = (u32) (agg[v] >> 32);
        if ((int) mol >= minM && (int) cells >= minC && mol > bm) { bm = mol; bc = cells; bb = (int) (K[v] & 3); }
    }
    if (bb >= 0) { flag[u] = 1; sit[u] = DSite{posK, bm, bc, (u32) bb, 0}; }
}

// ---- counting
__global__ void kRange(ChunkV C, u32 n, const int32_t* spos, const u32* sOff, u32* rf, u32* rl) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const AD::Read x = C.rd[i];
    const u32* o = C.ops + x.ops; long long end = x.pos;
    for (int k = 0; k < x.nops; ++k) { const int op = o[k] & 15; if (op == 0 || op == 2 || op == 3 || op == 7 || op == 8) end += o[k] >> 4; }
    const u32 lo = sOff[x.ci], hi = sOff[x.ci + 1];
    auto lb = [&](long long v) { u32 a = lo, b = hi; while (a < b) { const u32 m = (a + b) >> 1; if ((long long) spos[m] < v) a = m + 1; else b = m; } return a; };
    const u32 f = lb(x.pos), l = lb(end);
    rf[i] = f < l ? f : 0; rl[i] = f < l ? l : 0;
}
__device__ __forceinline__ u64 umiPack3(u64 u3, int L) {   // umi_pack3 of allsite.h's umi_chars (codes over 3 are N)
    u64 p = 1;
    for (int i = 0; i < L; ++i) { const u64 c = (u3 >> (3 * i)) & 7; p = (p << 3) | (c < 4 ? c : 4); }
    return p;
}
template <bool Write>
__global__ void kHits(ChunkV C, u32 n, const int32_t* spos, const u8* sra, const u32* rf, const u32* rl, u32 s0, u32 s1,
                      u32* cnt, const u64* off, u64* hi, u64* lo, int bcBits, int L) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const u32 a = max(rf[i], s0), b = min(rl[i], s1);
    if (a >= b) { if (!Write) cnt[i] = 0; return; }
    const AD::Read x = C.rd[i];
    u32 cur = a, k = 0; u64 o = Write ? off[i] : 0;
    const u64 p3 = Write ? (umiPack3(x.umi3, L) << 2) : 0;
    walkRead(C, x, [&](int32_t pos, u8 code, u32 ri) {
        while (cur < b && spos[cur] < pos) ++cur;
        if (cur >= b) return false;
        if (spos[cur] != pos || !passQ(C, x, ri, code)) return true;
        if (Write) {
            const int bi = baseIdx(code); const u32 ra = sra[cur];
            const u64 cls = bi == (int) (ra >> 2) ? 1 : (bi == (int) (ra & 3) ? 0 : 2);
            hi[o] = ((u64) (cur - s0) << bcBits) | (u64) (u32) x.bc; lo[o] = p3 | cls; ++o;
        } else ++k;
        return true;
    });
    if (!Write) cnt[i] = k;
}
__global__ void kHist(const u32* rf, const u32* rl, const u32* cnt, u32 n, u64* hist) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (rf[i] < rl[i] && cnt[i]) atomicAdd(&hist[rf[i]], (u64) cnt[i]);
}
__global__ void kEntryHead(const u64* hi, const u64* lo, u32 n, u8* f) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    f[i] = i == 0 || hi[i] != hi[i - 1] || (lo[i] >> 2) != (lo[i - 1] >> 2);
}
// per (barcode, site, UMI): the class counts of its hits, each saturated at 255 (SnpConsensus::add), and their sum (the read count R9 sorts by)
__global__ void kEntries(const u64* hi, const u64* lo, const u32* eS, const int* nE_, u32 nHits, u64* ehi, u64* eumi, u32* en, u32* er, u32* par) {
    const int nE = *nE_;
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nE) return;
    const u32 i0 = eS[e], i1 = e + 1 < nE ? eS[e + 1] : nHits;
    u32 c[3] = {0, 0, 0};
    for (u32 i = i0; i < i1; ++i) { const u32 k = (u32) (lo[i] & 3); if (k < 3 && c[k] < 255) ++c[k]; }
    ehi[e] = hi[i0]; eumi[e] = lo[i0] >> 2; en[e] = c[0] | (c[1] << 8) | (c[2] << 16); er[e] = c[0] + c[1] + c[2]; par[e] = (u32) e;
}
__global__ void kGroupHead(const u64* ehi, const int* nE_, u8* f) {
    const int nE = *nE_;
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nE) return;
    f[e] = e == 0 || ehi[e] != ehi[e - 1];
}
__device__ __forceinline__ bool oneMismatch(u64 a, u64 b) {   // umi3_one_mismatch
    if (a == b || __clzll((long long) a) != __clzll((long long) b)) return false;
    const u64 d = a ^ b;
    return __popcll((d | (d >> 1) | (d >> 2)) & 0x1249249249249249ULL) == 1;
}
// umi3_directional_roots joins i before j (reads descending) when count_i >= 2 count_j - 1: for two UMIs, larger >= 2 smaller - 1
__device__ __forceinline__ bool joins(u32 ca, u32 cb) { const long long mx = max(ca, cb), mn = min(ca, cb); return mx >= 2 * mn - 1; }
__device__ __forceinline__ u32 ufFind(u32* p, u32 x) {
    volatile u32* vp = p;
    u32 y;
    while ((y = vp[x]) != x) { const u32 z = vp[y]; if (z != y) vp[x] = z; x = y; }
    return x;
}
__device__ __forceinline__ void ufUnite(u32* p, u32 a, u32 b) {
    for (;;) {
        a = ufFind(p, a); b = ufFind(p, b);
        if (a == b) return;
        if (a < b) { const u32 t = a; a = b; b = t; }   // the larger root hooks under the smaller
        const u32 old = atomicCAS(&p[a], a, b);
        if (old == a) return;
        a = old;
    }
}
__global__ void kUnion(const u64* eumi, const u32* er, const u32* gid, const u32* gS, const int* nE_, const int* nG_, u32* par, int L) {
    const int nE = *nE_, nG = *nG_;
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nE) return;
    const u32 g = gid[e] - 1;
    const u32 g0 = gS[g], g1 = (int) g + 1 < nG ? gS[g + 1] : (u32) nE;
    if (g1 - g0 < 2) return;
    const u64 ui = eumi[e]; const u32 ci = er[e];
    if (g1 - g0 <= 64) {
        for (u32 j = e + 1; j < g1; ++j) if (oneMismatch(ui, eumi[j]) && joins(ci, er[j])) ufUnite(par, (u32) e, j);
        return;
    }
    for (int p = 0; p < L; ++p) {   // a deep group: the UMIs one base away, by binary search (the group's UMIs ascend)
        const u64 cur = (ui >> (3 * p)) & 7ull;
        for (u64 bb = 0; bb < 8; ++bb) {
            if (bb == cur) continue;
            const u64 cand = ui ^ ((cur ^ bb) << (3 * p));
            u32 a = g0, b = g1;
            while (a < b) { const u32 m = (a + b) >> 1; if (eumi[m] < cand) a = m + 1; else b = m; }
            if (a < g1 && eumi[a] == cand && a > (u32) e && joins(ci, er[a])) ufUnite(par, (u32) e, a);
        }
    }
}
__global__ void kRootAgg(u32* par, const u32* en, const int* nE_, u32* agg) {
    const int nE = *nE_;
    const int e = blockIdx.x * blockDim.x + threadIdx.x; if (e >= nE) return;
    const u32 r = ufFind(par, (u32) e);
    const u32 n = en[e];
    if (n & 0xff) atomicAdd(&agg[3 * r + 0], n & 0xff);
    if ((n >> 8) & 0xff) atomicAdd(&agg[3 * r + 1], (n >> 8) & 0xff);
    if ((n >> 16) & 0xff) atomicAdd(&agg[3 * r + 2], (n >> 16) & 0xff);
}
// per (barcode, site): the vote of each component (SnpConsensus::finalize's vote), DP and AD molecules, ambiguous and other molecules
struct GOut { u32 bc, bs, dp, ad; };   // barcode, site index within the block, DP and AD molecules
__global__ void kGroups(const u64* ehi, const u32* gS, const int* nE_, const int* nG_, const u32* par, const u32* agg,
                        u32 s0, const u32* sOff, u32 nC, int bcBits, GOut* go, u8* fdp, u8* fad, u64* stat) {
    const int nE = *nE_, nG = *nG_;
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    u32 dp = 0, ad = 0, amb = 0, oth = 0;
    if (g < nG) {
        const u32 e0 = gS[g], e1 = g + 1 < nG ? gS[g + 1] : (u32) nE;
        for (u32 e = e0; e < e1; ++e) {
            if (par[e] != e) continue;
            const u32* n = agg + 3 * (size_t) e;
            int best = 0; for (int c = 1; c < 3; ++c) if (n[c] > n[best]) best = c;
            bool tie = false; for (int c = 0; c < 3; ++c) if (c != best && n[c] == n[best]) tie = true;
            if (tie) ++amb; else if (best == 2) ++oth; else { ++dp; if (best == 1) ++ad; }
        }
        const u64 h = ehi[e0];
        const u32 s = s0 + (u32) (h >> bcBits);
        go[g] = GOut{(u32) (h & ((1ull << bcBits) - 1)), s - s0, dp, ad};
        fdp[g] = dp > 0; fad[g] = ad > 0;
    }
    u64 v[4] = {amb, oth, dp, ad};   // warp sums, one atomic per warp and counter
    for (int k = 0; k < 4; ++k) {
        for (int o = 16; o > 0; o >>= 1) v[k] += __shfl_down_sync(0xffffffffu, v[k], o);
        if ((threadIdx.x & 31) == 0 && v[k]) atomicAdd((unsigned long long*) &stat[k], (unsigned long long) v[k]);
    }
}
// The block's counts in the spill layout, built on the device: records (groups with DP > 0, in (site, barcode) order), their varint
// lengths, per-site barcode counts and byte offsets, then the bytes
struct ToU64 { const u32* c; u32 n; __host__ __device__ u64 operator()(u64 i) const { return i < n ? (u64) c[i] : 0ull; } };
struct ShrOp { int s; __host__ __device__ u64 operator()(u64 v) const { return v >> s; } };
struct U8ToU32 { __host__ __device__ u32 operator()(u8 v) const { return v; } };
struct ToU32Flag { const u8* f; u32 n; __host__ __device__ u32 operator()(u32 i) const { return i < n ? (u32) f[i] : 0u; } };
struct ToU64Plain { const u64* c; u32 n; __host__ __device__ u64 operator()(u64 i) const { return i < n ? c[i] : 0ull; } };
__device__ __forceinline__ u32 vlen(u32 v) { return v < (1u << 7) ? 1 : v < (1u << 14) ? 2 : v < (1u << 21) ? 3 : v < (1u << 28) ? 4 : 5; }
__device__ __forceinline__ u8* vput(u8* p, u32 v) { while (v >= 128) { *p++ = (u8) ((v & 127) | 128); v >>= 7; } *p++ = (u8) v; return p; }
__global__ void kRecs(const GOut* go, const u8* f, const u32* pre, const int* nG_, GOut* rec, u32* cntS, u64* sdp, u64* sad) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x; if (g >= *nG_ || !f[g]) return;
    const GOut x = go[g]; rec[pre[g]] = x; atomicAdd(&cntS[x.bs], 1u);
    atomicAdd((unsigned long long*) &sdp[x.bs], (unsigned long long) x.dp);
    if (x.ad) atomicAdd((unsigned long long*) &sad[x.bs], (unsigned long long) x.ad);
}
__global__ void kRecLen(const GOut* rec, u32 n, u32* len) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const GOut x = rec[i]; const u32 prev = (i > 0 && rec[i - 1].bs == x.bs) ? rec[i - 1].bc : 0u;
    len[i] = vlen(x.bc - prev) + vlen(x.dp) + vlen(x.ad);
}
// per site: its bytes = varint(count) + its records' bytes (records [rs[s], rs[s + 1]) of the exclusive length sums lx)
__global__ void kSiteBytes(const u32* cntS, const u32* rs, const u64* lx, const u8* kp, u32 nsb, u64* tb) {   // dropped sites: no bytes
    const u32 b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= nsb) return;
    tb[b] = kp[b] ? vlen(cntS[b]) + (lx[rs[b + 1]] - lx[rs[b]]) : 0;
}
__global__ void kSiteHead(const u32* cntS, const u64* so, const u8* kp, u32 nsb, u8* bytes) {
    const u32 b = blockIdx.x * blockDim.x + threadIdx.x; if (b >= nsb || !kp[b]) return;
    vput(bytes + so[b], cntS[b]);
}
__global__ void kRecWrite(const GOut* rec, u32 n, const u32* cntS, const u32* rs, const u64* lx, const u64* so, const u8* kp, u8* bytes) {
    const u32 i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const GOut x = rec[i]; if (!kp[x.bs]) return;
    const u32 prev = (i > 0 && rec[i - 1].bs == x.bs) ? rec[i - 1].bc : 0u;
    u8* p = bytes + so[x.bs] + vlen(cntS[x.bs]) + (lx[i] - lx[rs[x.bs]]);
    p = vput(p, x.bc - prev); p = vput(p, x.dp); vput(p, x.ad);
}
// Device to host through two pinned slices: the device fills one while host threads copy the other out
struct PinnedCopy {
    size_t SL = 64ull << 20, minPer = 4ull << 20;   // test knob SINGLET_ALLSITE_GPU_SLICE_BYTES shrinks both
    char* buf = nullptr; cudaEvent_t ev[2] = {nullptr, nullptr}; unsigned nth = 1; size_t turn = 0;
    void init(unsigned th) {
        nth = std::max(1u, std::min(th, 16u));
        if (const char* k = std::getenv("SINGLET_ALLSITE_GPU_SLICE_BYTES")) { SL = std::max<size_t>(64, std::strtoull(k, nullptr, 10)); minPer = 64; }
        if (cudaMallocHost(&buf, 2 * SL) != cudaSuccess) { buf = nullptr; cudaGetLastError(); return; }
        for (auto& e : ev) if (cudaEventCreateWithFlags(&e, cudaEventDisableTiming) != cudaSuccess) { release(); cudaGetLastError(); return; }
    }
    void release() { for (auto& e : ev) if (e) { cudaEventDestroy(e); e = nullptr; } if (buf) { cudaFreeHost(buf); buf = nullptr; } }
    void pcopy(char* d, const char* s, size_t n) const {
        const unsigned t = (unsigned) std::min<size_t>(nth, n / minPer);
        if (t <= 1) { std::memcpy(d, s, n); return; }
        const size_t per = ((n + t - 1) / t + 63) & ~size_t(63);   // t * per >= n
        std::vector<std::thread> th;
        for (unsigned i = 1; i < t; ++i) { const size_t a = i * per; if (a >= n) break; th.emplace_back([=] { std::memcpy(d + a, s + a, std::min(per, n - a)); }); }
        std::memcpy(d, s, std::min(per, n));
        for (auto& x : th) x.join();
    }
    void drain() { for (auto& e : ev) if (e) AS_CK(cudaEventSynchronize(e)); }
    void up(void* dst, const void* src, size_t bytes, cudaStream_t st) {   // host to device; the caller syncs the stream before reading dst
        if (!bytes) return;
        if (!buf) { AS_CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st)); return; }
        char* d = (char*) dst; const char* s = (const char*) src;
        for (size_t k = 0, n = (bytes + SL - 1) / SL; k < n; ++k) {
            const size_t len = std::min(SL, bytes - k * SL);
            const int slot = (int) (turn++ & 1);
            AS_CK(cudaEventSynchronize(ev[slot]));   // the slot's last transfer is done (an unrecorded event returns at once)
            pcopy(buf + slot * SL, s + k * SL, len);
            AS_CK(cudaMemcpyAsync(d + k * SL, buf + slot * SL, len, cudaMemcpyHostToDevice, st));
            AS_CK(cudaEventRecord(ev[slot], st));
        }
    }
    void copy(void* dst, const void* src, size_t bytes, cudaStream_t st) {
        if (!bytes) return;
        if (buf) drain();
        if (!buf) { AS_CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st)); return; }
        char* d = (char*) dst; const char* s = (const char*) src;
        const size_t n = (bytes + SL - 1) / SL;
        for (size_t k = 0; k <= n; ++k) {
            if (k < n) {   // slot k & 1 was emptied by the host in step k - 1
                AS_CK(cudaMemcpyAsync(buf + (k & 1) * SL, s + k * SL, std::min(SL, bytes - k * SL), cudaMemcpyDeviceToHost, st));
                AS_CK(cudaEventRecord(ev[k & 1], st));
            }
            if (k > 0) { const size_t j = k - 1; AS_CK(cudaEventSynchronize(ev[j & 1])); pcopy(d + j * SL, buf + (j & 1) * SL, std::min(SL, bytes - j * SL)); }
        }
    }
};

inline unsigned gridFor(u64 n, unsigned b = 256) { return (unsigned) std::max<u64>(1, (n + b - 1) / b); }

}  // namespace

bool allsiteDeviceRun(const AD::Input& in, AD::Output& out) {
    const auto T0 = std::chrono::steady_clock::now();
    out = AD::Output{};
    out.contigs.resize(in.contigs.size());
    if (const char* e = std::getenv("SINGLET_ALLSITE_GPU")) if (e[0] == '0') { out.why = "SINGLET_ALLSITE_GPU=0"; return false; }
    if (deviceCount() <= 0) { out.why = "no CUDA device"; return false; }
    if (in.umi_len < 1 || in.umi_len > 20) { out.why = "UMI length " + std::to_string(in.umi_len); return false; }
    if (in.n_barcodes == 0 || in.contigs.empty()) { out.why = "no barcodes or reads"; return false; }
    const int bcBits = bitsFor(in.n_barcodes - 1), L = in.umi_len, loBits = 3 * L + 3;
    u32 nCt = (u32) in.contigs.size();

    // the engine's device lock (a short holder). With SINGLET_GPU_PILEUP_UNLOCK=late the GPU pileup still holds it here, so the pass
    // takes the device without it (SINGLET_ALLSITE_GPU_LOCK=0 does the same); the arena is sized from the memory free either way
    int lockFd = -1, device = 0;
    const char* ul = std::getenv("SINGLET_GPU_PILEUP_UNLOCK"); const char* lk = std::getenv("SINGLET_ALLSITE_GPU_LOCK");
    if ((ul && std::string(ul) == "late") || (lk && lk[0] == '0')) {
        if (const char* dv = std::getenv("SINGLET_ENGINE_GPU_DEVICE")) device = std::atoi(dv);
    } else device = lockDevice(&lockFd);
    struct Unlock { int fd; ~Unlock() { unlockDevice(fd); } } unlock{lockFd};
    DrvCtx ctx;
    if (!ctx.make(device, out.why)) return false;
    Arena A;
    struct Free { Arena& a; ~Free() { if (a.base) cudaFree(a.base); } } freeA{A};
    cudaStream_t st = nullptr;
    struct SFree { cudaStream_t& s; ~SFree() { if (s) cudaStreamDestroy(s); } } freeS{st};
    PinnedCopy d2h;
    struct PFree { PinnedCopy& p; ~PFree() { p.release(); } } freeP{d2h};
    try {
        AS_CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
        d2h.init(in.threads);
        size_t fr = 0, tot = 0;
        AS_CK(cudaMemGetInfo(&fr, &tot));
        double frac = 0.6, maxGB = 24.0;
        if (const char* e = std::getenv("SINGLET_ALLSITE_GPU_FRAC")) { const double v = atof(e); if (v > 0 && v <= 0.95) frac = v; }
        if (const char* e = std::getenv("SINGLET_ALLSITE_GPU_MAXGB")) { const double v = atof(e); if (v > 0) maxGB = v; }
        size_t want = (size_t) std::min<double>(frac * (double) fr - 512.0 * 1048576.0, maxGB * 1073741824.0);
        const size_t minBytes = (size_t) 1536 << 20;
        if ((double) want < (double) minBytes) { out.why = "device memory: " + std::to_string(fr >> 20) + " MiB free"; return false; }
        while (cudaMalloc((void**) &A.base, want) != cudaSuccess) {
            cudaGetLastError(); A.base = nullptr; want = want / 4 * 3;
            if (want < minBytes) { out.why = "device memory: arena allocation failed"; return false; }
        }
        A.cap = want; out.arena_bytes = want;

        // chunks of consecutive contigs whose gathered reads take at most ~45% of the arena
        auto dataBytes = [](const AD::Contig& c) { return AD::contig_bytes(c); };
        {   // a contig too large for one chunk (or past the 32-bit offsets) is replaced by windows that fit, when the caller can split it
            u64 hard = (u64) (0.7 * (double) A.cap); const u64 lim32 = 0x7fffffffull;
            if (const char* e = std::getenv("SINGLET_ALLSITE_GPU_SPLIT_BYTES")) { const u64 v = strtoull(e, nullptr, 10); if (v > 0) hard = std::min(hard, v); }   // test knob
            bool over = false;
            for (const AD::Contig& C : in.contigs) over = over || dataBytes(C) > hard || C.reads >= lim32 || C.ops >= lim32 || C.seqb >= lim32;
            if (over && in.resplit) {
                if (!in.resplit(hard, lim32)) { out.why = "a contig could not be split to fit the arena (" + std::to_string(A.cap >> 20) + " MiB)"; return false; }
                nCt = (u32) in.contigs.size(); out.contigs.assign(nCt, AD::ContigOut{});
            }
        }
        std::vector<u32> cut{0};
        u64 chunkReads = 0x7fffffffull, blockHits = 0;   // test knobs: smaller chunks and blocks than memory allows
        if (const char* e = std::getenv("SINGLET_ALLSITE_GPU_CHUNK_READS")) { const u64 v = strtoull(e, nullptr, 10); if (v > 0) chunkReads = v; }
        if (const char* e = std::getenv("SINGLET_ALLSITE_GPU_BLOCK_HITS")) blockHits = strtoull(e, nullptr, 10);
        // a chunk's op and sequence offsets are 32-bit (Read::ops, Read::seq): chunks also end before either array reaches 2^31
        { const u64 lim = (u64) (0.45 * (double) A.cap), lim32 = 0x7fffffffull; u64 acc = 0, nr = 0, no = 0, ns = 0;
          for (u32 c = 0; c < nCt; ++c) {
              const AD::Contig& C = in.contigs[c];
              const u64 b = dataBytes(C);
              if (b > (u64) (0.7 * (double) A.cap) || C.reads >= lim32 || C.ops >= lim32 || C.seqb >= lim32) {
                  out.why = "a contig's reads exceed the device budget (arena " + std::to_string(A.cap >> 20) + " MiB)"; return false; }
              if (c > cut.back() && (acc + b > lim || c - cut.back() >= 65535 || nr + C.reads > chunkReads || no + C.ops >= lim32 || ns + C.seqb >= lim32)) {
                  cut.push_back(c); acc = 0; nr = 0; no = 0; ns = 0; }
              acc += b; nr += C.reads; no += C.ops; ns += C.seqb;
          }
          cut.push_back(nCt); }
        out.chunks = cut.size() - 1;

        // host staging for the largest chunk (pageable: page-locking GBs costs more than the copy saves)
        u64 mxR = 0, mxO = 0, mxS = 0, mxQ = 0;
        for (size_t k = 0; k + 1 < cut.size(); ++k) {
            u64 r = 0, o = 0, s = 0, q = 0;
            for (u32 c = cut[k]; c < cut[k + 1]; ++c) { r += in.contigs[c].reads; o += in.contigs[c].ops; s += in.contigs[c].seqb; q += in.contigs[c].lqw; }
            mxR = std::max(mxR, r); mxO = std::max(mxO, o); mxS = std::max(mxS, s); mxQ = std::max(mxQ, q);
        }
        if (mxO >= 0xffffffffull || mxS >= 0xffffffffull) { out.why = "chunk arrays past 32-bit offsets"; return false; }
        std::unique_ptr<AD::Read[]> hR(new AD::Read[mxR + 1]); std::unique_ptr<u32[]> hO(new u32[mxO + 1]);
        std::unique_ptr<u8[]> hS(new u8[mxS + 1]); std::unique_ptr<uint64_t[]> hQ(new uint64_t[mxQ + 1]);

        u64 stat[4] = {0, 0, 0, 0};
        const bool TM = std::getenv("SINGLET_ALLSITE_GPU_TIMING") != nullptr;
        auto tStage = std::chrono::steady_clock::now();
        auto stage = [&](double& acc) { if (!TM) return; AS_CK(cudaStreamSynchronize(st)); const auto t = std::chrono::steady_clock::now();
                                         acc += std::chrono::duration<double>(t - tStage).count(); tStage = t; };
        for (size_t k = 0; k + 1 < cut.size(); ++k) {
            const u32 c0 = cut[k], c1 = cut[k + 1], nC = c1 - c0;
            u64 nR = 0, nO = 0, nS = 0, nQ = 0, maxRef = 1;
            std::vector<u64> refOff(nC + 1, 0), refLen(nC, 0); std::vector<int32_t> wlo(nC), whi(nC);
            for (u32 c = c0; c < c1; ++c) {
                const AD::Contig& g = in.contigs[c];
                nR += g.reads; nO += g.ops; nS += g.seqb; nQ += g.lqw;
                wlo[c - c0] = g.lo; whi[c - c0] = g.hi;
                refLen[c - c0] = g.ref ? g.ref_len : 0; refOff[c - c0 + 1] = refOff[c - c0] + refLen[c - c0]; maxRef = std::max<u64>(maxRef, refLen[c - c0]);
            }
            const u32 n = (u32) nR;
            auto tg = std::chrono::steady_clock::now();
            in.gather(c0, c1, hR.get(), hO.get(), hS.get(), hQ.get());
            out.sec_gather += secSince(tg);
            const size_t mark0 = A.top;
            auto tu = std::chrono::steady_clock::now();
            AD::Read* dR = A.get<AD::Read>(nR); u32* dO = A.get<u32>(nO + 1); u8* dS = A.get<u8>(nS + 1); u64* dQ = A.get<u64>(nQ + 1);
            u8* dRef = A.get<u8>(refOff[nC] + 1); u64* dRefOff = A.get<u64>(nC + 1); u64* dRefLen = A.get<u64>(nC + 1);
            int32_t* dWlo = A.get<int32_t>(nC + 1); int32_t* dWhi = A.get<int32_t>(nC + 1);
            d2h.up(dR, hR.get(), nR * sizeof(AD::Read), st);
            d2h.up(dO, hO.get(), (size_t) nO * 4, st);
            d2h.up(dS, hS.get(), (size_t) nS, st);
            d2h.up(dQ, hQ.get(), (size_t) nQ * 8, st);
            for (u32 c = c0; c < c1; ++c) if (refLen[c - c0]) d2h.up(dRef + refOff[c - c0], in.contigs[c].ref, refLen[c - c0], st);
            AS_CK(cudaMemcpyAsync(dRefOff, refOff.data(), (nC + 1) * 8, cudaMemcpyHostToDevice, st));
            AS_CK(cudaMemcpyAsync(dRefLen, refLen.data(), nC * 8, cudaMemcpyHostToDevice, st));
            AS_CK(cudaMemcpyAsync(dWlo, wlo.data(), nC * 4, cudaMemcpyHostToDevice, st));
            AS_CK(cudaMemcpyAsync(dWhi, whi.data(), nC * 4, cudaMemcpyHostToDevice, st));
            AS_CK(cudaStreamSynchronize(st));
            out.sec_upload += secSince(tu);
            ChunkV C{dR, dO, dS, dQ, dRef, dRefOff, dRefLen, dWlo, dWhi, in.skip_n, in.use_qual};
            u32* dCnt = A.get<u32>(nR + 1); u64* dOff = A.get<u64>(nR + 1); u32* dRf = A.get<u32>(nR + 1); u32* dRl = A.get<u32>(nR + 1);
            int* dNum = A.get<int>(4); u64* dStat = A.get<u64>(4);
            AS_CK(cudaMemsetAsync(dStat, 0, 32, st));
            auto scanCounts = [&](u64& total) {   // dOff = exclusive prefix of dCnt over [0, n], total = dOff[n]
                cub::CountingInputIterator<u64> ci(0);
                cub::TransformInputIterator<u64, ToU64, cub::CountingInputIterator<u64>> it(ci, ToU64{dCnt, n});
                size_t tb = 0; AS_CK(cub::DeviceScan::ExclusiveSum(nullptr, tb, it, dOff, (int) n + 1, st));
                const size_t m = A.top; void* tmp = A.get<char>(tb);
                AS_CK(cub::DeviceScan::ExclusiveSum(tmp, tb, it, dOff, (int) n + 1, st)); A.top = m;
                AS_CK(cudaMemcpyAsync(&total, dOff + n, 8, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
            };

            // ---- 1. discovery
            auto td = std::chrono::steady_clock::now();
            const int posBits = bitsFor(maxRef - 1), ciBits = bitsFor(nC - 1);
            if (ciBits + posBits + 2 + bcBits > 64) { out.why = "discovery key past 64 bits"; return false; }
            const int shPos = bcBits + 2, shCi = shPos + posBits, keyBits = shCi + ciBits;
            kDiscCount<<<gridFor(n), 256, 0, st>>>(C, n, dCnt);
            u64 nObs = 0; scanCounts(nObs);
            out.obs += nObs;
            std::vector<DSite> hSites;
            if (nObs >= 0x7fffffffull) { out.why = "too many mismatch observations for one chunk"; return false; }
            if (nObs > 0) {
                const size_t m = A.top;
                u64* kB = A.get<u64>(nObs); u64* kB2 = A.get<u64>(nObs); u64* kU = A.get<u64>(nObs); u64* kU2 = A.get<u64>(nObs);
                kDiscWrite<<<gridFor(n), 256, 0, st>>>(C, n, dOff, kB, kU, shPos, shCi, bcBits);
                cub::DoubleBuffer<u64> dbU(kU, kU2), dbB(kB, kB2);
                size_t tb = 0; AS_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, dbU, dbB, (int) nObs, 0, 64, st));
                size_t tb2 = 0; AS_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb2, dbB, dbU, (int) nObs, 0, keyBits, st));
                void* tmp = A.get<char>(std::max(tb, tb2));
                AS_CK(cub::DeviceRadixSort::SortPairs(tmp, tb, dbU, dbB, (int) nObs, 0, 64, st));        // by UMI, then (stable) by key
                AS_CK(cub::DeviceRadixSort::SortPairs(tmp, tb2, dbB, dbU, (int) nObs, 0, keyBits, st));
                const u64* B = dbB.Current(); const u64* U = dbU.Current();
                u64* val = dbB.Alternate();   // free now
                kDiscVals<<<gridFor(nObs), 256, 0, st>>>(B, U, (u32) nObs, val);
                u64* uk = dbU.Alternate(); u64* ua = A.get<u64>(nObs);
                cub::TransformInputIterator<u64, ShrOp, const u64*> kit(B, ShrOp{bcBits});
                size_t tr = 0; AS_CK(cub::DeviceReduce::ReduceByKey(nullptr, tr, kit, uk, val, ua, dNum, cub::Sum(), (int) nObs, st));
                void* tmp2 = A.get<char>(tr);
                AS_CK(cub::DeviceReduce::ReduceByKey(tmp2, tr, kit, uk, val, ua, dNum, cub::Sum(), (int) nObs, st));
                int nU = 0; AS_CK(cudaMemcpyAsync(&nU, dNum, 4, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                u8* fl = A.get<u8>(nU + 1); DSite* sit = A.get<DSite>(nU + 1); DSite* sitc = A.get<DSite>(nU + 1);
                kSites<<<gridFor(nU), 256, 0, st>>>(uk, ua, dNum, in.min_alt_molecules, in.min_alt_cells, fl, sit);
                size_t ts = 0; AS_CK(cub::DeviceSelect::Flagged(nullptr, ts, sit, fl, sitc, dNum + 1, nU, st));
                void* tmp3 = A.get<char>(ts);
                AS_CK(cub::DeviceSelect::Flagged(tmp3, ts, sit, fl, sitc, dNum + 1, nU, st));
                int nSel = 0; AS_CK(cudaMemcpyAsync(&nSel, dNum + 1, 4, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                hSites.resize(nSel);
                if (nSel) AS_CK(cudaMemcpyAsync(hSites.data(), sitc, nSel * sizeof(DSite), cudaMemcpyDeviceToHost, st));
                AS_CK(cudaStreamSynchronize(st));
                A.top = m;
            }
            // the chunk's sites by (contig, position); sOff[c]: the first site of contig c
            const u32 nSite = (u32) hSites.size();
            std::vector<u32> sOff(nC + 1, 0);
            std::vector<int32_t> spos(nSite); std::vector<u8> sra(nSite);
            for (u32 i = 0; i < nSite; ++i) {
                const u32 c = (u32) (hSites[i].cp >> posBits); const int32_t p = (int32_t) (hSites[i].cp & ((1ull << posBits) - 1));
                const AD::Contig& g = in.contigs[c0 + c];
                const u8 rv = g.ref[p]; const u8 ref = rv < 4 ? rv : 4;   // discovery kept only ACGT reference bases
                spos[i] = p; sra[i] = (u8) ((ref & 3) | (hSites[i].alt << 2)); sOff[c + 1]++;
                AD::ContigOut& co = out.contigs[c0 + c];
                co.pos.push_back(p); co.ref.push_back(ref); co.alt.push_back((u8) hSites[i].alt); co.mol.push_back(hSites[i].mol); co.cells.push_back(hSites[i].cells);
            }
            for (u32 c = 0; c < nC; ++c) sOff[c + 1] += sOff[c];
            out.sec_disc += secSince(td);
            if (in.progress) in.progress();
            if (nSite == 0) { A.top = mark0; continue; }

            // ---- 2. counting
            auto tc = std::chrono::steady_clock::now();
            int32_t* dSpos = A.get<int32_t>(nSite); u8* dSra = A.get<u8>(nSite); u32* dSoff = A.get<u32>(nC + 1);
            AS_CK(cudaMemcpyAsync(dSpos, spos.data(), nSite * 4, cudaMemcpyHostToDevice, st));
            AS_CK(cudaMemcpyAsync(dSra, sra.data(), nSite, cudaMemcpyHostToDevice, st));
            AS_CK(cudaMemcpyAsync(dSoff, sOff.data(), (nC + 1) * 4, cudaMemcpyHostToDevice, st));
            kRange<<<gridFor(n), 256, 0, st>>>(C, n, dSpos, dSoff, dRf, dRl);
            // the hits of each block must fit: about 100 bytes per hit (two sort buffers of keys and values, the entry and group arrays)
            const size_t workMark = A.top;
            const u64 capHits = std::min<u64>(0x7ffffff0ull, (A.left() > ((size_t) 64 << 20) ? (A.left() - ((size_t) 64 << 20)) : 0) / 112);
            if (capHits < 1000000) { out.why = "too little device memory left for counting"; return false; }
            const u64 capHitsUse = blockHits ? std::min<u64>(capHits, blockHits) : capHits;
            kHits<false><<<gridFor(n), 256, 0, st>>>(C, n, dSpos, dSra, dRf, dRl, 0, nSite, dCnt, nullptr, nullptr, nullptr, bcBits, L);
            u64 allHits = 0; scanCounts(allHits);
            std::vector<std::pair<u32, u32>> todo;   // site ranges, processed from the back
            if (allHits <= capHitsUse) todo.push_back({0, nSite});
            else {   // cut by the hits of the reads whose first site falls in each range (spliced reads cross cuts; blocks that overflow split)
                u64* dHist = A.get<u64>(nSite);
                AS_CK(cudaMemsetAsync(dHist, 0, nSite * 8, st));
                kHist<<<gridFor(n), 256, 0, st>>>(dRf, dRl, dCnt, n, dHist);
                std::vector<u64> hist(nSite);
                AS_CK(cudaMemcpyAsync(hist.data(), dHist, nSite * 8, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                A.top = workMark;
                const u64 target = std::max<u64>(1, capHitsUse * 6 / 10);
                std::vector<std::pair<u32, u32>> fw;
                u32 a = 0; u64 acc = 0;
                for (u32 i = 0; i < nSite; ++i) { acc += hist[i]; if (acc >= target && i + 1 < nSite) { fw.push_back({a, i + 1}); a = i + 1; acc = 0; } }
                fw.push_back({a, nSite});
                for (auto it = fw.rbegin(); it != fw.rend(); ++it) todo.push_back(*it);
            }
            while (!todo.empty()) {
                const u32 s0 = todo.back().first, s1 = todo.back().second; todo.pop_back();
                A.top = workMark;
                if (TM) { AS_CK(cudaStreamSynchronize(st)); tStage = std::chrono::steady_clock::now(); }
                kHits<false><<<gridFor(n), 256, 0, st>>>(C, n, dSpos, dSra, dRf, dRl, s0, s1, dCnt, nullptr, nullptr, nullptr, bcBits, L);
                u64 N = 0; scanCounts(N);
                if (N > capHitsUse) {
                    if (s1 - s0 < 2) { out.why = "one site's hits exceed the device budget"; return false; }
                    const u32 mid = s0 + (s1 - s0) / 2; todo.push_back({mid, s1}); todo.push_back({s0, mid}); continue;
                }
                out.blocks++;
                if (N == 0) continue;
                out.hits += N;
                const int sBits = bitsFor(s1 - s0 - 1) + bcBits;
                u64* h1 = A.get<u64>(N); u64* h2 = A.get<u64>(N); u64* l1 = A.get<u64>(N); u64* l2 = A.get<u64>(N);
                kHits<true><<<gridFor(n), 256, 0, st>>>(C, n, dSpos, dSra, dRf, dRl, s0, s1, nullptr, dOff, h1, l1, bcBits, L);
                stage(out.sec_hits);
                cub::DoubleBuffer<u64> dbL(l1, l2), dbH(h1, h2);
                size_t tb = 0; AS_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb, dbL, dbH, (int) N, 0, loBits, st));
                size_t tb2 = 0; AS_CK(cub::DeviceRadixSort::SortPairs(nullptr, tb2, dbH, dbL, (int) N, 0, sBits, st));
                size_t tm = A.top; void* tmp = A.get<char>(std::max(tb, tb2));
                AS_CK(cub::DeviceRadixSort::SortPairs(tmp, tb, dbL, dbH, (int) N, 0, loBits, st));    // by (UMI, class), then (stable) by (site, barcode)
                AS_CK(cub::DeviceRadixSort::SortPairs(tmp, tb2, dbH, dbL, (int) N, 0, sBits, st));
                A.top = tm;
                stage(out.sec_sort);
                const u64* H = dbH.Current(); const u64* Lo = dbL.Current();
                u64* ehi = dbH.Alternate(); u64* eumi = dbL.Alternate();
                u8* f8 = A.get<u8>(N); u32* eS = A.get<u32>(N + 1);
                kEntryHead<<<gridFor(N), 256, 0, st>>>(H, Lo, (u32) N, f8);
                cub::CountingInputIterator<u32> cnt0(0);
                size_t t1 = 0; AS_CK(cub::DeviceSelect::Flagged(nullptr, t1, cnt0, f8, eS, dNum, (int) N, st));
                tm = A.top; tmp = A.get<char>(t1);
                AS_CK(cub::DeviceSelect::Flagged(tmp, t1, cnt0, f8, eS, dNum, (int) N, st)); A.top = tm;
                int nE = 0; AS_CK(cudaMemcpyAsync(&nE, dNum, 4, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                u32* en = A.get<u32>(nE); u32* er = A.get<u32>(nE); u32* par = A.get<u32>(nE); u32* agg = A.get<u32>(3 * (size_t) nE);
                u32* gid = A.get<u32>(nE); u32* gS = A.get<u32>(nE + 1);
                kEntries<<<gridFor(nE), 256, 0, st>>>(H, Lo, eS, dNum, (u32) N, ehi, eumi, en, er, par);
                kGroupHead<<<gridFor(nE), 256, 0, st>>>(ehi, dNum, f8);
                cub::TransformInputIterator<u32, U8ToU32, const u8*> f32(f8, U8ToU32{});
                size_t t2 = 0; AS_CK(cub::DeviceScan::InclusiveSum(nullptr, t2, f32, gid, nE, st));
                size_t t3 = 0; AS_CK(cub::DeviceSelect::Flagged(nullptr, t3, cnt0, f8, gS, dNum + 1, nE, st));
                tm = A.top; tmp = A.get<char>(std::max(t2, t3));
                AS_CK(cub::DeviceScan::InclusiveSum(tmp, t2, f32, gid, nE, st));
                AS_CK(cub::DeviceSelect::Flagged(tmp, t3, cnt0, f8, gS, dNum + 1, nE, st)); A.top = tm;
                int nG = 0; AS_CK(cudaMemcpyAsync(&nG, dNum + 1, 4, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                out.entries += (u64) nE; out.groups += (u64) nG;
                stage(out.sec_entries);
                if (in.directional) kUnion<<<gridFor(nE), 256, 0, st>>>(eumi, er, gid, gS, dNum, dNum + 1, par, L);
                AS_CK(cudaMemsetAsync(agg, 0, 3 * (size_t) nE * 4, st));
                kRootAgg<<<gridFor(nE), 256, 0, st>>>(par, en, dNum, agg);
                                stage(out.sec_r9);
                GOut* go = A.get<GOut>(nG); u8* fad = A.get<u8>(nG);
                kGroups<<<gridFor(nG), 256, 0, st>>>(ehi, gS, dNum, dNum + 1, par, agg, s0, dSoff, nC, bcBits, go, f8, fad, dStat);
                // the block's counts in the spill layout: records (groups with DP > 0) by an exclusive scan of the flags, then per site its
                // barcode count and byte offset, then the bytes; each contig of the block takes the bytes of its sites
                const u32 nsb = s1 - s0;
                u32* pre = A.get<u32>((size_t) nG + 1); u32* cntS = A.get<u32>(nsb); u32* rs = A.get<u32>((size_t) nsb + 1);
                u64* so = A.get<u64>((size_t) nsb + 1); u64* stb = A.get<u64>(nsb);
                u64* sdp = A.get<u64>(nsb); u64* sad = A.get<u64>(nsb); u8* kp = A.get<u8>(nsb);
                AS_CK(cudaMemsetAsync(cntS, 0, (size_t) nsb * 4, st));
                AS_CK(cudaMemsetAsync(sdp, 0, (size_t) nsb * 8, st)); AS_CK(cudaMemsetAsync(sad, 0, (size_t) nsb * 8, st));
                {
                    cub::TransformInputIterator<u32, ToU32Flag, cub::CountingInputIterator<u32>> fi(cub::CountingInputIterator<u32>(0), ToU32Flag{f8, (u32) nG});
                    size_t ts = 0; AS_CK(cub::DeviceScan::ExclusiveSum(nullptr, ts, fi, pre, nG + 1, st));
                    tm = A.top; tmp = A.get<char>(ts); AS_CK(cub::DeviceScan::ExclusiveSum(tmp, ts, fi, pre, nG + 1, st)); A.top = tm;
                }
                u32 nRec = 0; AS_CK(cudaMemcpyAsync(&nRec, pre + nG, 4, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                GOut* rec = A.get<GOut>(std::max<u32>(nRec, 1)); u32* len = A.get<u32>(std::max<u32>(nRec, 1)); u64* lx = A.get<u64>((size_t) nRec + 1);
                kRecs<<<gridFor(nG), 256, 0, st>>>(go, f8, pre, dNum + 1, rec, cntS, sdp, sad);
                // the ALT-fraction floor on the block's complete per-site sums (a block holds every hit of its sites): the host's rule decides
                std::vector<u64> hdp(nsb), had(nsb); std::vector<u8> hk(nsb, 1);
                AS_CK(cudaMemcpyAsync(hdp.data(), sdp, (size_t) nsb * 8, cudaMemcpyDeviceToHost, st));
                AS_CK(cudaMemcpyAsync(had.data(), sad, (size_t) nsb * 8, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                for (u32 b = 0; b < nsb; ++b) {
                    if (in.keep_site && !in.keep_site(had[b], hdp[b])) { hk[b] = 0; ++out.below_floor; continue; }
                    out.kept_dp += hdp[b]; out.kept_ad += had[b];
                }
                AS_CK(cudaMemcpyAsync(kp, hk.data(), nsb, cudaMemcpyHostToDevice, st));
                if (nRec) kRecLen<<<gridFor(nRec), 256, 0, st>>>(rec, nRec, len);
                {
                    cub::TransformInputIterator<u64, ToU64, cub::CountingInputIterator<u64>> li(cub::CountingInputIterator<u64>(0), ToU64{len, nRec});
                    size_t t1 = 0, t2 = 0; AS_CK(cub::DeviceScan::ExclusiveSum(nullptr, t1, li, lx, (int) nRec + 1, st));
                    AS_CK(cub::DeviceScan::ExclusiveSum(nullptr, t2, cntS, rs, (int) nsb, st));
                    tm = A.top; tmp = A.get<char>(std::max(t1, t2));
                    AS_CK(cub::DeviceScan::ExclusiveSum(tmp, t1, li, lx, (int) nRec + 1, st));
                    AS_CK(cub::DeviceScan::ExclusiveSum(tmp, t2, cntS, rs, (int) nsb, st));
                    A.top = tm;
                    AS_CK(cudaMemcpyAsync(rs + nsb, &nRec, 4, cudaMemcpyHostToDevice, st));
                    kSiteBytes<<<gridFor(nsb), 256, 0, st>>>(cntS, rs, lx, kp, nsb, stb);
                    cub::TransformInputIterator<u64, ToU64Plain, cub::CountingInputIterator<u64>> bi(cub::CountingInputIterator<u64>(0), ToU64Plain{stb, nsb});
                    size_t t3 = 0; AS_CK(cub::DeviceScan::ExclusiveSum(nullptr, t3, bi, so, (int) nsb + 1, st));
                    tm = A.top; tmp = A.get<char>(t3); AS_CK(cub::DeviceScan::ExclusiveSum(tmp, t3, bi, so, (int) nsb + 1, st)); A.top = tm;
                }
                std::vector<u64> hso((size_t) nsb + 1);
                AS_CK(cudaMemcpyAsync(hso.data(), so, ((size_t) nsb + 1) * 8, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
                u8* bytes = A.get<u8>(std::max<u64>(hso[nsb], 1));
                kSiteHead<<<gridFor(nsb), 256, 0, st>>>(cntS, so, kp, nsb, bytes);
                if (nRec) kRecWrite<<<gridFor(nRec), 256, 0, st>>>(rec, nRec, cntS, rs, lx, so, kp, bytes);
                for (u32 c = 0; c < nC; ++c) {   // the contigs whose site ranges meet [s0, s1), in site order
                    const u32 a0 = std::max(sOff[c], s0), a1 = std::min(sOff[c + 1], s1); if (a1 <= a0) continue;
                    const u64 b0 = hso[a0 - s0], b1 = hso[a1 - s0];
                    std::string buf; buf.resize(b1 - b0);
                    d2h.copy(&buf[0], bytes + b0, b1 - b0, st);
                    out.spill_bytes += b1 - b0;
                    uint32_t nk = 0; for (u32 b = a0; b < a1; ++b) nk += hk[b - s0];
                    if (in.sink) in.sink(c0 + c, a0 - sOff[c], a1 - a0, hk.data() + (a0 - s0), nk, buf);
                }
                stage(out.sec_out);
                if (in.progress) in.progress();
            }
            AS_CK(cudaMemcpyAsync(stat, dStat, 32, cudaMemcpyDeviceToHost, st)); AS_CK(cudaStreamSynchronize(st));
            out.ambiguous += stat[0]; out.other += stat[1]; out.molecules += stat[2]; out.ad_molecules += stat[3];
            out.sec_count += secSince(tc);
            A.top = mark0;
        }
    } catch (const std::exception& e) {
        out.why = e.what();
        for (auto& c : out.contigs) c = AD::ContigOut{};   // the caller drops any spill units already put
        return false;
    }
    out.sec = secSince(T0);
    return true;
}

}  // namespace singlet_engine_gpu
