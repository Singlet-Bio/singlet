// E10: the per-read products of STAR's loop for the reads with a GPU record, on the device (API and rules: hostloop_gpu.h).
#include "hostloop_gpu.h"
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h>
#include <thrust/copy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>
#include <thrust/functional.h>
#include <algorithm>
#include <cstring>
#include <vector>

namespace {

typedef unsigned long long ull;
typedef singlet_engine_gpu::Output::E10 E10;

#define HL_CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string(#x) + ": " + cudaGetErrorString(e_); return false; } } while (0)

// PileupEngine::hash_qname("r" + std::to_string(i)): FNV-1a with a murmur finaliser (the pileup's rRNA sample is hash & 63 == 0)
__device__ inline uint64_t hashR(uint64_t i) {
    uint64_t h = 14695981039346656037ULL;
    h ^= (uint8_t)'r'; h *= 1099511628211ULL;
    char d[24]; int k = 0; uint64_t x = i;
    do { d[k++] = (char)('0' + x % 10); x /= 10; } while (x);
    while (k) { h ^= (uint8_t)d[--k]; h *= 1099511628211ULL; }
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL; h ^= h >> 33;
    return h;
}

// ReadAlign::handoffReadEntry for read i (every read: products() zeroes the reads without a record). STAR's inputs and how they are here:
// Qual0[0][j] - 33 < minQ = bit j of the read's quality words (host decode); Read0[0][j] == 'N' = code 4 (the host checked that every base
// of code 4 is an 'N'); (Read0[0][j] | 32) == 'a' = code 0; clipMates' clippedN = the GPU clip (readLoad takes g_handoffClip5/3, uint16).
__global__ void k_entries(const char* seq, const uint64_t* off, const uint64_t* cs, const uint64_t* cl, uint64_t n, const uint64_t* aux, uint32_t nw,
                          bool hook, e6::ReadEntry* E, uint64_t* M) {
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t o = off[i], L = off[i + 1] - o;
    const uint8_t* s = (const uint8_t*)seq + o;
    uint64_t c5 = 0, c3 = 0;
    if (cs) { c5 = cs[i] - o; c3 = L - cl[i] - c5; c5 = c5 < 65535 ? c5 : 65535; c3 = c3 < 65535 ? c3 : 65535; }
    const uint64_t* a = aux + i * (2 + (uint64_t)nw);
    const uint64_t* q = a + 2;
    uint32_t lq = 0, ne = 0;
    for (uint64_t j = 0; j < L; j++) {
        const uint32_t low = (uint32_t)((q[j >> 6] >> (j & 63)) & 1), isN = s[j] == 4;
        lq += low; ne += low != isN;
    }
    e6::ReadEntry e;
    e.cr3 = a[0]; e.ur3 = a[1];
    e.lowQual = (uint16_t)(lq < 65535 ? lq : 65535); e.qualNeqN = (uint16_t)(ne < 65535 ? ne : 65535);
    uint32_t fl = 1;
    if (M) {
        uint64_t m0 = 0, m1 = 0;
        if (c5 + c3 > L || L - c5 - c3 > 128) {
            fl |= 64;
        } else {
            for (uint64_t j = c5; j < L - c3; j++)
                if ((q[j >> 6] >> (j & 63)) & 1) { const uint64_t b = j - c5; if (b < 64) m0 |= 1ull << b; else m1 |= 1ull << (b - 64); }
        }
        M[2 * i] = m0; M[2 * i + 1] = m1;
    }
    if (c5 + c3 > 0) {
        uint32_t nA = 0;
        for (uint64_t j = L - c3; j < L; j++) nA += s[j] == 0;
        if (c5 > 255 || c3 > 255) fl |= 32;
        fl |= ((uint32_t)(c5 < 255 ? c5 : 255) << 8) | ((uint32_t)(c3 < 255 ? c3 : 255) << 16) | ((nA < 255 ? nA : 255u) << 24);
        if (c3 > 0 && s[L - c3] == 0) fl |= 2;
        if (hook && L >= 21 && (hashR(i) & 63) == 0) fl |= 4;   // in the sample; the host adds the detector's verdict (bits 3, 4)
    }
    e.flags = fl;
    E[i] = e;
}

// R11 (r11_hist.cpp): a UMI without N that is not a homopolymer, an N-free CB that is an exact whitelist entry (STAR's 2-bit key, first base high)
__global__ void k_r11(const uint64_t* aux, uint32_t nw, uint64_t n, const uint64_t* wl, uint64_t nWL, uint32_t cbL, uint32_t umiL, uint32_t* hist) {
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t* a = aux + i * (2 + (uint64_t)nw);
    const uint64_t cr = a[0], ur = a[1];
    const uint32_t u0 = (uint32_t)(ur & 7);
    bool homo = true;
    for (uint32_t j = 0; j < umiL; j++) {
        const uint32_t c = (uint32_t)((ur >> (3 * j)) & 7);
        if (c > 3) return;
        homo = homo && c == u0;
    }
    if (homo) return;
    uint64_t key = 0;
    for (uint32_t j = 0; j < cbL; j++) {
        const uint64_t c = (cr >> (3 * j)) & 7;
        if (c > 3) return;
        key = key << 2 | c;
    }
    uint64_t lo = 0, hi = nWL;
    while (lo < hi) { const uint64_t mid = (lo + hi) >> 1; if (wl[mid] < key) lo = mid + 1; else hi = mid; }
    if (lo < nWL && wl[lo] == key) atomicAdd(&hist[lo], 1u);
}

__global__ void k_mult(const uint32_t* rr, uint64_t n, uint32_t* mult) {
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n && rr[i] != 0xFFFFFFFFu) atomicAdd(&mult[rr[i]], 1u);
}
__global__ void k_zero(const uint32_t* rr, uint64_t n, e6::ReadEntry* E, uint64_t* M) {
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n || rr[i] != 0xFFFFFFFFu) return;
    E[i] = e6::ReadEntry{0, 0, 0, 0, 0};
    if (M) { M[2 * i] = 0; M[2 * i + 1] = 0; }
}

// one record's blob (seqCacheEncode layout, decoded as seqCacheRestore does): the scalar fields of a transcript in forEachScalar order
struct Sc { int64_t v[29]; };
__device__ inline void scalars(const uint8_t*& p, Sc& t) { for (int k = 0; k < 29; k++) t.v[k] = e6::getV(p); }
enum { V_rLength = 2, V_nMatch = 9, V_nMM = 10, V_nDel = 15, V_nIns = 16, V_lDel = 17, V_lIns = 18, V_maxScore = 24 };

__device__ unsigned long long g_stat[E10::S_N + 2];   // + odd records, + junction start/gap out of the key's range
__device__ double g_portion;

// per record: mappedFilter (as e6::recordCount), Stats of its reads (x multiplicity), its junction entries (outputTranscriptSJ, before the
// in-read duplicate rule), and whether its reads replay a summary
__global__ void k_rec(e6::RecParams P, const uint8_t* blob, const uint64_t* bOff, const uint64_t* sOff, uint64_t nRec, const uint32_t* mult,
                      bool outSJ, bool sjAll, uint8_t* recMapped, uint8_t* recUniq, uint32_t* nJ) {
    const uint64_t r = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    ull st[E10::S_N + 1];
    for (int k = 0; k <= E10::S_N; k++) st[k] = 0;
    double portion = 0;
    if (r < nRec) {
        const uint64_t L = sOff[r + 1] - sOff[r], m = mult[r];
        const uint8_t* p = blob + bOff[r];
        const uint64_t nW = (uint64_t)e6::getV(p), nTr = (uint64_t)e6::getV(p);
        const int64_t iBest = e6::getV(p);
        uint64_t trInitRL = 0;
        if (nW == 0) trInitRL = (uint64_t)e6::getV(p);
        int bScore = 0; uint64_t bMatch = 0, bMM = 0, bRL = trInitRL;
        uint64_t t0[5] = {0, 0, 0, 0, 0}, mappedL = 0, spl[7] = {0, 0, 0, 0, 0, 0, 0}, sjdb = 0; bool hasEx = false;
        uint32_t nj = 0;
        for (uint64_t i = 0; i < nTr; i++) {
            const uint64_t nEx = (uint64_t)e6::getV(p);
            Sc t; scalars(p, t);
            if ((int64_t)i == iBest) { bScore = (int)t.v[V_maxScore]; bMatch = (uint64_t)t.v[V_nMatch]; bMM = (uint64_t)t.v[V_nMM]; bRL = (uint64_t)t.v[V_rLength]; }
            if (i == 0) { t0[0] = (uint64_t)t.v[V_nMM]; t0[1] = (uint64_t)t.v[V_nIns]; t0[2] = (uint64_t)t.v[V_nDel]; t0[3] = (uint64_t)t.v[V_lIns]; t0[4] = (uint64_t)t.v[V_lDel]; hasEx = nEx > 0; }
            for (uint64_t iex = 0; iex < nEx; iex++) {
                e6::skipV(p); e6::skipV(p);
                const uint64_t Lx = (uint64_t)e6::getV(p);
                e6::skipV(p); e6::skipV(p);
                if (i == 0) mappedL += Lx;
                if (iex + 1 < nEx) {
                    const int c = (int)e6::getV(p); const int an = (int)(uint8_t)e6::getV(p); e6::skipV(p);
                    if (c >= 0) { e6::skipV(p); e6::skipV(p); nj++; }
                    if (i == 0) { if (c >= 0 && c < 7) spl[c]++; if (an == 1) sjdb++; }
                }
            }
        }
        // mappedFilter (Lread = L, outFilterMismatchNmaxTotal as oneRead sets it for a single-end read of clipped length L)
        int unmapType = -1;
        const uint64_t mmCap = (uint64_t)(P.outFilterMismatchNoverReadLmax * (double)L);
        const uint64_t mmTotal = P.outFilterMismatchNmax < mmCap ? P.outFilterMismatchNmax : mmCap;
        if (nW == 0) unmapType = 0;
        else if ((bScore < P.outFilterScoreMin) || (bScore < (int)(P.outFilterScoreMinOverLread * (double)(L - 1)))
                 || (bMatch < P.outFilterMatchNmin) || (bMatch < (uint64_t)(P.outFilterMatchNminOverLread * (double)(L - 1)))) unmapType = 1;
        else if ((bMM > mmTotal) || (double(bMM) / double(bRL) > P.outFilterMismatchNoverLmax)) unmapType = 2;
        else if (nTr > P.outFilterMultimapNmax) unmapType = 3;
        const bool mapped = unmapType < 0 && nTr > 0;
        recMapped[r] = mapped; recUniq[r] = nTr == 1;
        nJ[r] = (mapped && outSJ && (sjAll || nTr == 1)) ? nj : 0;
        st[E10::S_readN] = m; st[E10::S_readBases] = m * L;
        if (unmapType < 0 && nTr == 0) st[E10::S_N] = 1;   // odd: mapped without a transcript (no summary, but not unmapped either)
        else if (mapped) {
            if (nTr > 1) st[E10::S_M] = m;
            else {
                st[E10::S_U] = m;
                st[E10::S_mm] = m * t0[0]; st[E10::S_insN] = m * t0[1]; st[E10::S_delN] = m * t0[2]; st[E10::S_insL] = m * t0[3]; st[E10::S_delL] = m * t0[4];
                if (hasEx) {
                    st[E10::S_bases] = m * mappedL;
                    for (int k = 0; k < 7; k++) st[E10::S_spl0 + k] = m * spl[k];
                    st[E10::S_sjdb] = m * sjdb;
                    portion = (double)m * (double(mappedL) / double(L));
                }
            }
        } else {
            st[E10::S_uOther + unmapType] = m;   // unmapType 0..3: other, short, mismatch, multi (mappedFilter's counters)
            st[E10::S_uAll] = m;
        }
    }
    // warp sums, one atomic per counter per warp
    for (int k = 0; k <= E10::S_N; k++) {
        ull v = st[k];
        for (int d = 16; d > 0; d >>= 1) v += __shfl_down_sync(0xffffffffu, v, d);
        if ((threadIdx.x & 31) == 0 && v) atomicAdd(&g_stat[k], v);
    }
    for (int d = 16; d > 0; d >>= 1) portion += __shfl_down_sync(0xffffffffu, portion, d);
    if ((threadIdx.x & 31) == 0 && portion != 0) atomicAdd(&g_portion, portion);
}

// per junction entry (record order, transcript order, exon order): key = start << 32 | gap, the record, overhang | motif << 16 | annot << 24
__global__ void k_sjfill(const uint8_t* blob, const uint64_t* bOff, uint64_t nRec, const uint32_t* nJ, const uint64_t* jOff, uint64_t* keys,
                         uint32_t* recs, uint32_t* vals) {
    const uint64_t r = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (r >= nRec || nJ[r] == 0) return;
    uint64_t o = jOff[r];
    const uint8_t* p = blob + bOff[r];
    const uint64_t nW = (uint64_t)e6::getV(p), nTr = (uint64_t)e6::getV(p);
    e6::getV(p);
    if (nW == 0) e6::getV(p);
    for (uint64_t i = 0; i < nTr; i++) {
        const uint64_t nEx = (uint64_t)e6::getV(p);
        for (int k = 0; k < 29; k++) e6::skipV(p);
        uint64_t Gp = 0, Lp = 0; int cp = -1, ap = 0;
        for (uint64_t iex = 0; iex < nEx; iex++) {
            e6::skipV(p);
            const uint64_t G = (uint64_t)e6::getV(p), Lx = (uint64_t)e6::getV(p);
            e6::skipV(p); e6::skipV(p);
            if (iex > 0 && cp >= 0) {   // junction iex-1 (outputTranscriptSJ: canonSJ >= 0 only)
                const uint64_t start = Gp + Lp;
                const uint32_t gap = (uint32_t)(G - start);
                const uint32_t ov32 = (uint32_t)Lp < (uint32_t)Lx ? (uint32_t)Lp : (uint32_t)Lx;
                if (start >> 32) atomicAdd(&g_stat[E10::S_N + 1], 1ULL);
                keys[o] = start << 32 | gap; recs[o] = (uint32_t)r;
                vals[o] = (uint32_t)(uint16_t)ov32 | (uint32_t)(uint8_t)(char)cp << 16 | (uint32_t)(uint8_t)ap << 24;
                o++;
            }
            if (iex + 1 < nEx) {
                cp = (int)e6::getV(p); ap = (int)(uint8_t)e6::getV(p); e6::skipV(p);
                if (cp >= 0) { e6::skipV(p); e6::skipV(p); }
            }
            Gp = G; Lp = Lx;
        }
    }
}

struct KeyRecEq {
    __host__ __device__ bool operator()(const thrust::tuple<uint64_t, uint32_t>& a, const thrust::tuple<uint64_t, uint32_t>& b) const {
        return thrust::get<0>(a) == thrust::get<0>(b) && thrust::get<1>(a) == thrust::get<1>(b);
    }
};
struct MaxU32 { __host__ __device__ uint32_t operator()(uint32_t a, uint32_t b) const { return a > b ? a : b; } };
__global__ void k_sjcounts(uint64_t nU, const uint32_t* recs, const uint8_t* recUniq, const uint32_t* mult, uint32_t* cU, uint32_t* cM) {
    const uint64_t k = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (k >= nU) return;
    const uint32_t r = recs[k];
    cU[k] = recUniq[r] ? mult[r] : 0; cM[k] = recUniq[r] ? 0 : mult[r];
}
typedef thrust::tuple<uint32_t, uint32_t, uint32_t> T3;
struct SumSumMax { __host__ __device__ T3 operator()(const T3& a, const T3& b) const {
    return T3(thrust::get<0>(a) + thrust::get<0>(b), thrust::get<1>(a) + thrust::get<1>(b), thrust::get<2>(a) > thrust::get<2>(b) ? thrust::get<2>(a) : thrust::get<2>(b)); } };
struct UnmappedRead {
    const uint32_t* rr; const uint8_t* rm;
    __device__ bool operator()(uint64_t i) const { const uint32_t r = rr[i]; return r != 0xFFFFFFFFu && !rm[r]; }
};
struct HookRead {
    const uint32_t* rr; const e6::ReadEntry* E;
    __device__ bool operator()(uint64_t i) const { return rr[i] != 0xFFFFFFFFu && (E[i].flags & 4u); }
};

struct DevBufs { std::vector<void*> p; ~DevBufs() { for (void* x : p) cudaFree(x); } };

} // namespace

namespace singlet_hostloop_gpu {

bool entries(const char* dSeq, const uint64_t* dOff, const uint64_t* dCs, const uint64_t* dCl, uint64_t n, const uint64_t* dAux, uint32_t nw,
             bool hook, e6::ReadEntry* dEnt, uint64_t* dMask, float& ms, std::string& err) {
    if (n == 0) { ms = 0; return true; }
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    k_entries<<<(unsigned)((n + 255) / 256), 256>>>(dSeq, dOff, dCs, dCl, n, dAux, nw, hook, dEnt, dMask);
    cudaEventRecord(b);
    HL_CK(cudaEventSynchronize(b));
    HL_CK(cudaGetLastError());
    cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
    return true;
}

bool r11(const uint64_t* dAux, uint32_t nw, uint64_t n, const uint64_t* dWL, uint64_t nWL, uint32_t cbL, uint32_t umiL, uint32_t* dHist, float& ms,
         std::string& err) {
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    HL_CK(cudaMemset(dHist, 0, nWL * 4));
    if (n > 0) k_r11<<<(unsigned)((n + 255) / 256), 256>>>(dAux, nw, n, dWL, nWL, cbL, umiL, dHist);
    cudaEventRecord(b);
    HL_CK(cudaEventSynchronize(b));
    HL_CK(cudaGetLastError());
    cudaEventElapsedTime(&ms, a, b); cudaEventDestroy(a); cudaEventDestroy(b);
    return true;
}

bool products(const e6::RecParams& P, const uint8_t* dBlob, const uint64_t* dBOff, const uint64_t* dSOff, uint64_t nRec, const uint32_t* dRR, uint64_t n,
              e6::ReadEntry* dEnt, uint64_t* dMask, bool outSJ, bool sjAll, E10& out, std::string& err) {
    DevBufs db;
    auto dal = [&](uint64_t bytes) -> void* { void* p = nullptr; if (cudaMalloc(&p, bytes + 64) != cudaSuccess) { cudaGetLastError(); return nullptr; } db.p.push_back(p); return p; };
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0);
    uint32_t* mult = (uint32_t*)dal(nRec * 4); uint8_t* rm = (uint8_t*)dal(nRec); uint8_t* ru = (uint8_t*)dal(nRec); uint32_t* nJ = (uint32_t*)dal(nRec * 4);
    uint64_t* jOff = (uint64_t*)dal((nRec + 1) * 8);
    if (!mult || !rm || !ru || !nJ || !jOff) { err = "out of device memory (E10 records)"; return false; }
    HL_CK(cudaMemset(mult, 0, nRec * 4));
    const unsigned GN = (unsigned)((n + 255) / 256), GR = (unsigned)((nRec + 255) / 256);
    k_mult<<<GN, 256>>>(dRR, n, mult);
    ull z[E10::S_N + 2] = {}; double zp = 0;
    HL_CK(cudaMemcpyToSymbol(g_stat, z, sizeof z)); HL_CK(cudaMemcpyToSymbol(g_portion, &zp, sizeof zp));
    k_rec<<<GR, 256>>>(P, dBlob, dBOff, dSOff, nRec, mult, outSJ, sjAll, rm, ru, nJ);
    HL_CK(cudaGetLastError());
    k_zero<<<GN, 256>>>(dRR, n, dEnt, dMask);
    ull st[E10::S_N + 2];
    HL_CK(cudaMemcpyFromSymbol(st, g_stat, sizeof st)); HL_CK(cudaMemcpyFromSymbol(&out.mappedPortion, g_portion, sizeof(double)));
    for (int k = 0; k < E10::S_N; k++) out.stats[k] = st[k];
    out.nOddRec = st[E10::S_N];
    // junctions: entries, (key, record) duplicates merged (the in-read rule: one count, the larger overhang), then per key summed and maxed
    HL_CK(cudaMemset(jOff, 0, 8));
    thrust::inclusive_scan(thrust::device_ptr<uint32_t>(nJ), thrust::device_ptr<uint32_t>(nJ + nRec), thrust::device_ptr<uint64_t>(jOff + 1));
    uint64_t nE = 0;
    HL_CK(cudaMemcpy(&nE, jOff + nRec, 8, cudaMemcpyDeviceToHost));
    out.nSJentries = nE; out.nSJ = 0; out.sj.clear();
    if (nE > 0) {
        uint64_t* keys = (uint64_t*)dal(nE * 8); uint32_t* recs = (uint32_t*)dal(nE * 4); uint32_t* vals = (uint32_t*)dal(nE * 4);
        if (!keys || !recs || !vals) { err = "out of device memory (E10 junctions)"; return false; }
        k_sjfill<<<GR, 256>>>(dBlob, dBOff, nRec, nJ, jOff, keys, recs, vals);
        HL_CK(cudaGetLastError());
        HL_CK(cudaMemcpyFromSymbol(st, g_stat, sizeof st));
        if (st[E10::S_N + 1]) { err = "junction start beyond 32 bits"; return false; }
        thrust::device_ptr<uint64_t> K(keys); thrust::device_ptr<uint32_t> R(recs), V(vals);
        thrust::stable_sort_by_key(K, K + nE, thrust::make_zip_iterator(thrust::make_tuple(R, V)));   // record order kept within a key
        uint64_t* k1 = (uint64_t*)dal(nE * 8); uint32_t* r1 = (uint32_t*)dal(nE * 4); uint32_t* v1 = (uint32_t*)dal(nE * 4);
        if (!k1 || !r1 || !v1) { err = "out of device memory (E10 junctions)"; return false; }
        auto kr = thrust::make_zip_iterator(thrust::make_tuple(K, R));
        auto kr1 = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<uint64_t>(k1), thrust::device_ptr<uint32_t>(r1)));
        auto e1r = thrust::reduce_by_key(kr, kr + nE, V, kr1, thrust::device_ptr<uint32_t>(v1), KeyRecEq(), MaxU32());
        const uint64_t nU = (uint64_t)(e1r.second - thrust::device_ptr<uint32_t>(v1));
        uint32_t* cU = (uint32_t*)dal(nU * 4); uint32_t* cM = (uint32_t*)dal(nU * 4);
        uint64_t* k2 = keys; uint32_t* cU2 = recs; uint32_t* cM2 = vals; uint32_t* v2 = (uint32_t*)dal(nU * 4);   // reuse the entry buffers for the output
        if (!cU || !cM || !v2) { err = "out of device memory (E10 junctions)"; return false; }
        k_sjcounts<<<(unsigned)((nU + 255) / 256), 256>>>(nU, r1, ru, mult, cU, cM);
        auto in3 = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<uint32_t>(cU), thrust::device_ptr<uint32_t>(cM), thrust::device_ptr<uint32_t>(v1)));
        auto out3 = thrust::make_zip_iterator(thrust::make_tuple(thrust::device_ptr<uint32_t>(cU2), thrust::device_ptr<uint32_t>(cM2), thrust::device_ptr<uint32_t>(v2)));
        auto e2r = thrust::reduce_by_key(thrust::device_ptr<uint64_t>(k1), thrust::device_ptr<uint64_t>(k1 + nU), in3, thrust::device_ptr<uint64_t>(k2), out3,
                                         thrust::equal_to<uint64_t>(), SumSumMax());
        const uint64_t nJu = (uint64_t)(e2r.first - thrust::device_ptr<uint64_t>(k2));
        std::vector<uint64_t> hk(nJu); std::vector<uint32_t> hu(nJu), hm(nJu), hv(nJu);
        HL_CK(cudaMemcpy(hk.data(), k2, nJu * 8, cudaMemcpyDeviceToHost)); HL_CK(cudaMemcpy(hu.data(), cU2, nJu * 4, cudaMemcpyDeviceToHost));
        HL_CK(cudaMemcpy(hm.data(), cM2, nJu * 4, cudaMemcpyDeviceToHost)); HL_CK(cudaMemcpy(hv.data(), v2, nJu * 4, cudaMemcpyDeviceToHost));
        // STAR's Junction layout (OutSJ.h): uint start, uint32 gap, char strand, motif, annot, uint32 countUnique, countMultiple, uint16 overhangs
        const size_t DS = 27;
        out.sj.assign((nJu + 1) * DS, 0);
        for (uint64_t k = 0; k < nJu; k++) {
            uint8_t* d = out.sj.data() + k * DS;
            const uint64_t start = hk[k] >> 32; const uint32_t gap = (uint32_t)hk[k];
            const char motif = (char)(uint8_t)(hv[k] >> 16), annot = (char)(uint8_t)(hv[k] >> 24);
            const char strand = (char)(motif == 0 ? 0 : (motif + 1) % 2 + 1);
            const uint16_t ov = (uint16_t)hv[k];
            std::memcpy(d, &start, 8); std::memcpy(d + 8, &gap, 4); d[12] = (uint8_t)strand; d[13] = (uint8_t)motif; d[14] = (uint8_t)annot;
            std::memcpy(d + 15, &hu[k], 4); std::memcpy(d + 19, &hm[k], 4); std::memcpy(d + 23, &ov, 2); std::memcpy(d + 25, &ov, 2);
        }
        std::memset(out.sj.data() + nJu * DS, 255, DS);   // outputSJ's end marker
        out.nSJ = nJu;
    }
    // the reads whose record is unmapped, and the reads in the hook's sample
    {
        thrust::counting_iterator<uint64_t> c0(0);
        UnmappedRead pu{dRR, rm};
        const uint64_t nUm = (uint64_t)thrust::count_if(thrust::device, c0, c0 + n, pu);
        out.unmappedReads.resize(nUm);
        if (nUm) {
            uint64_t* li = (uint64_t*)dal(nUm * 8);
            if (!li) { err = "out of device memory (E10 lists)"; return false; }
            thrust::copy_if(thrust::device, c0, c0 + n, thrust::device_ptr<uint64_t>(li), pu);
            std::vector<uint64_t> h(nUm);
            HL_CK(cudaMemcpy(h.data(), li, nUm * 8, cudaMemcpyDeviceToHost));
            for (uint64_t k = 0; k < nUm; k++) out.unmappedReads[k] = (uint32_t)h[k];
        }
        HookRead ph{dRR, dEnt};
        const uint64_t nH = (uint64_t)thrust::count_if(thrust::device, c0, c0 + n, ph);
        out.hookReads.resize(nH);
        if (nH) {
            uint64_t* li = (uint64_t*)dal(nH * 8);
            if (!li) { err = "out of device memory (E10 lists)"; return false; }
            thrust::copy_if(thrust::device, c0, c0 + n, thrust::device_ptr<uint64_t>(li), ph);
            HL_CK(cudaMemcpy(out.hookReads.data(), li, nH * 8, cudaMemcpyDeviceToHost));
        }
    }
    out.recMapped.resize(nRec);
    HL_CK(cudaMemcpy(out.recMapped.data(), rm, nRec, cudaMemcpyDeviceToHost));
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms = 0; cudaEventElapsedTime(&ms, e0, e1); out.msProducts = ms;
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    HL_CK(cudaGetLastError());
    return true;
}

} // namespace singlet_hostloop_gpu
