// M3 + M4: exact STAR seed search on the GPU, and U-array seed certificates, both checked against the STAR dump.
// docs/gpu-engine/exploration.md §4.4 (certificates), §9 (M3, M4 specs), §12 (results).
//
// Build: nvcc -O3 -std=c++17 -arch=sm_90 -o seedsearch seedsearch.cu
// Run:   seedsearch --genome-dir STAR_INDEX --dump PREFIX [--reps 3] [--json OUT] [--no-m4]
//          PREFIX: the stardump output prefix (files PREFIX.<tid>.bin, layout src/gpu/stardump/dump_format.h)
//
// M3. `seed_search` is a line-by-line port of STAR's ReadAlign::maxMappableLength2strands (gSAsparseD = 1) with
// maxMappableLength, findMultRange and the byte-genome compareSeqToGenome (SuffixArrayFuns.cpp, prod/campaign), over
// STAR's own Genome, SA and SAindex files loaded unchanged into HBM. One thread per seed call. For every dumped call it
// recomputes (maxL, Nrep, SA interval) and compares with STAR's answer. Seeds run in dump order and in prefix-sorted
// order (first 32 bases of the query in search direction); the steps counter gives SIMT efficiency per warp.
//
// M4. Builds the U-array from STAR's SA (lcp of SA neighbours over A/C/G/T, then U[t] = 1 + max(lcp_prev, lcp_next)),
// checks SA sortedness on the way, and runs the certificate of ucert.cu on every dumped seed, with candidate loci
// propagated from the read's earlier seeds that STAR answered uniquely (locus propagation, §4.4). Every certified
// answer is compared with STAR's answer: a certified seed must have Nrep = 1, the same locus and the same maxL.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <glob.h>
#include <sys/stat.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include "../../../src/gpu/stardump/dump_format.h"

typedef unsigned long long u64;
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------------------------------------------------------------
// Index (STAR's files, unchanged)
// ---------------------------------------------------------------------------------------------------------------------
struct Index {
    u64 nGenome, nSA, nSAi, GstrandBit, GstrandMask, saBits, saiBits;
    u64 SAiMarkNmaskC, SAiMarkNmask, SAiMarkAbsentMaskC;
    u64 gSAindexNbases;
    u64 saiStart[32];
    const char* G;          // device: G[-200 .. nGenome+200), as STAR's G = G1 + 200
    const u64* SA;          // device: STAR's packed SA bytes, 8-byte aligned, padded
    const u64* SAi;         // device: STAR's packed SAindex bytes
};
__constant__ Index cI;

// STAR's PackedArray::operator[]: an unaligned 8-byte little-endian load at bit offset ii * wordLength, masked.
__device__ __forceinline__ u64 packed_get(const u64* a, u64 ii, u64 bits) {
    u64 b = ii * bits, w = b >> 6, sh = b & 63;
    u64 v = a[w] >> sh;
    if (sh) v |= a[w + 1] << (64 - sh);
    return v & ((1ULL << bits) - 1);
}
__device__ __forceinline__ u64 SA_(u64 i) { return packed_get(cI.SA, i, cI.saBits); }
__device__ __forceinline__ u64 SAi_(u64 i) { return packed_get(cI.SAi, i, cI.saiBits); }

__device__ __forceinline__ u64 medianUint2(u64 a, u64 b) { return a / 2 + b / 2 + (a % 2 + b % 2) / 2; }

// s2[0] = read, s2[1] = STAR's complement of the read (both from the dump)
struct ReadPtr { const char* s0; const char* s1; };

// compareSeqToGenome, byte-genome version (SuffixArrayFuns.cpp, !GENOME_PACKED branch)
__device__ u64 compareSeqToGenome(ReadPtr s2, u64 S, u64 N, u64 L, u64 iSA, bool dirR, bool& compRes, unsigned& steps) {
    steps++;
    u64 SAstr = SA_(iSA);
    bool dirG = (SAstr >> cI.GstrandBit) == 0;
    SAstr &= cI.GstrandMask;
    const char* g = cI.G;
    long long ii;
    if (dirR && dirG) {
        const char* s = s2.s0 + S + L; g += SAstr + L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[ii] != g[ii]) { compRes = s[ii] > g[ii]; return ii + L; }
        return N;
    } else if (dirR && !dirG) {
        const char* s = s2.s1 + S + L; g += cI.nGenome - 1 - SAstr - L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[ii] != g[-ii]) { compRes = !(s[ii] > g[-ii] || g[-ii] > 3); return ii + L; }
        return N;
    } else if (!dirR && dirG) {
        const char* s = s2.s1 + S - L; g += SAstr + L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[-ii] != g[ii]) { compRes = s[-ii] > g[ii]; return ii + L; }
        return N;
    } else {
        const char* s = s2.s0 + S - L; g += cI.nGenome - 1 - SAstr - L;
        for (ii = 0; (u64)ii < N - L; ii++)
            if (s[-ii] != g[-ii]) { compRes = !(s[-ii] > g[-ii] || g[-ii] > 3); return ii + L; }
        return N;
    }
}

__device__ u64 findMultRange(u64 i3, u64 L3, u64 i1, u64 L1, u64 i1a, u64 L1a, u64 i1b, u64 L1b,
                             ReadPtr s, bool dirR, u64 S, unsigned& steps) {
    bool compRes;
    if (L1 < L3) { L1b = L1; i1b = i1; i1a = i3; }
    else if (L1a < L1) { L1b = L1a; i1b = i1a; i1a = i1; }
    while ((i1b + 1 < i1a) | (i1b > i1a + 1)) {
        u64 i1c = medianUint2(i1a, i1b);
        u64 L1c = compareSeqToGenome(s, S, L3, L1b, i1c, dirR, compRes, steps);
        if (L1c == L3) i1a = i1c;
        else { i1b = i1c; L1b = L1c; }
    }
    return i1a;
}

__device__ u64 maxMappableLength(ReadPtr s, u64 S, u64 N, u64 i1, u64 i2, bool dirR, u64& L, u64* ind, unsigned& steps) {
    bool compRes;
    u64 L1, L2, i3, L3, L1a, L1b, L2a, L2b, i1a, i1b, i2a, i2b;
    L1 = compareSeqToGenome(s, S, N, L, i1, dirR, compRes, steps);
    L2 = compareSeqToGenome(s, S, N, L, i2, dirR, compRes, steps);
    L = L1 < L2 ? L1 : L2;
    L1a = L1; L1b = L1; i1a = i1; i1b = i1;
    L2a = L2; L2b = L2; i2a = i2; i2b = i2;
    i3 = i1; L3 = L1;
    while (i1 + 1 < i2) {
        i3 = medianUint2(i1, i2);
        L3 = compareSeqToGenome(s, S, N, L, i3, dirR, compRes, steps);
        if (L3 == N) break;
        if (compRes) {
            if (L3 > L1) { L1b = L1a; L1a = L1; i1b = i1a; i1a = i1; }
            i1 = i3; L1 = L3;
        } else {
            if (L3 > L2) { L2b = L2a; L2a = L2; i2b = i2a; i2a = i2; }
            i2 = i3; L2 = L3;
        }
        L = L1 < L2 ? L1 : L2;
    }
    if (L3 < N) {
        if (L1 > L2) { i3 = i1; L3 = L1; } else { i3 = i2; L3 = L2; }
    }
    i1 = findMultRange(i3, L3, i1, L1, i1a, L1a, i1b, L1b, s, dirR, S, steps);
    i2 = findMultRange(i3, L3, i2, L2, i2a, L2a, i2b, L2b, s, dirR, S, steps);
    L = L3;
    ind[0] = i1; ind[1] = i2;
    return i2 - i1 + 1;
}

struct Answer { u64 Nrep, ind0, ind1; unsigned maxL; unsigned char branch; };

// One iDist pass of ReadAlign::maxMappableLength2strands (STAR loops iDist < gSAsparseD; the caller passes the shifted
// pieceStart and pieceLength of that pass), without storeAligns
__device__ Answer seed_search(ReadPtr r, u64 pieceStart, u64 pieceLength, bool dirR, unsigned& steps) {
    Answer a;
    u64 Lmax = cI.gSAindexNbases < pieceLength ? cI.gSAindexNbases : pieceLength;
    u64 ind1 = 0;
    if (dirR) for (u64 ii = 0; ii < Lmax; ii++) { ind1 <<= 2; ind1 += (u64)r.s0[pieceStart + ii]; }
    else      for (u64 ii = 0; ii < Lmax; ii++) { ind1 <<= 2; ind1 += 3 - (u64)r.s0[pieceStart - ii]; }
    u64 Lind = Lmax, iSA1 = 0, iSA2;
    while (Lind > 0) {
        steps++;
        iSA1 = SAi_(cI.saiStart[Lind - 1] + ind1);
        if ((iSA1 & cI.SAiMarkAbsentMaskC) == 0) break;
        --Lind; ind1 >>= 2;
    }
    bool iSA2good = true;
    if (cI.saiStart[Lind - 1] + ind1 + 1 < cI.saiStart[Lind]) {
        iSA2 = SAi_(cI.saiStart[Lind - 1] + ind1 + 1);
        if ((iSA2 & cI.SAiMarkAbsentMaskC) == 0) iSA2 = (iSA2 & cI.SAiMarkNmask) - 1;
        else { iSA2 = cI.nSA - 1; iSA2good = false; }
    } else { iSA2 = cI.nSA - 1; iSA2good = false; }
    bool iSA1noN = (iSA1 & cI.SAiMarkNmaskC) == 0;
    u64 maxL, ind[2];
    if (Lind < cI.gSAindexNbases && iSA1noN && iSA2good) {
        a.branch = stardump::BR_SAI_ONLY;
        ind[0] = iSA1; ind[1] = iSA2; a.Nrep = ind[1] - ind[0] + 1; maxL = Lind;
    } else if (iSA1 == iSA2 && iSA1noN && iSA2good) {
        a.branch = stardump::BR_UNIQUE_EXTEND;
        ind[0] = ind[1] = iSA1; a.Nrep = 1;
        bool cr;
        maxL = compareSeqToGenome(r, pieceStart, pieceLength, Lind, iSA1, dirR, cr, steps);
    } else {
        a.branch = stardump::BR_SA_SEARCH;
        maxL = (iSA2good && iSA1noN) ? Lind : 0;
        a.Nrep = maxMappableLength(r, pieceStart, pieceLength, iSA1 & cI.SAiMarkNmask, iSA2, dirR, maxL, ind, steps);
    }
    a.maxL = (unsigned)maxL; a.ind0 = ind[0]; a.ind1 = ind[1];
    return a;
}

// ---------------------------------------------------------------------------------------------------------------------
// Dumped seeds
// ---------------------------------------------------------------------------------------------------------------------
struct Seeds {   // SoA, device or host; one entry per (call, iDist) as dumped
    uint32_t* read; uint8_t* dir; uint8_t* idist; uint16_t* start; uint16_t* len;
    uint16_t* maxL; u64* Nrep; u64* ind0; u64* ind1; u64* sa0; uint8_t* branch;
};

__global__ void k_search(const char* rb, const u64* roff, Seeds s, const uint32_t* order, u64 n,
                         uint16_t* oL, u64* oN, u64* o0, u64* o1, uint8_t* oB, unsigned* osteps) {
    for (u64 k = blockIdx.x * (u64)blockDim.x + threadIdx.x; k < n; k += (u64)gridDim.x * blockDim.x) {
        u64 q = order ? order[k] : k;
        uint32_t ri = s.read[q];
        u64 o = roff[ri], L = (roff[ri + 1] - o) / 2;
        ReadPtr r{rb + o, rb + o + L};
        unsigned steps = 0;
        int d = s.idist[q];
        Answer a = seed_search(r, s.dir[q] == 0 ? s.start[q] + d : s.start[q] - d, s.len[q] - d, s.dir[q] == 0, steps);
        oL[q] = (uint16_t)a.maxL; oN[q] = a.Nrep; o0[q] = a.ind0; o1[q] = a.ind1; oB[q] = a.branch;
        if (osteps) osteps[k] = steps;   // in launch order, for per-warp SIMT efficiency
    }
}

// 64-bit key: first 32 bases of the query in search direction (seeds never contain N; a base > 3 is folded to & 3)
__global__ void k_prefix_keys(const char* rb, const u64* roff, Seeds s, u64 n, u64* keys) {
    for (u64 q = blockIdx.x * (u64)blockDim.x + threadIdx.x; q < n; q += (u64)gridDim.x * blockDim.x) {
        uint32_t ri = s.read[q];
        const char* r0 = rb + roff[ri];
        u64 key = 0; int ln = s.len[q] - s.idist[q], st = s.dir[q] == 0 ? s.start[q] + s.idist[q] : s.start[q] - s.idist[q];
        for (int k = 0; k < 32; k++) {
            u64 b = 0;
            if (k < ln) b = s.dir[q] == 0 ? (u64)(r0[st + k] & 3) : 3 - (u64)(r0[st - k] & 3);
            key = (key << 2) | b;
        }
        keys[q] = key;
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// M4: packed two-strand text, U-array, certificate
// text coordinate t = strand * nGenome + pos; base(t) = G[pos] (strand 0) or comp(G[nGenome-1-pos]) (strand 1)
// ---------------------------------------------------------------------------------------------------------------------
struct Text { const u64* w; const u64* nb; u64 nGenome, nBases; };   // w: 2-bit MSB-first, 32/word; nb: 1 bit/base, MSB-first

__global__ void k_pack_text(const char* G, u64 nGenome, u64 nWords, u64* w, u64* nb) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < nWords; i += (u64)gridDim.x * blockDim.x) {
        u64 word = 0, nbits = 0;
        for (int k = 0; k < 32; k++) {
            u64 t = i * 32 + k; int b;
            if (t < nGenome) b = G[t];
            else if (t < 2 * nGenome) { int c = G[nGenome - 1 - (t - nGenome)]; b = c < 4 ? 3 - c : c; }
            else b = 5;
            word = (word << 2) | (u64)(b & 3);
            nbits = (nbits << 1) | (u64)(b > 3);
        }
        w[i] = word;
        // 1 bit per base: 64 bases per u64 -> two text words per N-word; even word fills the high half
        unsigned long long* p = (unsigned long long*)nb + (i >> 1);
        atomicOr(p, (i & 1) ? nbits : (nbits << 32));   // integer OR: order-free
    }
}

__device__ __forceinline__ u64 word_at(const u64* w, u64 t) {            // 32 bases starting at t
    u64 i = t >> 5; unsigned sh = (unsigned)(t & 31) * 2;
    u64 a = w[i];
    return sh ? (a << sh) | (w[i + 1] >> (64 - sh)) : a;
}
__device__ __forceinline__ uint32_t nbits_at(const u64* nb, u64 t) {     // N flags of the 32 bases starting at t
    u64 i = t >> 6; unsigned sh = (unsigned)(t & 63);
    u64 a = nb[i];
    u64 v = sh ? (a << sh) | (nb[i + 1] >> (64 - sh)) : a;
    return (uint32_t)(v >> 32);
}

// lcp of the suffixes at text coordinates a and b over A/C/G/T, capped; dir = sign of first difference (a vs b)
__device__ int lcp_tt(const Text& T, u64 a, u64 b, int cap, int& cmp) {
    int l = 0; cmp = 0;
    while (l < cap) {
        int n = cap - l < 32 ? cap - l : 32;
        uint32_t nm = nbits_at(T.nb, a + l) | nbits_at(T.nb, b + l);
        u64 x = word_at(T.w, a + l) ^ word_at(T.w, b + l);
        int firstN = nm ? __clz(nm) : 32;
        int firstX = x ? (__clzll(x) >> 1) : 32;
        int stop = min(min(firstN, firstX), n);
        if (stop < n) {
            l += stop;
            if (firstX < firstN && firstX < n) {
                u64 ba = (word_at(T.w, a + l) >> 62), bb = (word_at(T.w, b + l) >> 62);
                cmp = ba < bb ? -1 : 1;
            }
            return l;
        }
        l += n;
    }
    return l;
}

__device__ __forceinline__ u64 sa_to_t(u64 sa) {
    u64 str = sa >> cI.GstrandBit, pos = sa & cI.GstrandMask;
    return str * cI.nGenome + pos;
}

__global__ void k_build_lcp(Text T, u64 nSA, uint8_t* lcp, unsigned long long* viol, unsigned long long* hist) {
    __shared__ unsigned sh[256];
    for (int k = threadIdx.x; k < 256; k += blockDim.x) sh[k] = 0;
    __syncthreads();
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < nSA; i += (u64)gridDim.x * blockDim.x) {
        if (i == 0) { lcp[0] = 0; continue; }
        int cmp;
        int l = lcp_tt(T, sa_to_t(SA_(i - 1)), sa_to_t(SA_(i)), 254, cmp);
        if (cmp > 0) atomicAdd(viol, 1ULL);   // suffix i-1 > suffix i within an A/C/G/T prefix: SA not sorted there
        lcp[i] = (uint8_t)l;
        atomicAdd(sh + l, 1u);                // integer histogram (order-free); flushed per block
    }
    __syncthreads();
    for (int k = threadIdx.x; k < 256; k += blockDim.x) if (sh[k]) atomicAdd(hist + k, (unsigned long long)sh[k]);
}

__global__ void k_hist_u8(const uint8_t* a, u64 n, unsigned long long* hist) {
    __shared__ unsigned sh[256];
    for (int k = threadIdx.x; k < 256; k += blockDim.x) sh[k] = 0;
    __syncthreads();
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < n; i += (u64)gridDim.x * blockDim.x) atomicAdd(sh + a[i], 1u);
    __syncthreads();
    for (int k = threadIdx.x; k < 256; k += blockDim.x) if (sh[k]) atomicAdd(hist + k, (unsigned long long)sh[k]);
}

__global__ void k_scatter_u(u64 nSA, const uint8_t* lcp, uint8_t* U, unsigned long long* smp) {
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < nSA; i += (u64)gridDim.x * blockDim.x) {
        int l = lcp[i], r = i + 1 < nSA ? lcp[i + 1] : 0;
        int u = 1 + (l > r ? l : r);
        u64 t = sa_to_t(SA_(i));
        U[t] = (uint8_t)(u >= 255 ? 255 : u);
        atomicOr(smp + (t >> 6), 1ULL << (63 - (t & 63)));   // t is a suffix start in STAR's (possibly sparse) SA
    }
}
__device__ __forceinline__ bool sampled(const u64* smp, u64 t) { return (smp[t >> 6] >> (63 - (t & 63))) & 1; }

// V-array (sparse SA only): for a text position t that is NOT a suffix start in STAR's SA, V[t] = the longest prefix of
// text[t..] that occurs at any sampled suffix (max lcp with its insertion neighbours in SA order), capped; 255 = unknown.
// A query that matches text[t..] for m > V[t] bases has maximal mappable length <= V[t] in STAR's sparse SA.
__global__ void k_build_v(Text T, u64 nSA, const u64* smp, uint8_t* V) {
    for (u64 t = blockIdx.x * (u64)blockDim.x + threadIdx.x; t < T.nBases; t += (u64)gridDim.x * blockDim.x) {
        if (sampled(smp, t)) continue;
        if ((nbits_at(T.nb, t) >> 31) & 1) { V[t] = 0; continue; }   // starts with N: queries never match here
        u64 lo = 0, hi = nSA;   // first i with suffix(SA[i]) > text[t..]
        int cmp, l, lmax = 0; bool tie = false;
        while (lo < hi) {
            u64 mid = lo + (hi - lo) / 2;
            l = lcp_tt(T, t, sa_to_t(SA_(mid)), 254, cmp);
            if (cmp == 0) { tie = true; break; }                     // equal to the cap or up to an N: no bound
            if (cmp < 0) hi = mid; else lo = mid + 1;
        }
        if (tie) { V[t] = 255; continue; }
        if (lo > 0)   { l = lcp_tt(T, t, sa_to_t(SA_(lo - 1)), 254, cmp); lmax = max(lmax, l); }
        if (lo < nSA) { l = lcp_tt(T, t, sa_to_t(SA_(lo)), 254, cmp);     lmax = max(lmax, l); }
        V[t] = (uint8_t)(lmax >= 254 ? 255 : lmax);
    }
}

// query base k of a seed (search direction), as STAR compares it: forward read[S+k]; reverse comp(read[S-k])
__device__ __forceinline__ int qbase(const char* r0, int S, int dir, int k) { return dir == 0 ? r0[S + k] : 3 - r0[S - k]; }

// anchor: the read's forward orientation on the genome. fstr 0: read[x] = G[c + x]; fstr 1: read[x] = comp(G[c - x])
struct Anchor { int fstr; long long c; };

__device__ bool anchor_from(const Anchor& A, int dir, int S, u64 nG, u64& t) {
    long long pos; int str;
    if (A.fstr == 0) { if (dir == 0) { str = 0; pos = A.c + S; } else { str = 1; pos = (long long)nG - 1 - A.c - S; } }
    else             { if (dir == 0) { str = 1; pos = (long long)nG - 1 - A.c + S; } else { str = 0; pos = A.c - S; } }
    if (pos < 0 || pos >= (long long)nG) return false;
    t = (u64)str * nG + (u64)pos;
    return true;
}
__device__ Anchor anchor_of(int dir, int S, u64 sa, u64 nG, u64 gsb) {
    int str = (int)(sa >> gsb); long long pos = (long long)(sa & ((1ULL << gsb) - 1));
    Anchor A;
    if (dir == 0) { if (str == 0) { A.fstr = 0; A.c = pos - S; } else { A.fstr = 1; A.c = (long long)nG - 1 - pos + S; } }
    else          { if (str == 0) { A.fstr = 1; A.c = pos + S; } else { A.fstr = 0; A.c = (long long)nG - 1 - pos - S; } }
    return A;
}

// Per dumped record (one iDist pass of one call), outcome:
//   0 no anchor; 1 declined; 2 certified, equal to STAR; 3 certified, DIFFERENT from STAR (must never happen);
//   4 bounded (unsampled candidate, maxL <= V proved) and STAR's maxL obeys the bound; 5 bounded but STAR exceeds it (never)
// Per call (group of iDist records), callres on its first record: 1 unresolved, 2 resolved (every record certified or
// bounded below the best certified maxL + iDist, so storeAligns' input is known without any SA search), 3 resolved WRONG.
__device__ int lcp_query(const Text& T, const char* r0, int S, int dir, int len, u64 t, bool& stopN) {
    int m = 0; stopN = false;
    while (m < len) {
        uint32_t nm = nbits_at(T.nb, t + m); u64 w = word_at(T.w, t + m);
        int n = len - m < 32 ? len - m : 32;
        for (int k = 0; k < n; k++) {
            if ((nm >> (31 - k)) & 1) { stopN = true; return m + k; }
            if ((int)((w >> (62 - 2 * k)) & 3) != qbase(r0, S, dir, m + k)) return m + k;
        }
        m += n;
    }
    return m;
}

__global__ void k_certify(const char* rb, const u64* roff, const u64* rseed, uint32_t nReads, Seeds s, Text T,
                          const uint8_t* U, const uint8_t* V, const u64* smp, uint8_t* outcome, uint8_t* callres,
                          unsigned* uhist_used, unsigned long long* why, int minAnchorL) {
    for (u64 ri = blockIdx.x * (u64)blockDim.x + threadIdx.x; ri < nReads; ri += (u64)gridDim.x * blockDim.x) {
        const char* r0 = rb + roff[ri];
        Anchor anc[4]; int na = 0;
        u64 end = rseed[ri + 1];
        for (u64 q = rseed[ri]; q < end;) {
            u64 g1 = q + 1;
            while (g1 < end && g1 - q < 4 && s.dir[g1] == s.dir[q] && s.start[g1] == s.start[q] && s.len[g1] == s.len[q]
                   && s.idist[g1] > s.idist[g1 - 1]) g1++;
            int bestCert = -1; bool allAnswered = true, wrong = false; int bound[4];
            for (u64 r = q; r < g1; r++) {
                int d = s.idist[r], dir = s.dir[r];
                int S = dir == 0 ? s.start[r] + d : s.start[r] - d, len = s.len[r] - d;
                uint8_t oc = 0; bound[r - q] = -1;
                for (int a = 0; a < na; a++) {
                    u64 t;
                    if (!anchor_from(anc[a], dir, S, T.nGenome, t)) continue;
                    if (oc == 0) oc = 1;
                    bool stopN; int m;
                    if (sampled(smp, t)) {
                        int u = U[t];
                        if (u == 255) { atomicAdd(why + 0, 1ULL); continue; }
                        if (u > len) { atomicAdd(why + 1, 1ULL); continue; }
                        m = lcp_query(T, r0, S, dir, len, t, stopN);
                        if (m < u) { atomicAdd(why + 2, 1ULL); continue; }
                        if (stopN) { atomicAdd(why + 3, 1ULL); continue; }
                        bool same = s.Nrep[r] == 1 && s.maxL[r] == m && sa_to_t(s.sa0[r]) == t;
                        oc = same ? 2 : 3;
                        atomicAdd(uhist_used + u, 1u);
                        bestCert = max(bestCert, m + d);
                        break;
                    } else {
                        int v = V[t];
                        if (v == 255) { atomicAdd(why + 4, 1ULL); continue; }
                        m = lcp_query(T, r0, S, dir, len, t, stopN);
                        if (m <= v) { atomicAdd(why + 5, 1ULL); continue; }
                        oc = s.maxL[r] <= v ? 4 : 5;
                        bound[r - q] = v + d;
                        break;
                    }
                }
                outcome[r] = oc;
                if (oc != 2 && oc != 4) allAnswered = false;
                if (oc == 3 || oc == 5) wrong = true;
            }
            bool resolved = allAnswered && bestCert >= 0;
            for (u64 r = q; r < g1 && resolved; r++) if (bound[r - q] >= 0 && bound[r - q] >= bestCert) resolved = false;
            callres[q] = resolved ? (wrong ? 3 : 2) : 1;
            for (u64 r = q + 1; r < g1; r++) callres[r] = 0;
            for (u64 r = q; r < g1; r++)   // anchors: records STAR answered with one locus (a certified answer equals it)
                if (s.Nrep[r] == 1 && s.maxL[r] >= minAnchorL && na < 4) {
                    int d = s.idist[r], S = s.dir[r] == 0 ? s.start[r] + d : s.start[r] - d;
                    anc[na++] = anchor_of(s.dir[r], S, s.sa0[r], T.nGenome, cI.GstrandBit);
                }
            q = g1;
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// host
// ---------------------------------------------------------------------------------------------------------------------
static u64 fileSize(const std::string& p) { struct stat st; if (stat(p.c_str(), &st)) { perror(p.c_str()); exit(2); } return st.st_size; }

static void readInto(const std::string& path, u64 off, char* dst, u64 n) {
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { perror(path.c_str()); exit(2); }
    fseeko(f, (off_t)off, SEEK_SET);
    u64 got = 0; while (got < n) { size_t k = fread(dst + got, 1, std::min<u64>(n - got, 1ULL << 30), f); if (!k) break; got += k; }
    fclose(f);
    if (got != n) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
}

// stream a host file region into device memory through a pinned staging buffer
static void fileToDevice(const std::string& path, u64 off, char* d, u64 n) {
    const u64 B = 1ULL << 28; char* h; CK(cudaMallocHost(&h, B));
    FILE* f = fopen(path.c_str(), "rb"); if (!f) { perror(path.c_str()); exit(2); }
    fseeko(f, (off_t)off, SEEK_SET);
    for (u64 done = 0; done < n;) {
        u64 k = std::min(B, n - done);
        if (fread(h, 1, k, f) != k) { fprintf(stderr, "short read %s\n", path.c_str()); exit(2); }
        CK(cudaMemcpy(d + done, h, k, cudaMemcpyHostToDevice)); done += k;
    }
    fclose(f); CK(cudaFreeHost(h));
}

struct HostDump {
    std::vector<char> rb; std::vector<u64> roff{0}, rseed{0}; std::vector<uint32_t> rcount; std::vector<uint16_t> rlen;
    std::vector<uint32_t> read; std::vector<uint8_t> dir, idist, branch, nsplit; std::vector<uint16_t> start, len, maxL;
    std::vector<u64> Nrep, ind0, ind1, sa0;
};

static void loadDump(const std::string& prefix, HostDump& D) {
    glob_t g; std::string pat = prefix + ".*.bin";
    if (glob(pat.c_str(), 0, nullptr, &g)) { fprintf(stderr, "no dump files %s\n", pat.c_str()); exit(2); }
    for (size_t fi = 0; fi < g.gl_pathc; fi++) {
        std::string p = g.gl_pathv[fi]; u64 n = fileSize(p);
        std::vector<char> b(n); readInto(p, 0, b.data(), n);
        u64 o = sizeof(stardump::FileHeader);
        if (n < o || memcmp(b.data(), "STARDMP1", 8)) { fprintf(stderr, "bad header %s\n", p.c_str()); exit(2); }
        while (o < n) {
            if (b[o] == 'R') {
                stardump::ReadRec r; memcpy(&r, b.data() + o, sizeof r); o += sizeof r;
                o += 6ULL * r.Nsplit;
                if (!D.rcount.empty()) D.rseed.push_back(D.read.size());
                D.rcount.push_back(r.count); D.rlen.push_back(r.Lread); D.nsplit.push_back(r.Nsplit);
                D.rb.insert(D.rb.end(), b.data() + o, b.data() + o + 2ULL * r.Lread); o += 2ULL * r.Lread;
                D.roff.push_back(D.rb.size());
            } else if (b[o] == 'S') {
                stardump::SeedRec s; memcpy(&s, b.data() + o, sizeof s); o += sizeof s;
                D.read.push_back((uint32_t)(D.rcount.size() - 1)); D.dir.push_back(s.iDir); D.idist.push_back(s.iDist); D.branch.push_back(s.branch);
                D.start.push_back(s.pieceStart); D.len.push_back(s.pieceLength); D.maxL.push_back(s.maxL);
                D.Nrep.push_back(s.Nrep); D.ind0.push_back(s.ind0); D.ind1.push_back(s.ind1); D.sa0.push_back(s.sa0);
            } else { fprintf(stderr, "bad record tag %d at %llu in %s\n", b[o], o, p.c_str()); exit(2); }
        }
    }
    D.rseed.push_back(D.read.size());
    globfree(&g);
}

template <class T> static T* toDev(const std::vector<T>& v) {
    T* d; CK(cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T)));
    if (!v.empty()) CK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
template <class T> static T* devAlloc(u64 n) { T* d; CK(cudaMalloc(&d, std::max<u64>(1, n) * sizeof(T))); return d; }

static double median(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

int main(int argc, char** argv) {
    std::string gdir, dump, json; int reps = 3; bool m4 = true; int minAnchorL = 20;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--genome-dir") gdir = argv[++i];
        else if (a == "--dump") dump = argv[++i];
        else if (a == "--reps") reps = atoi(argv[++i]);
        else if (a == "--json") json = argv[++i];
        else if (a == "--no-m4") m4 = false;
        else if (a == "--min-anchor") minAnchorL = atoi(argv[++i]);
    }
    if (gdir.empty() || dump.empty()) { fprintf(stderr, "usage: seedsearch --genome-dir D --dump PREFIX [--json F]\n"); return 2; }
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    int drv, rtv; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rtv);
    fprintf(stderr, "device %s, %d SMs, L2 %d MB, driver %d runtime %d\n", prop.name, prop.multiProcessorCount, prop.l2CacheSize >> 20, drv, rtv);

    // ---- index
    double t0 = now();
    Index I{};
    I.nGenome = fileSize(gdir + "/Genome");
    u64 nSAbyte = fileSize(gdir + "/SA");
    I.GstrandBit = 0;
    {   // genomeParameters.txt may define GstrandBit
        FILE* f = fopen((gdir + "/genomeParameters.txt").c_str(), "r"); char line[4096];
        while (f && fgets(line, sizeof line, f)) { unsigned v; if (sscanf(line, "GstrandBit %u", &v) == 1) I.GstrandBit = v; }
        if (f) fclose(f);
    }
    if (I.GstrandBit == 0) { I.GstrandBit = (u64)floor(log((double)I.nGenome) / log(2.0)) + 1; if (I.GstrandBit < 32) I.GstrandBit = 32; }
    I.GstrandMask = ~(1ULL << I.GstrandBit);
    I.saBits = I.GstrandBit + 1; I.saiBits = I.GstrandBit + 3;
    I.nSA = nSAbyte * 8 / I.saBits;
    I.SAiMarkNmaskC = 1ULL << (I.GstrandBit + 1); I.SAiMarkNmask = ~I.SAiMarkNmaskC;
    I.SAiMarkAbsentMaskC = 1ULL << (I.GstrandBit + 2);
    std::string saiPath = gdir + "/SAindex";
    readInto(saiPath, 0, (char*)&I.gSAindexNbases, 8);
    if (I.gSAindexNbases > 30) { fprintf(stderr, "gSAindexNbases %llu\n", I.gSAindexNbases); return 2; }
    readInto(saiPath, 8, (char*)I.saiStart, 8 * (I.gSAindexNbases + 1));
    I.nSAi = I.saiStart[I.gSAindexNbases];
    u64 saiHdr = 8 + 8 * (I.gSAindexNbases + 1);
    u64 saiBytes = (I.nSAi - 1) * I.saiBits / 8 + 8;
    fprintf(stderr, "nGenome %llu nSA %llu GstrandBit %llu nSAi %llu Nbases %llu\n", I.nGenome, I.nSA, I.GstrandBit, I.nSAi, I.gSAindexNbases);

    const u64 PAD = 200;
    char* dG1 = devAlloc<char>(I.nGenome + 2 * PAD + 64);
    CK(cudaMemset(dG1, 5, I.nGenome + 2 * PAD + 64));                 // STAR pads with K-1 = 5 on both sides
    fileToDevice(gdir + "/Genome", 0, dG1 + PAD, I.nGenome);
    u64* dSA = devAlloc<u64>(nSAbyte / 8 + 4); CK(cudaMemset(dSA, 0, (nSAbyte / 8 + 4) * 8));
    fileToDevice(gdir + "/SA", 0, (char*)dSA, nSAbyte);
    u64* dSAi = devAlloc<u64>(saiBytes / 8 + 4); CK(cudaMemset(dSAi, 0, (saiBytes / 8 + 4) * 8));
    fileToDevice(saiPath, saiHdr, (char*)dSAi, saiBytes);
    I.G = dG1 + PAD; I.SA = dSA; I.SAi = dSAi;
    CK(cudaMemcpyToSymbol(cI, &I, sizeof I));
    double tIndex = now() - t0;
    fprintf(stderr, "index loaded to HBM in %.1f s (%.1f GB)\n", tIndex, (I.nGenome + nSAbyte + saiBytes) / 1e9);

    // ---- dump
    t0 = now();
    HostDump D; loadDump(dump, D);
    u64 nS = D.read.size(); uint32_t nR = (uint32_t)D.rcount.size();
    u64 nReadsW = 0; for (auto c : D.rcount) nReadsW += c;
    fprintf(stderr, "dump: %u distinct reads (%llu reads), %llu seed calls, loaded in %.1f s\n", nR, nReadsW, nS, now() - t0);

    char* dRb = toDev(D.rb); u64* dRoff = toDev(D.roff); u64* dRseed = toDev(D.rseed);
    Seeds S{toDev(D.read), toDev(D.dir), toDev(D.idist), toDev(D.start), toDev(D.len), toDev(D.maxL), toDev(D.Nrep), toDev(D.ind0),
            toDev(D.ind1), toDev(D.sa0), toDev(D.branch)};
    uint16_t* oL = devAlloc<uint16_t>(nS); u64* oN = devAlloc<u64>(nS); u64* o0 = devAlloc<u64>(nS); u64* o1 = devAlloc<u64>(nS);
    uint8_t* oB = devAlloc<uint8_t>(nS); unsigned* osteps = devAlloc<unsigned>(nS);

    // ---- M3: dump order and prefix-sorted order
    uint32_t* dOrder = devAlloc<uint32_t>(nS);
    {
        u64* keys = devAlloc<u64>(nS);
        k_prefix_keys<<<4096, 256>>>(dRb, dRoff, S, nS, keys); CK(cudaGetLastError());
        thrust::sequence(thrust::device_ptr<uint32_t>(dOrder), thrust::device_ptr<uint32_t>(dOrder + nS));
        thrust::stable_sort_by_key(thrust::device_ptr<u64>(keys), thrust::device_ptr<u64>(keys + nS), thrust::device_ptr<uint32_t>(dOrder));
        CK(cudaDeviceSynchronize()); CK(cudaFree(keys));
    }
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    struct Run { const char* name; const uint32_t* order; std::vector<double> ms; double simt; u64 stepsTotal; };
    Run runs[2] = {{"dump_order", nullptr, {}, 0, 0}, {"prefix_sorted", dOrder, {}, 0, 0}};
    std::vector<unsigned> hsteps(nS);
    for (auto& R : runs) {
        for (int rep = 0; rep <= reps; rep++) {   // rep 0 = warm-up
            cudaEventRecord(e0);
            k_search<<<prop.multiProcessorCount * 16, 256>>>(dRb, dRoff, S, R.order, nS, oL, oN, o0, o1, oB, osteps);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            float ms; cudaEventElapsedTime(&ms, e0, e1); if (rep) R.ms.push_back(ms);
        }
        CK(cudaMemcpy(hsteps.data(), osteps, nS * 4, cudaMemcpyDeviceToHost));
        double sumMean = 0; u64 nw = 0, tot = 0;
        for (u64 w = 0; w + 32 <= nS; w += 32) {
            unsigned mx = 0; u64 sm = 0;
            for (int j = 0; j < 32; j++) { mx = std::max(mx, hsteps[w + j]); sm += hsteps[w + j]; }
            if (mx) { sumMean += (double)sm / 32.0 / mx; nw++; }
        }
        for (u64 k = 0; k < nS; k++) tot += hsteps[k];
        R.simt = nw ? sumMean / nw : 0; R.stepsTotal = tot;
        fprintf(stderr, "M3 %s: %.2f ms median -> %.1f M seeds/s, steps/seed %.1f, SIMT(step) %.3f\n", R.name, median(R.ms),
                nS / (median(R.ms) * 1e3), (double)tot / nS, R.simt);
    }
    // identity vs STAR (outputs of the last run; both runs compute the same function)
    std::vector<uint16_t> hL(nS); std::vector<u64> hN(nS), h0(nS), h1(nS); std::vector<uint8_t> hB(nS);
    CK(cudaMemcpy(hL.data(), oL, nS * 2, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hN.data(), oN, nS * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h0.data(), o0, nS * 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(h1.data(), o1, nS * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(hB.data(), oB, nS, cudaMemcpyDeviceToHost));
    u64 diff = 0, diffBranch = 0; u64 byBranch[3] = {0, 0, 0}, nrepGt1 = 0, nrep0 = 0;
    for (u64 k = 0; k < nS; k++) {
        bool same = hL[k] == D.maxL[k] && hN[k] == D.Nrep[k] && (D.Nrep[k] == 0 || (h0[k] == D.ind0[k] && h1[k] == D.ind1[k]));
        if (!same) { if (diff < 10) fprintf(stderr, "  DIFF seed %llu read %u dir %d S %d len %d: gpu L %d N %llu [%llu,%llu] star L %d N %llu [%llu,%llu]\n",
                 k, D.read[k], D.dir[k], D.start[k], D.len[k], hL[k], hN[k], h0[k], h1[k], D.maxL[k], D.Nrep[k], D.ind0[k], D.ind1[k]); diff++; }
        if (hB[k] != D.branch[k]) diffBranch++;
        byBranch[D.branch[k] < 3 ? D.branch[k] : 0]++;
        if (D.Nrep[k] > 1) nrepGt1++; if (D.Nrep[k] == 0) nrep0++;
    }
    fprintf(stderr, "M3 identity: %llu / %llu seeds differ (branch differs on %llu)\n", diff, nS, diffBranch);

    // per-branch timing (prefix-sorted order restricted to one branch)
    double branchMs[3] = {0, 0, 0}; u64 branchN[3] = {0, 0, 0};
    for (int b = 0; b < 3; b++) {
        std::vector<uint32_t> ord;
        std::vector<uint32_t> hOrder(nS); CK(cudaMemcpy(hOrder.data(), dOrder, nS * 4, cudaMemcpyDeviceToHost));
        for (u64 k = 0; k < nS; k++) if (D.branch[hOrder[k]] == b) ord.push_back(hOrder[k]);
        branchN[b] = ord.size(); if (ord.empty()) continue;
        uint32_t* dO = toDev(ord); std::vector<double> ms;
        for (int rep = 0; rep <= reps; rep++) {
            cudaEventRecord(e0);
            k_search<<<prop.multiProcessorCount * 16, 256>>>(dRb, dRoff, S, dO, ord.size(), oL, oN, o0, o1, oB, nullptr);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1));
            float m; cudaEventElapsedTime(&m, e0, e1); if (rep) ms.push_back(m);
        }
        branchMs[b] = median(ms); CK(cudaFree(dO));
        fprintf(stderr, "M3 branch %d: %llu seeds, %.1f M seeds/s\n", b, branchN[b], branchN[b] / (branchMs[b] * 1e3));
    }
    // seed calls per read (distribution, read-weighted)
    std::vector<u64> perRead(64, 0);
    for (uint32_t r = 0; r < nR; r++) { u64 c = D.rseed[r + 1] - D.rseed[r]; perRead[std::min<u64>(c, 63)] += D.rcount[r]; }

    // ---- M4
    double tPack = 0, tLcp = 0, tScatter = 0, tV = 0, tCert = 0; unsigned long long hViol = 0;
    std::vector<unsigned long long> lcpHist(256, 0); std::vector<unsigned> uUsed(256, 0);
    u64 oc[6] = {0}, ocW[6] = {0}, certByBranch[3] = {0, 0, 0};
    u64 calls = 0, callsW = 0, res[4] = {0}, resW[4] = {0};
    u64 classCalls[3] = {0}, classRes[3] = {0}, classCallsW[3] = {0}, classResW[3] = {0};
    u64 uHist[256] = {0}, vHist[256] = {0}; unsigned long long why[8] = {0};
    if (m4) {
        u64 nBases = 2 * I.nGenome, nWords = (nBases + 31) / 32 + 4, nNb = (nWords + 1) / 2 + 4;
        u64* dW = devAlloc<u64>(nWords); u64* dNb = devAlloc<u64>(nNb); CK(cudaMemset(dNb, 0, nNb * 8));
        cudaEventRecord(e0);
        k_pack_text<<<4096, 256>>>(I.G, I.nGenome, nWords, dW, dNb);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        float ms; cudaEventElapsedTime(&ms, e0, e1); tPack = ms / 1e3;
        Text T{dW, dNb, I.nGenome, nBases};
        uint8_t* dLcp = devAlloc<uint8_t>(I.nSA);
        unsigned long long *dViol = devAlloc<unsigned long long>(1), *dHist = devAlloc<unsigned long long>(256);
        CK(cudaMemset(dViol, 0, 8)); CK(cudaMemset(dHist, 0, 256 * 8));
        cudaEventRecord(e0);
        k_build_lcp<<<prop.multiProcessorCount * 16, 256>>>(T, I.nSA, dLcp, dViol, dHist);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        cudaEventElapsedTime(&ms, e0, e1); tLcp = ms / 1e3;
        CK(cudaMemcpy(&hViol, dViol, 8, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(lcpHist.data(), dHist, 256 * 8, cudaMemcpyDeviceToHost));
        uint8_t* dU = devAlloc<uint8_t>(nBases + 64); CK(cudaMemset(dU, 255, nBases + 64));
        u64 nSmp = nBases / 64 + 8; u64* dSmp = devAlloc<u64>(nSmp); CK(cudaMemset(dSmp, 0, nSmp * 8));
        cudaEventRecord(e0);
        k_scatter_u<<<prop.multiProcessorCount * 16, 256>>>(I.nSA, dLcp, dU, (unsigned long long*)dSmp);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        cudaEventElapsedTime(&ms, e0, e1); tScatter = ms / 1e3;
        CK(cudaFree(dLcp));
        uint8_t* dV = devAlloc<uint8_t>(nBases + 64); CK(cudaMemset(dV, 255, nBases + 64));
        cudaEventRecord(e0);
        k_build_v<<<prop.multiProcessorCount * 16, 256>>>(T, I.nSA, dSmp, dV);
        cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
        cudaEventElapsedTime(&ms, e0, e1); tV = ms / 1e3;
        fprintf(stderr, "M4 build: pack %.2f s, lcp %.2f s, scatter %.2f s, V %.2f s; SA order violations %llu\n", tPack, tLcp, tScatter, tV, hViol);
        {   // U and V distributions over all text positions (255 = no bound / not a suffix start)
            CK(cudaMemset(dHist, 0, 256 * 8));
            k_hist_u8<<<prop.multiProcessorCount * 8, 256>>>(dU, nBases, dHist); CK(cudaGetLastError());
            CK(cudaMemcpy(uHist, dHist, 256 * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemset(dHist, 0, 256 * 8));
            k_hist_u8<<<prop.multiProcessorCount * 8, 256>>>(dV, nBases, dHist); CK(cudaGetLastError());
            CK(cudaMemcpy(vHist, dHist, 256 * 8, cudaMemcpyDeviceToHost));
        }
        uint8_t* dOc = devAlloc<uint8_t>(nS); uint8_t* dCr = devAlloc<uint8_t>(nS); unsigned* dUsed = devAlloc<unsigned>(256);
        unsigned long long* dWhy = devAlloc<unsigned long long>(8);
        std::vector<double> cms;
        for (int rep = 0; rep <= reps; rep++) {
            CK(cudaMemset(dUsed, 0, 256 * 4)); CK(cudaMemset(dWhy, 0, 8 * 8));
            cudaEventRecord(e0);
            k_certify<<<prop.multiProcessorCount * 8, 128>>>(dRb, dRoff, dRseed, nR, S, T, dU, dV, dSmp, dOc, dCr, dUsed, dWhy, minAnchorL);
            cudaEventRecord(e1); CK(cudaEventSynchronize(e1)); CK(cudaGetLastError());
            cudaEventElapsedTime(&ms, e0, e1); if (rep) cms.push_back(ms);
        }
        tCert = median(cms) / 1e3;
        std::vector<uint8_t> hOc(nS), hCr(nS);
        CK(cudaMemcpy(hOc.data(), dOc, nS, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hCr.data(), dCr, nS, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(uUsed.data(), dUsed, 256 * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(why, dWhy, 8 * 8, cudaMemcpyDeviceToHost));
        // read classes: 0 perfect (first call maps the whole piece uniquely), 1 heavy (any record with Nrep > 50, or too
        // short to anchor: no record with maxL >= 20), 2 other (mismatch, splice, multimapper with few loci)
        for (uint32_t r = 0; r < nR; r++) {
            u64 a = D.rseed[r], b = D.rseed[r + 1]; if (a == b) continue;
            bool perfect = D.Nrep[a] == 1 && D.maxL[a] == D.len[a] - D.idist[a] && D.dir[a] == 0;
            bool heavy = false, anchorable = false;
            for (u64 q = a; q < b; q++) { if (D.Nrep[q] > 50) heavy = true; if (D.maxL[q] >= 20) anchorable = true; }
            int cls = perfect ? 0 : (heavy || !anchorable) ? 1 : 2;
            for (u64 q = a; q < b; q++) {
                if (hCr[q] == 0) continue;   // not the first record of a call
                classCalls[cls]++; classCallsW[cls] += D.rcount[r];
                if (hCr[q] == 2) { classRes[cls]++; classResW[cls] += D.rcount[r]; }
            }
        }
        for (u64 k = 0; k < nS; k++) {
            u64 w = D.rcount[D.read[k]];
            oc[hOc[k]]++; ocW[hOc[k]] += w; if (hOc[k] == 2) certByBranch[D.branch[k]]++;
            if (hCr[k]) { calls++; callsW += w; res[hCr[k]]++; resW[hCr[k]] += w; }
        }
        fprintf(stderr, "M4 certify: %.2f ms -> %.1f M records/s; records none/declined/cert/CERT_WRONG/bounded/BOUND_WRONG = "
                "%llu/%llu/%llu/%llu/%llu/%llu; calls %llu resolved %llu RESOLVED_WRONG %llu\n",
                tCert * 1e3, nS / (tCert * 1e6), oc[0], oc[1], oc[2], oc[3], oc[4], oc[5], calls, res[2], res[3]);
        CK(cudaFree(dU)); CK(cudaFree(dV)); CK(cudaFree(dSmp)); CK(cudaFree(dW)); CK(cudaFree(dNb));
    }

    // ---- JSON
    FILE* jf = json.empty() ? stdout : fopen(json.c_str(), "w");
    fprintf(jf, "{\n \"device\": \"%s\", \"sms\": %d, \"driver\": %d, \"runtime\": %d,\n", prop.name, prop.multiProcessorCount, drv, rtv);
    fprintf(jf, " \"index\": {\"dir\": \"%s\", \"nGenome\": %llu, \"nSA\": %llu, \"GstrandBit\": %llu, \"nSAi\": %llu, \"load_s\": %.2f},\n",
            gdir.c_str(), I.nGenome, I.nSA, I.GstrandBit, I.nSAi, tIndex);
    fprintf(jf, " \"dump\": {\"distinct_reads\": %u, \"reads\": %llu, \"seed_calls\": %llu, \"seed_calls_nrep0\": %llu, \"seed_calls_nrep_gt1\": %llu,\n"
                "   \"by_branch\": {\"sai_only\": %llu, \"unique_extend\": %llu, \"sa_search\": %llu},\n   \"calls_per_read_weighted\": [",
            nR, nReadsW, nS, nrep0, nrepGt1, byBranch[0], byBranch[1], byBranch[2]);
    for (int i = 0; i < 64; i++) fprintf(jf, "%s%llu", i ? "," : "", perRead[i]);
    fprintf(jf, "]},\n \"m3\": {\"identical\": %llu, \"different\": %llu, \"branch_different\": %llu, \"reps\": %d,\n", nS - diff, diff, diffBranch, reps);
    for (int i = 0; i < 2; i++)
        fprintf(jf, "   \"%s\": {\"ms_median\": %.3f, \"ms_all\": [%.3f,%.3f,%.3f], \"Mseeds_per_s\": %.2f, \"steps_per_seed\": %.3f, \"simt_steps\": %.4f},\n",
                runs[i].name, median(runs[i].ms), runs[i].ms[0], runs[i].ms[std::min<size_t>(1, runs[i].ms.size() - 1)],
                runs[i].ms.back(), nS / (median(runs[i].ms) * 1e3), (double)runs[i].stepsTotal / nS, runs[i].simt);
    fprintf(jf, "   \"sorted_over_unsorted\": %.3f,\n   \"by_branch_sorted\": {", median(runs[0].ms) / median(runs[1].ms));
    const char* bn[3] = {"sai_only", "unique_extend", "sa_search"};
    for (int b = 0; b < 3; b++) fprintf(jf, "%s\"%s\": {\"seeds\": %llu, \"Mseeds_per_s\": %.2f}", b ? ", " : "", bn[b], branchN[b],
                                        branchMs[b] > 0 ? branchN[b] / (branchMs[b] * 1e3) : 0.0);
    fprintf(jf, "}},\n");
    if (m4) {
        fprintf(jf, " \"m4\": {\"build_s\": {\"pack_text\": %.3f, \"lcp\": %.3f, \"scatter_u\": %.3f, \"v_array\": %.3f}, \"sa_order_violations\": %llu,\n",
                tPack, tLcp, tScatter, tV, hViol);
        fprintf(jf, "   \"lcp_hist\": ["); for (int i = 0; i < 256; i++) fprintf(jf, "%s%llu", i ? "," : "", lcpHist[i]); fprintf(jf, "],\n");
        fprintf(jf, "   \"u_hist\": ["); for (int i = 0; i < 256; i++) fprintf(jf, "%s%llu", i ? "," : "", uHist[i]); fprintf(jf, "],\n");
        fprintf(jf, "   \"v_hist\": ["); for (int i = 0; i < 256; i++) fprintf(jf, "%s%llu", i ? "," : "", vHist[i]); fprintf(jf, "],\n");
        fprintf(jf, "   \"certify_ms\": %.3f, \"certify_Mrecords_per_s\": %.2f,\n", tCert * 1e3, nS / (tCert * 1e6));
        fprintf(jf, "   \"min_anchor_L\": %d, \"decline_attempts\": {\"u_none\": %llu, \"u_gt_len\": %llu, \"m_lt_u\": %llu, \"n_in_text\": %llu, \"v_none\": %llu, \"m_le_v\": %llu},\n",
                minAnchorL, why[0], why[1], why[2], why[3], why[4], why[5]);
        const char* on[6] = {"no_anchor", "declined", "certified_equal", "certified_WRONG", "bounded_ok", "bounded_WRONG"};
        fprintf(jf, "   \"records\": {"); for (int i = 0; i < 6; i++) fprintf(jf, "%s\"%s\": %llu", i ? ", " : "", on[i], oc[i]); fprintf(jf, "},\n");
        fprintf(jf, "   \"records_read_weighted\": {"); for (int i = 0; i < 6; i++) fprintf(jf, "%s\"%s\": %llu", i ? ", " : "", on[i], ocW[i]); fprintf(jf, "},\n");
        fprintf(jf, "   \"calls\": {\"total\": %llu, \"resolved\": %llu, \"resolved_WRONG\": %llu, \"total_w\": %llu, \"resolved_w\": %llu},\n",
                calls, res[2], res[3], callsW, resW[2]);
        fprintf(jf, "   \"certified_by_branch\": {\"sai_only\": %llu, \"unique_extend\": %llu, \"sa_search\": %llu},\n", certByBranch[0], certByBranch[1], certByBranch[2]);
        const char* cn[3] = {"perfect", "heavy_or_short", "other"};
        fprintf(jf, "   \"calls_by_read_class\": {");
        for (int c = 0; c < 3; c++) fprintf(jf, "%s\"%s\": {\"calls\": %llu, \"resolved\": %llu, \"calls_w\": %llu, \"resolved_w\": %llu}",
                                            c ? ", " : "", cn[c], classCalls[c], classRes[c], classCallsW[c], classResW[c]);
        fprintf(jf, "},\n   \"u_of_certified\": ["); for (int i = 0; i < 256; i++) fprintf(jf, "%s%u", i ? "," : "", uUsed[i]); fprintf(jf, "]}\n");
    } else fprintf(jf, " \"m4\": null\n");
    fprintf(jf, "}\n");
    if (jf != stdout) fclose(jf);
    return diff == 0 && oc[3] == 0 && oc[5] == 0 && res[3] == 0 ? 0 : 1;
}
