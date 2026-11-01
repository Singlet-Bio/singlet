// U-array seed certificates: build and use (docs/gpu-engine/exploration.md, sections 3.3 and 4.4).
//
// Sketch of the two kernels that make "locus propagation" exact. Not wired into CMake; it shows data layout and the
// exactness rule, not tuned code.
//
// Text. STAR's suffix array (SA) indexes suffixes of two strands. SA[i] = (strand bit << GstrandBit) | pos.
//   strand 0: suffix = G[pos], G[pos+1], ...
//   strand 1: suffix = comp(G[nGenome-1-pos]), comp(G[nGenome-2-pos]), ...   (reverse complement)
// We store G and its reverse complement GRC as 2-bit packed arrays (0.8 GB each), so both strands are forward reads:
//   text coordinate t = strand * nGenome + pos ;  base(t + k) = strand ? GRC[pos + k] : G[pos + k]
// N and spacer bases are kept in a separate bitmap (GenomePacked.h does the same); they never match anything here.
//
// U array. For every text coordinate t that starts a suffix in SA (i = ISA[t]):
//   lcp_prev(i) = LCP(suffix SA[i-1], suffix SA[i]) over A/C/G/T only (stops at N/spacer), capped at 254
//   U[t] = 1 + max(lcp_prev(i), lcp_prev(i+1))   = the shortest prefix length that occurs only at t in the text
//   U[t] = 255 means "not certifiable" (cap reached, or t is not a suffix start)
// Size: 2 * nGenome bytes ~ 6.4 GB for GRCh38 (4-bit variant: 3.2 GB, U in 12..26, 15 = not certifiable).
//
// Exactness (the certificate). Let Q be a seed query as STAR builds it (read[S..] for a forward search, or the reverse
// complement of read[..S] for a reverse search; seeds never contain N because qualitySplit cuts reads at N). Let
// m = lcp(Q, text[t..]) for a candidate t, and suppose m >= U[t] and (m == |Q| or Q[m] != base(t+m)). Then:
//   1. any suffix sharing m >= U[t] bases with Q shares U[t] bases with suffix t; by the definition of U and the
//      sortedness of SA, only suffix t has that prefix -> the length-m match set is exactly {t};
//   2. Q[0..m] (m+1 bases) does not match at t (mismatch or end of Q) and nowhere else (step 1) -> MMP = m.
// So the seed's result is L = m, Nrep = 1, loci = {t}: the mathematical maximal-mappable-prefix answer. STAR's
// maxMappableLength2strands returns the same answer whenever its SAi+binary search is correct for that query; M4
// checks this equality on every seed of the real test datasets (and SA sortedness is re-checked for free by build_lcp).
// When the certificate fails, the seed goes to the exact SA search (a port of STAR's code), so the output never differs.
//
// Where t comes from ("locus propagation"): after one seed of a read is answered as a single locus t0 (by search or by
// the locator), STAR's other seeds of the same read start at known read offsets; under the ungapped hypothesis their
// candidates are t0 + offset on the same strand (or the mirrored coordinate on the other strand for reverse searches).
// A wrong hypothesis (indel, splice, other locus) just fails the test m >= U[t] and falls back. No heuristic can change
// an answer; it can only change which code path computes it.
#include <cstdint>

struct PackedText {
    const uint64_t* w;       // G words then GRC words, MSB-first 2-bit (A=0 C=1 G=2 T=3), 32 bases per word
    const uint64_t* nmask;   // 1 bit per 32-base word: word contains N or spacer
    uint64_t nGenome;        // bases per strand
};

// 32 bases starting at text coordinate t (may straddle two words). Same shift trick as GenomePacked.h genomeWordAt.
__device__ __forceinline__ uint64_t word_at(const PackedText& T, uint64_t t) {
    uint64_t i = t >> 5; unsigned sh = (unsigned)(t & 31) * 2;
    uint64_t a = __ldg(T.w + i);
    return sh ? (a << sh) | (__ldg(T.w + i + 1) >> (64 - sh)) : a;
}

__device__ __forceinline__ bool has_n(const PackedText& T, uint64_t t, int len) {
    for (uint64_t i = t >> 5; i <= (t + len - 1) >> 5; i++)
        if ((__ldg(T.nmask + (i >> 6)) >> (i & 63)) & 1) return true;   // conservative: whole word flagged
    return false;
}

// lcp of query q (packed MSB-first, qlen bases) with text suffix t, A/C/G/T only, capped at cap.
__device__ int lcp_query_text(const PackedText& T, const uint64_t* q, int qlen, uint64_t t, int cap) {
    int m = 0, lim = qlen < cap ? qlen : cap;
    while (m < lim) {
        int n = lim - m < 32 ? lim - m : 32;
        if (has_n(T, t + m, n)) return m;      // the slow exact path handles windows with N; certificate just declines
        uint64_t qw = (m & 31) ? (q[m >> 5] << ((m & 31) * 2)) | (q[(m >> 5) + 1] >> (64 - (m & 31) * 2)) : q[m >> 5];
        uint64_t x = (qw ^ word_at(T, t + m)) & (n == 32 ? ~0ULL : ~0ULL << (64 - 2 * n));
        if (x) return m + (__clzll(x) >> 1);
        m += n;
    }
    return m;
}

// ---- build: one thread per SA entry. SA is STAR's packed array (33-bit words for GRCh38); readSA unpacks it.
__device__ __forceinline__ uint64_t sa_to_t(uint64_t sa, unsigned gstrandBit, uint64_t nGenome) {
    uint64_t strand = sa >> gstrandBit, pos = sa & ((1ULL << gstrandBit) - 1);
    return strand * nGenome + pos;
}

template <class SAReader>
__global__ void build_lcp(SAReader SA, uint64_t nSA, PackedText T, unsigned gsb, uint8_t* lcp_prev, int* unsorted) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < nSA; i += (uint64_t)gridDim.x * blockDim.x) {
        if (i == 0) { lcp_prev[0] = 0; continue; }
        uint64_t a = sa_to_t(SA(i - 1), gsb, T.nGenome), b = sa_to_t(SA(i), gsb, T.nGenome);
        int l = 0;
        while (l < 254) {                                   // 2-3 random sectors per pair; ~6e9 pairs -> seconds
            int n = 254 - l < 32 ? 254 - l : 32;
            if (has_n(T, a + l, n) || has_n(T, b + l, n)) break;
            uint64_t x = (word_at(T, a + l) ^ word_at(T, b + l)) & (n == 32 ? ~0ULL : ~0ULL << (64 - 2 * n));
            if (x) {
                int k = __clzll(x) >> 1; l += k;
                if (k < n && ((word_at(T, a + l - k) << (2 * k)) > (word_at(T, b + l - k) << (2 * k)))) atomicAdd(unsorted, 1);
                break;
            }
            l += n;
        }
        lcp_prev[i] = (uint8_t)l;                           // SA-order array, 1 B per entry, written coalesced
    }
}

template <class SAReader>
__global__ void scatter_u(SAReader SA, uint64_t nSA, const uint8_t* lcp_prev, unsigned gsb, uint64_t nGenome, uint8_t* U) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < nSA; i += (uint64_t)gridDim.x * blockDim.x) {
        int l = lcp_prev[i], r = i + 1 < nSA ? lcp_prev[i + 1] : 0;
        int u = 1 + (l > r ? l : r);
        U[sa_to_t(SA(i), gsb, nGenome)] = (uint8_t)(u >= 255 ? 255 : u);   // random 1-B writes; U pre-filled with 255
    }
}

// ---- use: one seed. Returns true and fills (L, locus) when the certificate holds; false sends the seed to the SA search.
struct SeedAnswer { int L; uint64_t t; };

__device__ bool certify_seed(const PackedText& T, const uint8_t* __restrict__ U,
                             const uint64_t* q, int qlen, uint64_t t_candidate, SeedAnswer& out) {
    uint8_t u = __ldg(U + t_candidate);                    // 1 random sector
    if (u == 255 || u > qlen) return false;
    int m = lcp_query_text(T, q, qlen, t_candidate, qlen);  // usually L1-hot: previous seed of this read read these words
    if (m < u) return false;
    // m == qlen (end of seed) or q[m] != base(t+m) (lcp stopped on a mismatch). An N in the window returned early with
    // m < qlen and no mismatch: decline, because STAR's comparison semantics at N are handled only by the exact path.
    if (m < qlen && has_n(T, t_candidate + m, 1)) return false;
    out.L = m; out.t = t_candidate;                         // Nrep = 1; PC_SAstart/End carry the locus, tagged "direct"
    return true;
}
