// SPDX-License-Identifier: MIT
// Idea 1 / Idea 5, stage 4: the fast decoder core (format "v2").  Independent of the harness (works with
// -DIDEA1_LIGHT, i.e. without the singlet/zstd headers) so it can be micro-benchmarked and compiled quickly.
//
// Stream format v2 (same probability model and symbols as the reference codec, reorganised):
//   * independent rANS streams instead of one state:
//       B  one stream with the depth-bucket symbol of every column (decoded up front into a byte array),
//       G  per lane: gap symbols (+ escape positions)       -- the long dependency chain
//       C  per lane: count symbols                          -- depends only on the decoded feature id
//     so the count chain runs in parallel with the gap chain (instruction-level parallelism).
//   * LANES lanes = contiguous column ranges balanced by nnz, each with its own G and C stream; the decoder
//     interleaves the lanes step by step (more ILP) and each lane writes straight into the output arrays at
//     the offsets stored in the header.  Lanes are also the natural unit for multi-threading.
//   * the search for the next feature is a 16-way SIMD tree (AVX-512, 64-bit keys) over the log-survival
//     axis.  For the rank-1 (factored) model the tree is over the *shared* prefix sums S and the target is
//     rescaled by a per-bucket reciprocal, so all depth buckets share one 2.5 MB array; for the dense
//     per-(feature, bucket) table there is one tree per bucket.
//   * renormalisation is branchless; count symbols are decoded with one 16-lane 16-bit compare.
#pragma once
#include "idea1.h"

namespace cs {
namespace i1 {

#ifndef IDEA1_UNROLL
#define I1_UNROLL _Pragma("GCC unroll 16")
#else
#define I1_UNROLL _Pragma("GCC unroll 1")
#endif
constexpr int MAXL = 16;          // upper bound on lanes in a stream
constexpr int DEFAULT_LANES = 4;  // lanes written by the encoder unless overridden
#if defined(__AVX512F__)
#define I1_AVX512 1
#endif

// Per-thread cycle counter (Windows): immune to time-slicing by the other agents sharing the machine.
#ifdef _WIN32
extern "C" {
void* __stdcall GetCurrentThread(void);
int __stdcall QueryThreadCycleTime(void*, unsigned long long*);
}
inline double thread_cycles() { unsigned long long c = 0; QueryThreadCycleTime(GetCurrentThread(), &c); return (double)c; }
#else
inline double thread_cycles() { return (double)__builtin_ia32_rdtsc(); }
#endif
inline double g_pay_cyc_min = 1e300;
// cycles per iteration of a dependent imul+add chain (4 cycles on an idle core): contention calibration
inline double calibrate_chain() {
    volatile uint64_t sink = 0;
    uint64_t a = 1;
    double best = 1e300;
    for (int r = 0; r < 3; ++r) {
        double c0 = thread_cycles();
        for (uint64_t i = 0; i < 20000000ull; ++i) a = a * 6364136223846793005ull + 1442695040888963407ull;
        best = std::min(best, (thread_cycles() - c0) / 2e7);
    }
    sink = a; (void)sink;
    return best;
}

struct Tree {
    // Level 0 holds the exact 64-bit keys (leaf blocks of 16).  Levels >= 1 hold the keys sampled every 16 entries of the
    // level below, truncated to 32 bits (key >> KSH): half the bytes and one 16-lane compare per level.  Truncation can only
    // make the search land one or two blocks too far right (ties); the leaf scan then returns the end of the previous block
    // and the exact CDF check in the decoder walks the last few steps, so the result is always exact.
    static constexpr int KSH = 12;
    std::vector<uint64_t> buf;       // level 0
    std::vector<uint32_t> buf32;     // levels >= 1
    const uint64_t* lv[8] = {nullptr};       // lv[0] = exact leaf keys (S or L), padded with INF
    const uint32_t* lv32[8] = {nullptr};
    int top = 0;
    void build(const uint64_t* V, size_t n) {
        std::vector<std::vector<uint64_t>> L64;
        L64.emplace_back(V, V + n);
        while (L64.back().size() > 16) {
            const auto& p = L64.back();
            std::vector<uint64_t> q;
            for (size_t i = 0; i < p.size(); i += 16) q.push_back(p[i]);
            L64.push_back(std::move(q));
        }
        top = (int)L64.size() - 1;
        buf.assign(((n + 15) / 16) * 16 + 32, INF);
        std::memcpy(buf.data(), V, n * 8);
        lv[0] = buf.data();
        size_t tot = 0;
        for (int k = 1; k <= top; ++k) tot += ((L64[k].size() + 15) / 16) * 16 + 32;
        buf32.assign(tot + 16, 0xFFFFFFFFu);
        size_t off = 0;
        for (int k = 1; k <= top; ++k) {
            uint32_t* d = buf32.data() + off;
            for (size_t i = 0; i < L64[k].size(); ++i) {
                uint64_t v = L64[k][i];
                d[i] = v >= INF ? 0xFFFFFFFFu : (uint32_t)std::min<uint64_t>(v >> KSH, 0xFFFFFFFEull);
            }
            lv32[k] = d;
            off += ((L64[k].size() + 15) / 16) * 16 + 32;
        }
    }
    // index of the largest key <= t (requires key[0] <= t)
    inline uint32_t find(uint64_t t) const {
        uint32_t idx = 0;
#if I1_AVX512
        const uint64_t th = t >> KSH;
        const __m512i tv32 = _mm512_set1_epi32((int)(th >= 0xFFFFFFFFull ? 0xFFFFFFFEu : (uint32_t)th));
        for (int l = top; l >= 1; --l) {
            const uint32_t* q = lv32[l] + (size_t)idx * 16;
            unsigned m = _mm512_cmple_epu32_mask(_mm512_loadu_si512((const void*)q), tv32);
            idx = idx * 16 + (unsigned)__builtin_popcount(m) - 1;
        }
        const __m512i tv = _mm512_set1_epi64((long long)t);
        const uint64_t* q = lv[0] + (size_t)idx * 16;
        unsigned m0 = _mm512_cmple_epu64_mask(_mm512_loadu_si512((const void*)q), tv);
        unsigned m1 = _mm512_cmple_epu64_mask(_mm512_loadu_si512((const void*)(q + 8)), tv);
        idx = idx * 16 + (unsigned)__builtin_popcount(m0 | (m1 << 8)) - 1;
#else
        const uint64_t th = t >> KSH;
        const uint32_t t32 = th >= 0xFFFFFFFFull ? 0xFFFFFFFEu : (uint32_t)th;
        for (int l = top; l >= 1; --l) {
            const uint32_t* q = lv32[l] + (size_t)idx * 16;
            unsigned c = 0;
            for (int i = 0; i < 16; ++i) c += (q[i] <= t32);
            idx = idx * 16 + c - 1;
        }
        const uint64_t* q = lv[0] + (size_t)idx * 16;
        unsigned c = 0;
        for (int i = 0; i < 16; ++i) c += (q[i] <= t);
        idx = idx * 16 + c - 1;
#endif
        return idx;
    }
};

struct FastModel {
    const Model* md = nullptr;
    std::vector<Tree> trees;               // type 0: nb trees over Ld[b]; type 1: one tree over S (leaf level = S)
    std::vector<unsigned __int128> RB;     // type 1: 2^75 / Cb
    std::vector<uint16_t> cum17;           // ncls*cnb rows of 24 u16: [0]=0, [1..16]=cum
    std::vector<uint8_t> clsb;
    const uint64_t* S0() const { return trees[0].lv[0]; }   // type 1: S[0..m], S[m+1] = INF
};

inline std::shared_ptr<FastModel> fast_model(const std::shared_ptr<Model>& md) {
    static std::mutex mu;
    static std::map<const Model*, std::shared_ptr<FastModel>> cache;
    std::lock_guard<std::mutex> g(mu);
    auto it = cache.find(md.get());
    if (it != cache.end()) return it->second;
    auto F = std::make_shared<FastModel>();
    F->md = md.get();
    const uint32_t m = md->m;
    if (md->type == 0) {
        F->trees.resize(md->nb);
        for (uint32_t b = 0; b < md->nb; ++b) F->trees[b].build(md->L(b), m + 2);
    } else {
        std::vector<uint64_t> Sx(md->S.begin(), md->S.end());
        Sx.push_back(INF);                                         // S[m+1]
        F->trees.resize(1);
        F->trees[0].build(Sx.data(), m + 2);
        F->RB.resize(md->nb);
        for (uint32_t b = 0; b < md->nb; ++b) F->RB[b] = ((unsigned __int128)1 << 75) / md->Cb[b];
    }
    const size_t rows = (size_t)md->ncls * md->cnb;
    F->cum17.assign(rows * 24 + 32, 0);
    for (size_t r = 0; r < rows; ++r) {
        F->cum17[r * 24] = 0;
        for (int s = 1; s <= 16; ++s) F->cum17[r * 24 + s] = (uint16_t)cum_at(md->ccum.data() + r * 17, s);
    }
    F->clsb = md->cls;
    cache[md.get()] = F;
    return F;
}

// ------------------------------------------------------------------------------------------------
inline uint32_t ld32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint64_t ld64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// branchless 64-bit renormalisation
static inline void renorm(uint64_t& x, const uint8_t*& p) {
    uint64_t need = x < RL;
    uint64_t nx = (x << 32) | ld32(p);
    x = need ? nx : x;
    p += need << 2;
}

struct V2Hdr {
    uint32_t m, n, nb, nl;
    uint64_t nnz;
    uint8_t vt;
    std::shared_ptr<Model> model;
    std::vector<uint16_t> bf;
    const uint8_t* bwords; uint64_t bn;
    struct LaneH { uint32_t c0, c1; uint64_t nnz0; const uint8_t *g, *c; } lane[MAXL];
    const uint8_t* names;
};

constexpr uint32_t V2_MAGIC = 0x325A5A49u;   // "IZZ2"

inline V2Hdr parse_v2(const uint8_t* p, size_t) {
    V2Hdr h;
    auto g32 = [&]() { uint32_t v = ld32(p); p += 4; return v; };
    auto g64 = [&]() { uint64_t v = ld64(p); p += 8; return v; };
    if (g32() != V2_MAGIC) throw std::runtime_error("idea1 v2: bad magic");
    if (g64() != tabs().sum) throw std::runtime_error("idea1: integer table checksum mismatch");
    uint32_t tl = g32();
    std::string tag((const char*)p, tl); p += tl;
    h.m = g32(); h.n = g32(); h.nnz = g64(); h.vt = *p++; h.nb = g32(); h.nl = g32();
    if (h.nl == 0 || h.nl > (uint32_t)MAXL) throw std::runtime_error("idea1 v2: bad lane count");
    h.model = load_model(tag);
    h.bf.resize(h.nb);
    for (uint32_t i = 0; i < h.nb; ++i) { h.bf[i] = (uint16_t)(p[0] | (p[1] << 8)); p += 2; }
    h.bn = g64(); h.bwords = p; p += h.bn * 4;
    for (uint32_t l = 0; l < h.nl; ++l) {
        h.lane[l].c0 = g32(); h.lane[l].c1 = g32(); h.lane[l].nnz0 = g64();
        uint64_t ng = g64(), nc = g64();
        h.lane[l].g = p; h.lane[l].c = p + ng * 4; p += (ng + nc) * 4;
    }
    h.names = p;
    return h;
}

struct Lane {
    uint64_t xg, xc;
    const uint8_t *pg, *pc;
    uint32_t col, col_end, i1, b;
    uint64_t pos, x0;
    const Tree* tree;
    const uint64_t* Lb;          // type 0: L table of the column's bucket
    uint64_t Cb;                 // type 1
    unsigned __int128 rb;        // type 1
    bool active;
};

// ablation / instrumentation switches for the micro benchmark (never set in production use)
inline uint32_t g_abl = 0;                       // bit0: skip count decoding (all counts = 1)
inline std::vector<uint64_t>* g_rec = nullptr;   // if set, records the tree search targets
inline const uint32_t *g_ixo = nullptr, *g_ipo = nullptr;   // abl bit2: oracle j from a previous decode (replaces the tree search)

// Phased variant: the four lanes advance in lock step through the phases (A: rANS slot -> target on the log axis,
// B: tree search, C: exact CDF verification + state update + count + output) so that the independent searches of
// the lanes (the cache-missing part) are issued back to back.  Identical output, identical stream.
template <int TYPE, int NL>
inline void decode_payload_v2p(const V2Hdr& h, const FastModel& F, uint32_t* ip, uint32_t* ix, uint32_t* dv) {
    const Tabs& T = tabs();
    const Model& md = *h.model;
    const uint32_t m = h.m, n = h.n, nb = h.nb;
    const int PB = md.pos_bits;
    const bool no_count = (g_abl & 1u) != 0;
    const bool cls0 = (g_abl & 2u) != 0;
    const bool oracle = (g_abl & 4u) != 0;
    std::vector<uint8_t> bk(n);
    {
        std::vector<uint8_t> blut(1u << BB);
        std::vector<uint16_t> bcum(nb + 1, 0);
        for (uint32_t b = 0; b < nb; ++b) {
            bcum[b + 1] = (uint16_t)(bcum[b] + h.bf[b]);
            for (uint32_t s = bcum[b]; s < bcum[b + 1]; ++s) blut[s] = (uint8_t)b;
        }
        uint64_t x = ((uint64_t)ld32(h.bwords) << 32) | ld32(h.bwords + 4);
        const uint8_t* p = h.bwords + 8;
        for (uint32_t c = 0; c < n; ++c) {
            uint32_t slot = (uint32_t)(x & ((1u << BB) - 1));
            uint32_t b = blut[slot];
            bk[c] = (uint8_t)b;
            x = (uint64_t)h.bf[b] * (x >> BB) + slot - bcum[b];
            renorm(x, p);
        }
    }
    Lane lane[NL];
    const uint64_t* const Sx = TYPE == 1 ? F.S0() : nullptr;
    auto setup_col = [&](Lane& L) {
        const uint32_t b = bk[L.col];
        L.b = b; L.i1 = 0; L.x0 = 0;
        if (TYPE == 0) { L.tree = &F.trees[b]; L.Lb = md.L(b); }
        else { L.tree = &F.trees[0]; L.Cb = md.Cb[b]; L.rb = F.RB[b]; }
    };
    int nactive = 0;
    for (int l = 0; l < NL; ++l) {
        Lane& L = lane[l];
        const auto& H = h.lane[l];
        L.xg = ((uint64_t)ld32(H.g) << 32) | ld32(H.g + 4); L.pg = H.g + 8;
        L.xc = ((uint64_t)ld32(H.c) << 32) | ld32(H.c + 4); L.pc = H.c + 8;
        L.col = H.c0; L.col_end = H.c1; L.pos = H.nnz0;
        ip[H.c0] = (uint32_t)H.nnz0;
        L.active = H.c0 < H.c1;
        if (L.active) { setup_col(L); ++nactive; }
    }
    const uint16_t* const cum17 = F.cum17.data();
    const uint8_t* const clsb = F.clsb.data();
    const uint32_t cnb = md.cnb;

    while (nactive) {
        uint64_t slotv[NL], tgt[NL];
        uint32_t jv[NL];
        uint8_t st[NL];
I1_UNROLL
        for (int l = 0; l < NL; ++l) {                       // phase A
            Lane& L = lane[l];
            if (!L.active) { st[l] = 0; continue; }
            const uint64_t slot = L.xg & (MSC - 1);
            slotv[l] = slot;
            if (__builtin_expect(slot >= FCUT, 0)) { st[l] = 2; continue; }
            st[l] = 1;
            const uint64_t E = ((uint64_t)31 << FR) - log2fix(T, (uint32_t)(MSC - slot));
            const uint64_t t = L.x0 + E;
            if (TYPE == 0) tgt[l] = t;
            else tgt[l] = (uint64_t)(((unsigned __int128)t * L.rb) >> 40);
        }
I1_UNROLL
        for (int l = 0; l < NL; ++l) {                       // phase B
            if (st[l] != 1) continue;
            if (__builtin_expect(g_rec != nullptr, 0)) g_rec->push_back((tgt[l] << 4) | lane[l].b);
            uint32_t jj;
            if (oracle) jj = (lane[l].pos == g_ipo[lane[l].col + 1]) ? m : g_ixo[lane[l].pos];
            else jj = lane[l].tree->find(tgt[l]);
            jv[l] = jj < lane[l].i1 ? lane[l].i1 : jj;
        }
I1_UNROLL
        for (int l = 0; l < NL; ++l) {                       // phase C
            if (st[l] == 0) continue;
            Lane& L = lane[l];
            uint64_t xg = L.xg;
            const uint64_t slot = slotv[l];
            uint32_t j;
            uint64_t Lj1v;
            if (__builtin_expect(st[l] == 2, 0)) {
                const uint8_t* pg = L.pg;
                xg = (uint64_t)(MSC - FCUT) * (xg >> MB) + (slot - FCUT);
                renorm(xg, pg);
                j = (uint32_t)(xg & ((1u << PB) - 1));
                xg = (xg >> PB);
                renorm(xg, pg);
                L.pg = pg;
                if (TYPE == 0) Lj1v = j < m ? L.Lb[j + 1] : 0;
                else Lj1v = j < m ? (uint64_t)(((unsigned __int128)Sx[j + 1] * L.Cb) >> 35) : 0;
            } else {
                uint32_t jj = jv[l];
                const uint64_t x0 = L.x0;
                uint64_t Lj, Lj1, c0, c1;
                for (;;) {
                    if (TYPE == 0) { Lj = L.Lb[jj]; Lj1 = L.Lb[jj + 1]; }
                    else {
                        Lj = (uint64_t)(((unsigned __int128)Sx[jj] * L.Cb) >> 35);
                        Lj1 = (uint64_t)(((unsigned __int128)Sx[jj + 1] * L.Cb) >> 35);
                    }
                    c0 = MSC - G(T, Lj - x0);
                    c1 = MSC - G(T, Lj1 - x0);
                    if (__builtin_expect(c0 <= slot && slot < c1, 1)) break;
                    if (c0 > slot) --jj; else ++jj;
                }
                j = jj; Lj1v = Lj1;
                xg = (c1 - c0) * (xg >> MB) + slot - c0;
                const uint8_t* pg = L.pg;
                renorm(xg, pg);
                L.pg = pg;
            }
            L.xg = xg;
            if (__builtin_expect(j >= m, 0)) {
                ip[L.col + 1] = (uint32_t)L.pos;
                ++L.col;
                if (L.col == L.col_end) { L.active = false; --nactive; }
                else setup_col(L);
                continue;
            }
            if (no_count) {
                ix[L.pos] = j; dv[L.pos] = 1; ++L.pos;
            } else {
                uint64_t xc = L.xc;
                const uint32_t row = ((cls0 ? 0u : (uint32_t)clsb[j]) * cnb + (cnb > 1 ? L.b : 0)) * 24;
                const uint32_t cs = (uint32_t)(xc & ((1u << CB) - 1));
                const uint32_t hi1 = cum17[row + 1];
                uint32_t s, lo, hi;
                if (cs < hi1) { s = 0; lo = 0; hi = hi1; }          // count == 1: 75-99% of the non-zeros
                else {
#if I1_AVX512
                    __m256i cv = _mm256_loadu_si256((const __m256i*)(cum17 + row + 1));
                    unsigned mk = (unsigned)_mm256_movemask_epi8(_mm256_cmpgt_epi16(cv, _mm256_set1_epi16((short)cs)));
                    s = (uint32_t)__builtin_ctz(mk | 0x80000000u) >> 1;
#else
                    s = 0;
                    while (cum17[row + 1 + s] <= cs) ++s;
#endif
                    lo = cum17[row + s]; hi = cum17[row + s + 1];
                }
                xc = (uint64_t)(hi - lo) * (xc >> CB) + cs - lo;
                const uint8_t* pc = L.pc;
                renorm(xc, pc);
                uint32_t cnt = s + 1;
                if (__builtin_expect(s == 15, 0)) {
                    uint32_t nbv = (uint32_t)(xc & 31); xc >>= 5; renorm(xc, pc);
                    uint32_t low = 0;
                    if (nbv > 16) {
                        uint32_t hi2 = (uint32_t)(xc & ((1u << (nbv - 16)) - 1)); xc >>= (nbv - 16); renorm(xc, pc);
                        uint32_t lo2 = (uint32_t)(xc & 0xFFFF); xc >>= 16; renorm(xc, pc);
                        low = (hi2 << 16) | lo2;
                    } else if (nbv > 0) {
                        low = (uint32_t)(xc & ((1u << nbv) - 1)); xc >>= nbv; renorm(xc, pc);
                    }
                    cnt = 15 + (1u << nbv) + low;
                }
                L.xc = xc; L.pc = pc;
                ix[L.pos] = j; dv[L.pos] = cnt; ++L.pos;
            }
            L.i1 = j + 1;
            L.x0 = Lj1v;
        }
    }
}

inline void decode_payload_v2(const V2Hdr& h, const FastModel& F, uint32_t* ip, uint32_t* ix, uint32_t* dv) {
    const int t = h.model->type;
    switch (h.nl) {
        case 1: t == 0 ? decode_payload_v2p<0, 1>(h, F, ip, ix, dv) : decode_payload_v2p<1, 1>(h, F, ip, ix, dv); break;
        case 2: t == 0 ? decode_payload_v2p<0, 2>(h, F, ip, ix, dv) : decode_payload_v2p<1, 2>(h, F, ip, ix, dv); break;
        case 4: t == 0 ? decode_payload_v2p<0, 4>(h, F, ip, ix, dv) : decode_payload_v2p<1, 4>(h, F, ip, ix, dv); break;
        case 8: t == 0 ? decode_payload_v2p<0, 8>(h, F, ip, ix, dv) : decode_payload_v2p<1, 8>(h, F, ip, ix, dv); break;
        case 16: t == 0 ? decode_payload_v2p<0, 16>(h, F, ip, ix, dv) : decode_payload_v2p<1, 16>(h, F, ip, ix, dv); break;
        default: throw std::runtime_error("idea1 v2: unsupported lane count (use 1, 2, 4, 8 or 16)");
    }
}


// ------------------------------------------------------------------------------------------------
// Payload encoder (everything except the name streams), raw CSC arrays in, v2 stream bytes out.
struct EncStats { double bucket_bits = 0; };
inline void put32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back((uint8_t)(v >> (8 * i))); }
inline void put64(std::vector<uint8_t>& o, uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back((uint8_t)(v >> (8 * i))); }

inline std::vector<uint8_t> encode_v2_payload(const Model& md, uint32_t m, uint32_t n, uint64_t nnz, uint8_t vt,
                                              const uint32_t* indptr, const uint32_t* indices, const uint32_t* data,
                                              int nl, EncStats* stats = nullptr) {
    const Tabs& T = tabs();
    const uint32_t nb = md.nb;
    std::vector<uint8_t> bk(n, 0);
    std::vector<uint32_t> hist(nb, 0);
    for (uint32_t c = 0; c < n; ++c) {
        uint64_t D = 0;
        for (uint32_t k = indptr[c]; k < indptr[c + 1]; ++k) D += data[k];
        bk[c] = (uint8_t)md.bucket_of(D);
        ++hist[bk[c]];
    }
    std::vector<uint16_t> bf(nb, 0), bcum(nb + 1, 0);
    {
        const uint32_t tot = 1u << BB;
        uint32_t acc = 0, best = 0;
        for (uint32_t b = 0; b < nb; ++b) {
            uint32_t f = (uint32_t)std::max<uint64_t>(1, (uint64_t)hist[b] * tot / std::max<uint32_t>(n, 1));
            bf[b] = (uint16_t)f; acc += f;
            if (hist[b] > hist[best]) best = b;
        }
        int64_t fixed = (int64_t)bf[best] + (int64_t)tot - (int64_t)acc;
        if (fixed < 1 || fixed > (int64_t)tot) throw std::runtime_error("bucket freq overflow");
        bf[best] = (uint16_t)fixed;
        for (uint32_t b = 0; b < nb; ++b) bcum[b + 1] = (uint16_t)(bcum[b] + bf[b]);
    }
    // lane partition: contiguous column ranges balanced on nnz + 4 per column
    std::vector<uint32_t> cb(nl + 1, 0);
    {
        double total = (double)nnz + 4.0 * n, acc = 0;
        uint32_t c = 0;
        for (int l = 1; l < nl; ++l) {
            double target = total * l / nl;
            while (c < n && acc < target) { acc += (indptr[c + 1] - indptr[c]) + 4.0; ++c; }
            cb[l] = c;
        }
        cb[nl] = n;
    }
    Enc eb;
    for (int64_t c = (int64_t)n - 1; c >= 0; --c) eb.put({bcum[bk[c]], bf[bk[c]], (uint8_t)BB});
    eb.flush();
    std::reverse(eb.out.begin(), eb.out.end());

    std::vector<uint8_t> o;
    put32(o, V2_MAGIC);
    put64(o, T.sum);
    put32(o, (uint32_t)md.tag.size());
    o.insert(o.end(), md.tag.begin(), md.tag.end());
    put32(o, m); put32(o, n); put64(o, nnz); o.push_back(vt); put32(o, nb); put32(o, (uint32_t)nl);
    for (uint32_t b = 0; b < nb; ++b) { o.push_back(bf[b] & 0xFF); o.push_back(bf[b] >> 8); }
    put64(o, eb.out.size());
    { size_t pre = o.size(); o.resize(pre + eb.out.size() * 4); std::memcpy(o.data() + pre, eb.out.data(), eb.out.size() * 4); }

    std::vector<Sym> gs, cs_;
    for (int l = 0; l < nl; ++l) {
        Enc eg, ec;
        uint64_t nnz_begin = indptr[cb[l]];
        for (int64_t c = (int64_t)cb[l + 1] - 1; c >= (int64_t)cb[l]; --c) {
            gs.clear(); cs_.clear();
            const uint32_t b = bk[c];
            const uint64_t* L = md.L(b);
            uint32_t i1 = 0;
            for (uint32_t k = indptr[c]; k < indptr[c + 1]; ++k) {
                const uint32_t j = indices[k];
                put_gap(gs, T, md, L, i1, j);
                put_count(cs_, md, j, b, data[k]);
                i1 = j + 1;
            }
            put_gap(gs, T, md, L, i1, m);
            for (size_t s2 = gs.size(); s2-- > 0;) eg.put(gs[s2]);
            for (size_t s2 = cs_.size(); s2-- > 0;) ec.put(cs_[s2]);
        }
        eg.flush(); ec.flush();
        std::reverse(eg.out.begin(), eg.out.end());
        std::reverse(ec.out.begin(), ec.out.end());
        put32(o, cb[l]); put32(o, cb[l + 1]); put64(o, nnz_begin);
        put64(o, eg.out.size()); put64(o, ec.out.size());
        size_t pre = o.size();
        o.resize(pre + (eg.out.size() + ec.out.size()) * 4);
        std::memcpy(o.data() + pre, eg.out.data(), eg.out.size() * 4);
        std::memcpy(o.data() + pre + eg.out.size() * 4, ec.out.data(), ec.out.size() * 4);
    }
    if (stats) {
        double sb = 0;
        for (uint32_t b = 0; b < nb; ++b) if (hist[b]) sb += hist[b] * std::log2((double)(1u << BB) / bf[b]);
        stats->bucket_bits = sb;
    }
    return o;
}

}  // namespace i1
}  // namespace cs
