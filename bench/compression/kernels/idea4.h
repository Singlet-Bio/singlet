// SPDX-License-Identifier: MIT
// Idea 4: low-rank manifold prior at gene level, coded as Multinomial(depth, pi) with the survival-gap trick.
//
// Decode path is integer only: pi_c,g ∝ pi0_g * sum_j theta_j * 2^(E_gj/4) (int8 E, 6-bit theta), lambda_g = N * pi_g
// is bucketed into quarter octaves through an integer log2 (Q8), and every probability comes from small calibration
// tables indexed by (depth context, lambda bucket). Presence of genes is coded as the gap to the next present gene
// (survival function of the per-gene absence costs, prefix-summed in Q32 natural-log units and exponentiated with
// an integer table), counts given presence from a per-bucket table with escape, and the interval split of each gene
// with the same machinery over the gene's intervals (static per-interval level table). Empty droplets (N < T) use
// an ambient profile that both sides accumulate from the decoded droplets.
//
// Model file (results/idea4_models/*.i4m, produced by tools/idea4_export.py + harness/idea4_tables.cpp):
//   "I4M3" G m k NB NDEP CMAX flags | gstart[G+1] u32 | pi0[G] u32 Q32 | E[G*k] i8 | cb[256*k] u8 | lvl[m] u8
//   | vocab: u64 hash, u32 raw_len, u32 z_len, bytes | tables (flags&1): 5 x { cabs[NCTX] u64 ; cfreq[NCTX*CMAX] u16 }
#pragma once
#include "common.h"
#include "idea4_core.h"
#include "ref_zstd_csc.h"
#include <cmath>
#include <map>
#include <memory>
#include <unordered_map>

namespace cs { namespace i4 {

static const int NB = 128, NDEP = 8, CMAX = 24, CB = 14, NCTX = NDEP * NB, SB = 30;
enum { TAB_STATIC = 0, TAB_Q6 = 1, TAB_CB = 2, TAB_AMB = 3, TAB_SPLIT = 4, NTAB = 5 };
enum Mode { MODE_STATIC = 0, MODE_Q6 = 1, MODE_CB = 2 };
static const u32 BLOCK_COLS = 1024;

// ---------------------------------------------------------------------------------------------
// integer log2 with 8 fractional bits
// ---------------------------------------------------------------------------------------------
struct Log2Tab {
    u32 t[256];
    Log2Tab() {
        for (int i = 0; i < 256; ++i) {
            // y = 1 + i/256 in Q62; extract 8 fractional bits of log2(y) by squaring
            u128 y = ((u128)(256 + i)) << 54;   // (256+i)/256 * 2^62
            u32 bits = 0;
            for (int b = 0; b < 8; ++b) {
                y = (y * y) >> 62;
                bits <<= 1;
                if (y >= ((u128)1 << 63)) { bits |= 1; y >>= 1; }
            }
            t[i] = bits;
        }
    }
};
inline const Log2Tab& log2tab() { static const Log2Tab t; return t; }
inline int64_t ilog2q8(u64 x) {            // x >= 1
    int e = 63 - __builtin_clzll(x);
    u32 mant = e >= 8 ? (u32)((x >> (e - 8)) & 0xFF) : (u32)((x << (8 - e)) & 0xFF);
    return (int64_t)e * 256 + log2tab().t[mant];
}
inline int64_t ilog2q8_128(u128 x) {
    if ((u64)(x >> 64) == 0) return ilog2q8((u64)x);
    return ilog2q8((u64)(x >> 32)) + 32 * 256;
}
inline int bucket_of(int64_t lb) {         // lb in Q8 log2 units -> quarter-octave bucket from 2^-24
    int64_t v = lb + 24 * 256;
    if (v < 0) return 0;
    int64_t b = v >> 6;
    return b >= NB ? NB - 1 : (int)b;
}
inline int depth_ctx(u64 N) {              // floor(log2 N / 2) - 2 clamped to [0, 7]
    int e = N ? 63 - __builtin_clzll(N) : 0;
    int d = e / 2 - 2;
    return d < 0 ? 0 : (d > 7 ? 7 : d);
}
inline int split_ctx(u64 x) {              // min(7, floor(log2 x))
    int e = x ? 63 - __builtin_clzll(x) : 0;
    return e > 7 ? 7 : e;
}
inline int bitlen(u64 v) { return v ? 64 - __builtin_clzll(v) : 0; }

// ---------------------------------------------------------------------------------------------
// 64-bit range coder (LZMA-style carry handling, arbitrary totals <= 2^31)
// ---------------------------------------------------------------------------------------------
struct RangeEnc {
    std::vector<uint8_t> out;
    u128 low = 0; u64 range = ~0ull; uint8_t cache = 0; u64 cache_size = 1;
    double bits = 0;   // analytic cost accumulator (for section accounting)
    void shift_low() {
        if ((u64)low < 0xFF00000000000000ull || (u64)(low >> 64) != 0) {
            uint8_t carry = (uint8_t)(low >> 64);
            uint8_t temp = cache;
            do { out.push_back((uint8_t)(temp + carry)); temp = 0xFF; } while (--cache_size != 0);
            cache = (uint8_t)(((u64)low) >> 56);
        }
        cache_size++;
        low = ((u128)(((u64)low) & 0x00FFFFFFFFFFFFFFull)) << 8;
    }
    inline void encode(u64 start, u64 freq, u64 total) {
        u64 r = range / total;
        low += (u128)r * start;
        if (start + freq == total) range -= r * start; else range = r * freq;
        while (range < ((u64)1 << 56)) { range <<= 8; shift_low(); }
    }
    void put_bits(u64 v, int nb) {
        while (nb > 0) { int c = nb > 16 ? 16 : nb; encode(v & ((1ull << c) - 1), 1, 1ull << c); v >>= c; nb -= c; }
    }
    std::vector<uint8_t> finish() { for (int i = 0; i < 9; ++i) shift_low(); return std::move(out); }
};
struct RangeDec {
    const uint8_t* p = nullptr; const uint8_t* end = nullptr;
    u64 code = 0, range = ~0ull, r_cur = 0;
    inline uint8_t next() { return p < end ? *p++ : 0; }
    void init(const uint8_t* b, size_t n) { p = b; end = b + n; ++p; code = 0; range = ~0ull; for (int i = 0; i < 8; ++i) code = (code << 8) | next(); }
    inline u64 slot(u64 total) { r_cur = range / total; u64 s = code / r_cur; return s >= total ? total - 1 : s; }
    inline void consume(u64 start, u64 freq, u64 total) {
        code -= r_cur * start;
        if (start + freq == total) range -= r_cur * start; else range = r_cur * freq;
        while (range < ((u64)1 << 56)) { range <<= 8; code = (code << 8) | next(); }
    }
    u64 get_bits(int nb) {
        u64 v = 0; int sh = 0;
        while (nb > 0) { int c = nb > 16 ? 16 : nb; u64 s = slot(1ull << c); consume(s, 1, 1ull << c); v |= s << sh; sh += c; nb -= c; }
        return v;
    }
};

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------
struct TabSet {
    std::vector<u64> cabs;     // NCTX: -ln(1-p) in Q32
    std::vector<u16> cfreq;    // NCTX*CMAX, sums to 2^CB per ctx
    std::vector<u32> ccum;     // NCTX*(CMAX+1)
    void build_cum() {
        ccum.assign((size_t)NCTX * (CMAX + 1), 0);
        for (int c = 0; c < NCTX; ++c) {
            u32 a = 0;
            for (int s = 0; s < CMAX; ++s) { ccum[(size_t)c * (CMAX + 1) + s] = a; a += cfreq[(size_t)c * CMAX + s]; }
            ccum[(size_t)c * (CMAX + 1) + CMAX] = a;
            if (a != (1u << CB)) throw std::runtime_error("idea4: count table total mismatch");
        }
    }
};

struct Model {
    u32 G = 0, m = 0, k = 0, flags = 0;
    std::vector<u32> gstart, pi0;
    std::vector<int8_t> E;
    std::vector<uint8_t> cb, lvl;
    u64 vocab_hash = 0; u32 vocab_raw_len = 0; std::vector<uint8_t> vocab_z;
    TabSet tab[NTAB];
    size_t file_bytes = 0;
    u64 L[128];                       // 2^(e/4) in Q24, e in [-64, 63] -> index e+64
    std::vector<float> Phi_f;         // encoder-side float reconstruction (G*k)
    bool has_tables() const { return flags & 1; }

    void derive() {
        u64 q = root2_q62(4);          // 2^(1/4) in Q62
        u64 pw[65]; pw[0] = (u64)1 << 62;
        for (int i = 1; i <= 64; ++i) pw[i] = (u64)(((u128)pw[i - 1] * q) >> 62);
        for (int e = -64; e < 64; ++e) {
            if (e >= 0) L[e + 64] = (u64)((((u128)1 << 24) * pw[e]) >> 62);
            else L[e + 64] = (u64)((((u128)1 << (24 + 62)) / pw[-e]));
        }
        if (k) {
            Phi_f.resize((size_t)G * k);
            for (u32 g = 0; g < G; ++g)
                for (u32 j = 0; j < k; ++j) Phi_f[(size_t)g * k + j] = (float)((double)pi0[g] / 4294967296.0 * std::ldexp(1.0, 0) * std::pow(2.0, E[(size_t)g * k + j] / 4.0));
        }
        for (int t = 0; t < NTAB; ++t) if (has_tables()) tab[t].build_cum();
    }
};

inline void save_model(const std::string& path, const Model& M) {
    std::vector<uint8_t> o;
    using namespace detail;
    o.push_back('I'); o.push_back('4'); o.push_back('M'); o.push_back('3');
    put_u32(o, M.G); put_u32(o, M.m); put_u32(o, M.k); put_u32(o, NB); put_u32(o, NDEP); put_u32(o, CMAX); put_u32(o, M.flags);
    auto put_arr = [&](const void* p, size_t n) { size_t s = o.size(); o.resize(s + n); std::memcpy(o.data() + s, p, n); };
    put_arr(M.gstart.data(), M.gstart.size() * 4);
    put_arr(M.pi0.data(), M.pi0.size() * 4);
    put_arr(M.E.data(), M.E.size());
    put_arr(M.cb.data(), M.cb.size());
    put_arr(M.lvl.data(), M.lvl.size());
    put_u64(o, M.vocab_hash); put_u32(o, M.vocab_raw_len); put_u32(o, (u32)M.vocab_z.size());
    put_arr(M.vocab_z.data(), M.vocab_z.size());
    if (M.flags & 1)
        for (int t = 0; t < NTAB; ++t) { put_arr(M.tab[t].cabs.data(), M.tab[t].cabs.size() * 8); put_arr(M.tab[t].cfreq.data(), M.tab[t].cfreq.size() * 2); }
    write_file(path, o);
}

inline std::shared_ptr<Model> load_model(const std::string& path) {
    auto b = read_file(path);
    auto M = std::make_shared<Model>();
    M->file_bytes = b.size();
    const uint8_t* p = b.data();
    using namespace detail;
    if (std::memcmp(p, "I4M3", 4) != 0) throw std::runtime_error("idea4: bad model magic");
    p += 4;
    M->G = get_u32(p); M->m = get_u32(p); M->k = get_u32(p);
    u32 nb = get_u32(p), nd = get_u32(p), cm = get_u32(p); M->flags = get_u32(p);
    if (nb != (u32)NB || nd != (u32)NDEP || cm != (u32)CMAX) throw std::runtime_error("idea4: model constants mismatch");
    auto get_arr = [&](void* d, size_t n) { std::memcpy(d, p, n); p += n; };
    M->gstart.resize(M->G + 1); get_arr(M->gstart.data(), (M->G + 1) * 4);
    M->pi0.resize(M->G); get_arr(M->pi0.data(), M->G * 4);
    M->E.resize((size_t)M->G * M->k); get_arr(M->E.data(), M->E.size());
    M->cb.resize(256 * (size_t)M->k); get_arr(M->cb.data(), M->cb.size());
    M->lvl.resize(M->m); get_arr(M->lvl.data(), M->m);
    M->vocab_hash = get_u64(p); M->vocab_raw_len = get_u32(p); u32 zl = get_u32(p);
    M->vocab_z.resize(zl); get_arr(M->vocab_z.data(), zl);
    if (M->flags & 1)
        for (int t = 0; t < NTAB; ++t) {
            M->tab[t].cabs.resize(NCTX); get_arr(M->tab[t].cabs.data(), NCTX * 8);
            M->tab[t].cfreq.resize((size_t)NCTX * CMAX); get_arr(M->tab[t].cfreq.data(), (size_t)NCTX * CMAX * 2);
        }
    M->derive();
    return M;
}

// ---------------------------------------------------------------------------------------------
// Histograms (table training) and table construction
// ---------------------------------------------------------------------------------------------
struct Hist {
    std::vector<u64> all, pres, cnt;   // per tab: NCTX, NCTX, NCTX*CMAX
    Hist() : all((size_t)NTAB * NCTX, 0), pres((size_t)NTAB * NCTX, 0), cnt((size_t)NTAB * NCTX * CMAX, 0) {}
    void absent(int t, int ctx) { all[(size_t)t * NCTX + ctx]++; }
    void present(int t, int ctx) { all[(size_t)t * NCTX + ctx]++; pres[(size_t)t * NCTX + ctx]++; }
    void count(int t, int ctx, u64 x) { int s = x > (u64)CMAX ? CMAX : (int)x; cnt[((size_t)t * NCTX + ctx) * CMAX + s - 1]++; }
    void add(const Hist& o) { for (size_t i = 0; i < all.size(); ++i) { all[i] += o.all[i]; pres[i] += o.pres[i]; } for (size_t i = 0; i < cnt.size(); ++i) cnt[i] += o.cnt[i]; }
};

inline void tables_from_hist(const Hist& H, Model& M) {
    for (int t = 0; t < NTAB; ++t) {
        TabSet& T = M.tab[t];
        T.cabs.assign(NCTX, 0); T.cfreq.assign((size_t)NCTX * CMAX, 0);
        for (int c = 0; c < NCTX; ++c) {
            double all = (double)H.all[(size_t)t * NCTX + c], pres = (double)H.pres[(size_t)t * NCTX + c];
            // presence: shrink toward the bucket's Poisson prediction when data are thin
            double lam = std::pow(2.0, (double)(c % NB) / 4.0 - 24.0);
            double ppois = 1.0 - std::exp(-lam);
            double p = (pres + 4.0 * ppois) / (all + 4.0);
            if (p < 1.0 / 16777216.0) p = 1.0 / 16777216.0;
            if (p > 1.0 - 1.0 / 65536.0) p = 1.0 - 1.0 / 65536.0;
            double cabs = -std::log(1.0 - p);
            T.cabs[c] = (u64)std::llround(cabs * 4294967296.0);
            if (T.cabs[c] == 0) T.cabs[c] = 1;
            // counts: smoothed toward a geometric-ish prior
            double f[CMAX]; double s = 0;
            for (int i = 0; i < CMAX; ++i) { f[i] = (double)H.cnt[((size_t)t * NCTX + c) * CMAX + i] + 0.5 * std::pow(0.5, i); s += f[i]; }
            u32 total = 1u << CB; u32 acc = 0; int big = 0;
            for (int i = 0; i < CMAX; ++i) {
                u32 v = 1 + (u32)std::floor(f[i] / s * (total - CMAX));
                T.cfreq[(size_t)c * CMAX + i] = (u16)v; acc += v;
                if (f[i] > f[big]) big = i;
            }
            T.cfreq[(size_t)c * CMAX + big] += (u16)(total - acc);
        }
    }
    M.flags |= 1;
    for (int t = 0; t < NTAB; ++t) M.tab[t].build_cum();
}

// ---------------------------------------------------------------------------------------------
// Per-cell gene contexts
// ---------------------------------------------------------------------------------------------
struct GeneCtx {
    std::vector<u16> ctx;    // G
    std::vector<u64> S;      // G+1 prefix of cabs
    std::vector<u64> w;      // G scratch
};

// From unnormalised weights w (Z = sum w), depth N: ctx and prefix sums.
inline void contexts_from_w(const Model& M, const TabSet& T, const u64* w, u128 Z, u64 N, int dctx, u16* ctx, u64* S) {
    int64_t lN = ilog2q8(N), lZ = ilog2q8_128(Z);
    u64 acc = 0; S[0] = 0;
    for (u32 g = 0; g < M.G; ++g) {
        int b = w[g] ? bucket_of(lN + ilog2q8(w[g]) - lZ) : 0;
        u16 c = (u16)(dctx * NB + b);
        ctx[g] = c; acc += T.cabs[c]; S[g + 1] = acc;
    }
}
// Manifold weights: w_g = pi0_g * sum_j q_j * L[E_gj] >> 32 ; static (q == nullptr): w_g = pi0_g
inline u128 weights(const Model& M, const uint8_t* q, u64* w) {
    u128 Z = 0;
    if (!q) { for (u32 g = 0; g < M.G; ++g) { w[g] = M.pi0[g]; Z += w[g]; } return Z; }
    const u32 k = M.k;
    for (u32 g = 0; g < M.G; ++g) {
        const int8_t* e = &M.E[(size_t)g * k];
        u64 R = 0;
        for (u32 j = 0; j < k; ++j) R += (u64)q[j] * M.L[e[j] + 64];
        u64 v = (u64)(((u128)M.pi0[g] * R) >> 32);
        w[g] = v; Z += v;
    }
    return Z;
}

// ---------------------------------------------------------------------------------------------
// Survival chain helpers (shared by encoder and decoder)
// ---------------------------------------------------------------------------------------------
// F'(j) for candidates j in (i, n]: 2^SB - exp(-(S[j]-base)) + (j-i-1)
inline u64 chain_F(const u64* S, u64 base, int64_t i, int64_t j) {
    u64 D = S[j] - base;
    return ((u64)1 << SB) - exp_neg_q30(D) + (u64)(j - i - 1);
}
// Decode: find largest j in (i, n) with F'(j) <= slot. F' strictly increasing.
inline int64_t chain_find(const u64* S, u64 base, int64_t i, int64_t n, u64 slot) {
    int64_t lo = i + 1, hi = n - 1;
    while (lo < hi) {
        int64_t mid = (lo + hi + 1) >> 1;
        if (chain_F(S, base, i, mid) <= slot) lo = mid; else hi = mid - 1;
    }
    return lo;
}

struct Section { double bits[12] = {0}; };
enum { SEC_HDR = 0, SEC_THETA, SEC_HPRES, SEC_HCNT, SEC_EPRES, SEC_ECNT, SEC_SPRES, SEC_SCNT, SEC_ESC, SEC_NAMES, SEC_N };
static const char* SEC_NAMES_STR[] = {"col_hdr", "theta", "high_pres", "high_cnt", "emp_pres", "emp_cnt", "split_pres", "split_cnt", "escape", "names"};

// ---------------------------------------------------------------------------------------------
// Ambient state (shared encoder/decoder, deterministic)
// ---------------------------------------------------------------------------------------------
struct Ambient {
    std::vector<u64> amb;            // G: pseudo counts
    std::vector<u64> high;           // G: sum of high-cell gene counts
    std::vector<u64> emp;            // G: sum of empty-droplet gene counts
    u64 n_emp = 0;
    std::unordered_map<u32, std::shared_ptr<GeneCtx>> cache;   // by Nq
    void init(const Model& M) { amb.assign(M.G, 0); high.assign(M.G, 0); emp.assign(M.G, 0); n_emp = 0; cache.clear(); }
    void refresh(const Model& M) {
        // amb = round(pi0 * 2^14) + high/8 + emp  (pseudo-UMIs)
        for (u32 g = 0; g < M.G; ++g) amb[g] = ((u64)M.pi0[g] * 16384 + (1ull << 31)) >> 32 | 0 ;
        for (u32 g = 0; g < M.G; ++g) { if (!amb[g]) amb[g] = 1; amb[g] += (high[g] >> 3) + emp[g]; }
        cache.clear();
    }
    const GeneCtx& get(const Model& M, const TabSet& T, u64 N) {
        u32 Nq = (u32)(ilog2q8(N) >> 6);     // quarter-octave quantised depth
        auto it = cache.find(Nq);
        if (it != cache.end()) return *it->second;
        auto gc = std::make_shared<GeneCtx>();
        gc->ctx.resize(M.G); gc->S.resize(M.G + 1);
        u128 Z = 0; for (u32 g = 0; g < M.G; ++g) Z += amb[g];
        u64 Nrep = 1ull << 20;                 // represent quantised N as 2^(Nq/4): lN = Nq*64 exactly
        (void)Nrep;
        // contexts with lN = Nq*64
        int64_t lN = (int64_t)Nq * 64, lZ = ilog2q8_128(Z);
        int d = depth_ctx(N);
        u64 acc = 0; gc->S[0] = 0;
        for (u32 g = 0; g < M.G; ++g) {
            int b = bucket_of(lN + ilog2q8(amb[g]) - lZ);
            u16 c = (u16)(d * NB + b);
            gc->ctx[g] = c; acc += T.cabs[c]; gc->S[g + 1] = acc;
        }
        cache[Nq] = gc;
        return *gc;
    }
};

// ---------------------------------------------------------------------------------------------
// Codec state shared by encoder and decoder (adaptive models)
// ---------------------------------------------------------------------------------------------
struct Adapt {
    AdaptModel depth_len, esc_len[NTAB];
    void init() { depth_len.init(41); for (auto& a : esc_len) a.init(41); }
};

// Split contexts for one gene with total x over intervals [a, b)
inline void split_contexts(const Model& M, const TabSet& T, u32 a, u32 b, u64 x, u16* ctx, u64* S) {
    int64_t lx = ilog2q8(x); int d = split_ctx(x);
    u64 acc = 0; S[0] = 0;
    for (u32 t = a; t < b; ++t) {
        int bk = bucket_of(lx - (int64_t)M.lvl[t] * 64);
        u16 c = (u16)(d * NB + bk);
        ctx[t - a] = c; acc += T.cabs[c]; S[t - a + 1] = acc;
    }
}

// ---------------------------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------------------------
struct Params { Mode mode = MODE_STATIC; u64 T = 100; int em_iters = 30; };

struct Encoder {
    const Model& M; Params P; Hist* hist = nullptr;   // hist != nullptr: training mode (no output)
    Section sec;
    Ambient amb;
    Adapt ad;
    RangeEnc* rc = nullptr;
    GeneCtx gc;
    std::vector<u64> gene_cnt;        // G scratch (dense)
    std::vector<std::pair<u32, u64>> genes;   // (g, x) present genes
    std::vector<u16> sctx; std::vector<u64> sS;
    std::vector<float> theta;
    Encoder(const Model& m, Params p) : M(m), P(p) {
        gc.ctx.resize(M.G); gc.S.resize(M.G + 1); gc.w.resize(M.G);
        gene_cnt.assign(M.G, 0); sctx.resize(512); sS.resize(513); amb.init(M); ad.init();
    }
    int tab_high() const { return P.mode == MODE_STATIC ? TAB_STATIC : (P.mode == MODE_Q6 ? TAB_Q6 : TAB_CB); }

    inline void put(int sec_id, u64 start, u64 freq, u64 total) {
        sec.bits[sec_id] += std::log2((double)total / (double)freq);
        if (rc) rc->encode(start, freq, total);
    }
    void put_gamma(int sec_id, AdaptModel& am, u64 v) {   // v >= 0: code bitlen(v+1) then the low bits
        int nb = bitlen(v + 1);
        u32 st, fr; am.get(nb, st, fr); put(sec_id, st, fr, 1u << AdaptModel::SB); am.update(nb);
        if (nb > 1) { sec.bits[sec_id] += nb - 1; if (rc) rc->put_bits((v + 1) & ((1ull << (nb - 1)) - 1), nb - 1); }
    }
    void put_count(int sec_id, int tab, int ctx, u64 x, u64 rem) {
        if (rem <= 1) return;                      // x == rem == 1 implicit
        if (hist) { hist->count(tab, ctx, x); return; }
        const TabSet& T = M.tab[tab];
        const u32* cum = &T.ccum[(size_t)ctx * (CMAX + 1)];
        if (rem < (u64)CMAX) {                     // alphabet 1..rem, no escape
            u64 total = cum[rem];
            put(sec_id, cum[x - 1], cum[x] - cum[x - 1], total);
        } else {
            u64 total = cum[CMAX];
            if (x < (u64)CMAX) put(sec_id, cum[x - 1], cum[x] - cum[x - 1], total);
            else { put(sec_id, cum[CMAX - 1], cum[CMAX] - cum[CMAX - 1], total); put_gamma(SEC_ESC, ad.esc_len[tab], x - CMAX); }
        }
    }
    // survival chain over present items idx[] (ascending) in [0, n) with prefix S; items carry counts cnt[]
    template <class CountFn>
    void chain(int sec_pres, int tab, const u16* ctx, const u64* S, int64_t n, const u32* idx, const u64* cnt, size_t np, u64 Ntot, CountFn cf) {
        int64_t i = -1; u64 rem = Ntot;
        for (size_t t = 0; t < np; ++t) {
            int64_t j = idx[t];
            if (hist) { for (int64_t g = i + 1; g < j; ++g) hist->absent(tab, ctx[g]); hist->present(tab, ctx[j]); }
            else {
                u64 base = S[i + 1];
                u64 Fj = chain_F(S, base, i, j), Fj1 = chain_F(S, base, i, j + 1), Mtot = chain_F(S, base, i, n);
                put(sec_pres, Fj, Fj1 - Fj, Mtot);
            }
            cf(ctx[j], cnt[t], rem, j == n - 1);
            rem -= cnt[t]; i = j;
            if (rem == 0) break;
        }
        if (rem != 0) throw std::runtime_error("idea4: chain did not consume the column total");
    }

    void fit_theta(uint8_t* q) {
        const u32 k = M.k;
        theta.assign(k, 1.0f / k);
        std::vector<float> tn(k);
        for (int it = 0; it < P.em_iters; ++it) {
            std::fill(tn.begin(), tn.end(), 0.f);
            for (auto& [g, x] : genes) {
                const float* ph = &M.Phi_f[(size_t)g * k];
                float p = 0.f; for (u32 j = 0; j < k; ++j) p += theta[j] * ph[j];
                if (p < 1e-30f) p = 1e-30f;
                float r = (float)x / p;
                for (u32 j = 0; j < k; ++j) tn[j] += r * ph[j];
            }
            float s = 0.f; for (u32 j = 0; j < k; ++j) { theta[j] *= tn[j]; s += theta[j]; }
            if (s <= 0.f) { theta.assign(k, 1.0f / k); break; }
            for (u32 j = 0; j < k; ++j) theta[j] /= s;
        }
        if (P.mode == MODE_Q6) {
            u32 best = 0; int sum = 0;
            for (u32 j = 0; j < k; ++j) { int v = (int)std::lround(theta[j] * 63.f); if (v < 0) v = 0; if (v > 63) v = 63; q[j] = (uint8_t)v; sum += v; if (theta[j] > theta[best]) best = j; }
            if (sum == 0) q[best] = 1;
        } else {   // codebook: maximise the multinomial log-likelihood over all 256 codewords
            double bestll = -1e300; int bi = 0;
            std::vector<float> cw(k);
            for (int c = 0; c < 256; ++c) {
                float s = 0.f; for (u32 j = 0; j < k; ++j) { cw[j] = M.cb[(size_t)c * k + j]; s += cw[j]; }
                if (s <= 0.f) continue;
                double ll = 0;
                for (auto& [g, x] : genes) {
                    const float* ph = &M.Phi_f[(size_t)g * k];
                    float p = 0.f; for (u32 j = 0; j < k; ++j) p += cw[j] * ph[j];
                    ll += (double)x * std::log((double)(p > 1e-30f ? p : 1e-30f));
                }
                if (ll > bestll) { bestll = ll; bi = c; }
            }
            q[0] = (uint8_t)bi;
        }
    }

    void code_column(const CanonicalMatrix& X, u32 col) {
        // gene aggregation
        genes.clear();
        u64 N = 0;
        for (u32 t = X.indptr[col]; t < X.indptr[col + 1]; ++t) {
            u32 r = X.indices[t]; u64 x = X.data[t]; N += x;
            // gene of r: binary search in gstart
            u32 g = (u32)(std::upper_bound(M.gstart.begin(), M.gstart.end(), r) - M.gstart.begin()) - 1;
            if (!genes.empty() && genes.back().first == g) genes.back().second += x; else genes.emplace_back(g, x);
        }
        put_gamma(SEC_HDR, ad.depth_len, N);
        if (N == 0) return;
        bool high = N >= P.T;
        int tab; const u16* ctx; const u64* S;
        uint8_t q[64] = {0};
        if (high) {
            tab = tab_high();
            if (P.mode != MODE_STATIC) {
                fit_theta(q);
                if (P.mode == MODE_Q6) { for (u32 j = 0; j < M.k; ++j) { sec.bits[SEC_THETA] += 6; if (rc) rc->put_bits(q[j], 6); } }
                else { sec.bits[SEC_THETA] += 8; if (rc) rc->put_bits(q[0], 8); }
                const uint8_t* qq = P.mode == MODE_CB ? &M.cb[(size_t)q[0] * M.k] : q;
                u128 Z = weights(M, qq, gc.w.data());
                contexts_from_w(M, M.tab[tab], gc.w.data(), Z, N, depth_ctx(N), gc.ctx.data(), gc.S.data());
            } else {
                u128 Z = weights(M, nullptr, gc.w.data());
                contexts_from_w(M, M.tab[tab], gc.w.data(), Z, N, depth_ctx(N), gc.ctx.data(), gc.S.data());
            }
            ctx = gc.ctx.data(); S = gc.S.data();
        } else {
            tab = TAB_AMB;
            if (amb.n_emp == 0) amb.refresh(M);
            const GeneCtx& a = amb.get(M, M.tab[tab], N);
            ctx = a.ctx.data(); S = a.S.data();
        }
        std::vector<u32> gi(genes.size()); std::vector<u64> gx(genes.size());
        for (size_t t = 0; t < genes.size(); ++t) { gi[t] = genes[t].first; gx[t] = genes[t].second; }
        int sp = high ? SEC_HPRES : SEC_EPRES, sc = high ? SEC_HCNT : SEC_ECNT;
        chain(sp, tab, ctx, S, M.G, gi.data(), gx.data(), gi.size(), N,
              [&](u16 c, u64 x, u64 rem, bool) { put_count(sc, tab, c, x, rem); });
        // interval split per gene
        u32 t0 = X.indptr[col];
        for (size_t t = 0; t < genes.size(); ++t) {
            u32 g = gi[t]; u64 x = gx[t];
            u32 a = M.gstart[g], b = M.gstart[g + 1];
            u32 I = b - a;
            // collect this gene's intervals from the column
            std::vector<u32> ii; std::vector<u64> ix;
            while (t0 < X.indptr[col + 1] && X.indices[t0] < b) { ii.push_back(X.indices[t0] - a); ix.push_back(X.data[t0]); ++t0; }
            if (I == 1) continue;
            if (sctx.size() < I) { sctx.resize(I); sS.resize(I + 1); }
            split_contexts(M, M.tab[TAB_SPLIT], a, b, x, sctx.data(), sS.data());
            chain(SEC_SPRES, TAB_SPLIT, sctx.data(), sS.data(), I, ii.data(), ix.data(), ii.size(), x,
                  [&](u16 c, u64 xx, u64 rem, bool last) { if (!last) put_count(SEC_SCNT, TAB_SPLIT, c, xx, rem); });
        }
        if (high) { for (auto& [g, x] : genes) amb.high[g] += x; }
        else {
            for (auto& [g, x] : genes) amb.emp[g] += x;
            if (++amb.n_emp % 1024 == 0) amb.refresh(M);
        }
    }
};

// ---------------------------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------------------------
struct Decoder {
    const Model& M; Params P;
    Ambient amb; Adapt ad; RangeDec* rd = nullptr; GeneCtx gc;
    std::vector<u16> sctx; std::vector<u64> sS;
    std::vector<std::pair<u32, u64>> genes;
    double ms_matvec = 0;
    Decoder(const Model& m, Params p) : M(m), P(p) { gc.ctx.resize(M.G); gc.S.resize(M.G + 1); gc.w.resize(M.G); sctx.resize(512); sS.resize(513); amb.init(M); ad.init(); }
    int tab_high() const { return P.mode == MODE_STATIC ? TAB_STATIC : (P.mode == MODE_Q6 ? TAB_Q6 : TAB_CB); }

    u64 get_gamma(AdaptModel& am) {
        u32 st, fr; u64 s = rd->slot(1u << AdaptModel::SB);
        int nb = am.find((u32)s, st, fr); rd->consume(st, fr, 1u << AdaptModel::SB); am.update(nb);
        u64 v = 1;
        if (nb > 1) v = (1ull << (nb - 1)) | rd->get_bits(nb - 1);
        return v - 1;
    }
    u64 get_count(int tab, int ctx, u64 rem) {
        if (rem <= 1) return 1;
        const TabSet& T = M.tab[tab];
        const u32* cum = &T.ccum[(size_t)ctx * (CMAX + 1)];
        u64 total = rem < (u64)CMAX ? cum[rem] : cum[CMAX];
        int hi = rem < (u64)CMAX ? (int)rem : CMAX;
        u64 s = rd->slot(total);
        int lo = 1, h = hi;      // find symbol x in [1, hi] with cum[x-1] <= s < cum[x]
        while (lo < h) { int mid = (lo + h + 1) >> 1; if (cum[mid - 1] <= s) lo = mid; else h = mid - 1; }
        rd->consume(cum[lo - 1], cum[lo] - cum[lo - 1], total);
        if (rem < (u64)CMAX || lo < CMAX) return (u64)lo;
        return (u64)CMAX + get_gamma(ad.esc_len[tab]);
    }
    template <class CountFn>
    void chain(int tab, const u16* ctx, const u64* S, int64_t n, u64 Ntot, CountFn cf) {
        int64_t i = -1; u64 rem = Ntot;
        while (rem > 0) {
            u64 base = S[i + 1];
            u64 Mtot = chain_F(S, base, i, n);
            u64 s = rd->slot(Mtot);
            int64_t j = chain_find(S, base, i, n, s);
            u64 Fj = chain_F(S, base, i, j), Fj1 = chain_F(S, base, i, j + 1);
            rd->consume(Fj, Fj1 - Fj, Mtot);
            u64 x = cf(j, ctx[j], rem, j == n - 1);
            rem -= x; i = j;
        }
    }

    void decode_column(CanonicalMatrix& X, u32 col) {
        u64 N = get_gamma(ad.depth_len);
        if (N == 0) return;
        bool high = N >= P.T;
        int tab; const u16* ctx; const u64* S;
        if (high) {
            tab = tab_high();
            Timer tm;
            if (P.mode != MODE_STATIC) {
                uint8_t q[64]; const uint8_t* qq = q;
                if (P.mode == MODE_Q6) { for (u32 j = 0; j < M.k; ++j) q[j] = (uint8_t)rd->get_bits(6); }
                else { int ci = (int)rd->get_bits(8); qq = &M.cb[(size_t)ci * M.k]; }
                u128 Z = weights(M, qq, gc.w.data());
                contexts_from_w(M, M.tab[tab], gc.w.data(), Z, N, depth_ctx(N), gc.ctx.data(), gc.S.data());
            } else {
                u128 Z = weights(M, nullptr, gc.w.data());
                contexts_from_w(M, M.tab[tab], gc.w.data(), Z, N, depth_ctx(N), gc.ctx.data(), gc.S.data());
            }
            ms_matvec += tm.ms();
            ctx = gc.ctx.data(); S = gc.S.data();
        } else {
            tab = TAB_AMB;
            if (amb.n_emp == 0) amb.refresh(M);
            const GeneCtx& a = amb.get(M, M.tab[tab], N);
            ctx = a.ctx.data(); S = a.S.data();
        }
        genes.clear();
        chain(tab, ctx, S, M.G, N, [&](int64_t j, u16 c, u64 rem, bool) { u64 x = get_count(tab, c, rem); genes.emplace_back((u32)j, x); return x; });
        for (auto& [g, x] : genes) {
            u32 a = M.gstart[g], b = M.gstart[g + 1]; u32 I = b - a;
            if (I == 1) { X.indices.push_back(a); X.data.push_back((u32)x); continue; }
            if (sctx.size() < I) { sctx.resize(I); sS.resize(I + 1); }
            split_contexts(M, M.tab[TAB_SPLIT], a, b, x, sctx.data(), sS.data());
            chain(TAB_SPLIT, sctx.data(), sS.data(), I, x, [&](int64_t j, u16 c, u64 rem, bool last) {
                u64 xx = last ? rem : get_count(TAB_SPLIT, c, rem);
                X.indices.push_back(a + (u32)j); X.data.push_back((u32)xx); return xx; });
        }
        if (high) { for (auto& [g, x] : genes) amb.high[g] += x; }
        else { for (auto& [g, x] : genes) amb.emp[g] += x; if (++amb.n_emp % 1024 == 0) amb.refresh(M); }
    }
};

// ---------------------------------------------------------------------------------------------
// Names
// ---------------------------------------------------------------------------------------------
inline bool pack_barcodes(const std::vector<std::string>& v, std::vector<uint8_t>& out) {
    out.clear(); out.reserve(v.size() * 4);
    for (auto& s : v) {
        if (s.size() != 16) return false;
        u32 code = 0;
        for (char c : s) { int b = c == 'A' ? 0 : c == 'C' ? 1 : c == 'G' ? 2 : c == 'T' ? 3 : -1; if (b < 0) return false; code = (code << 2) | b; }
        detail::put_u32(out, code);
    }
    return true;
}
inline std::vector<std::string> unpack_barcodes(const std::vector<uint8_t>& b, size_t n) {
    std::vector<std::string> v(n);
    const uint8_t* p = b.data();
    for (size_t i = 0; i < n; ++i) {
        u32 code = detail::get_u32(p); std::string s(16, 'A');
        for (int j = 15; j >= 0; --j) { s[j] = "ACGT"[code & 3]; code >>= 2; }
        v[i] = s;
    }
    return v;
}

// ---------------------------------------------------------------------------------------------
// Codec: container format
//   "I4C1" | u32 m n | u64 nnz | u8 vt | u8 mode | u64 T | u8 name_flags | [names streams] | u32 nblocks | blocks: u32 len + bytes
// ---------------------------------------------------------------------------------------------
struct EncodeResult { std::vector<uint8_t> bytes; Section sec; };

inline EncodeResult encode_matrix(const Model& M, Params P, const CanonicalMatrix& X) {
    using namespace detail;
    if (X.m != M.m) throw std::runtime_error("idea4: matrix rows != model intervals");
    std::vector<uint8_t> o;
    o.push_back('I'); o.push_back('4'); o.push_back('C'); o.push_back('1');
    put_u32(o, X.m); put_u32(o, X.n); put_u64(o, X.nnz); o.push_back(X.vt_code); o.push_back((uint8_t)P.mode); put_u64(o, P.T);
    // names
    auto rn = join_names(X.rownames);
    u64 h = fnv64(rn.data(), rn.size());
    uint8_t nf = 0;
    std::vector<uint8_t> packed;
    bool bc = pack_barcodes(X.colnames, packed);
    if (h == M.vocab_hash && rn.size() == M.vocab_raw_len) nf |= 1;
    if (bc) nf |= 2;
    o.push_back(nf);
    size_t names0 = o.size();
    if (!(nf & 1)) put_stream(o, rn.data(), rn.size(), 19);
    if (nf & 2) put_stream(o, packed.data(), packed.size(), 9);
    else { auto cn = join_names(X.colnames); put_stream(o, cn.data(), cn.size(), 19); }
    std::vector<uint8_t> kv;
    for (auto& [k, v] : X.user_kv) { kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0); kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0); }
    put_u32(o, (u32)X.user_kv.size()); put_stream(o, kv.data(), kv.size(), 9);
    Section sec; sec.bits[SEC_NAMES] = 8.0 * (o.size() - names0);
    // blocks
    Encoder enc(M, P);
    u32 nblocks = (X.n + BLOCK_COLS - 1) / BLOCK_COLS;
    put_u32(o, nblocks);
    for (u32 b = 0; b < nblocks; ++b) {
        RangeEnc rc; enc.rc = &rc;
        u32 c0 = b * BLOCK_COLS, c1 = std::min(X.n, c0 + BLOCK_COLS);
        for (u32 c = c0; c < c1; ++c) enc.code_column(X, c);
        auto bytes = rc.finish();
        put_u32(o, (u32)bytes.size()); o.insert(o.end(), bytes.begin(), bytes.end());
    }
    for (int i = 0; i < SEC_NAMES; ++i) sec.bits[i] = enc.sec.bits[i];
    return {std::move(o), sec};
}

inline CanonicalMatrix decode_matrix(const Model& M, const std::vector<uint8_t>& b, double* ms_matvec = nullptr) {
    using namespace detail;
    const uint8_t* p = b.data();
    if (std::memcmp(p, "I4C1", 4) != 0) throw std::runtime_error("idea4: bad container");
    p += 4;
    CanonicalMatrix X;
    X.m = get_u32(p); X.n = get_u32(p); X.nnz = get_u64(p); X.vt_code = *p++;
    Params P; P.mode = (Mode)*p++; P.T = get_u64(p);
    uint8_t nf = *p++;
    if (nf & 1) { auto raw = zstd_d(M.vocab_z.data(), M.vocab_z.size(), M.vocab_raw_len); X.rownames = split_names(raw, X.m); }
    else X.rownames = split_names(get_stream(p), X.m);
    if (nf & 2) X.colnames = unpack_barcodes(get_stream(p), X.n);
    else X.colnames = split_names(get_stream(p), X.n);
    u32 nkv = get_u32(p); auto kv = get_stream(p); auto parts = split_names(kv, 2 * (size_t)nkv);
    for (u32 i = 0; i < nkv; ++i) X.user_kv[parts[2 * i]] = parts[2 * i + 1];
    X.indptr.resize(X.n + 1); X.indices.reserve(X.nnz); X.data.reserve(X.nnz);
    Decoder dec(M, P);
    u32 nblocks = get_u32(p);
    u32 col = 0;
    for (u32 bi = 0; bi < nblocks; ++bi) {
        u32 len = get_u32(p);
        RangeDec rd; rd.init(p, len); dec.rd = &rd; p += len;
        u32 c1 = std::min(X.n, col + BLOCK_COLS);
        for (; col < c1; ++col) { X.indptr[col] = (u32)X.indices.size(); dec.decode_column(X, col); }
    }
    X.indptr[X.n] = (u32)X.indices.size();
    if (ms_matvec) *ms_matvec = dec.ms_matvec;
    return X;
}

// ---------------------------------------------------------------------------------------------
// Codec factory. Section accounting is appended to `sections_path` (if set) keyed by g_file.
// ---------------------------------------------------------------------------------------------
inline std::string& g_file() { static std::string s; return s; }
inline std::string& g_sections_path() { static std::string s; return s; }
inline double& g_last_matvec_ms() { static double v = 0; return v; }

inline Codec make_idea4(std::shared_ptr<Model> M, Params P, const std::string& name) {
    Codec c;
    c.name = name;
    c.model_bytes = M->file_bytes;
    c.encode = [M, P, name](const CanonicalMatrix& X) {
        auto r = encode_matrix(*M, P, X);
        if (!g_sections_path().empty()) {
            bool nf = !std::ifstream(g_sections_path()).good();
            std::ofstream f(g_sections_path(), std::ios::app);
            if (nf) { f << "sample\tcodec\tnnz\tn\tbytes"; for (int i = 0; i < SEC_NAMES + 1; ++i) f << '\t' << SEC_NAMES_STR[i]; f << "\tmodel_bytes\tk\tT\n"; }
            f << g_file() << '\t' << name << '\t' << X.nnz << '\t' << X.n << '\t' << r.bytes.size();
            for (int i = 0; i < SEC_NAMES + 1; ++i) f << '\t' << std::llround(r.sec.bits[i] / 8.0);
            f << '\t' << M->file_bytes << '\t' << M->k << '\t' << P.T << '\n';
        }
        return std::move(r.bytes);
    };
    c.decode = [M](const std::vector<uint8_t>& b) { return decode_matrix(*M, b, &g_last_matvec_ms()); };
    return c;
}

}}  // namespace cs::i4
