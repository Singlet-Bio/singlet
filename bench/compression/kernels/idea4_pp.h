// SPDX-License-Identifier: MIT
// Idea 4 ("pp" = Poisson-process variant): generative-prior codec. Counts of a column are modelled as a
// Poisson process over the interval axis whose rate is  lambda_j = Nhat * pi_g(j) * q_j  (g(j) = gene of
// interval j, q = static split of a gene's mass over its exon intervals).  pi_g is either
//   K = 0 : a static profile pi0 (the "per-gene x depth" baseline), or
//   K > 0 : pi_g ~ pi0_g * sum_k theta_k u_gk (low-rank manifold prior, theta sent per cell).
// Presence gaps use the survival function of the Poisson process (O(log m) per non-zero), counts
// given presence use small empirical tables bucketed by lambda. Empty droplets use an ambient
// profile built by the decoder itself from the already decoded called cells (+ decoded empties).
// The coding distribution is computed with integer arithmetic only (see idea4_pp_core.h).
// (Namespace cs::ipp, files idea4_pp*: kept apart from the second Idea-4 implementation in kernels/idea4.h.)
#pragma once
#include "common.h"
#include "idea4_pp_core.h"
#include <zstd.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>

namespace cs { namespace ipp {

static const int NSYM = 16;            // count symbols: c-1 in 0..14, 15 = escape
static const int NCB = 40;             // count buckets (half octaves of lambda)
static const int CSB = 14;             // count table scale bits
static const int TB = 28;              // gap symbol scale bits
static const u64 TT = (u64)1 << TB;
static const int NLEV = 16;            // theta level alphabet
static const u32 LEVEL_TQ[NLEV] = {0, 2, 3, 4, 6, 8, 11, 16, 23, 32, 45, 64, 91, 128, 181, 256};
static const int NOCT = 20;            // depth-scale octaves
static const int SCALE_R_CELL = 8, SCALE_R_AMB = 4;   // scale classes per octave

// ---------------------------------------------------------------------------------------------
// Statistics (encoder side only, information-content accounting in bits)
// ---------------------------------------------------------------------------------------------
struct Stats {
    double bits_flag = 0, bits_scale = 0, bits_theta = 0, bits_gap_cell = 0, bits_cnt_cell = 0,
           bits_gap_amb = 0, bits_cnt_amb = 0, bits_end = 0;
    size_t n_cell = 0, n_amb = 0, nnz_cell = 0, nnz_amb = 0, names_bytes = 0, hdr_bytes = 0, rans_bytes = 0;
    double enc_fit_ms = 0, dec_matvec_ms = 0, dec_total_ms = 0; size_t dec_cells = 0;
    double sc_cell = 0, sc_amb = 0;
    double dec_share_gapc = 0, dec_share_ambr = 0, dec_share_gapa = 0;   // wall-clock shares of the decode loop
    double ideal_bits() const { return bits_flag + sc_cell + sc_amb + bits_theta + bits_gap_cell + bits_cnt_cell + bits_gap_amb + bits_cnt_amb; }
    std::string str() const {
        char b[1024];
        std::snprintf(b, sizeof b,
            "cells=%zu amb=%zu nnz_cell=%zu nnz_amb=%zu | bits/nnz cell: gap %.3f cnt %.3f side %.3f | amb: gap %.3f cnt %.3f scale %.3f | "
            "flag_b/col %.3f end_b %.0f | names_B %zu rans_B %zu ideal_B %.0f coder_overhead %.4f%%",
            n_cell, n_amb, nnz_cell, nnz_amb,
            nnz_cell ? bits_gap_cell / nnz_cell : 0, nnz_cell ? bits_cnt_cell / nnz_cell : 0,
            nnz_cell ? (bits_theta + sc_cell) / nnz_cell : 0,
            nnz_amb ? bits_gap_amb / nnz_amb : 0, nnz_amb ? bits_cnt_amb / nnz_amb : 0,
            nnz_amb ? sc_amb / nnz_amb : 0, (n_cell + n_amb) ? bits_flag / (n_cell + n_amb) : 0, bits_end,
            names_bytes, rans_bytes, ideal_bits() / 8.0, ideal_bits() > 0 ? 100.0 * ((double)rans_bytes * 8.0 / ideal_bits() - 1.0) : 0.0);
        return b;
    }
};
inline Stats& stats() { static Stats s; return s; }

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------
template <class T> inline void rd_vec(const uint8_t*& p, const uint8_t* end, std::vector<T>& v, size_t n) {
    if (p + n * sizeof(T) > end) throw std::runtime_error("idea4pp model: truncated");
    v.resize(n); std::memcpy(v.data(), p, n * sizeof(T)); p += n * sizeof(T);
}

struct Model {
    u32 m = 0, G = 0, K = 0, flags = 0;
    std::vector<u32> gstart;
    std::vector<uint16_t> pi0code;
    std::vector<uint8_t> qcode;
    std::vector<uint8_t> Rcode;       // K*G topic-major
    std::vector<uint16_t> Rsq;        // K*G, Rcode^2 (derived)
    std::vector<u32> tscale;          // K, quadratic companding scale: u = tscale * c^2 / 2^16
    std::vector<uint16_t> ctab[2];    // cumulative count tables [mode][NCB*(NSYM+1)]
    std::vector<uint16_t> thprior;    // K*NLEV counts
    std::vector<uint8_t> names_blob;  // zstd of '\n'-joined rownames
    size_t file_bytes = 0;
    // derived
    std::vector<u32> gene_of, qcum;
    std::vector<u64> pi0q;            // G, Q62
    std::vector<u64> P0;              // m, static interval profile (Q62, sums to ~2^62)
    std::vector<float> phi;           // encoder only: G*K float topic profiles (normalised)
    std::vector<std::string> rownames;

    void load(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f) throw std::runtime_error("idea4pp: cannot open model " + path);
        f.seekg(0, std::ios::end); size_t sz = (size_t)f.tellg(); f.seekg(0);
        std::vector<uint8_t> buf(sz); f.read((char*)buf.data(), (std::streamsize)sz);
        file_bytes = sz;
        const uint8_t* p = buf.data(); const uint8_t* end = p + sz;
        u32 hdr[8]; std::memcpy(hdr, p, 32); p += 32;
        if (hdr[0] != 0x314D3449u) throw std::runtime_error("idea4pp: bad model magic");
        m = hdr[2]; G = hdr[3]; K = hdr[4]; flags = hdr[5];
        rd_vec(p, end, gstart, G + 1);
        rd_vec(p, end, pi0code, G);
        rd_vec(p, end, qcode, m);
        if (K) { rd_vec(p, end, Rcode, (size_t)K * G); rd_vec(p, end, tscale, K); rd_vec(p, end, thprior, (size_t)K * NLEV); }
        rd_vec(p, end, ctab[0], (size_t)NCB * (NSYM + 1));
        rd_vec(p, end, ctab[1], (size_t)NCB * (NSYM + 1));
        if (flags & 1) { u32 nb; std::memcpy(&nb, p, 4); p += 4; rd_vec(p, end, names_blob, nb); }
        derive();
    }

    static u64 pow2neg_q30(int c16) {          // 2^(-c16/16) in Q30 via the integer root chain
        static u64 tab[16]; static bool init = false;
        if (!init) {
            u64 r = root2_q62(16);              // 2^(1/16) Q62
            u64 x = (u64)1 << 62;
            for (int i = 0; i < 16; ++i) { tab[i] = x; x = (u64)(((u128)x * r) >> 62); }   // 2^(i/16)
            for (int i = 0; i < 16; ++i) tab[i] = (u64)(((u128)1 << 92) / tab[i]);        // 2^(-i/16) Q30
            init = true;
        }
        int sh = c16 >> 4;
        return sh >= 40 ? 0 : (tab[c16 & 15] >> sh);
    }

    void derive() {
        gene_of.resize(m); qcum.resize(m); pi0q.resize(G); P0.resize(m);
        for (u32 g = 0; g < G; ++g) {
            u32 a = gstart[g], b = gstart[g + 1];
            u64 tot = 0; std::vector<u64> w(b - a);
            for (u32 j = a; j < b; ++j) { w[j - a] = std::max<u64>(1, pow2neg_q30(qcode[j])); tot += w[j - a]; }
            u64 cum = 0;
            for (u32 j = a; j < b; ++j) {
                gene_of[j] = g; cum += w[j - a];
                qcum[j] = (j + 1 == b) ? (u32)((u64)1 << 31) : (u32)((((u128)cum) << 31) / tot);
            }
            pi0q[g] = exp_neg_q62((u64)pi0code[g] << 21);       // exp(-code/2048)
            for (u32 j = a; j < b; ++j) {
                u32 qlo = (j == a) ? 0 : qcum[j - 1];
                P0[j] = (u64)(((u128)pi0q[g] * (qcum[j] - qlo)) >> 31);
            }
        }
        if (K) {
            Rsq.resize((size_t)K * G);
            for (size_t i = 0; i < Rsq.size(); ++i) Rsq[i] = (uint16_t)((uint32_t)Rcode[i] * Rcode[i]);
            phi.assign((size_t)G * K, 0.f);
            std::vector<double> sums(K, 0.0);
            for (u32 k = 0; k < K; ++k)
                for (u32 g = 0; g < G; ++g) {
                    double c = Rcode[(size_t)k * G + g];
                    double u = (double)tscale[k] * c * c / 65536.0;
                    double v = (double)pi0q[g] / 4611686018427387904.0 * u;
                    phi[(size_t)g * K + k] = (float)v; sums[k] += v;
                }
            for (u32 g = 0; g < G; ++g) for (u32 k = 0; k < K; ++k) phi[(size_t)g * K + k] /= (float)sums[k];
        }
    }
    void load_rownames() {
        if (!rownames.empty() || !(flags & 1)) return;
        size_t raw = ZSTD_getFrameContentSize(names_blob.data(), names_blob.size());
        std::vector<uint8_t> o(raw);
        size_t z = ZSTD_decompress(o.data(), raw, names_blob.data(), names_blob.size());
        if (ZSTD_isError(z)) throw std::runtime_error("idea4pp model: names");
        rownames.reserve(m);
        size_t i = 0;
        for (u32 r = 0; r < m; ++r) { size_t j = i; while (j < raw && o[j] != '\n') ++j; rownames.emplace_back((const char*)&o[i], j - i); i = j + 1; }
    }
};

// ---------------------------------------------------------------------------------------------
// Scale classes: Nhat(s) = 2^((s - 2R)/R) in Q32.
// ---------------------------------------------------------------------------------------------
inline u64 scale_q32(int s, int R) {
    static u64 pw[2][16]; static bool init = false;
    if (!init) {
        int Rs[2] = {SCALE_R_CELL, SCALE_R_AMB};
        for (int t = 0; t < 2; ++t) {
            u64 r = root2_q62(Rs[t]); u64 x = (u64)1 << 62;
            for (int i = 0; i < Rs[t]; ++i) { pw[t][i] = x >> 30; x = (u64)(((u128)x * r) >> 62); }   // 2^(i/R) in Q32
        }
        init = true;
    }
    int t = (R == SCALE_R_CELL) ? 0 : 1;
    int e = s / R - 2, r = s % R;
    u64 v = pw[t][r];
    return e >= 0 ? (v << e) : (v >> (-e));
}
inline int scale_levels(int R) { return NOCT * R; }

// ---------------------------------------------------------------------------------------------
// Hazards
// ---------------------------------------------------------------------------------------------
inline u64 mulsh31(u64 a, u32 q) { return (u64)(((u128)a * q) >> 31); }

struct CalledHaz {
    const Model* M = nullptr;
    std::vector<u64> mu, P;      // mu: G, P: G+1
    std::vector<u64> acc, zs;    // scratch
    const u64* pi0q_p = nullptr; const u32* qcum_p = nullptr;   // base profile / interval split in use (model or adaptive)
    static inline int bitlen64(u64 v) { return v ? 64 - __builtin_clzll(v) : 0; }
    void build(const Model& m_, u64 nhat, const uint16_t* tq /*K or null*/, const u64* pi0q_over = nullptr, const u32* qcum_over = nullptr) {
        M = &m_;
        pi0q_p = pi0q_over ? pi0q_over : M->pi0q.data(); qcum_p = qcum_over ? qcum_over : M->qcum.data();
        const u32 G = M->G, K = M->K;
        mu.resize(G); P.resize(G + 1); zs.resize(G);
        if (K) {
            acc.assign(G, 0);
            u64* __restrict ac = acc.data();
            for (u32 k = 0; k < K; ++k) {
                if (!tq[k]) continue;
                const u64 a = (u64)tq[k] * M->tscale[k];
                const uint16_t* __restrict r = &M->Rsq[(size_t)k * G];
                for (u32 g = 0; g < G; ++g) ac[g] += a * (u64)r[g];
            }
            u64 mx = 0;
            for (u32 g = 0; g < G; ++g) mx = ac[g] > mx ? ac[g] : mx;
            if (mx == 0) throw std::runtime_error("idea4pp: degenerate theta (all-zero weights)");
            const int sa = 63 - bitlen64(mx);
            for (u32 g = 0; g < G; ++g) zs[g] = (u64)(((u128)pi0q_p[g] * (ac[g] << sa)) >> 64);
        } else {
            for (u32 g = 0; g < G; ++g) zs[g] = pi0q_p[g] >> 3;
        }
        u64 Zt = 0;
        for (u32 g = 0; g < G; ++g) Zt += zs[g];
        if (Zt == 0) throw std::runtime_error("idea4pp: degenerate profile");
        const int s2 = 61 - bitlen64(Zt);          // normalise the total to [2^60, 2^61]
        if (s2 >= 0) { Zt <<= s2; } else { Zt >>= -s2; }
        const u64 scale = (u64)((((u128)nhat) << 64) / Zt);
        P[0] = 0;
        u64 run = 0;
        for (u32 g = 0; g < G; ++g) {
            u64 z = s2 >= 0 ? (zs[g] << s2) : (zs[g] >> -s2);
            u64 m_ = (u64)(((u128)z * scale) >> 64);
            mu[g] = m_; run += m_; P[g + 1] = run;
        }
    }
    inline u64 Mc(u32 j) const { u32 g = M->gene_of[j]; return P[g] + mulsh31(mu[g], qcum_p[j]); }
    // smallest j in [lo, m-1] with Mc(j) >= t ; m if none
    u32 lower_bound(u64 t, u32 lo) const {
        const u32 m = M->m;
        if (t > P[M->G]) return m;
        if (t == 0) return lo;
        u32 a = 0, b = M->G;     // P[a] < t <= P[b]
        while (b - a > 1) { u32 mid = (a + b) >> 1; if (P[mid] < t) a = mid; else b = mid; }
        u32 g = a;
        u32 x = M->gstart[g], y = M->gstart[g + 1] - 1;
        while (x < y) { u32 mid = (x + y) >> 1; if (Mc(mid) >= t) y = mid; else x = mid + 1; }
        return std::max(x, lo);
    }
    u64 total() const { return P[M->G]; }
};

struct AmbHaz {
    const Model* M = nullptr;
    const u64* CAn = nullptr;     // normalised cumulative profile, CAn[m-1] = 2^56
    u64 nhat = 0;
    inline u64 Mc(u32 j) const { return (u64)(((u128)nhat * CAn[j]) >> 56); }
    u32 lower_bound(u64 t, u32 lo) const {
        u32 a = lo, b = M->m;     // smallest j in [lo, m) with Mc(j) >= t
        while (a < b) { u32 mid = (a + b) >> 1; if (Mc(mid) >= t) b = mid; else a = mid + 1; }
        return a;
    }
    u64 total() const { return Mc(M->m - 1); }
};

// ---------------------------------------------------------------------------------------------
// Log helper for the decoder's search start (accuracy irrelevant for correctness).
// ---------------------------------------------------------------------------------------------
inline u64 neg_ln_ratio_q32(u64 num, u64 den) {      // ln(den/num) in Q32, num,den > 0
    if (num >= den) return 0;
    static u32 lt[257]; static bool init = false;
    if (!init) { for (int i = 0; i <= 256; ++i) lt[i] = (u32)(std::log(1.0 + i / 256.0) * 1073741824.0); init = true; }
    auto ln_q30 = [&](u64 x) -> int64_t {            // ln(x) in Q30 (up to a constant that cancels)
        int e = 63 - __builtin_clzll(x);
        u64 mant = (e >= 8) ? (x >> (e - 8)) & 255 : (x << (8 - e)) & 255;
        u64 frac = (e >= 16) ? ((x >> (e - 16)) & 255) : (((x << (16 - e)) & 255));
        int64_t base = (int64_t)lt[mant];
        int64_t nxt = (int64_t)lt[mant + 1];
        int64_t interp = base + (((nxt - base) * (int64_t)frac) >> 8);
        return (int64_t)e * 744261117 /* ln2 Q30 */ + interp;
    };
    int64_t d = ln_q30(den) - ln_q30(num);
    if (d < 0) d = 0;
    return (u64)d << 2;
}

// ---------------------------------------------------------------------------------------------
// Count coding helpers
// ---------------------------------------------------------------------------------------------
inline int cbucket(u64 lam_q32) {
    if (lam_q32 == 0) return 0;
    int lz = 63 - __builtin_clzll(lam_q32);
    int e = lz - 32;
    int half = lz >= 1 ? (int)((lam_q32 >> (lz - 1)) & 1) : 0;
    int b = 2 * (e + 10) + half;
    return b < 0 ? 0 : (b >= NCB ? NCB - 1 : b);
}
struct CountCoder {
    const uint16_t* tab;     // NCB*(NSYM+1) cumulative, scale 2^CSB
    explicit CountCoder(const uint16_t* t) : tab(t) {}
    void enc(RansEnc& e, u32 c, u64 lam_q32, double* bits) const {
        const uint16_t* cum = tab + cbucket(lam_q32) * (NSYM + 1);
        u32 s = c - 1 < (u32)(NSYM - 1) ? c - 1 : (u32)(NSYM - 1);
        e.put(cum[s], cum[s + 1] - cum[s], CSB);
        if (bits) *bits += CSB - std::log2((double)(cum[s + 1] - cum[s]));
        if (s == (u32)(NSYM - 1)) {
            u32 v = c - NSYM + 1;                      // c>=16 -> v>=1
            int nb = 0; { u32 t = v; while (t) { ++nb; t >>= 1; } }   // 1..32
            e.put(nb - 1, 1, 5);
            if (nb > 1) e.put_bits(v & ((nb - 1 >= 32) ? 0xFFFFFFFFu : ((1u << (nb - 1)) - 1)), nb - 1);
            if (bits) *bits += 5 + (nb - 1);
        }
    }
    u32 dec(RansDec& d, u64 lam_q32) const {
        const uint16_t* cum = tab + cbucket(lam_q32) * (NSYM + 1);
        u32 slot = d.slot(CSB);
        int s = 0;
        while (s < NSYM - 1 && cum[s + 1] <= slot) ++s;
        d.advance(cum[s], cum[s + 1] - cum[s], CSB);
        if (s < NSYM - 1) return (u32)s + 1;
        int nb = (int)d.slot(5) + 1; d.advance((u32)(nb - 1), 1, 5);
        u32 v = 1u << (nb - 1);
        if (nb > 1) v |= d.get_bits(nb - 1);
        return v + NSYM - 1;
    }
};

// ---------------------------------------------------------------------------------------------
// Gap coder (survival function of the Poisson process), templated on the hazard
// ---------------------------------------------------------------------------------------------
inline std::vector<u64>& enc_trace() { static std::vector<u64> v; return v; }
inline std::vector<u64>& dec_trace() { static std::vector<u64> v; return v; }
inline bool& trace_on() { static bool b = false; return b; }

template <class H> struct Gap {
    const H& h; u32 m;
    Gap(const H& h_, u32 m_) : h(h_), m(m_) {}
    // cum_hi(j) for current position i (i=-1 allowed), Mi = Mc(i)
    inline u64 cum_hi(u32 j, int64_t i, u64 Mi, u64 L) const {
        u64 D = h.Mc(j) - Mi;
        u64 e = exp_neg_q30(D);
        return (u64)(j - i) + ((((u64)TT - L) * (((u64)1 << 30) - e)) >> 30);
    }
    template <class CC>
    void encode(RansEnc& enc, const u32* idx, const u32* val, size_t n, const CC& cc, double* bits_gap, double* bits_cnt, double* bits_end) const {
        int64_t i = -1; u64 Mi = 0;
        for (size_t t = 0; t < n; ++t) {
            u32 j = idx[t];
            u64 L = (u64)(m - i);          // m - i  (i=-1 -> m+1)
            u64 hi = cum_hi(j, i, Mi, L);
            u64 lo = (j == (u32)(i + 1)) ? 0 : cum_hi(j - 1, i, Mi, L);
            enc.put((u32)lo, (u32)(hi - lo), TB);
            if (trace_on()) enc_trace().push_back((lo << 32) | (hi - lo));
            if (bits_gap) *bits_gap += TB - std::log2((double)(hi - lo));
            u64 Mj = h.Mc(j);
            u64 lam = Mj - (j ? h.Mc(j - 1) : 0);
            cc.enc(enc, val[t], lam, bits_cnt);
            i = j; Mi = Mj;
        }
        if (i != (int64_t)m - 1) {
            u64 L = (u64)(m - i);
            u64 lo = cum_hi(m - 1, i, Mi, L);
            enc.put((u32)lo, (u32)(TT - lo), TB);
            if (trace_on()) enc_trace().push_back((lo << 32) | (TT - lo));
            if (bits_end) *bits_end += TB - std::log2((double)(TT - lo));
        }
    }
    // decoder-side symbol search: returns j (m means END), fills lo/hi
    u32 find(u64 s, int64_t i, u64 Mi, u64 L, u64& lo, u64& hi) const {
        u64 chi_end = cum_hi(m - 1, i, Mi, L);
        if (s >= chi_end) { lo = chi_end; hi = TT; return m; }
        u64 span = TT - L;
        u64 num = (span > s) ? (span - s) : 1;
        u64 Dt = neg_ln_ratio_q32(num ? num : 1, span ? span : 1);
        u32 j = h.lower_bound(Mi + Dt, (u32)(i + 1));
        if (j >= m) j = m - 1;
        if (j < (u32)(i + 1)) j = (u32)(i + 1);
        hi = cum_hi(j, i, Mi, L);
        while (hi <= s) { ++j; hi = cum_hi(j, i, Mi, L); }          // terminates: cum_hi(m-1) > s here
        for (;;) {
            if (j == (u32)(i + 1)) { lo = 0; break; }
            lo = cum_hi(j - 1, i, Mi, L);
            if (lo > s) { --j; hi = lo; continue; }
            break;
        }
        return j;
    }
    template <class CC>
    void decode(RansDec& dec, std::vector<u32>& idx, std::vector<u32>& val, const CC& cc) const {
        int64_t i = -1; u64 Mi = 0;
        while (i != (int64_t)m - 1) {
            u64 L = (u64)(m - i);
            u32 s = dec.slot(TB);
            u64 lo, hi;
            u32 j = find(s, i, Mi, L, lo, hi);
            if (trace_on()) dec_trace().push_back((lo << 32) | (hi - lo));
            dec.advance((u32)lo, (u32)(hi - lo), TB);
            if (j >= m) break;
            u64 Mj = h.Mc(j);
            u64 lam = Mj - (j ? h.Mc(j - 1) : 0);
            idx.push_back(j);
            val.push_back(cc.dec(dec, lam));
            i = j; Mi = Mj;
        }
    }
    // self-check used under IDEA4_DEBUG: every encoded symbol must be found again by the decoder-side search
    void verify(const u32* idx, size_t n) const {
        int64_t i = -1; u64 Mi = 0;
        for (size_t t = 0; t <= n; ++t) {
            u64 L = (u64)(m - i);
            u32 j = (t < n) ? idx[t] : m;
            if (t == n && i == (int64_t)m - 1) break;
            u64 lo, hi;
            if (j < m) { hi = cum_hi(j, i, Mi, L); lo = (j == (u32)(i + 1)) ? 0 : cum_hi(j - 1, i, Mi, L); }
            else { lo = cum_hi(m - 1, i, Mi, L); hi = TT; }
            for (int pass = 0; pass < 2; ++pass) {
                u64 sl = pass ? hi - 1 : lo, lo2, hi2;
                u32 j2 = find(sl, i, Mi, L, lo2, hi2);
                if (j2 != j || lo2 != lo || hi2 != hi) {
                    static int shown = 0;
                    if (shown++ < 5)
                        std::printf("  VERIFY FAIL t=%zu j=%u (i=%lld) slot=%llu: found j2=%u lo/hi enc %llu/%llu dec %llu/%llu Mi=%llu L=%llu Mc(j)=%llu\n", t, j, (long long)i, (unsigned long long)sl, j2,
                                    (unsigned long long)lo, (unsigned long long)hi, (unsigned long long)lo2, (unsigned long long)hi2, (unsigned long long)Mi, (unsigned long long)L,
                                    (unsigned long long)(j < m ? h.Mc(j) : 0));
                    return;
                }
            }
            if (j >= m) break;
            i = j; Mi = h.Mc(j);
        }
    }
};

}}  // namespace cs::ipp
