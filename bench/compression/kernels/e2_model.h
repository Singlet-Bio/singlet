// SPDX-License-Identifier: MIT
// E2 model: block-adaptive rank-1 Poisson presence model + hazard-context count model.
//
//   hazard of feature f in a column of depth-bucket b   h_fb = A(z_f) * C_b    (bits: h/ln2)
//   z_f      running count of decoded columns (of this class) in which f was present
//   A(z)     per-z-bin hazard rate, fitted between blocks from the z histogram so that
//            nc (1 - exp(-A cbar)) = z*   (z* = Good-Turing adjusted count, cbar = mean exposure)
//   C_b      per-depth-bucket exposure, fitted so that sum_f (1 - exp(-A_f C_b)) = mean nnz of the bucket
//   S[f]     prefix sums (u32, total ~2^32) of the integer hazard rates a_f = A(z_f)/ln2 * scale (shared by all depth buckets)
//   D        survival distance (Q.24 bits) of a hazard-axis difference dS in bucket b:  D = (dS * M_b) >> k_b
//
// The fit runs in IEEE double using only + - * / (own exp/log, no libm), compiled with fp-contract=off,
// so encoder and decoder compute bit-identical tables. Everything that touches the coded symbols (S, M_b, the
// direct index, count tables) is integer.
#pragma once
#include "e2_core.h"
#include <array>
#include <cmath>

#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")

namespace cs {
namespace e2 {

// ---------------------------------------------------------------------------------------------
// deterministic double math (no libm): Taylor/atanh series with fixed evaluation order
namespace dm {
constexpr double LN2 = 0.6931471805599453094;
constexpr double LN2_HI = 0.693147180369123816490;     // 33 significant bits
constexpr double LN2_LO = 1.90821492927058770002e-10;
inline double pow2i(int k) {           // 2^k exactly, k in [-1000, 1000]
    u64 b = (u64)(k + 1023) << 52; double d; std::memcpy(&d, &b, 8); return d;
}
// e^x for x in [-700, 700]
inline double exp(double x) {
    if (x < -700) return 0.0;
    if (x > 700) x = 700;
    double kf = x * 1.4426950408889634074;
    int k = (int)(kf < 0 ? kf - 0.5 : kf + 0.5);
    double r = (x - (double)k * LN2_HI) - (double)k * LN2_LO;          // |r| <= 0.35
    double p = 1.0 / 87178291200.0;                                    // 1/14!
    p = p * r + 1.0 / 6227020800.0;
    p = p * r + 1.0 / 479001600.0;
    p = p * r + 1.0 / 39916800.0;
    p = p * r + 1.0 / 3628800.0;
    p = p * r + 1.0 / 362880.0;
    p = p * r + 1.0 / 40320.0;
    p = p * r + 1.0 / 5040.0;
    p = p * r + 1.0 / 720.0;
    p = p * r + 1.0 / 120.0;
    p = p * r + 1.0 / 24.0;
    p = p * r + 1.0 / 6.0;
    p = p * r + 0.5;
    p = p * r + 1.0;
    p = p * r + 1.0;
    int k1 = k / 2, k2 = k - k1;
    return p * pow2i(k1) * pow2i(k2);
}
// 1 - e^{-x} for x >= 0, accurate for small x
inline double ex1(double x) {
    if (x < 0.25) {
        double p = 1.0 / 355687428096000.0;                            // 1/17!
        p = p * x - 1.0 / 20922789888000.0;
        p = p * x + 1.0 / 1307674368000.0;
        p = p * x - 1.0 / 87178291200.0;
        p = p * x + 1.0 / 6227020800.0;
        p = p * x - 1.0 / 479001600.0;
        p = p * x + 1.0 / 39916800.0;
        p = p * x - 1.0 / 3628800.0;
        p = p * x + 1.0 / 362880.0;
        p = p * x - 1.0 / 40320.0;
        p = p * x + 1.0 / 5040.0;
        p = p * x - 1.0 / 720.0;
        p = p * x + 1.0 / 120.0;
        p = p * x - 1.0 / 24.0;
        p = p * x + 1.0 / 6.0;
        p = p * x - 0.5;
        p = p * x + 1.0;
        return p * x;
    }
    return 1.0 - exp(-x);
}
// natural log for x > 0
inline double ln(double x) {
    u64 b; std::memcpy(&b, &x, 8);
    int e = (int)((b >> 52) & 0x7FF) - 1023;
    b = (b & 0x000FFFFFFFFFFFFFull) | 0x3FF0000000000000ull;           // mantissa in [1,2)
    double mm; std::memcpy(&mm, &b, 8);
    if (mm > 1.4142135623730951) { mm *= 0.5; ++e; }
    double s = (mm - 1.0) / (mm + 1.0), s2 = s * s;
    double p = 1.0 / 23.0;
    p = p * s2 + 1.0 / 21.0;
    p = p * s2 + 1.0 / 19.0;
    p = p * s2 + 1.0 / 17.0;
    p = p * s2 + 1.0 / 15.0;
    p = p * s2 + 1.0 / 13.0;
    p = p * s2 + 1.0 / 11.0;
    p = p * s2 + 1.0 / 9.0;
    p = p * s2 + 1.0 / 7.0;
    p = p * s2 + 1.0 / 5.0;
    p = p * s2 + 1.0 / 3.0;
    p = p * s2 + 1.0;
    return (double)e * LN2 + 2.0 * s * p;
}
// binary exponent of a positive double: floor(log2(x))
inline int ilog2d(double x) { u64 b; std::memcpy(&b, &x, 8); return (int)((b >> 52) & 0x7FF) - 1023; }
}  // namespace dm

// ---------------------------------------------------------------------------------------------
struct Params {
    u32 bpo_log = 2;        // depth buckets per octave = 2^bpo_log
    u32 n0 = 16;            // first block (columns)
    u32 rho_x16 = 32;       // block growth factor (x16): next block = max(n0, (rho-1) * columns so far), capped
    u32 Bmax = 1024;        // block cap (columns)
    u32 Umax = 0;           // block cap (sum of column totals N, in units of 1024); 0 = none
    u32 lanes = 4;          // interleaved rANS lanes (columns of a block are dealt round-robin)
    u32 gtmax = 8;          // Good-Turing adjustment of z* for z <= gtmax
    u32 rounds = 1;         // alternating A/C fit rounds per refit
    u32 ctrust = 20;        // weight of a bucket's own exposure estimate: nh / (nh + ctrust)
    u32 flat_x100 = 70;     // flat prior (first block): expected nnz = min(flat * N, 0.2 m)
    u32 zlim = 2;           // empties: features with z <= zlim get their class mass redistributed by the cells profile
    u32 theta_x16 = 16;     // ... in proportion to (cells hazard)^theta; 0 = flat start for empties
    u32 seed = 12345;
    u32 pad = 0;
};
constexpr int NPARAMS = 14;
inline void params_pack(const Params& P, u32* r) {
    u32 raw[NPARAMS] = {P.bpo_log, P.n0, P.rho_x16, P.Bmax, P.Umax, P.lanes, P.gtmax, P.rounds, P.ctrust, P.flat_x100, P.zlim, P.theta_x16, P.seed, P.pad};
    std::memcpy(r, raw, sizeof raw);
}
inline Params params_unpack(const u32* r) {
    Params P;
    P.bpo_log = r[0]; P.n0 = r[1]; P.rho_x16 = r[2]; P.Bmax = r[3]; P.Umax = r[4]; P.lanes = r[5]; P.gtmax = r[6]; P.rounds = r[7];
    P.ctrust = r[8]; P.flat_x100 = r[9]; P.zlim = r[10]; P.theta_x16 = r[11]; P.seed = r[12]; P.pad = r[13];
    return P;
}

// depth bucket of a column total N >= 1: 2^bpo_log buckets per octave (linear sub-buckets of the mantissa)
inline u32 bucket_of(u64 N, u32 q) {
    int lg = ilog2(N);
    u32 sub;
    if (lg >= (int)q) sub = (u32)((N >> (lg - q)) & ((1u << q) - 1));
    else sub = (u32)((N << (q - lg)) & ((1u << q) - 1));
    return ((u32)lg << q) + sub;
}

constexpr int ZB = 160;                       // z bins: exact below 32, then 8 per octave
inline u32 zbin(u32 z) {
    if (z < 32) return z;
    int lg = 31 - __builtin_clz(z);
    return 32 + ((lg - 5) << 3) + ((z >> (lg - 3)) & 7);
}

constexpr int NCX = 48, NSYM = 16, CPB = 10;  // count contexts, symbols, probability bits
constexpr int IDXB = 17;                      // direct-index bits over the hazard axis
constexpr u32 SINF = 0xFFFFFFFFu;

// count-context of a hazard w (Q.FR bits): half-octave of w
inline u32 count_ctx(u64 w) {
    if (w < 256) return 0;
    int lg = ilog2(w);
    int c = (lg << 1) + (int)((w >> (lg - 1)) & 1) - 16;
    return c < 0 ? 0 : (c >= NCX ? NCX - 1 : (u32)c);
}

struct BucketParams { u32 M, R; u8 k, s; u16 pad; u32 pad2; };    // 16 bytes

struct ClassModel {
    Params P;
    u32 m = 0;
    int nbk = 0;
    // statistics
    std::vector<u32> z;                     // per feature
    std::vector<double> nh, Tn, Ns;         // per depth bucket: columns, nnz, sum of N
    u64 ncols = 0;
    std::vector<double> Cd;                 // persistent exposures (double)
    std::vector<u32> cn;                    // NCX * NSYM count statistics (units of 1/64)
    // derived by refit()
    std::vector<u32> S;                     // m + 2 (+ padding) prefix sums, S[m+1] = SINF
    std::vector<BucketParams> bp;           // per bucket
    std::vector<u32> idx;                   // direct index over the hazard axis (2^IDXB + 2 entries)
    std::vector<u32> clut;                  // NCX << CPB decode entries: sym | freq << 8 | bias << 20
    std::vector<u16> cfreq, ccum;           // NCX * NSYM, NCX * (NSYM+1)
    const std::vector<u32>* prior_a = nullptr;   // per-feature prior hazards (cells model's a_f), empties only
    std::vector<u32> prior_w;               // own copy
    double scale = 1;                       // S units per hazard bit (at unit exposure)

    ClassModel() {}
    ClassModel(const Params& p, u32 m_, int nbk_) { init(p, m_, nbk_); }
    void init(const Params& p, u32 m_, int nbk_) {
        P = p; m = m_; nbk = nbk_;
        z.assign(m, 0);
        nh.assign(nbk, 0.0); Tn.assign(nbk, 0.0); Ns.assign(nbk, 0.0);
        Cd.assign(nbk, 1.0);
        ncols = 0;
        cn.assign((size_t)NCX * NSYM, 0);
        S.assign(m + 2 + 64, SINF);
        bp.assign(nbk, BucketParams());
        idx.assign((1u << IDXB) + 2, m + 1);
        clut.assign((size_t)NCX << CPB, 0);
        cfreq.assign((size_t)NCX * NSYM, 0); ccum.assign((size_t)NCX * (NSYM + 1), 0);
    }
    // per-feature hazard a_f (S differences) of this model, used as the empties prior
    void export_prior(std::vector<u32>& w) const {
        w.resize(m);
        for (u32 f = 0; f < m; ++f) w[f] = S[f + 1] - S[f];
    }

    inline void add_column(u32 b, u32 k, u64 N) { nh[b] += 1; Tn[b] += k; Ns[b] += (double)N; ++ncols; }

    // ---- refit -----------------------------------------------------------------------------------------------
    static double bisect_C(const double* gA, const double* gH, int ng, double target, double y0) {
        // solve sum_g gH (1 - exp(-gA * e^y)) = target for y, monotone increasing in y
        double lo = -14.0, hi = 14.0;
        (void)y0;
        for (int it = 0; it < 40; ++it) {
            double mid = 0.5 * (lo + hi), e = dm::exp(mid), s = 0;
            for (int i = 0; i < ng; ++i) s += gH[i] * dm::ex1(gA[i] * e);
            if (s < target) lo = mid; else hi = mid;
        }
        return 0.5 * (lo + hi);
    }

    void refit() {
        // pass 1: z histogram (binned)
        u32 zh[ZB]; double zs[ZB];
        for (int i = 0; i < ZB; ++i) { zh[i] = 0; zs[i] = 0; }
        for (u32 f = 0; f < m; ++f) { u32 zb = zbin(z[f]); ++zh[zb]; zs[zb] += (double)z[f]; }
        double Ab[ZB];                              // hazard rate (bits per unit exposure) per z bin
        for (int i = 0; i < ZB; ++i) Ab[i] = 0;
        const double nc = (double)ncols;
        std::vector<double> Nbar(nbk);
        for (int b = 0; b < nbk; ++b) Nbar[b] = nh[b] > 0 ? Ns[b] / nh[b] : dm::exp((b + 0.5) / (double)(1u << P.bpo_log) * dm::LN2);
        if (ncols == 0) {
            double a = 1.0 / (double)m / dm::LN2;
            for (int b = 0; b < nbk; ++b) {
                double kt = std::min(P.flat_x100 / 100.0 * Nbar[b], 0.2 * (double)m);
                Cd[b] = -(double)m * dm::ln(1.0 - kt / (double)m);
            }
            for (int i = 0; i < ZB; ++i) Ab[i] = a;
        } else {
            double rep[ZB], tgt[ZB];
            for (int i = 0; i < ZB; ++i) rep[i] = zh[i] ? zs[i] / (double)zh[i] : 0.0;
            for (int i = 0; i < ZB; ++i) tgt[i] = rep[i];
            {
                double n0 = zh[0], n1 = zh[1];
                tgt[0] = n0 > 0 ? std::max(n1 / n0, 1e-5) : 1e-5;
                for (u32 zz = 1; zz <= P.gtmax && zz + 1 < 32; ++zz)
                    if (zh[zz] >= 5 && zh[zz + 1] >= 5) {
                        double g = (zz + 1.0) * zh[zz + 1] / zh[zz];
                        tgt[zz] = std::min(std::max(g, 0.4 * zz), (double)zz);
                    }
                for (int i = 0; i < ZB; ++i) tgt[i] = std::min(tgt[i], nc - 0.5);
            }
            auto a_step = [&]() {
                double cbar = 0;
                for (int b = 0; b < nbk; ++b) cbar += nh[b] * Cd[b];
                cbar /= nc;
                for (int i = 0; i < ZB; ++i) if (zh[i]) Ab[i] = -dm::ln(1.0 - tgt[i] / nc) / cbar / dm::LN2;
            };
            a_step();
            for (u32 r = 0; r < P.rounds; ++r) {
                // C step over grouped z bins
                double gA[40], gH[40], gS[40];
                int ng = 0;
                {
                    for (int g = 0; g < 40; ++g) { gH[g] = 0; gS[g] = 0; }
                    for (int i = 0; i < ZB; ++i) if (zh[i]) {
                        int g = i < 8 ? i : 8 + (i - 8) / 6;
                        if (g > 39) g = 39;
                        gH[g] += zh[i]; gS[g] += zh[i] * dm::ln(Ab[i] * dm::LN2);
                    }
                    for (int g = 0; g < 40; ++g) if (gH[g] > 0) { gH[ng] = gH[g]; gA[ng] = dm::exp(gS[g] / gH[g]); ++ng; }
                }
                std::vector<double> Cn(nbk, 0.0);
                for (int b = 0; b < nbk; ++b) if (nh[b] > 0) Cn[b] = dm::exp(bisect_C(gA, gH, ng, std::max(Tn[b] / nh[b], 1e-3), dm::ln(Cd[b])));
                double sw = 0, mx = 0, my = 0;
                for (int b = 0; b < nbk; ++b) if (nh[b] > 0) { double w = std::min(nh[b], 50.0); sw += w; mx += w * dm::ln(Nbar[b]); my += w * dm::ln(Cn[b]); }
                mx /= sw; my /= sw;
                double sxx = 0, sxy = 0;
                for (int b = 0; b < nbk; ++b) if (nh[b] > 0) { double w = std::min(nh[b], 50.0); double dx = dm::ln(Nbar[b]) - mx; sxx += w * dx * dx; sxy += w * dx * (dm::ln(Cn[b]) - my); }
                double slope = sxx > 1e-9 ? sxy / sxx : 1.0;
                slope = std::min(std::max(slope, 0.3), 1.5);
                for (int b = 0; b < nbk; ++b) {
                    double fitv = my + slope * (dm::ln(Nbar[b]) - mx);
                    double lam = nh[b] / (nh[b] + (double)P.ctrust);
                    Cd[b] = dm::exp(nh[b] > 0 ? lam * dm::ln(Cn[b]) + (1 - lam) * fitv : fitv);
                }
                a_step();
            }
        }
        // ---- integer tables ------------------------------------------------------------------------------
        double Atot = 0;
        for (int i = 0; i < ZB; ++i) Atot += zh[i] * Ab[i];
        scale = (4294967296.0 - 2.0 - (double)m - 64.0) / Atot;
        u32 aq[ZB];
        for (int i = 0; i < ZB; ++i) { double v = Ab[i] * scale; aq[i] = v < 1.0 ? 1u : (u32)v; }
        // optional redistribution of the low-z mass by a prior profile (empties from cells)
        double wmean[8];
        const bool use_prior = !prior_w.empty() && P.theta_x16 > 0;
        const u32 zl = std::min<u32>(P.zlim, 7);
        if (use_prior) {
            double sw[8] = {0, 0, 0, 0, 0, 0, 0, 0}; double cz[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            for (u32 f = 0; f < m; ++f) if (z[f] <= zl) { sw[z[f]] += (double)prior_w[f]; cz[z[f]] += 1; }
            for (u32 i = 0; i <= zl; ++i) wmean[i] = cz[i] > 0 ? sw[i] / cz[i] : 1.0;
        }
        u64 acc = 0;
        if (!use_prior) {
            for (u32 f = 0; f < m; ++f) { S[f] = (u32)acc; acc += aq[zbin(z[f])]; }
        } else {
            for (u32 f = 0; f < m; ++f) {
                S[f] = (u32)acc;
                u32 zf = z[f];
                u32 a;
                if (zf <= zl) {
                    double v = (Ab[zf] * scale) * ((double)prior_w[f] / wmean[zf]);
                    a = v < 1.0 ? 1u : (v > 4.0e9 ? 4000000000u : (u32)v);
                } else a = aq[zbin(zf)];
                acc += a;
            }
        }
        if (acc > 0xFFFFFFFEull - 32) {              // safety: renormalise (cannot happen for a sane model)
            throw std::runtime_error("e2: hazard prefix overflow");
        }
        S[m] = (u32)acc; S[m + 1] = SINF;
        for (u32 f = m + 2; f < S.size(); ++f) S[f] = SINF;
        // exposures -> fixed point
        for (int b = 0; b < nbk; ++b) {
            double g24 = Cd[b] / scale * 16777216.0;       // Q24 bits per S unit
            int e = dm::ilog2d(g24);
            int k = 31 - e;
            if (k < 0) k = 0;
            if (k > 62) k = 62;
            double Md = g24 * dm::pow2i(k);
            u64 M = Md >= 4294967295.0 ? 4294967295ull : (u64)Md;
            if (M < 1) M = 1;
            u64 R = (1ull << 62) / M;
            bp[b].M = (u32)M; bp[b].k = (u8)k; bp[b].R = (u32)(R > 0xFFFFFFFFull ? 0xFFFFFFFFull : R); bp[b].s = (u8)(62 - k);
        }
        // direct index: idx[k] = #{f in [0, m] : S[f] < k << (32 - IDXB)}
        {
            const int sh = 32 - IDXB;
            const size_t nk = (1u << IDXB) + 1;
            size_t k = 0;
            for (u32 f = 0; f <= m; ++f) {
                while (k < nk && (((u64)k << sh) <= (u64)S[f])) { idx[k] = f; ++k; }
            }
            while (k < nk + 1) idx[k++] = m + 1;
        }
        build_count_tables();
    }

    void build_count_tables() {
        for (int c = 0; c < NCX; ++c) {
            u64 cnt[NSYM], tot = 0;
            for (int s = 0; s < NSYM; ++s) {
                u64 pr = s < 15 ? (128ull >> (s + 1)) : 1;
                if (pr < 1) pr = 1;
                cnt[s] = (u64)cn[(size_t)c * NSYM + s] + pr;
                tot += cnt[s];
            }
            u32 fr[NSYM];
            const u32 Mt = 1u << CPB;
            u32 sum = 0, best = 0;
            for (int s = 0; s < NSYM; ++s) {
                u32 f = (u32)((cnt[s] * Mt) / tot);
                if (f < 1) f = 1;
                fr[s] = f; sum += f;
                if (cnt[s] > cnt[best]) best = s;
            }
            while (sum != Mt) {
                if (sum < Mt) { fr[best] += Mt - sum; sum = Mt; }
                else {
                    u32 d = sum - Mt;
                    u32 take = std::min<u32>(d, fr[best] - 1);
                    fr[best] -= take; sum -= take;
                    if (take == 0) for (int s = 0; s < NSYM && sum > Mt; ++s) if (fr[s] > 1) { --fr[s]; --sum; }
                }
            }
            u32 cu = 0;
            u32* lut = clut.data() + ((size_t)c << CPB);
            for (int s = 0; s < NSYM; ++s) {
                cfreq[(size_t)c * NSYM + s] = (u16)fr[s];
                ccum[(size_t)c * (NSYM + 1) + s] = (u16)cu;
                for (u32 t = 0; t < fr[s]; ++t) lut[cu + t] = (u32)s | (fr[s] << 8) | (t << 20);
                cu += fr[s];
            }
            ccum[(size_t)c * (NSYM + 1) + NSYM] = (u16)Mt;
        }
    }

    // ---- symbol-level helpers (shared by encoder and decoder) ----------------------------------------------
    // survival distance (Q.FR bits) of a hazard-axis difference for bucket b
    inline u64 dist(u32 dS, u32 b) const { return ((u64)dS * bp[b].M) >> bp[b].k; }

    // number of S[f] <= Ts over f in [0, m]  (Ts < S[m])
    inline u32 cnt_le(u32 Ts) const {
        u32 k = Ts >> (32 - IDXB);
        u32 a = idx[k], b = idx[k + 1];
        const u32* s = S.data();
        if (b - a <= 24) {
            u32 c = a;
            for (u32 f = a; f < b; ++f) c += (s[f] <= Ts);
            return c;
        }
        u32 lo = a, hi = b;
        while (lo < hi) { u32 mid = (lo + hi) >> 1; if (s[mid] <= Ts) lo = mid + 1; else hi = mid; }
        return lo;
    }
};

}  // namespace e2
}  // namespace cs
#pragma GCC pop_options
