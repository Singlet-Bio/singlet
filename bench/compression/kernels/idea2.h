// SPDX-License-Identifier: MIT
// Idea 2: "the file is its own model".
//   * called cells coded against a per-file Poisson rank-1 prior (transmitted: per-feature nnz z_f over called cells
//     + per-cell nnz), adaptive neighbour / cluster inputs mixed by an integer logistic mixer for the top-T ranked
//     features, order-statistic (set) coding for the tail;
//   * empty droplets coded last against the ambient profile estimated from the decoded cells (exact multinomial via
//     order statistics), with optional depth-bucketed / cluster-mixture / set+count variants.
// Everything on the decode path is integer arithmetic. No shipped model.
#pragma once
#include "common.h"
#include "idea2_core.h"
#include "idea2_order.h"
#include "ref_zstd_csc.h"
#include <cstdlib>
#include <fstream>
#include <numeric>
#include <sstream>
#include <unordered_set>

namespace cs { namespace i2 {

// ---------------------------------------------------------------------------------------------
struct Params {
    u32 T = 4096;          // number of top features modelled with binary decisions
    u32 K = 1;             // number of cell clusters (labels transmitted when K > 1)
    u32 NB = 8;            // depth buckets for the (feature, depth) counters
    u32 mask = 0;          // mixer inputs: 1 (f,depth) ctr | 2 (f,cluster) ctr | 4 EMA fast | 8 EMA slow | 16 (f,cluster) offset | 32 neighbour offset
    u32 mixshift = 14;     // mixer learning rate shift
    u32 limA = 255, limC = 120;
    u32 ema1 = 3, ema2 = 5;        // EMA shifts (rates 1/8, 1/32)
    u32 n0A = 3, n0C = 3;          // prior pseudo-counts of counters
    u32 order_mode = 0;    // 0: (label, depth) implicit | 1: NN chain within cluster, explicit permutation | 2: stored order | 3: depth only
    u32 amb_gamma = 100000;  // pseudo-UMI weight of the cells profile in the ambient estimate
    u32 amb_delta_x8 = 8;    // smoothing pseudo-UMI per feature (in 1/8)
    u32 amb_adapt = 1;       // adapt ambient to decoded empties
    u32 val_mode = 1;        // 0: static value ctx only, 1: + per-feature counters
    u32 fit_rounds = 4;
    u32 amb_mode = 0;        // 0 pooled | 1 depth-bucketed adaptive | 2 cluster mixture w/ droplet label | 3 set + counts
    u32 amb_beta = 4;        // mode 1: weight of the bucket's own counts
    u32 amb_mu_x8 = 8;       // mode 2: cluster-profile mass relative to the ambient mass (x1/8)
    u32 offshC = 0;          // mode 16: 0 = count-based rate, else fixed shift
    u32 offshN = 10;         // mode 32: fixed shift (10 = 1/4 nat per unit error)
    u32 pad[3] = {};
};
constexpr int NPARAM = 24;

inline u32 env_u32(const char* n, u32 d) { const char* s = std::getenv(n); return s ? (u32)std::strtoul(s, nullptr, 10) : d; }
inline Params params_from_env(Params p) {
    p.T = env_u32("I2_T", p.T); p.K = env_u32("I2_K", p.K); p.NB = env_u32("I2_NB", p.NB);
    p.mask = env_u32("I2_MASK", p.mask); p.mixshift = env_u32("I2_MIXSHIFT", p.mixshift);
    p.limA = env_u32("I2_LIMA", p.limA); p.limC = env_u32("I2_LIMC", p.limC);
    p.ema1 = env_u32("I2_EMA1", p.ema1); p.ema2 = env_u32("I2_EMA2", p.ema2);
    p.n0A = env_u32("I2_N0A", p.n0A); p.n0C = env_u32("I2_N0C", p.n0C);
    p.order_mode = env_u32("I2_ORDER", p.order_mode);
    p.amb_gamma = env_u32("I2_AMBG", p.amb_gamma); p.amb_delta_x8 = env_u32("I2_AMBD", p.amb_delta_x8);
    p.amb_adapt = env_u32("I2_AMBA", p.amb_adapt); p.val_mode = env_u32("I2_VALM", p.val_mode);
    p.fit_rounds = env_u32("I2_FIT", p.fit_rounds);
    p.amb_mode = env_u32("I2_AMBM", p.amb_mode); p.amb_beta = env_u32("I2_AMBB", p.amb_beta);
    p.amb_mu_x8 = env_u32("I2_AMBMU", p.amb_mu_x8);
    p.offshC = env_u32("I2_OFFC", p.offshC); p.offshN = env_u32("I2_OFFN", p.offshN);
    return p;
}
inline void params_pack(const Params& P, u32* r) {
    u32 raw[NPARAM] = {P.T, P.K, P.NB, P.mask, P.mixshift, P.limA, P.limC, P.ema1, P.ema2, P.n0A, P.n0C, P.order_mode,
                       P.amb_gamma, P.amb_delta_x8, P.amb_adapt, P.val_mode, P.fit_rounds, P.amb_mode, P.amb_beta,
                       P.amb_mu_x8, P.offshC, P.offshN, 0, 0};
    std::memcpy(r, raw, sizeof raw);
}
inline Params params_unpack(const u32* r) {
    Params P;
    P.T = r[0]; P.K = r[1]; P.NB = r[2]; P.mask = r[3]; P.mixshift = r[4]; P.limA = r[5]; P.limC = r[6];
    P.ema1 = r[7]; P.ema2 = r[8]; P.n0A = r[9]; P.n0C = r[10]; P.order_mode = r[11]; P.amb_gamma = r[12];
    P.amb_delta_x8 = r[13]; P.amb_adapt = r[14]; P.val_mode = r[15]; P.fit_rounds = r[16]; P.amb_mode = r[17];
    P.amb_beta = r[18]; P.amb_mu_x8 = r[19]; P.offshC = r[20]; P.offshN = r[21];
    return P;
}

// global hooks used by the bench driver
inline std::string g_calls_path;      // cell_calls.tsv of the sample being encoded
struct Report {
    double bits[24] = {};
    size_t names_bytes = 0, hdr_bytes = 0, payload_bytes = 0;
    u64 nnz_cells = 0, nnz_empt = 0, ncells = 0, nempt = 0;
    double hk_cells = 0, hk_empt = 0;     // header bits for k per class
    u64 S = 0, Tn = 0;
    double fit_ms = 0, t_hdr = 0, t_cells = 0, t_empt = 0, t_order = 0;
};
inline Report g_rep;
enum Cat { C_HK = 0, C_FLAG, C_LAB, C_NNZ, C_Z, C_TOP, C_TAIL, C_VAL, C_EMPT, C_PERM, C_ELAB, C_EVAL, C_NCAT };
inline const char* cat_name(int c) {
    static const char* n[] = {"hdr_k", "flag", "label", "nnz_excess", "side_z", "cell_top", "cell_tail", "cell_val",
                              "empt_set", "perm", "empt_lab", "empt_val"};
    return n[c];
}
inline bool cat_is_empties(int c) { return c == C_EMPT || c == C_ELAB || c == C_EVAL; }

inline int ilog2u(u64 v) { return v ? 63 - __builtin_clzll(v) : 0; }

// ---------------------------------------------------------------------------------------------
// Poisson fit (integer): lw[r] (per rank), ls[c] (per cell) in 1/1024 log units so that p = ptab[lw + ls + OFF]
// reproduces the transmitted per-feature nnz (zr) and per-cell nnz (cnnz) margins.
inline void poisson_fit(const std::vector<u32>& zr, const std::vector<u32>& cnnz, int rounds,
                        std::vector<i32>& lw, std::vector<i32>& ls) {
    const Tabs& t = TB();
    const int NPT = Tabs::NPT, OFF = Tabs::OFF;
    size_t S = zr.size(), nc = cnnz.size();
    ls.assign(nc, 0); lw.assign(S, 0);
    std::vector<u64> F(NPT);
    auto clampi = [&](i64 v) { return (int)(v < 0 ? 0 : (v >= NPT ? NPT - 1 : v)); };
    auto feature_step = [&]() {
        std::vector<u32> h((size_t)(2 * NPT / 32 + 2), 0);
        for (size_t j = 0; j < nc; ++j) ++h[(size_t)((ls[j] + NPT) >> 5)];
        std::fill(F.begin(), F.end(), 0ull);
        for (size_t b = 0; b < h.size(); ++b) {
            if (!h[b]) continue;
            i64 lsb = (i64)(b << 5) + 16 - NPT;
            for (int gi = 0; gi < NPT; ++gi) F[gi] += (u64)h[b] * t.ptab[clampi(gi + lsb)];
        }
        for (size_t r = 0; r < S; ++r) {
            u64 tgt = (u64)zr[r] << 32;
            int gi = (int)(std::lower_bound(F.begin(), F.end(), tgt) - F.begin());
            if (gi >= NPT) gi = NPT - 1;
            lw[r] = gi - OFF;
        }
    };
    auto cell_step = [&]() {
        std::vector<u32> h((size_t)(2 * NPT / 32 + 2), 0);
        for (size_t r = 0; r < S; ++r) ++h[(size_t)((lw[r] + NPT) >> 5)];
        std::fill(F.begin(), F.end(), 0ull);
        for (size_t b = 0; b < h.size(); ++b) {
            if (!h[b]) continue;
            i64 lwb = (i64)(b << 5) + 16 - NPT;
            for (int si = 0; si < NPT; ++si) F[si] += (u64)h[b] * t.ptab[clampi(si + lwb)];
        }
        for (size_t j = 0; j < nc; ++j) {
            u64 tgt = (u64)cnnz[j] << 32;
            int si = (int)(std::lower_bound(F.begin(), F.end(), tgt) - F.begin());
            if (si >= NPT) si = NPT - 1;
            ls[j] = si - OFF;
        }
    };
    for (int r = 0; r < rounds; ++r) { feature_step(); cell_step(); }
    feature_step();
}

// ZTP prior for value coding: P(N >= t+1 | N >= t) for Poisson(lambda), lambda = exp(L), as p16.
inline int ztp_p16(i64 L_q32, int t) {
    if (L_q32 > (i64)3 << 32) return 62000;        // lambda > 20
    u128 lam = exp_q64(L_q32);                      // Q.64
    i64 lam_q32 = (i64)(lam >> 32);
    u128 pmf = exp_q64(-lam_q32);                   // P(N=0)
    u128 one = (u128)1 << 64;
    u128 cum = pmf;
    for (int i = 1; i < t; ++i) {
        pmf = (pmf * (u64)lam_q32 >> 32) / (u32)i;
        cum += pmf;
    }
    u128 Tt = (cum >= one) ? 0 : one - cum;
    pmf = (pmf * (u64)lam_q32 >> 32) / (u32)t;
    cum += pmf;
    u128 Tn = (cum >= one) ? 0 : one - cum;
    if (Tt == 0) return 32768;
    u128 p = (Tn << 16) / Tt;
    if (p < 64) p = 64;
    if (p > 65472) p = 65472;
    return (int)p;
}

// "u >= t ?" ladder coder for excess counts with an adaptive (ctx, t) table + optional per-feature counters.
struct ValModel {
    static constexpr int NVT = 10, TCAP = 20;
    int nctx;
    std::vector<u32> ctr;         // nctx x NVT
    std::vector<i32> w;           // NVT x 3 mixer
    std::vector<u32> fctr;        // per-feature x 2 (optional)
    NumCoder esc{2};
    ValModel(int nctx_, size_t nfeat) : nctx(nctx_), ctr((size_t)nctx_ * NVT), w((size_t)NVT * 3, 0), fctr(nfeat * 2, 0) {
        for (int t = 0; t < NVT; ++t) { w[t * 3 + 0] = 65536 * 6 / 10; w[t * 3 + 1] = nfeat ? 65536 * 4 / 10 : 0; }
    }
    // code u (excess over 1) with u <= R; feature slot fs < 0 disables the per-feature counters
    template <class RC>
    u32 code(RC& rc, int ctx, i64 fs, u32 R, u32 u_in) {
        u32 uu = 0;
        for (u32 t = 1;; ++t) {
            if (t > R) { uu = t - 1; break; }
            if (t > (u32)TCAP) { u32 rem = (u32)esc.code(rc, 0, RC::ENC ? u_in - (u32)TCAP : 0); uu = TCAP + rem; break; }
            int ct = (int)std::min<u32>(t, NVT) - 1;
            u32& c0 = ctr[(size_t)ctx * NVT + ct];
            int xs0 = stretch(ctr_p16(c0));
            u32* c1 = nullptr; int xs1 = 0;
            if (fs >= 0 && t <= 2) { c1 = &fctr[(size_t)fs * 2 + (t - 1)]; if (!*c1) *c1 = ctr_init(ctr_p16(c0), 2); xs1 = stretch(ctr_p16(*c1)); }
            i32* vw = &w[(size_t)ct * 3];
            i64 dot = (i64)vw[0] * xs0 + (i64)vw[1] * xs1 + (i64)vw[2] * 256;
            int p = squash((int)(dot >> 16));
            int bit = rc.bit(p, RC::ENC ? (u_in >= t) : 0);
            i64 err = ((i64)bit << 16) - p;
            vw[0] += (i32)((xs0 * err) >> 14); vw[1] += (i32)((xs1 * err) >> 14); vw[2] += (i32)((256 * err) >> 14);
            ctr_upd(c0, bit, 255);
            if (c1) ctr_upd(*c1, bit, 60);
            if (!bit) { uu = t - 1; break; }
        }
        return uu;
    }
};

// ---------------------------------------------------------------------------------------------
template <class RC>
struct Engine {
    Params P;
    u32 m = 0, n = 0;
    const CanonicalMatrix* M = nullptr;           // ENC only
    std::vector<u8> isc_in;                       // ENC only
    std::vector<u16> label_in;                    // ENC only
    std::vector<u32> order_in;                    // ENC only: explicit coding order of called cells (column ids), order_mode 1
    // column headers
    std::vector<u32> kcol, nnzc;
    std::vector<u8> isc;
    std::vector<u16> labc;
    // output store (DEC)
    std::vector<u32> o_idx, o_val;
    std::vector<u64> o_off;
    std::vector<u32> o_len;

    static int qbucket(u32 k) {            // quarter-octave bucket of k
        if (k < 4) return (int)k;
        int l = ilog2u(k);
        return 4 * l + (int)((k >> (l - 2)) & 3) - 4;
    }
    static int kbucket4(u32 k) { return k < 4 ? 0 : (k < 16 ? 1 : (k < 64 ? 2 : 3)); }

    void run(RC& rc) {
        const Tabs& tb = TB();
        Timer th, tall_;
        auto TR = [&](const char* w) { if (std::getenv("I2_TRACE")) std::fprintf(stderr, "  [i2] %s @%.0f ms\n", w, tall_.ms()); };
        const u32 K = std::max(1u, P.K);
        // ===================== column headers (original column order) =====================
        NumCoder nk(1), nE(40);
        std::vector<BitM> flagm(256);
        SymCoder labc_coder(K > 1 ? ilog2u(K - 1) + 1 : 1, 1);
        kcol.assign(n, 0); nnzc.assign(n, 0); isc.assign(n, 0); labc.assign(n, 0);
        std::vector<u32> cells, empt;
        for (u32 j = 0; j < n; ++j) {
            u32 k = 0, nz = 0;
            if constexpr (RC::ENC) {
                u64 s = 0;
                for (u32 q = M->indptr[j]; q < M->indptr[j + 1]; ++q) s += M->data[q];
                k = (u32)s; nz = M->indptr[j + 1] - M->indptr[j];
            }
            rc.cat = C_HK;
            double c0 = 0; if constexpr (RC::ENC) c0 = rc.cost[C_HK];
            k = (u32)nk.code(rc, 0, k);
            double dk = 0; if constexpr (RC::ENC) dk = rc.cost[C_HK] - c0;
            kcol[j] = k;
            rc.cat = C_FLAG;
            int qb = qbucket(k); if (qb > 255) qb = 255;
            int f = code_bitm(rc, flagm[qb], RC::ENC ? isc_in[j] : 0, 5);
            isc[j] = (u8)f;
            if (f) {
                cells.push_back(j);
                if constexpr (RC::ENC) g_rep.hk_cells += dk;
                if (K > 1) { rc.cat = C_LAB; labc[j] = (u16)labc_coder.code(rc, 0, RC::ENC ? label_in[j] : 0); }
                rc.cat = C_NNZ;
                u32 E = k ? (RC::ENC ? k - nz : 0) : 0;
                if (k) E = (u32)nE.code(rc, std::min(ilog2u(k), 39), E);
                nnzc[j] = k ? k - E : 0;
            } else {
                empt.push_back(j);
                if constexpr (RC::ENC) g_rep.hk_empt += dk;
            }
        }
        g_rep.t_hdr = th.ms();
        const u32 nc = (u32)cells.size();
        std::vector<u32> cellord(n, 0xFFFFFFFFu);
        for (u32 i = 0; i < nc; ++i) cellord[cells[i]] = i;
        TR("headers");

        // ===================== coding order of the called cells =====================
        std::vector<u32> corder(cells);
        if (P.order_mode == 1) {
            rc.cat = C_PERM;
            Fenwick fw(nc);
            for (u32 i = 0; i < nc; ++i) {
                u32 rank = 0, idx = 0;
                if constexpr (RC::ENC) { idx = cellord[order_in[i]]; rank = fw.prefix(idx); }
                rank = rc.uniform(rank, nc - i);
                if constexpr (!RC::ENC) idx = fw.select(rank);
                fw.dec(idx);
                corder[i] = cells[idx];
            }
        } else if (P.order_mode == 0) {
            std::stable_sort(corder.begin(), corder.end(), [&](u32 a, u32 b) {
                if (labc[a] != labc[b]) return labc[a] < labc[b];
                return kcol[a] < kcol[b];
            });
        } else if (P.order_mode == 3) {
            std::stable_sort(corder.begin(), corder.end(), [&](u32 a, u32 b) { return kcol[a] < kcol[b]; });
        }   // 2: stored order

        // ===================== per-feature nnz over called cells (side info) =====================
        std::vector<u32> z(m, 0);
        if constexpr (RC::ENC) {
            for (u32 j : cells) for (u32 q = M->indptr[j]; q < M->indptr[j + 1]; ++q) ++z[M->indices[q]];
        }
        {
            rc.cat = C_Z;
            NumCoder nz(18);
            u32 prev = 0;
            for (u32 f = 0; f < m; ++f) {
                int ctx = prev ? std::min(1 + ilog2u(prev), 17) : 0;
                z[f] = (u32)nz.code(rc, ctx, z[f]);
                prev = z[f];
            }
        }
        TR("z coded");
        std::vector<u32> fid_of_rank, rank_of_fid(m, 0xFFFFFFFFu), zr;
        {
            u32 zmax = 0; for (u32 f = 0; f < m; ++f) zmax = std::max(zmax, z[f]);
            std::vector<u32> cnt(zmax + 2, 0);
            for (u32 f = 0; f < m; ++f) ++cnt[z[f]];
            std::vector<u64> start(zmax + 2, 0);
            u64 acc = 0;
            for (i64 v = zmax; v >= 1; --v) { start[v] = acc; acc += cnt[v]; }
            fid_of_rank.assign(acc, 0); zr.assign(acc, 0);
            for (u32 f = 0; f < m; ++f) if (z[f]) { u64 r = start[z[f]]++; fid_of_rank[r] = f; zr[r] = z[f]; rank_of_fid[f] = (u32)r; }
        }
        const u32 S = (u32)fid_of_rank.size();
        const u32 Tn = std::min(P.T, S);
        g_rep.S = S; g_rep.Tn = Tn; g_rep.ncells = nc; g_rep.nempt = (u32)empt.size();

        // ===================== Poisson fit =====================
        Timer tf;
        std::vector<u32> cnnz(nc);
        for (u32 i = 0; i < nc; ++i) cnnz[i] = nnzc[cells[i]];
        std::vector<i32> lw, ls;
        poisson_fit(zr, cnnz, (int)P.fit_rounds, lw, ls);
        g_rep.fit_ms = tf.ms();
        TR("fit");
        std::vector<i32> thr;
        {
            std::vector<i32> sorted(ls.begin(), ls.end());
            std::sort(sorted.begin(), sorted.end());
            for (u32 b = 1; b < P.NB; ++b) thr.push_back(nc ? sorted[(size_t)((u64)b * nc / P.NB)] : 0);
        }
        auto depth_bucket = [&](i32 l) { int b = 0; for (i32 t : thr) b += (l >= t); return b; };

        // ===================== cell model state =====================
        Timer tc;
        std::vector<u32> ctrA, ctrC, ema1, ema2, offC;
        std::vector<i32> offN;
        if (P.mask & 1) ctrA.assign((size_t)Tn * P.NB, 0);
        if (P.mask & 2) ctrC.assign((size_t)Tn * K, 0);
        if (P.mask & 4) ema1.assign(Tn, 0xFFFFFFFFu);
        if (P.mask & 8) ema2.assign(Tn, 0xFFFFFFFFu);
        if (P.mask & 16) offC.assign((size_t)Tn * K, 0x80000000u);      // (d + 2^23) << 8 | n
        if (P.mask & 32) offN.assign(Tn, 0);
        const int NSET = 16;
        const u32 NI = 8;      // 0 prior, 1 A, 2 C, 3 E1, 4 E2, 5 bias, 6 prior+offC, 7 prior+offN
        std::vector<i32> mw((size_t)NSET * NI, 0);
        {
            int en[8] = {1, (P.mask & 1) != 0, (P.mask & 2) != 0, (P.mask & 4) != 0, (P.mask & 8) != 0, 0, (P.mask & 16) != 0, (P.mask & 32) != 0};
            int ne = 0; for (int i = 0; i < 8; ++i) ne += en[i];
            for (int s = 0; s < NSET; ++s)
                for (u32 i = 0; i < NI; ++i)
                    if (en[i]) mw[(size_t)s * NI + i] = (i32)(65536 / ne);
        }
        const int NVB = 64;
        ValModel vm(NVB, P.val_mode ? Tn : 0);
        for (int b = 0; b < NVB; ++b) {
            i64 Lq = (i64)(b * 512 + 256 - Tabs::OFF) * (1 << 22);
            for (int t = 0; t < ValModel::NVT; ++t) vm.ctr[(size_t)b * ValModel::NVT + t] = ctr_init(ztp_p16(Lq, t + 1), 4);
        }
        // tail mass table over ranks [Tn, S)
        std::vector<u64> Qt(S - Tn + 1, 0);
        for (u32 r = S; r > Tn; --r) {
            u128 e = exp_q64((i64)lw[r - 1] * (1 << 22));
            u64 w = (u64)(e >> 24);
            if (w < 1) w = 1;
            if (w > (1ull << 46)) w = 1ull << 46;
            Qt[r - 1 - Tn] = Qt[r - Tn] + w;
        }
        TR("model init");

        std::vector<u32> st_idx, st_val;                 // per column concatenated (original feature ids)
        std::vector<u64> st_off(n, 0);
        std::vector<u32> st_len(n, 0);
        if constexpr (!RC::ENC) { st_idx.reserve(1 << 20); st_val.reserve(1 << 20); }

        std::vector<std::pair<u32, u32>> ent;
        std::vector<u32> hits, hval;
        const int L_off = Tabs::OFF, NPT = Tabs::NPT;
        auto L_index = [&](i64 v) { return (int)(v < 0 ? 0 : (v >= NPT ? NPT - 1 : v)); };

        for (u32 ci = 0; ci < nc; ++ci) {
            u32 j = corder[ci];
            u32 nnz = nnzc[j];
            i32 lsj = ls[cellord[j]];
            int bkt = depth_bucket(lsj);
            u32 lab = labc[j];
            if (nnz == 0) { st_off[j] = st_idx.size(); st_len[j] = 0; continue; }
            if constexpr (RC::ENC) {
                ent.clear();
                for (u32 q = M->indptr[j]; q < M->indptr[j + 1]; ++q) ent.emplace_back(rank_of_fid[M->indices[q]], M->data[q]);
                std::sort(ent.begin(), ent.end());
            }
            hits.clear(); hval.clear();
            size_t ei = 0;
            u32 h = 0;
            rc.cat = C_TOP;
            for (u32 r = 0; r < Tn; ++r) {
                if (h == nnz) break;
                int bit_in = 0;
                if constexpr (RC::ENC) bit_in = (ei < ent.size() && ent[ei].first == r);
                u32 p0q = tb.ptab[L_index((i64)lw[r] + lsj + L_off)];
                int p0 = (int)(p0q >> 16); if (p0 < 1) p0 = 1;
                int p;
                u32 *pa = nullptr, *pc = nullptr, *pe1 = nullptr, *pe2 = nullptr, *po = nullptr;
                i32* pn = nullptr;
                int x[8]; i32* w = nullptr;
                int x0 = 0;
                if (P.mask == 0) {
                    p = p0;
                } else {
                    x0 = stretch(p0);
                    x[0] = x0; x[1] = x[2] = x[3] = x[4] = x[6] = x[7] = 0; x[5] = 256;
                    if (P.mask & 1) { pa = &ctrA[(size_t)r * P.NB + bkt]; if (!*pa) *pa = ctr_init(p0, P.n0A); x[1] = stretch(ctr_p16(*pa)); }
                    if (P.mask & 2) { pc = &ctrC[(size_t)lab * Tn + r]; if (!*pc) *pc = ctr_init(p0, P.n0C); x[2] = stretch(ctr_p16(*pc)); }
                    if (P.mask & 4) { pe1 = &ema1[r]; if (*pe1 == 0xFFFFFFFFu) *pe1 = (u32)p0 << 6; x[3] = stretch((int)(*pe1 >> 6)); }
                    if (P.mask & 8) { pe2 = &ema2[r]; if (*pe2 == 0xFFFFFFFFu) *pe2 = (u32)p0 << 6; x[4] = stretch((int)(*pe2 >> 6)); }
                    if (P.mask & 16) { po = &offC[(size_t)lab * Tn + r]; i32 d = (i32)(*po >> 8) - (1 << 23); x[6] = std::clamp(x0 + d, -Tabs::XS, Tabs::XS); }
                    if (P.mask & 32) { pn = &offN[r]; x[7] = std::clamp(x0 + *pn, -Tabs::XS, Tabs::XS); }
                    int ws = std::min(ilog2u(r + 1), NSET - 1);
                    w = &mw[(size_t)ws * NI];
                    i64 dot = 0;
                    for (u32 i = 0; i < NI; ++i) dot += (i64)w[i] * x[i];
                    p = squash((int)(dot >> 16));
                }
                int bit = rc.bit(p, bit_in);
                if (P.mask) {
                    i64 err = ((i64)bit << 16) - p;
                    for (u32 i = 0; i < NI; ++i) w[i] += (i32)((x[i] * err) >> P.mixshift);
                    if (pa) ctr_upd(*pa, bit, (int)P.limA);
                    if (pc) ctr_upd(*pc, bit, (int)P.limC);
                    if (pe1) { i32 v = (i32)*pe1; v += (((bit ? 4194303 : 0) - v) >> P.ema1); *pe1 = (u32)v; }
                    if (pe2) { i32 v = (i32)*pe2; v += (((bit ? 4194303 : 0) - v) >> P.ema2); *pe2 = (u32)v; }
                    if (po) {
                        u32 nn_ = *po & 255; i32 d = (i32)(*po >> 8) - (1 << 23);
                        i64 e6 = ((i64)bit << 16) - squash(x[6]);
                        if (P.offshC) d += (i32)(e6 >> P.offshC);
                        else d += (i32)((e6 * tb.rec[nn_]) >> 24);
                        d = std::clamp(d, -4096, 4096);
                        if (nn_ < 200) ++nn_;
                        *po = ((u32)(d + (1 << 23)) << 8) | nn_;
                    }
                    if (pn) { i64 e7 = ((i64)bit << 16) - squash(x[7]); *pn = std::clamp(*pn + (i32)(e7 >> P.offshN), -4096, 4096); }
                }
                if (bit) { hits.push_back(r); ++h; if constexpr (RC::ENC) ++ei; }
            }
            rc.cat = C_TAIL;
            u32 Ntail = nnz - h;
            if (Ntail) {
                u32 lo = 0, end = S - Tn;
                for (u32 i = 0; i < Ntail; ++i) {
                    u32 xin = 0;
                    if constexpr (RC::ENC) xin = ent[ei + i].first - Tn;
                    u32 x_ = code_orderstat(rc, Qt.data(), lo, end, Ntail - i, xin);
                    hits.push_back(x_ + Tn);
                    lo = x_ + 1;
                }
            }
            rc.cat = C_VAL;
            hval.assign(nnz, 1);
            {
                u32 R = kcol[j] - nnz;
                for (u32 i = 0; i < nnz; ++i) {
                    u32 rk = hits[i];
                    u32 vin = 1;
                    if constexpr (RC::ENC) vin = ent[i].second;
                    if (i == nnz - 1) { hval[i] = 1 + R; R = 0; continue; }
                    if (R == 0) { hval[i] = 1; continue; }
                    i64 Lv = (i64)lw[rk] + lsj + L_off;
                    int bq = (int)(Lv < 0 ? 0 : Lv >> 9); if (bq >= NVB) bq = NVB - 1;
                    u32 u = vm.code(rc, bq, (P.val_mode && rk < Tn) ? (i64)rk : -1, R, vin - 1);
                    hval[i] = 1 + u;
                    R -= u;
                }
            }
            if constexpr (!RC::ENC) {
                st_off[j] = st_idx.size(); st_len[j] = nnz;
                std::vector<std::pair<u32, u32>> tmp(nnz);
                for (u32 i = 0; i < nnz; ++i) tmp[i] = {fid_of_rank[hits[i]], hval[i]};
                std::sort(tmp.begin(), tmp.end());
                for (auto& pr : tmp) { st_idx.push_back(pr.first); st_val.push_back(pr.second); }
            }
        }
        g_rep.t_cells = tc.ms();
        TR("cells");

        // ===================== empties =====================
        Timer te;
        code_empties(rc, cells, empt, K, st_idx, st_val, st_off, st_len);
        g_rep.t_empt = te.ms();
        TR("empties");

        if constexpr (!RC::ENC) {
            o_off.assign(n, 0); o_len = st_len;
            u64 tot = 0;
            for (u32 j = 0; j < n; ++j) { o_off[j] = tot; tot += st_len[j]; }
            o_idx.resize(tot); o_val.resize(tot);
            for (u32 j = 0; j < n; ++j) {
                std::memcpy(&o_idx[o_off[j]], &st_idx[st_off[j]], st_len[j] * 4);
                std::memcpy(&o_val[o_off[j]], &st_val[st_off[j]], st_len[j] * 4);
            }
        }
    }

    // ---- empties: ambient profile from the decoded called cells, droplet = multiset of k draws (exact multinomial)
    void code_empties(RC& rc, const std::vector<u32>& cells, const std::vector<u32>& empt, u32 K,
                      std::vector<u32>& st_idx, std::vector<u32>& st_val, std::vector<u64>& st_off, std::vector<u32>& st_len) {
        const u32 mode = P.amb_mode;
        // per-cluster and pooled cell profiles
        const u32 KP = (mode == 2) ? K : 1;
        std::vector<u64> ucell((size_t)KP * m, 0), utot(KP, 0);
        auto add_cell = [&](u32 j, u32 f, u32 v) { u32 c = (mode == 2) ? labc[j] : 0; ucell[(size_t)c * m + f] += v; utot[c] += v; };
        if constexpr (RC::ENC) {
            for (u32 j : cells) for (u32 q = M->indptr[j]; q < M->indptr[j + 1]; ++q) add_cell(j, M->indices[q], M->data[q]);
        } else {
            for (u32 j : cells) for (u32 q = 0; q < st_len[j]; ++q) add_cell(j, st_idx[st_off[j] + q], st_val[st_off[j] + q]);
        }
        u64 Utot = 0; for (u32 c = 0; c < KP; ++c) Utot += utot[c];
        const u64 delta8 = P.amb_delta_x8;
        // ambient prior (pooled cells profile), Q.40
        std::vector<u64> base(m);
        {
            u128 denom = (u128)Utot * 8 + (u128)m * delta8;
            for (u32 f = 0; f < m; ++f) {
                u64 u = 0; for (u32 c = 0; c < KP; ++c) u += ucell[(size_t)c * m + f];
                base[f] = (u64)((((u128)u * 8 + delta8) << 40) / denom);
            }
        }
        // cluster profiles (mode 2), Q.40
        std::vector<u64> cprof;
        if (mode == 2) {
            cprof.assign((size_t)K * m, 0);
            for (u32 c = 0; c < K; ++c) {
                u128 denom = (u128)utot[c] * 8 + (u128)m * delta8;
                for (u32 f = 0; f < m; ++f) cprof[(size_t)c * m + f] = (u64)((((u128)ucell[(size_t)c * m + f] * 8 + delta8) << 40) / denom);
            }
        }
        const u32 NQ = (mode == 1) ? 4 : (mode == 2 ? K + 1 : 1);     // number of prefix-mass tables
        std::vector<u64> Ef(m, 0), Efb;                                  // decoded empty-droplet UMIs per feature (+ per depth bucket)
        if (mode == 1) Efb.assign((size_t)4 * m, 0);
        std::vector<u64> Q((size_t)NQ * (m + 1), 0);
        u64 Etot = 0, pending = 0;
        auto rebuild = [&]() {
            for (u32 t = 0; t < NQ; ++t) {
                u64* q = &Q[(size_t)t * (m + 1)];
                q[m] = 0;
                if (mode == 2 && t > 0) {
                    const u64* cp = &cprof[(size_t)(t - 1) * m];
                    u64 scale = ((Etot + P.amb_gamma) * P.amb_mu_x8) >> 3;        // cluster mass in UMI units
                    for (u32 f = m; f > 0; --f) {
                        u64 w = (u64)P.amb_gamma * base[f - 1] + (Ef[f - 1] << 40) + scale * cp[f - 1];
                        if (w < 1) w = 1;
                        q[f - 1] = q[f] + w;
                    }
                } else if (mode == 1) {
                    const u64* eb = &Efb[(size_t)t * m];
                    for (u32 f = m; f > 0; --f) {
                        u64 w = (u64)P.amb_gamma * base[f - 1] + (Ef[f - 1] << 40) + ((u64)P.amb_beta * eb[f - 1] << 40);
                        if (w < 1) w = 1;
                        q[f - 1] = q[f] + w;
                    }
                } else {
                    for (u32 f = m; f > 0; --f) {
                        u64 w = (u64)P.amb_gamma * base[f - 1] + (Ef[f - 1] << 40);
                        if (w < 1) w = 1;
                        q[f - 1] = q[f] + w;
                    }
                }
            }
            pending = 0;
        };
        rebuild();
        SymCoder dlab(mode == 2 ? ilog2u(K) + 1 : 1, 4);     // droplet label, context = depth bucket
        NumCoder dnz(40);                                      // mode 3: excess k - nnz per droplet
        ValModel vm3(48, 0);
        for (int c = 0; c < 48; ++c) for (int t = 0; t < ValModel::NVT; ++t) vm3.ctr[(size_t)c * ValModel::NVT + t] = ctr_init(2048, 2);
        std::vector<u32> xs, cnts;
        for (u32 j : empt) {
            u32 k = kcol[j];
            if constexpr (!RC::ENC) { st_off[j] = st_idx.size(); st_len[j] = 0; }
            if (!k) continue;
            int kb = kbucket4(k);
            u32 tsel = 0;
            if (mode == 1) tsel = (u32)kb;
            if (mode == 2) {
                u32 lab = 0;
                if constexpr (RC::ENC) {
                    double best = -1e300; u32 bl = 0;
                    for (u32 t = 0; t < NQ; ++t) {
                        const u64* q = &Q[(size_t)t * (m + 1)];
                        double ll = 0, lw = std::log2((double)q[0]);
                        for (u32 qq = M->indptr[j]; qq < M->indptr[j + 1]; ++qq) {
                            u32 f = M->indices[qq];
                            ll += M->data[qq] * (std::log2((double)(q[f] - q[f + 1])) - lw);
                        }
                        if (t) ll -= std::log2((double)NQ) + 1.0;
                        if (ll > best) { best = ll; bl = t; }
                    }
                    lab = bl;
                }
                rc.cat = C_ELAB;
                lab = dlab.code(rc, kb, lab);
                tsel = lab;
            }
            const u64* q = &Q[(size_t)tsel * (m + 1)];
            if (mode == 3) {
                rc.cat = C_ELAB;
                u32 nz = 0;
                if constexpr (RC::ENC) nz = M->indptr[j + 1] - M->indptr[j];
                u32 E = (u32)dnz.code(rc, std::min(ilog2u(k), 39), RC::ENC ? k - nz : 0);
                nz = k - E;
                xs.assign(nz, 0); cnts.assign(nz, 1);
                rc.cat = C_EMPT;
                u32 lo = 0;
                for (u32 i = 0; i < nz; ++i) {
                    u32 xin = 0;
                    if constexpr (RC::ENC) xin = M->indices[M->indptr[j] + i];
                    u32 x_ = code_orderstat(rc, q, lo, m, nz - i, xin);
                    xs[i] = x_; lo = x_ + 1;
                }
                rc.cat = C_EVAL;
                u32 R = E;
                int lW = ilog2u(q[0]);
                for (u32 i = 0; i < nz; ++i) {
                    if (i == nz - 1) { cnts[i] = 1 + R; R = 0; break; }
                    if (R == 0) break;
                    u64 w = q[xs[i]] - q[xs[i] + 1];
                    int lq = ilog2u((u64)k * w) - lW;             // log2(k * p_f), octave resolution
                    int ctx = std::clamp(lq + 36, 0, 47);
                    u32 vin = 1;
                    if constexpr (RC::ENC) vin = M->data[M->indptr[j] + i];
                    u32 u = vm3.code(rc, ctx, -1, R, vin - 1);
                    cnts[i] = 1 + u; R -= u;
                }
                if constexpr (!RC::ENC) {
                    for (u32 i = 0; i < nz; ++i) { st_idx.push_back(xs[i]); st_val.push_back(cnts[i]); }
                    st_len[j] = nz;
                }
                if (P.amb_adapt) {
                    for (u32 i = 0; i < nz; ++i) { Ef[xs[i]] += cnts[i]; if (mode == 1) Efb[(size_t)kb * m + xs[i]] += cnts[i]; }
                    Etot += k; pending += k;
                    if (pending >= std::max<u64>(2048, Etot / 4)) rebuild();
                }
                continue;
            }
            rc.cat = C_EMPT;
            xs.clear();
            if constexpr (RC::ENC) {
                for (u32 qq = M->indptr[j]; qq < M->indptr[j + 1]; ++qq) for (u32 c = 0; c < M->data[qq]; ++c) xs.push_back(M->indices[qq]);
            } else xs.assign(k, 0);
            u32 lo = 0;
            for (u32 i = 0; i < k; ++i) {
                u32 x_ = code_orderstat(rc, q, lo, m, k - i, RC::ENC ? xs[i] : 0);
                xs[i] = x_; lo = x_;
            }
            if constexpr (!RC::ENC) {
                u32 cnt = 0;
                for (u32 i = 0; i < k; ++i) {
                    if (i + 1 < k && xs[i + 1] == xs[i]) { ++cnt; continue; }
                    st_idx.push_back(xs[i]); st_val.push_back(cnt + 1); ++st_len[j]; cnt = 0;
                }
            }
            if (P.amb_adapt) {
                for (u32 i = 0; i < k; ++i) { Ef[xs[i]] += 1; if (mode == 1) Efb[(size_t)kb * m + xs[i]] += 1; }
                Etot += k; pending += k;
                if (pending >= std::max<u64>(2048, Etot / 4)) rebuild();
            }
        }
    }
};

// ---------------------------------------------------------------------------------------------
// encoder-side labelling and ordering of the called cells (floats allowed: the result is transmitted)
inline void assign_order(const CanonicalMatrix& M, const Params& P, const std::vector<u8>& isc, std::vector<u16>& label,
                         std::vector<u32>& order) {
    const u32 K = std::max(1u, P.K);
    if (K <= 1 && P.order_mode != 1) return;
    Embedding E = embed_cells(M, isc);
    std::vector<u16> lab = (K > 1) ? kmeans_labels(E, (int)K) : std::vector<u16>(E.cells.size(), 0);
    for (size_t i = 0; i < E.cells.size(); ++i) label[E.cells[i]] = lab[i];
    if (P.order_mode == 1) {
        order.clear();
        for (u32 c = 0; c < K; ++c) {
            std::vector<size_t> mem;
            for (size_t i = 0; i < E.cells.size(); ++i) if (lab[i] == c) mem.push_back(i);
            auto ch = nn_chain(E, mem);
            for (size_t i : ch) order.push_back(E.cells[i]);
        }
    }
}

inline std::vector<u8> encode_idea2(const CanonicalMatrix& M, const Params& P0) {
    using namespace detail;
    Params P = P0;
    std::vector<u8> o;
    const char magic[4] = {'I', '2', 'v', '2'};
    o.insert(o.end(), magic, magic + 4);
    put_u32(o, M.m); put_u32(o, M.n); put_u64(o, M.nnz); o.push_back(M.vt_code);
    {
        u32 raw[NPARAM]; params_pack(P, raw);
        for (int i = 0; i < NPARAM; ++i) put_u32(o, raw[i]);
    }
    size_t hdr = o.size();
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 3);
    put_stream(o, cn.data(), cn.size(), 3);
    std::vector<u8> kv;
    for (auto& [k, v] : M.user_kv) { kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0); kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0); }
    put_u32(o, (u32)M.user_kv.size());
    put_stream(o, kv.data(), kv.size(), 3);
    size_t names_end = o.size();

    Engine<RcEnc> E;
    E.P = P; E.m = M.m; E.n = M.n; E.M = &M;
    E.isc_in.assign(M.n, 0);
    E.label_in.assign(M.n, 0);
    {
        std::ifstream f(g_calls_path);
        if (!f) throw std::runtime_error("idea2: cannot open cell_calls.tsv at '" + g_calls_path + "'");
        std::unordered_set<std::string> s;
        std::string line; std::getline(f, line);
        while (std::getline(f, line)) { auto t = line.find('\t'); s.insert(line.substr(0, t)); }
        for (u32 j = 0; j < M.n; ++j) E.isc_in[j] = s.count(M.colnames[j]) ? 1 : 0;
    }
    g_rep = Report();
    Timer t_all;
    assign_order(M, P, E.isc_in, E.label_in, E.order_in);
    g_rep.t_order = t_all.ms();
    std::vector<u8> pay;
    RcEnc rc(pay);
    E.run(rc);
    rc.finish();
    if (std::getenv("I2_TRACE")) std::fprintf(stderr, "[i2] run %.0f ms\n", t_all.ms());
    put_u64(o, pay.size());
    o.insert(o.end(), pay.begin(), pay.end());
    for (int c = 0; c < 24; ++c) g_rep.bits[c] = rc.cost[c];
    g_rep.names_bytes = names_end - hdr; g_rep.hdr_bytes = hdr; g_rep.payload_bytes = pay.size();
    for (u32 j = 0; j < M.n; ++j) {
        u64 nz = M.indptr[j + 1] - M.indptr[j];
        if (E.isc_in[j]) g_rep.nnz_cells += nz; else g_rep.nnz_empt += nz;
    }
    return o;
}

inline CanonicalMatrix decode_idea2(const std::vector<u8>& b) {
    using namespace detail;
    const u8* p = b.data();
    if (std::memcmp(p, "I2v2", 4) != 0) throw std::runtime_error("idea2: bad magic");
    p += 4;
    CanonicalMatrix M;
    M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
    u32 raw[NPARAM];
    for (int i = 0; i < NPARAM; ++i) raw[i] = get_u32(p);
    Params P = params_unpack(raw);
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    u32 nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * (size_t)nkv);
    for (u32 i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    u64 plen = get_u64(p);
    RcDec rc(p, p + plen);
    Engine<RcDec> E;
    E.P = P; E.m = M.m; E.n = M.n;
    E.run(rc);
    M.indptr.assign(M.n + 1, 0);
    for (u32 j = 0; j < M.n; ++j) M.indptr[j + 1] = M.indptr[j] + E.o_len[j];
    M.indices = std::move(E.o_idx); M.data = std::move(E.o_val);
    return M;
}

inline Codec make_idea2(const std::string& name, Params P, bool from_env = false) {
    Codec c;
    c.name = name;
    c.encode = [P, from_env](const CanonicalMatrix& M) { return encode_idea2(M, from_env ? params_from_env(P) : P); };
    c.decode = [](const std::vector<u8>& b) { return decode_idea2(b); };
    return c;
}

}}  // namespace cs::i2
