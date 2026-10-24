// SPDX-License-Identifier: MIT
// Idea 4 codec assembly (encoder fits, column coders, stream layout). See idea4.h for the model.
#pragma once
#include "idea4_pp.h"
#include "ref_zstd_csc.h"
#include <chrono>
#include <unordered_map>

namespace cs { namespace ipp {

inline double wall_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
#ifdef _WIN32
}}
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
namespace cs { namespace ipp {
// process CPU time (user+kernel) in ms: robust against the other agents sharing the machine
inline double now_ms() {
    FILETIME c, e, k, u;
    GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u);
    auto v = [](FILETIME f) { return (double)(((unsigned long long)f.dwHighDateTime << 32) | f.dwLowDateTime) / 10000.0; };
    return v(k) + v(u);
}
#else
inline double now_ms() { return wall_ms(); }
#endif
inline const CanonicalMatrix*& dbg_ref() { static const CanonicalMatrix* p = nullptr; return p; }
inline std::string& calls_path() { static std::string p; return p; }
inline double env_d(const char* k, double d) { const char* v = std::getenv(k); return v ? std::atof(v) : d; }

// ---------------------------------------------------------------------------------------------
// Ambient profile (empty droplets): decoder-derived from decoded cells and decoded empties
// ---------------------------------------------------------------------------------------------
struct AmbProfile {
    const Model* M = nullptr;
    std::vector<u32> Acell, Aemp;
    u64 totCell = 0, totEmp = 0;
    std::vector<u64> CAn;
    u64 kc = 20000, k0 = 2000;
    void init(const Model& m_) { M = &m_; Acell.assign(M->m, 0); Aemp.assign(M->m, 0); CAn.assign(M->m, 0); totCell = totEmp = 0; }
    void add_cell(const u32* idx, const u32* val, size_t n) { for (size_t t = 0; t < n; ++t) { Acell[idx[t]] += val[t]; totCell += val[t]; } }
    void add_emp(const u32* idx, const u32* val, size_t n) { for (size_t t = 0; t < n; ++t) { Aemp[idx[t]] += val[t]; totEmp += val[t]; } }
    void rebuild() {
        const u32 m = M->m;
        const u64 S = (u64)1 << 24;
        u64 kcS = kc * S, k0S = k0 * S;
        std::vector<u64> ca(m);
        u64 cum = 0;
        for (u32 j = 0; j < m; ++j) {
            u64 a = (u64)Aemp[j] * S;
            if (totCell) a += (u64)(((u128)Acell[j] * kcS) / totCell);
            a += (u64)(((u128)M->P0[j] * k0S) >> 62);
            cum += a; ca[j] = cum;
        }
        if (cum == 0) { ca[m - 1] = 1; cum = 1; for (u32 j = 0; j + 1 < m; ++j) ca[j] = 0; }
        for (u32 j = 0; j < m; ++j) CAn[j] = (u64)((((u128)ca[j]) << 56) / cum);
        CAn[m - 1] = (u64)1 << 56;
    }
};

// ---------------------------------------------------------------------------------------------
// Adaptive base profile for called cells: gene profile and interval split re-estimated by the decoder from
// the called cells decoded so far (counts + pseudo-counts toward the shipped model), rebuilt on a doubling schedule.
// ---------------------------------------------------------------------------------------------
struct AdaptBase {
    bool on = false; u64 kg = 200, kq = 3; u32 since = 0, block = 32; bool built = false;
    std::vector<u32> Cg, Ci; std::vector<u64> pi0q; std::vector<u32> qcum; u64 tot = 0;
    void init(const Model& M, bool on_, u64 kg_, u64 kq_) {
        on = on_; kg = kg_; kq = kq_; since = 0; block = 32; built = false; tot = 0;
        if (on) { Cg.assign(M.G, 0); Ci.assign(M.m, 0); pi0q.assign(M.G, 0); qcum.assign(M.m, 0); }
    }
    void add(const Model& M, const u32* idx, const u32* val, size_t n) {
        if (!on) return;
        for (size_t t = 0; t < n; ++t) { Ci[idx[t]] += val[t]; Cg[M.gene_of[idx[t]]] += val[t]; tot += val[t]; }
        ++since;
    }
    bool maybe_rebuild(const Model& M) {
        if (!on || since < block || tot == 0) return false;
        rebuild(M); since = 0; block = std::min<u32>(1024, block * 2); built = true; return true;
    }
    void rebuild(const Model& M) {
        const u32 G = M.G;
        std::vector<u128> w(G); u128 W = 0;
        for (u32 g = 0; g < G; ++g) { w[g] = ((u128)Cg[g] << 30) + (u128)(M.pi0q[g] >> 32) * kg; W += w[g]; }
        for (u32 g = 0; g < G; ++g) pi0q[g] = (u64)((w[g] << 62) / W);
        for (u32 g = 0; g < G; ++g) {
            u32 a = M.gstart[g], b = M.gstart[g + 1];
            u128 V = 0; std::vector<u128> v(b - a);
            for (u32 j = a; j < b; ++j) {
                u32 qlo = (j == a) ? 0 : M.qcum[j - 1];
                v[j - a] = ((u128)Ci[j] << 30) + (((u128)(M.qcum[j] - qlo) * kq) >> 1);
                V += v[j - a];
            }
            u128 cum = 0;
            for (u32 j = a; j < b; ++j) {
                cum += v[j - a];
                qcum[j] = (j + 1 == b) ? (u32)((u64)1 << 31) : (u32)((cum << 31) / V);
            }
        }
    }
    const u64* pi0q_ptr() const { return built ? pi0q.data() : nullptr; }
    const u32* qcum_ptr() const { return built ? qcum.data() : nullptr; }
};

// ---------------------------------------------------------------------------------------------
// Encoder-side float helpers
// ---------------------------------------------------------------------------------------------
struct Col { const u32* idx; const u32* val; size_t n; u64 N; };

struct Fitter {
    const Model& M;
    std::vector<float> qf;         // interval split q (float), encoder only
    std::vector<float> p0f;        // static interval profile (float)
    std::vector<float> phi_cur;    // topic profiles currently in use (shipped or rebased on the adaptive base)
    explicit Fitter(const Model& m_) : M(m_) {
        phi_cur = M.phi;
        qf.resize(M.m); p0f.resize(M.m);
        for (u32 j = 0; j < M.m; ++j) {
            u32 g = M.gene_of[j]; u32 lo = (j == M.gstart[g]) ? 0 : M.qcum[j - 1];
            qf[j] = (float)((double)(M.qcum[j] - lo) / 2147483648.0);
            p0f[j] = (float)((double)M.P0[j] / 4611686018427387904.0);
        }
    }
    void rebase(const AdaptBase& ab) {
        const u32 G = M.G, K = M.K;
        if (!K) return;
        if (!ab.built) { phi_cur = M.phi; return; }
        std::vector<double> sums(K, 0.0);
        for (u32 g = 0; g < G; ++g) {
            double base = (double)ab.pi0q[g] / 4611686018427387904.0;
            for (u32 k = 0; k < K; ++k) {
                double c = M.Rcode[(size_t)k * G + g];
                double v = base * (double)M.tscale[k] * c * c / 65536.0;
                phi_cur[(size_t)g * K + k] = (float)v; sums[k] += v;
            }
        }
        for (u32 g = 0; g < G; ++g) for (u32 k = 0; k < K; ++k) phi_cur[(size_t)g * K + k] /= (float)sums[k];
    }
    void agg(const Col& c, std::vector<u32>& g, std::vector<float>& x) const {
        g.clear(); x.clear();
        for (size_t t = 0; t < c.n; ++t) {
            u32 gg = M.gene_of[c.idx[t]];
            if (!g.empty() && g.back() == gg) x.back() += (float)c.val[t]; else { g.push_back(gg); x.push_back((float)c.val[t]); }
        }
    }
    void em(const std::vector<u32>& g, const std::vector<float>& x, std::vector<float>& th, int iters) const {
        const u32 K = M.K;
        th.assign(K, 1.f / K);
        std::vector<float> tn(K);
        for (int it = 0; it < iters; ++it) {
            std::fill(tn.begin(), tn.end(), 0.f);
            for (size_t t = 0; t < g.size(); ++t) {
                const float* ph = &phi_cur[(size_t)g[t] * K];
                float p = 0; for (u32 k = 0; k < K; ++k) p += th[k] * ph[k];
                float r = x[t] / std::max(p, 1e-30f);
                for (u32 k = 0; k < K; ++k) tn[k] += r * ph[k];
            }
            float s = 0; for (u32 k = 0; k < K; ++k) { tn[k] *= th[k]; s += tn[k]; }
            for (u32 k = 0; k < K; ++k) th[k] = tn[k] / std::max(s, 1e-30f);
        }
    }
    // log-likelihood (nats) of gene-level counts under theta given as quantised level codes
    double ll_tq(const std::vector<u32>& g, const std::vector<float>& x, const uint8_t* codes) const {
        const u32 K = M.K; float w[64]; float s = 0;
        for (u32 k = 0; k < K; ++k) { w[k] = (float)LEVEL_TQ[codes[k]]; s += w[k]; }
        if (s <= 0) return -1e30;
        double ll = 0;
        for (size_t t = 0; t < g.size(); ++t) {
            const float* ph = &phi_cur[(size_t)g[t] * K];
            float p = 0; for (u32 k = 0; k < K; ++k) p += w[k] * ph[k];
            ll += (double)x[t] * std::log(std::max((double)(p / s), 1e-30));
        }
        return ll;
    }
    // Rate-distortion aware quantisation: minimise  -log2 L(theta codes) + side-information bits under the live adaptive
    // models, starting from the better of {rounded EM fit, per-component modal codes}; coordinate moves
    // {c-1, c+1, 0, modal}.  Encoder-only (the stream format is unchanged).
    template <class SM>
    void quantise_rd(const std::vector<float>& th, const std::vector<u32>& g, const std::vector<float>& x, uint8_t* codes, SM& sm, int passes) const {
        const u32 K = M.K; const size_t ng = g.size();
        uint8_t cA[64], cB[64];
        quantise(th, g, x, cA, false);
        bool anyB = false;
        for (u32 k = 0; k < K; ++k) { cB[k] = (uint8_t)sm.th[k].mode(); anyB |= cB[k] != 0; }
        if (!anyB) for (u32 k = 0; k < K; ++k) cB[k] = cA[k];
        std::vector<float> pv(ng);
        double sw = 0;
        auto load = [&](const uint8_t* c) {
            sw = 0; for (u32 k = 0; k < K; ++k) sw += LEVEL_TQ[c[k]];
            for (size_t t = 0; t < ng; ++t) {
                const float* ph = &phi_cur[(size_t)g[t] * K]; float p = 0;
                for (u32 k = 0; k < K; ++k) p += (float)LEVEL_TQ[c[k]] * ph[k];
                pv[t] = p;
            }
        };
        auto ll_of = [&](double s_) { double ll = 0; for (size_t t = 0; t < ng; ++t) ll += (double)x[t] * std::log(std::max((double)pv[t] / s_, 1e-30)); return ll; };
        auto side_of = [&](const uint8_t* c) { double b = 0; for (u32 k = 0; k < K; ++k) b += sm.th[k].bits(c[k]); return b; };
        const double L2E = 1.4426950408889634;
        double jA, jB;
        load(cA); jA = (sw > 0) ? -ll_of(sw) * L2E + side_of(cA) : 1e300;
        load(cB); jB = (sw > 0) ? -ll_of(sw) * L2E + side_of(cB) : 1e300;
        const uint8_t* start = (jA <= jB) ? cA : cB; double J = std::min(jA, jB);
        for (u32 k = 0; k < K; ++k) codes[k] = start[k];
        load(codes);
        double sideTot = side_of(codes);
        for (int pass = 0; pass < passes; ++pass) {
            bool any = false;
            for (u32 k = 0; k < K; ++k) {
                int cur = codes[k]; double bk = sm.th[k].bits(cur);
                int cands[4] = {cur - 1, cur + 1, 0, (int)cB[k]};
                double bestJ = J - 1e-7; int bestC = -1;
                for (int ci = 0; ci < 4; ++ci) {
                    int cnd = cands[ci];
                    if (cnd < 0 || cnd > 15 || cnd == cur) continue;
                    double dw = (double)LEVEL_TQ[cnd] - (double)LEVEL_TQ[cur];
                    double s2 = sw + dw; if (s2 <= 0) continue;
                    double ll = 0;
                    for (size_t t = 0; t < ng; ++t) ll += (double)x[t] * std::log(std::max(((double)pv[t] + dw * phi_cur[(size_t)g[t] * K + k]) / s2, 1e-30));
                    double Jn = -ll * L2E + (sideTot - bk + sm.th[k].bits(cnd));
                    if (Jn < bestJ) { bestJ = Jn; bestC = cnd; }
                }
                if (bestC >= 0) {
                    double dw = (double)LEVEL_TQ[bestC] - (double)LEVEL_TQ[cur];
                    for (size_t t = 0; t < ng; ++t) pv[t] += (float)(dw * phi_cur[(size_t)g[t] * K + k]);
                    sw += dw; sideTot += sm.th[k].bits(bestC) - bk; codes[k] = (uint8_t)bestC; J = bestJ; any = true;
                }
            }
            if (!any) break;
        }
    }
    void quantise(const std::vector<float>& th, const std::vector<u32>& g, const std::vector<float>& x, uint8_t* codes, bool refine) const {
        const u32 K = M.K;
        float mx = 0; for (u32 k = 0; k < K; ++k) mx = std::max(mx, th[k]);
        for (u32 k = 0; k < K; ++k) {
            double v = th[k] / mx * 256.0;
            int c = 0;
            if (v >= 1.41421356) { c = (int)std::lround(2.0 * std::log2(v / 2.0)) + 1; if (c < 1) c = 1; if (c > 15) c = 15; }
            codes[k] = (uint8_t)c;
        }
        if (!refine) return;
        double best = ll_tq(g, x, codes);
        for (int pass = 0; pass < 2; ++pass) {
            bool any = false;
            for (u32 k = 0; k < K; ++k) {
                for (int d = -1; d <= 1; d += 2) {
                    int c = (int)codes[k] + d; if (c < 0 || c > 15) continue;
                    uint8_t old = codes[k]; codes[k] = (uint8_t)c;
                    double ll = ll_tq(g, x, codes);
                    if (ll > best + 1e-9) { best = ll; any = true; } else codes[k] = old;
                }
            }
            if (!any) break;
        }
    }
};

// Poisson cost (bits) of a column; rate_of(j) is the rate per unit nhat
template <class F> inline double poisson_bits(const Col& c, double nhat, F&& rate_of) {
    double b = nhat;
    for (size_t t = 0; t < c.n; ++t) {
        double lam = nhat * rate_of(c.idx[t]);
        double x = c.val[t];
        b += -x * std::log(std::max(lam, 1e-300)) + std::lgamma(x + 1.0);
    }
    return b * 1.4426950408889634;
}

inline int pick_scale(u64 N, int R, int levels) {
    if (N == 0) return 0;
    int s0 = (int)std::lround(R * std::log2((double)N)) + 2 * R;
    int best = -1; double bf = 1e300;
    for (int s = s0 - 1; s <= s0 + 1; ++s) {
        if (s < 0 || s >= levels) continue;
        double nh = (double)scale_q32(s, R) / 4294967296.0;
        double f = nh - (double)N * std::log(nh);
        if (f < bf) { bf = f; best = s; }
    }
    if (best < 0) best = std::max(0, std::min(levels - 1, s0));
    return best;
}

// ---------------------------------------------------------------------------------------------
// Side-information models shared by encoder and decoder
// ---------------------------------------------------------------------------------------------
struct SideModels {
    AdaptModel flag[2];
    AdaptModel oct_cell, oct_amb;
    std::vector<AdaptModel> th;
    void init(const Model& M) {
        for (auto& f : flag) f.init(2, 1, 24, 1 << 13);
        oct_cell.init(NOCT, 1, 32, 1 << 13); oct_amb.init(NOCT, 1, 32, 1 << 13);
        th.resize(M.K);
        for (u32 k = 0; k < M.K; ++k) {
            th[k].init(NLEV, 1, 32, 1 << 14);
            th[k].total = 0;
            for (int l = 0; l < NLEV; ++l) { th[k].cnt[l] = 1 + M.thprior[(size_t)k * NLEV + l]; th[k].total += th[k].cnt[l]; }
        }
    }
};

inline void put_model_sym(RansEnc& e, AdaptModel& am, int s, double* bits) {
    u32 st, fr; am.get(s, st, fr);
    e.put(st, fr, AdaptModel::SB);
    if (bits) *bits += AdaptModel::SB - std::log2((double)fr);
    am.update(s);
}
inline int get_model_sym(RansDec& d, AdaptModel& am) {
    u32 slot = d.slot(AdaptModel::SB), st, fr;
    int s = am.find(slot, st, fr);
    d.advance(st, fr, AdaptModel::SB);
    am.update(s);
    return s;
}

// ---------------------------------------------------------------------------------------------
// Names (rownames from the model when identical, barcodes 2-bit packed, user_kv as in the reference)
// ---------------------------------------------------------------------------------------------
inline void put_colnames(std::vector<uint8_t>& o, const std::vector<std::string>& cn) {
    bool pack = !cn.empty();
    size_t L = pack ? cn[0].size() : 0;
    if (L == 0 || L > 32) pack = false;
    for (auto& s : cn) {
        if (!pack) break;
        if (s.size() != L) { pack = false; break; }
        for (char ch : s) if (ch != 'A' && ch != 'C' && ch != 'G' && ch != 'T') { pack = false; break; }
    }
    o.push_back(pack ? 1 : 0);
    if (pack) {
        o.push_back((uint8_t)L);
        std::vector<uint8_t> b((cn.size() * L * 2 + 7) / 8, 0);
        size_t bit = 0;
        for (auto& s : cn) for (char ch : s) {
            int v = ch == 'A' ? 0 : ch == 'C' ? 1 : ch == 'G' ? 2 : 3;
            b[bit >> 3] |= (uint8_t)(v << (bit & 7)); bit += 2;
        }
        detail::put_stream(o, b.data(), b.size(), 19);
    } else {
        auto j = detail::join_names(cn);
        detail::put_stream(o, j.data(), j.size(), 19);
    }
}
inline std::vector<std::string> get_colnames(const uint8_t*& p, size_t n) {
    uint8_t pack = *p++;
    if (pack) {
        size_t L = *p++;
        auto b = detail::get_stream(p);
        std::vector<std::string> v(n);
        size_t bit = 0;
        for (size_t c = 0; c < n; ++c) {
            v[c].resize(L);
            for (size_t i = 0; i < L; ++i) { v[c][i] = "ACGT"[(b[bit >> 3] >> (bit & 7)) & 3]; bit += 2; }
        }
        return v;
    }
    return detail::split_names(detail::get_stream(p), n);
}

inline void dbg_check(u32 c, const std::vector<u32>& idx, const std::vector<u32>& val, const char* what, int s) {
    const CanonicalMatrix* M = dbg_ref();
    if (!M) return;
    u32 a = M->indptr[c], b = M->indptr[c + 1];
    bool same = (b - a == idx.size());
    size_t bad = 0;
    for (size_t t = 0; t < idx.size() && t < b - a; ++t) if (M->indices[a + t] != idx[t] || M->data[a + t] != val[t]) { same = false; bad = t; break; }
    if (!same) {
        static int shown = 0;
        if (shown++ < 3) {
            std::printf("  DBG column %u (%s, scale %d) mismatch: orig n=%u dec n=%zu first diff at %zu: orig (%u,%u) dec (%u,%u)\n", c, what, s, b - a, idx.size(), bad,
                        bad < b - a ? M->indices[a + bad] : 0u, bad < b - a ? M->data[a + bad] : 0u, bad < idx.size() ? idx[bad] : 0u, bad < idx.size() ? val[bad] : 0u);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// The codec
// ---------------------------------------------------------------------------------------------
struct Idea4Codec {
    std::string model_path;
    std::shared_ptr<Model> model;
    std::shared_ptr<Fitter> fit;
    double kc = 20000, k0 = 2000, tmin = 8, theta_refine = 1, adapt_on = 0, kg = 200, kq = 3;

    static void tr(const char* what) { if (std::getenv("IDEA4_TRACE")) { std::fprintf(stderr, "[trace] %s\n", what); std::fflush(stderr); } }
    void ensure() {
        if (model) return;
        tr("loading model");
        auto m = std::make_shared<Model>();
        m->load(model_path);
        tr("model loaded");
        model = m;
        fit = std::make_shared<Fitter>(*model);
        tr("fitter built");
    }

    // ------------------------------------------------------------------------------------ encode
    std::vector<uint8_t> encode(const CanonicalMatrix& Mx) {
        ensure();
        Stats& st = stats(); st = Stats();
        dbg_ref() = std::getenv("IDEA4_DEBUG") ? &Mx : nullptr;
        trace_on() = dbg_ref() != nullptr; enc_trace().clear(); dec_trace().clear();
        const Model& M = *model;
        if (Mx.m != M.m) throw std::runtime_error("idea4: matrix rows != model rows");
        const u32 K = M.K, n = Mx.n;
        bool rn_in_model = false;
        if (M.flags & 1) {
            const_cast<Model&>(M).load_rownames();
            rn_in_model = (M.rownames == Mx.rownames);
        }
        std::vector<uint8_t> out;
        out.push_back('I'); out.push_back('4'); out.push_back('C'); out.push_back('1');
        detail::put_u32(out, Mx.m); detail::put_u32(out, Mx.n); detail::put_u64(out, Mx.nnz);
        out.push_back(Mx.vt_code); out.push_back(rn_in_model ? 1 : 0); out.push_back((uint8_t)K);
        size_t h0 = out.size();
        if (!rn_in_model) { auto rn = detail::join_names(Mx.rownames); detail::put_stream(out, rn.data(), rn.size(), 19); }
        put_colnames(out, Mx.colnames);
        {
            std::vector<uint8_t> kv;
            for (auto& [k, v] : Mx.user_kv) { kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0); kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0); }
            detail::put_u32(out, (uint32_t)Mx.user_kv.size());
            detail::put_stream(out, kv.data(), kv.size(), 19);
        }
        st.names_bytes = out.size() - h0;
        st.hdr_bytes = h0;

        std::vector<Col> cols(n);
        for (u32 c = 0; c < n; ++c) {
            u32 a = Mx.indptr[c], b = Mx.indptr[c + 1];
            u64 N = 0; for (u32 t = a; t < b; ++t) N += Mx.data[t];
            cols[c] = Col{Mx.indices.data() + a, Mx.data.data() + a, (size_t)(b - a), N};
        }
        tr("encode: columns ready");
        double t0 = now_ms();
        std::vector<uint8_t> mode(n, 0);
        std::vector<std::vector<uint8_t>> codes(n);
        std::vector<int> scl(n, 0);
        if (!calls_path().empty()) {
            // modes given by the sample's cell_calls.tsv (the producer knows the calls; flags cost ~0.03 bit/column)
            std::ifstream f(calls_path());
            if (!f) throw std::runtime_error("idea4: cannot open " + calls_path());
            std::string line; std::getline(f, line);
            std::unordered_map<std::string, bool> isc;
            while (std::getline(f, line)) {
                size_t t1 = line.find('	'); if (t1 == std::string::npos) continue;
                size_t t2 = line.find('	', t1 + 1);
                isc[line.substr(0, t1)] = line.substr(t1 + 1, t2 - t1 - 1) == "TRUE";
            }
            for (u32 c = 0; c < n; ++c) { auto it = isc.find(Mx.colnames[c]); mode[c] = (it != isc.end() && it->second && cols[c].n > 0) ? 1 : 0; }
        } else {
            decide_modes(cols, mode);
        }
        st.enc_fit_ms = 0;
        if (K) fit->phi_cur = M.phi;

        tr("encode: modes decided");
        RansEnc enc;
        SideModels sm; sm.init(M);
        tr("encode: side models ready");
        { int prev = 0; for (u32 c = 0; c < n; ++c) { put_model_sym(enc, sm.flag[prev], mode[c], &st.bits_flag); prev = mode[c]; } }
        tr("encode: flags done");
        AmbProfile amb; amb.init(M); amb.kc = (u64)kc; amb.k0 = (u64)k0;
        CountCoder ccC(M.ctab[0].data()), ccA(M.ctab[1].data());
        CalledHaz ch;
        std::vector<uint16_t> tq(K ? K : 1);
        AdaptBase ab; ab.init(M, adapt_on != 0, (u64)kg, (u64)kq);
        std::vector<u32> fg; std::vector<float> fx, fth; std::vector<uint8_t> fcodes(K ? K : 1);
        for (u32 c = 0; c < n; ++c) {
            if (!mode[c]) continue;
            if (ab.maybe_rebuild(M) && K) fit->rebase(ab);
            double tf = now_ms();
            if (K) {
                fit->agg(cols[c], fg, fx);
                fit->em(fg, fx, fth, 20);
                if (theta_refine >= 2) fit->quantise_rd(fth, fg, fx, fcodes.data(), sm, 3);
                else fit->quantise(fth, fg, fx, fcodes.data(), theta_refine > 0);
                bool anyc = false; for (u32 k = 0; k < K; ++k) anyc |= fcodes[k] != 0;
                if (!anyc) {
                    std::fprintf(stderr, "[warn] col %u: all-zero theta codes (N=%llu, genes=%zu); th:", c, (unsigned long long)cols[c].N, fg.size());
                    for (u32 k = 0; k < K; ++k) std::fprintf(stderr, " %g", fth[k]);
                    if (!fg.empty()) { std::fprintf(stderr, " | gene0=%u x0=%g phi0:", fg[0], fx[0]); for (u32 k = 0; k < K; ++k) std::fprintf(stderr, " %g", fit->phi_cur[(size_t)fg[0] * K + k]); }
                    std::fprintf(stderr, "\n");
                    for (u32 k = 0; k < K; ++k) fcodes[k] = 8;
                }
            }
            st.enc_fit_ms += now_ms() - tf;
            if (st.n_cell == 0) tr("encode: first cell fitted");
            int R = SCALE_R_CELL;
            int s = pick_scale(cols[c].N, R, scale_levels(R));
            put_model_sym(enc, sm.oct_cell, s / R, &st.sc_cell);
            enc.put(s % R, 1, 3); st.sc_cell += 3;
            if (st.n_cell == 0) tr("encode: scale picked");
            for (u32 k = 0; k < K; ++k) { put_model_sym(enc, sm.th[k], fcodes[k], &st.bits_theta); tq[k] = (uint16_t)LEVEL_TQ[fcodes[k]]; }
            if (st.n_cell == 0) { tr("encode: theta coded"); if (std::getenv("IDEA4_TRACE")) { for (u32 k = 0; k < K; ++k) std::fprintf(stderr, "code[%u]=%d ", k, (int)fcodes[k]); std::fprintf(stderr, "\n"); } }
            ch.build(M, scale_q32(s, R), K ? tq.data() : nullptr, ab.pi0q_ptr(), ab.qcum_ptr());
            if (st.n_cell == 0) tr("encode: build done");
            Gap<CalledHaz> gp(ch, M.m);
            if (trace_on()) gp.verify(cols[c].idx, cols[c].n);
            gp.encode(enc, cols[c].idx, cols[c].val, cols[c].n, ccC, &st.bits_gap_cell, &st.bits_cnt_cell, &st.bits_gap_cell);
            if (st.n_cell == 0) tr("encode: first column coded");
            if (std::getenv("IDEA4_TRACE") && (st.n_cell % 500) == 0) std::fprintf(stderr, "[trace] cell %zu (col %u, nnz %zu)\n", st.n_cell, c, cols[c].n);
            amb.add_cell(cols[c].idx, cols[c].val, cols[c].n);
            ab.add(M, cols[c].idx, cols[c].val, cols[c].n);
            st.n_cell++; st.nnz_cell += cols[c].n;
        }
        {
            u32 since = 0, block = 64; bool first = true;
            for (u32 c = 0; c < n; ++c) {
                if (mode[c]) continue;
                if (first || since >= block) { amb.rebuild(); since = 0; block = std::min<u32>(4096, first ? 64 : block * 2); first = false; }
                int R = SCALE_R_AMB;
                int s = pick_scale(cols[c].N, R, scale_levels(R));
                put_model_sym(enc, sm.oct_amb, s / R, &st.sc_amb);
                enc.put(s % R, 1, 2); st.sc_amb += 2;
                AmbHaz ah; ah.M = &M; ah.CAn = amb.CAn.data(); ah.nhat = scale_q32(s, R);
                Gap<AmbHaz> gp(ah, M.m);
                gp.encode(enc, cols[c].idx, cols[c].val, cols[c].n, ccA, &st.bits_gap_amb, &st.bits_cnt_amb, &st.bits_gap_amb);
                amb.add_emp(cols[c].idx, cols[c].val, cols[c].n);
                st.n_amb++; st.nnz_amb += cols[c].n; ++since;
            }
        }
        auto r = enc.finish();
        st.rans_bytes = r.size();
        out.insert(out.end(), r.begin(), r.end());
        return out;
    }

    // Mode decision: called (theta model) vs ambient. Float estimates, encoder only.
    void decide_modes(const std::vector<Col>& cols, std::vector<uint8_t>& mode) {
        const Model& M = *model;
        const u32 n = (u32)cols.size(), K = M.K;
        std::vector<u64> ds; for (auto& c : cols) if (c.N >= tmin) ds.push_back(c.N);
        if (ds.empty()) return;
        std::sort(ds.begin(), ds.end());
        u64 T0 = std::max<u64>(40, ds[ds.size() * 3 / 4] / 4);
        AmbProfile amb; amb.init(M); amb.kc = (u64)kc; amb.k0 = (u64)k0;
        for (u32 c = 0; c < n; ++c) if (cols[c].N >= T0) amb.add_cell(cols[c].idx, cols[c].val, cols[c].n);
        amb.rebuild();
        const std::vector<float>& qf = fit->qf; const std::vector<float>& p0f = fit->p0f;
        auto amb_rate = [&](u32 j) { double lo = j ? (double)amb.CAn[j - 1] : 0.0; return ((double)amb.CAn[j] - lo) / 72057594037927936.0; };
        std::vector<u32> g; std::vector<float> x, th; std::vector<uint8_t> codes(K ? K : 1);
        for (u32 c = 0; c < n; ++c) {
            const Col& cl = cols[c];
            if (cl.N < tmin || cl.n == 0) { mode[c] = 0; continue; }
            double nh = (double)cl.N;
            double b_amb = poisson_bits(cl, nh, amb_rate) + 2.0 + 2.5;
            double b_cell;
            if (K) {
                fit->agg(cl, g, x);
                fit->em(g, x, th, 12);
                fit->quantise(th, g, x, codes.data(), false);
                float w[64]; float sw = 0; for (u32 k = 0; k < K; ++k) { w[k] = (float)LEVEL_TQ[codes[k]]; sw += w[k]; }
                b_cell = poisson_bits(cl, nh, [&](u32 j) {
                    u32 gg = M.gene_of[j]; const float* ph = &M.phi[(size_t)gg * K];
                    float p = 0; for (u32 k = 0; k < K; ++k) p += w[k] * ph[k];
                    return (double)(p / sw) * qf[j];
                });
                b_cell += 5.0 + 3.0 + 0.7 * K;
            } else {
                b_cell = poisson_bits(cl, nh, [&](u32 j) { return (double)p0f[j]; }) + 5.0 + 3.0;
            }
            mode[c] = (b_cell < b_amb) ? 1 : 0;
        }
    }

    // ------------------------------------------------------------------------------------ decode
    CanonicalMatrix decode(const std::vector<uint8_t>& b) {
        ensure();
        Stats& st = stats();
        double tdec0 = now_ms();
        const Model& M = *model;
        const uint8_t* p = b.data();
        if (b.size() < 4 || p[0] != 'I' || p[1] != '4' || p[2] != 'C') throw std::runtime_error("idea4: bad stream");
        p += 4;
        CanonicalMatrix X;
        X.m = detail::get_u32(p); X.n = detail::get_u32(p); X.nnz = detail::get_u64(p);
        X.vt_code = *p++; uint8_t rn_in_model = *p++; uint8_t K8 = *p++;
        if (K8 != M.K || X.m != M.m) throw std::runtime_error("idea4: model mismatch");
        const u32 K = M.K, n = X.n;
        if (rn_in_model) { const_cast<Model&>(M).load_rownames(); X.rownames = M.rownames; }
        else X.rownames = detail::split_names(detail::get_stream(p), X.m);
        X.colnames = get_colnames(p, n);
        uint32_t nkv = detail::get_u32(p);
        auto kv = detail::get_stream(p);
        auto parts = detail::split_names(kv, 2 * (size_t)nkv);
        for (uint32_t i = 0; i < nkv; ++i) X.user_kv[parts[2 * i]] = parts[2 * i + 1];
        RansDec dec; dec.init(p, b.size() - (p - b.data()));
        SideModels sm; sm.init(M);
        std::vector<uint8_t> mode(n);
        { int prev = 0; for (u32 c = 0; c < n; ++c) { mode[c] = (uint8_t)get_model_sym(dec, sm.flag[prev]); prev = mode[c]; } }
        std::vector<std::vector<u32>> ci(n), cv(n);
        AmbProfile amb; amb.init(M); amb.kc = (u64)kc; amb.k0 = (u64)k0;
        CountCoder ccC(M.ctab[0].data()), ccA(M.ctab[1].data());
        CalledHaz ch; std::vector<uint16_t> tq(K ? K : 1);
        AdaptBase ab; ab.init(M, adapt_on != 0, (u64)kg, (u64)kq);
        double t_build = 0, w_gapc = 0, w_ambr = 0, w_gapa = 0; size_t ncell = 0;
        double w_start = wall_ms();
        for (u32 c = 0; c < n; ++c) {
            if (!mode[c]) continue;
            ab.maybe_rebuild(M);
            int R = SCALE_R_CELL;
            int oct = get_model_sym(dec, sm.oct_cell);
            int fine = (int)dec.slot(3); dec.advance((u32)fine, 1, 3);
            int s = oct * R + fine;
            for (u32 k = 0; k < K; ++k) { int code = get_model_sym(dec, sm.th[k]); tq[k] = (uint16_t)LEVEL_TQ[code]; }
            double tb = now_ms();
            ch.build(M, scale_q32(s, R), K ? tq.data() : nullptr, ab.pi0q_ptr(), ab.qcum_ptr());
            t_build += now_ms() - tb; ++ncell;
            Gap<CalledHaz> gp(ch, M.m);
            double wg0 = wall_ms();
            gp.decode(dec, ci[c], cv[c], ccC);
            w_gapc += wall_ms() - wg0;
            dbg_check(c, ci[c], cv[c], "called", s);
            amb.add_cell(ci[c].data(), cv[c].data(), ci[c].size());
            ab.add(M, ci[c].data(), cv[c].data(), ci[c].size());
        }
        {
            u32 since = 0, block = 64; bool first = true;
            for (u32 c = 0; c < n; ++c) {
                if (mode[c]) continue;
                if (first || since >= block) { double wr0 = wall_ms(); amb.rebuild(); w_ambr += wall_ms() - wr0; since = 0; block = std::min<u32>(4096, first ? 64 : block * 2); first = false; }
                int R = SCALE_R_AMB;
                int oct = get_model_sym(dec, sm.oct_amb);
                int fine = (int)dec.slot(2); dec.advance((u32)fine, 1, 2);
                int s = oct * R + fine;
                AmbHaz ah; ah.M = &M; ah.CAn = amb.CAn.data(); ah.nhat = scale_q32(s, R);
                Gap<AmbHaz> gp(ah, M.m);
                double wg1 = wall_ms();
                gp.decode(dec, ci[c], cv[c], ccA);
                w_gapa += wall_ms() - wg1;
                dbg_check(c, ci[c], cv[c], "amb", s);
                amb.add_emp(ci[c].data(), cv[c].data(), ci[c].size());
                ++since;
            }
        }
        X.indptr.assign(n + 1, 0);
        for (u32 c = 0; c < n; ++c) X.indptr[c + 1] = X.indptr[c] + (u32)ci[c].size();
        X.indices.resize(X.indptr[n]); X.data.resize(X.indptr[n]);
        for (u32 c = 0; c < n; ++c) {
            if (ci[c].empty()) continue;
            std::memcpy(&X.indices[X.indptr[c]], ci[c].data(), ci[c].size() * 4);
            std::memcpy(&X.data[X.indptr[c]], cv[c].data(), cv[c].size() * 4);
        }
        if (trace_on()) {
            auto& et = enc_trace(); auto& dt = dec_trace();
            size_t nn = std::min(et.size(), dt.size());
            for (size_t t = 0; t < nn; ++t) if (et[t] != dt[t]) {
                std::printf("  TRACE first divergence at symbol %zu: enc lo=%llu f=%llu  dec lo=%llu f=%llu (enc total %zu dec total %zu)\n", t, (unsigned long long)(et[t] >> 32), (unsigned long long)(et[t] & 0xFFFFFFFFull),
                            (unsigned long long)(dt[t] >> 32), (unsigned long long)(dt[t] & 0xFFFFFFFFull), et.size(), dt.size());
                break;
            }
            trace_on() = false;
        }
        st.dec_matvec_ms = t_build; st.dec_cells = ncell; st.dec_total_ms = now_ms() - tdec0;
        { double wt = wall_ms() - w_start; st.dec_share_gapc = w_gapc / wt; st.dec_share_ambr = w_ambr / wt; st.dec_share_gapa = w_gapa / wt; }
        return X;
    }
};

inline Codec make_idea4(const std::string& name, const std::string& model_path) {
    auto impl = std::make_shared<Idea4Codec>();
    impl->model_path = model_path;
    impl->kc = env_d("IDEA4_KC", 50000); impl->k0 = env_d("IDEA4_K0", 2000); impl->tmin = env_d("IDEA4_TMIN", 8);
    impl->theta_refine = env_d("IDEA4_REFINE", 2);
    impl->adapt_on = env_d("IDEA4_ADAPT", 0); impl->kg = env_d("IDEA4_KG", 200); impl->kq = env_d("IDEA4_KQ", 3);
    Codec c;
    c.name = name;
    c.encode = [impl](const CanonicalMatrix& M) { return impl->encode(M); };
    c.decode = [impl](const std::vector<uint8_t>& b) { return impl->decode(b); };
    {
        std::ifstream f(model_path, std::ios::binary | std::ios::ate);
        c.model_bytes = f ? (size_t)f.tellg() : 0;
    }
    return c;
}

inline void register_idea4(std::vector<Codec>& v) {
    const char* tag = std::getenv("IDEA4_TAG");
    std::string t = tag ? std::string("_") + tag : "";
    const char* ms = std::getenv("IDEA4_MODEL_STATIC");
    const char* mn = std::getenv("IDEA4_MODEL_NMF");
    v.push_back(make_idea4("idea4_static" + t, ms ? ms : "results/idea4_model_static_human.bin"));
    v.push_back(make_idea4("idea4_nmf" + t, mn ? mn : "results/idea4_model_nmf16_human.bin"));
}

}}  // namespace cs::ipp
