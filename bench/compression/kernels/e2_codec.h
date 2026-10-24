// SPDX-License-Identifier: MIT
// E2 codec: block-adaptive survival-gap coder with interleaved rANS lanes.
//
// Stream:  "E2v2" | m n nnz vt | params | names (zstd-3 streams, Track A's problem) | header stream | payload
//   header stream (adaptive binary range coder): per column in file order  depth N_j, called-cell flag
//   payload: class 0 (called cells), then class 1 (empties); each class is cut into blocks of columns in a
//   pseudo-random coding order; for every block  [lane headers][lane gap streams][lane count streams].
//   Within a block all tables are frozen; they are refitted from the already-decoded blocks (e2_model.h).
// Coding order inside a column is ascending row id (no permutation: the decoded column is canonical).
#pragma once
#include "e2_model.h"
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <unordered_set>

namespace cs {
namespace e2 {

// ---- reporting / harness hooks ------------------------------------------------------------------------------
inline std::string g_calls_path;               // cell_calls.tsv of the sample being encoded
struct Report {
    size_t names_bytes = 0, hdr_bytes = 0, payload_bytes = 0, blockhdr_bytes = 0;
    size_t cells_bytes = 0, empt_bytes = 0;
    u64 nnz_cells = 0, nnz_empt = 0, ncells = 0, nempt = 0;
    double enc_refit_ms = 0;
    // decode timing (min over calls), milliseconds
    double dec_total_ms = 1e300, dec_payload_ms = 1e300, dec_refit_ms = 1e300, dec_gather_ms = 1e300, dec_hdr_ms = 1e300, dec_payload_cyc = 1e300;
    double dec_lanes_ms = 1e300, dec_lanes_cyc = 1e300;
    u64 nesc = 0, ncorr = 0, nsym = 0, nblocks = 0;
    std::vector<double> block_bits;
    std::vector<u32> block_cols; std::vector<u64> block_nnz; std::vector<int> block_class;
    void reset_dec() { dec_total_ms = dec_payload_ms = dec_refit_ms = dec_gather_ms = dec_hdr_ms = dec_payload_cyc = dec_lanes_ms = dec_lanes_cyc = 1e300; }
};
inline Report g_rep;
#ifdef E2_STATS
inline u64 g_win[4096];
#endif

inline u32 env_u32(const char* n, u32 d) { const char* s = std::getenv(n); return s ? (u32)std::strtoul(s, nullptr, 10) : d; }
inline Params params_from_env(Params p) {
    p.bpo_log = env_u32("E2_BPO", p.bpo_log); p.n0 = env_u32("E2_N0", p.n0); p.rho_x16 = env_u32("E2_RHO16", p.rho_x16);
    p.Bmax = env_u32("E2_B", p.Bmax); p.Umax = env_u32("E2_UMAX", p.Umax); p.lanes = env_u32("E2_LANES", p.lanes);
    p.gtmax = env_u32("E2_GTMAX", p.gtmax); p.rounds = env_u32("E2_ROUNDS", p.rounds); p.ctrust = env_u32("E2_CTRUST", p.ctrust);
    p.flat_x100 = env_u32("E2_FLAT", p.flat_x100); p.zlim = env_u32("E2_ZLIM", p.zlim); p.theta_x16 = env_u32("E2_THETA16", p.theta_x16);
    p.seed = env_u32("E2_SEED", p.seed);
    return p;
}

struct Sym { u32 cum, freq; u8 bits; };

inline u32 block_size(const Params& P, u64 pos) {
    u64 sz;
    if (pos == 0) sz = P.n0;
    else {
        sz = ((u64)(P.rho_x16 - 16) * pos) / 16;
        if (sz < P.n0) sz = P.n0;
    }
    if (sz > P.Bmax) sz = P.Bmax;
    return (u32)std::max<u64>(sz, 1);
}
inline size_t block_end(const Params& P, const std::vector<u64>& N, const std::vector<u32>& order, size_t pos) {
    size_t e = std::min<size_t>(order.size(), pos + block_size(P, pos));
    if (P.Umax) {
        u64 U = 0, lim = (u64)P.Umax * 1024;
        size_t q = pos;
        for (; q < e; ++q) { if (q > pos && U + N[order[q]] > lim) break; U += N[order[q]]; }
        e = q;
    }
    return e;
}

// ---------------------------------------------------------------------------------------------------------
// header stream: depths and called-cell flags, original column order
struct HdrModel {
    TreeM e_tree{6, 64};          // exponent given previous exponent
    TreeM m_tree{3, 64};          // top mantissa bits given exponent
    BitM flag[64];
};
inline void hdr_encode(RcEnc& rc, const std::vector<u64>& N, const std::vector<u8>& isc) {
    HdrModel H;
    u32 prev = 0;
    for (size_t j = 0; j < N.size(); ++j) {
        u64 v = N[j] + 1;
        u32 e = (u32)ilog2(v);
        H.e_tree.enc(rc, (int)std::min<u32>(prev, 63), std::min<u32>(e, 63));
        int nb = (int)std::min<u32>(e, 3);
        if (nb) {
            u32 top = (u32)((v >> (e - nb)) & ((1u << nb) - 1));
            size_t base = (size_t)e << 3; u32 node = 1;
            for (int k = nb - 1; k >= 0; --k) {
                int bit = (top >> k) & 1;
                BitM& bm = H.m_tree.t[base + node];
                rc.codebit(bm.P(), bit); bm.upd(bit);
                node = node * 2 + bit;
            }
        }
        if (e > 3) rc_raw_enc(rc, (u32)(v & ((1ull << (e - 3)) - 1)), (int)(e - 3));
        if (N[j] > 0) {
            BitM& fb = H.flag[std::min<u32>(e, 63)];
            rc.codebit(fb.P(), isc[j]); fb.upd(isc[j]);
        }
        prev = e;
    }
}
inline void hdr_decode(RcDec& rc, size_t n, std::vector<u64>& N, std::vector<u8>& isc) {
    HdrModel H;
    N.assign(n, 0); isc.assign(n, 0);
    u32 prev = 0;
    for (size_t j = 0; j < n; ++j) {
        u32 e = H.e_tree.dec(rc, (int)std::min<u32>(prev, 63));
        u64 v = 1;
        int nb = (int)std::min<u32>(e, 3);
        if (e > 0) {
            size_t base = (size_t)e << 3; u32 node = 1;
            for (int k = nb - 1; k >= 0; --k) {
                BitM& bm = H.m_tree.t[base + node];
                int bit = rc.codebit(bm.P()); bm.upd(bit);
                node = node * 2 + bit;
            }
            u32 top = node - (1u << nb);
            v = (1ull << e) | ((u64)top << (e - nb));
            if (e > 3) v |= rc_raw_dec(rc, (int)(e - 3));
        }
        N[j] = v - 1;
        if (N[j] > 0) {
            BitM& fb = H.flag[std::min<u32>(e, 63)];
            int bit = rc.codebit(fb.P()); fb.upd(bit);
            isc[j] = (u8)bit;
        }
        prev = e;
    }
}

// ---------------------------------------------------------------------------------------------------------
inline std::vector<u32> class_columns(const std::vector<u64>& N, const std::vector<u8>& isc, int cls, u64 seed) {
    std::vector<u32> v;
    for (u32 j = 0; j < N.size(); ++j)
        if (N[j] > 0 && (int)isc[j] == (cls == 0)) v.push_back(j);
    shuffle_cols(v, seed + (u64)cls * 0x9E3779B97F4A7C15ull);
    return v;
}

inline void put_stream_names(std::vector<u8>& o, const CanonicalMatrix& M) {
    using namespace detail;
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 3);
    put_stream(o, cn.data(), cn.size(), 3);
    std::vector<u8> kv;
    for (auto& [k, v] : M.user_kv) {
        kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
        kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
    }
    put_u32(o, (u32)M.user_kv.size());
    put_stream(o, kv.data(), kv.size(), 3);
}

inline int pos_bits_of(u32 m) { int pb = 1; while ((1u << pb) < m + 1) ++pb; return pb; }

// ---------------------------------------------------------------------------------------------------------
// encoder
inline std::vector<u8> encode_e2(const CanonicalMatrix& M, const Params& P) {
    using namespace detail;
    const u32 m = M.m, n = M.n;
    g_rep = Report();
    std::vector<u64> N(n, 0);
    for (u32 j = 0; j < n; ++j) { u64 s = 0; for (u32 q = M.indptr[j]; q < M.indptr[j + 1]; ++q) s += M.data[q]; N[j] = s; }
    std::vector<u8> isc(n, 0);
    {
        std::ifstream f(g_calls_path);
        if (!f) throw std::runtime_error("e2: cannot open cell_calls.tsv at '" + g_calls_path + "'");
        std::unordered_set<std::string> s;
        std::string line; std::getline(f, line);
        while (std::getline(f, line)) s.insert(line.substr(0, line.find('\t')));
        for (u32 j = 0; j < n; ++j) isc[j] = (N[j] > 0 && s.count(M.colnames[j])) ? 1 : 0;
    }
    u64 Nmax = 1;
    for (u32 j = 0; j < n; ++j) Nmax = std::max(Nmax, N[j]);
    const int nbk = (int)bucket_of(Nmax, P.bpo_log) + 1;

    std::vector<u8> o;
    o.push_back('E'); o.push_back('2'); o.push_back('v'); o.push_back('2');
    put_u32(o, m); put_u32(o, n); put_u64(o, M.nnz); o.push_back(M.vt_code);
    { u32 raw[NPARAMS]; params_pack(P, raw); for (int i = 0; i < NPARAMS; ++i) put_u32(o, raw[i]); }
    size_t nb0 = o.size();
    put_stream_names(o, M);
    g_rep.names_bytes = o.size() - nb0;
    size_t h0 = o.size();
    {
        RcEnc rc;
        hdr_encode(rc, N, isc);
        rc.finish();
        put_u64(o, rc.out.size());
        o.insert(o.end(), rc.out.begin(), rc.out.end());
    }
    g_rep.hdr_bytes = o.size() - h0;

    const int PBITS = pos_bits_of(m);
    const int LG = (int)P.lanes;
    std::vector<u8> bk(n, 0);
    for (u32 j = 0; j < n; ++j) if (N[j]) bk[j] = (u8)bucket_of(N[j], P.bpo_log);
    size_t pay0 = o.size();
    Timer t_enc_all;
    const bool trace = std::getenv("E2_TRACE") != nullptr;
    std::vector<u32> prior;
    for (int cls = 0; cls < 2; ++cls) {
        std::vector<u32> order = class_columns(N, isc, cls, P.seed);
        size_t cls0 = o.size();
        ClassModel md(P, m, nbk);
        if (cls == 1 && !prior.empty() && P.theta_x16 > 0) md.prior_w = prior;
        u64 pos = 0;
        std::vector<Sym> gs, cs;
        while (pos < order.size()) {
            size_t e = block_end(P, N, order, pos);
            Timer tr; md.refit(); g_rep.enc_refit_ms += tr.ms();
            std::vector<std::vector<u32>> gw(LG), cw(LG);
            std::vector<u64> lane_nnz(LG, 0);
            u64 blk_nnz = 0;
            for (int l = 0; l < LG; ++l) {
                gs.clear(); cs.clear();
                for (size_t p = pos + l; p < e; p += LG) {
                    const u32 j = order[p];
                    const u32 b = bk[j];
                    u32 i1 = 0;
                    for (u32 q = M.indptr[j]; q < M.indptr[j + 1]; ++q) {
                        const u32 f = M.indices[q], v = M.data[q];
                        const u32 x0 = md.S[i1];
                        const u64 D0 = md.dist(md.S[f] - x0, b), D1 = md.dist(md.S[f + 1] - x0, b);
                        const u64 c0 = MSC - G(D0), c1 = MSC - G(D1);
                        if (D0 >= XCUT || c1 <= c0) {
                            gs.push_back({(u32)FCUT, (u32)(MSC - FCUT), (u8)MB});
                            gs.push_back({f, 1u, (u8)PBITS});
                            ++g_rep.nesc;
                        } else {
                            gs.push_back({(u32)c0, (u32)(c1 - c0), (u8)MB});
                        }
                        const u32 ctx = count_ctx(D1 - D0);
                        const u32 s = v - 1 < 15 ? v - 1 : 15;
                        cs.push_back({md.ccum[(size_t)ctx * (NSYM + 1) + s], md.cfreq[(size_t)ctx * NSYM + s], (u8)CPB});
                        if (s == 15) {
                            u32 u = v - 15;
                            int nbv = ilog2(u);
                            cs.push_back({(u32)nbv, 1u, 5});
                            u32 low = u - (1u << nbv);
                            if (nbv > 16) { cs.push_back({low >> 16, 1u, (u8)(nbv - 16)}); cs.push_back({low & 0xFFFF, 1u, 16}); }
                            else if (nbv > 0) cs.push_back({low, 1u, (u8)nbv});
                        }
                        md.cn[(size_t)ctx * NSYM + s] += 64;
                        md.z[f]++;
                        i1 = f + 1;
                        ++lane_nnz[l];
                    }
                    {   // END: interval [c0, FCUT)
                        const u32 x0 = md.S[i1];
                        const u64 D0 = md.dist(md.S[m] - x0, b);
                        const u64 c0 = MSC - G(D0), c1 = FCUT;
                        if (D0 >= XCUT || c1 <= c0) {
                            gs.push_back({(u32)FCUT, (u32)(MSC - FCUT), (u8)MB});
                            gs.push_back({m, 1u, (u8)PBITS});
                            ++g_rep.nesc;
                        } else {
                            gs.push_back({(u32)c0, (u32)(c1 - c0), (u8)MB});
                        }
                    }
                    md.add_column(b, M.indptr[j + 1] - M.indptr[j], N[j]);
                }
                RansEnc eg, ec;
                for (size_t i = gs.size(); i-- > 0;) eg.put(gs[i].cum, gs[i].freq, gs[i].bits);
                for (size_t i = cs.size(); i-- > 0;) ec.put(cs[i].cum, cs[i].freq, cs[i].bits);
                eg.finish(gw[l]); ec.finish(cw[l]);
                blk_nnz += lane_nnz[l];
            }
            size_t bh0 = o.size();
            for (int l = 0; l < LG; ++l) { put_vi(o, lane_nnz[l]); put_vi(o, gw[l].size()); put_vi(o, cw[l].size()); }
            g_rep.blockhdr_bytes += o.size() - bh0;
            for (int l = 0; l < LG; ++l) {
                size_t p0 = o.size();
                o.resize(p0 + (gw[l].size() + cw[l].size()) * 4);
                std::memcpy(o.data() + p0, gw[l].data(), gw[l].size() * 4);
                std::memcpy(o.data() + p0 + gw[l].size() * 4, cw[l].data(), cw[l].size() * 4);
            }
            g_rep.block_bits.push_back((double)(o.size() - bh0) * 8.0);
            g_rep.block_cols.push_back((u32)(e - pos)); g_rep.block_nnz.push_back(blk_nnz); g_rep.block_class.push_back(cls);
            ++g_rep.nblocks;
            pos = e;
        }
        if (cls == 0 && P.theta_x16 > 0) { md.refit(); md.export_prior(prior); }
        size_t cb = o.size() - cls0;
        if (cls == 0) g_rep.cells_bytes = cb; else g_rep.empt_bytes = cb;
        if (trace) std::fprintf(stderr, "[e2 enc] class %d: %zu cols, %zu bytes, elapsed %.0f ms (refit so far %.0f ms)\n", cls, order.size(), cb, t_enc_all.ms(), g_rep.enc_refit_ms);
    }
    for (int i = 0; i < 64; ++i) o.push_back(0);       // read padding for the branchless renormalisation
    g_rep.payload_bytes = o.size() - pay0;
    for (u32 j = 0; j < n; ++j) {
        u64 k = M.indptr[j + 1] - M.indptr[j];
        if (isc[j]) { g_rep.nnz_cells += k; ++g_rep.ncells; } else { g_rep.nnz_empt += k; ++g_rep.nempt; }
    }
    return o;
}

// ---------------------------------------------------------------------------------------------------------
// decoder
struct DLane {
    u64 xg, xc;
    const u32 *pg, *pc;
    u32 t, ncol;
    u32 cid, b, i1;
    u32 x0;
    u32 *ix, *dv, *start_ix;
    bool active;
};

template <class T> struct RawBuf {                 // uninitialised array (no zero-fill pass)
    std::unique_ptr<T[]> p;
    RawBuf() {}
    explicit RawBuf(size_t n) : p(new T[n]) {}
    void alloc(size_t n) { p.reset(new T[n]); }
    T* data() { return p.get(); }
    const T* data() const { return p.get(); }
    T& operator[](size_t i) { return p[i]; }
};

// scalar reference decoder for one block (all lanes, round-robin)
inline void decode_block_scalar(ClassModel& md, DLane* lane, int LG, const std::vector<u32>& order, size_t pos,
                                const std::vector<u8>& bk, const std::vector<u64>& N, u32* kcol, u64* off,
                                u32* stI_base, int PBITS) {
    const u32 m = md.m;
    const u32* const S = md.S.data();
    const BucketParams* const bp = md.bp.data();
    const u32* const idx = md.idx.data();
    const u32* const clut = md.clut.data();
    u32* const z = md.z.data();
    u32* const cn = md.cn.data();
    u64 nesc = 0, ncorr = 0;
    int nactive = 0;
    for (int l = 0; l < LG; ++l) if (lane[l].active) ++nactive;
    while (nactive) {
        for (int l = 0; l < LG; ++l) {
            DLane& L = lane[l];
            if (!L.active) continue;
            u64 xg = L.xg;
            const u64 slot = xg & (MSC - 1);
            const u32 b = L.b;
            const u32 x0 = L.x0;
            const BucketParams& B = bp[b];
            u32 j; u64 D0, D1;
            if (__builtin_expect(slot >= FCUT, 0)) {
                const u32* pg = L.pg;
                xg = (MSC - FCUT) * (xg >> MB) + (slot - FCUT);
                renorm(xg, pg);
                j = (u32)(xg & ((1u << PBITS) - 1));
                xg >>= PBITS;
                renorm(xg, pg);
                L.pg = pg;
                ++nesc;
                D0 = ((u64)(S[j] - x0) * B.M) >> B.k; D1 = ((u64)(S[j + 1] - x0) * B.M) >> B.k;
            } else {
                const u32 rem = (u32)(MSC - slot);
                const u64 E = ((u64)31 << FR) - lg2fix(rem);
                const u64 Ts = (u64)x0 + ((E * B.R) >> B.s);
                u32 jj;
                if (Ts >= S[m]) jj = m;
                else {
                    const u32 Ts32 = (u32)Ts;
                    const u32 k = Ts32 >> (32 - IDXB);
                    const u32 a = idx[k], bnd = idx[k + 1];
#ifdef E2_STATS
                    ++g_win[std::min<u32>(bnd - a, 4095)];
#endif
                    u32 c;
                    if (bnd - a <= 24) {
                        c = a;
                        for (u32 f = a; f < bnd; ++f) c += (S[f] <= Ts32);
                    } else {
                        u32 lo = a, hi = bnd;
                        while (lo < hi) { u32 mid = (lo + hi) >> 1; if (S[mid] <= Ts32) lo = mid + 1; else hi = mid; }
                        c = lo;
                    }
                    jj = c - 1;
                }
                if (jj < L.i1) jj = L.i1;
                u64 c0, c1;
                for (;;) {
                    D0 = ((u64)(S[jj] - x0) * B.M) >> B.k;
                    c0 = MSC - G(D0);
                    if (c0 > slot) { --jj; ++ncorr; continue; }
                    D1 = ((u64)(S[jj + 1] - x0) * B.M) >> B.k;
                    c1 = (jj >= m) ? FCUT : MSC - G(D1);
                    if (c1 <= slot) { ++jj; ++ncorr; continue; }
                    break;
                }
                j = jj;
                const u32* pg = L.pg;
                xg = (c1 - c0) * (xg >> MB) + slot - c0;
                renorm(xg, pg);
                L.pg = pg;
            }
            L.xg = xg;
            if (__builtin_expect(j >= m, 0)) {
                const u32 k = (u32)(L.ix - L.start_ix);
                kcol[L.cid] = k;
                md.add_column(L.b, k, N[L.cid]);
                ++L.t;
                if (L.t >= L.ncol) { L.active = false; --nactive; }
                else {
                    L.cid = order[pos + l + (size_t)L.t * LG]; L.b = bk[L.cid]; L.i1 = 0; L.x0 = 0;
                    L.start_ix = L.ix; off[L.cid] = (u64)(L.ix - stI_base);
                }
                continue;
            }
            {   // count
                const u32 ctx = count_ctx(D1 - D0);
                u64 xc = L.xc;
                const u32 e32 = clut[((size_t)ctx << CPB) + (xc & ((1u << CPB) - 1))];
                const u32 s = e32 & 0xFF;
                xc = (u64)((e32 >> 8) & 0xFFF) * (xc >> CPB) + (e32 >> 20);
                const u32* pc = L.pc;
                renorm(xc, pc);
                u32 cnt = s + 1;
                if (__builtin_expect(s == 15, 0)) {
                    u32 nbv = (u32)(xc & 31); xc >>= 5; renorm(xc, pc);
                    u32 low = 0;
                    if (nbv > 16) {
                        u32 hi2 = (u32)(xc & ((1u << (nbv - 16)) - 1)); xc >>= (nbv - 16); renorm(xc, pc);
                        u32 lo2 = (u32)(xc & 0xFFFF); xc >>= 16; renorm(xc, pc);
                        low = (hi2 << 16) | lo2;
                    } else if (nbv > 0) {
                        low = (u32)(xc & ((1u << nbv) - 1)); xc >>= nbv; renorm(xc, pc);
                    }
                    cnt = 15 + (1u << nbv) + low;
                }
                L.xc = xc; L.pc = pc;
                cn[(size_t)ctx * NSYM + s] += 64;
                *L.ix++ = j; *L.dv++ = cnt;
                z[j]++;
            }
            L.i1 = j + 1;
            L.x0 = S[j + 1];
        }
    }
    g_rep.nesc += nesc; g_rep.ncorr += ncorr;
}

inline CanonicalMatrix decode_e2(const std::vector<u8>& buf) {
    using namespace detail;
    Timer t_all;
    const u8* p = buf.data();
    if (std::memcmp(p, "E2v2", 4) != 0) throw std::runtime_error("e2: bad magic");
    p += 4;
    CanonicalMatrix M;
    M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
    u32 raw[NPARAMS];
    for (int i = 0; i < NPARAMS; ++i) raw[i] = get_u32(p);
    const Params P = params_unpack(raw);
    const u32 m = M.m, n = M.n;
    const u8* names_p = p;
    { const u8* q = p; get_stream(q); get_stream(q); get_u32(q); get_stream(q); p = q; }
    Timer t_hdr;
    u64 hlen = get_u64(p);
    std::vector<u64> N; std::vector<u8> isc;
    {
        RcDec rc(p, p + hlen);
        hdr_decode(rc, n, N, isc);
    }
    p += hlen;
    g_rep.dec_hdr_ms = std::min(g_rep.dec_hdr_ms, t_hdr.ms());
    Timer t_pay; const double cyc0 = thread_cycles();
    u64 Nmax = 1;
    for (u32 j = 0; j < n; ++j) Nmax = std::max(Nmax, N[j]);
    const int nbk = (int)bucket_of(Nmax, P.bpo_log) + 1;
    const int PBITS = pos_bits_of(m);
    const int LG = (int)P.lanes;
    std::vector<u8> bk(n, 0);
    for (u32 j = 0; j < n; ++j) if (N[j]) bk[j] = (u8)bucket_of(N[j], P.bpo_log);
    RawBuf<u32> stI(M.nnz + 64), stV(M.nnz + 64);
    std::vector<u64> off(n, 0);
    std::vector<u32> kcol(n, 0);
    u64 stpos = 0;
    double refit_ms = 0, lanes_ms = 0, lanes_cyc = 0;
    DLane lane[16];
    std::vector<u32> prior;
    for (int cls = 0; cls < 2; ++cls) {
        std::vector<u32> order = class_columns(N, isc, cls, P.seed);
        ClassModel md(P, m, nbk);
        if (cls == 1 && !prior.empty() && P.theta_x16 > 0) md.prior_w = prior;
        u64 pos = 0;
        while (pos < order.size()) {
            size_t e = block_end(P, N, order, pos);
            Timer tr; md.refit(); refit_ms += tr.ms();
            u64 lnnz[16], lgw[16], lcw[16];
            for (int l = 0; l < LG; ++l) { lnnz[l] = get_vi(p); lgw[l] = get_vi(p); lcw[l] = get_vi(p); }
            const u32* wp = reinterpret_cast<const u32*>(p);
            u64 sp = stpos;
            for (int l = 0; l < LG; ++l) {
                DLane& L = lane[l];
                const u32* gwp = wp; const u32* cwp = wp + lgw[l];
                wp += lgw[l] + lcw[l];
                L.xg = ((u64)ld32w(gwp) << 32) | ld32w(gwp + 1); L.pg = gwp + 2;
                L.xc = ((u64)ld32w(cwp) << 32) | ld32w(cwp + 1); L.pc = cwp + 2;
                L.t = 0;
                L.ncol = 0;
                for (size_t q = pos + l; q < e; q += LG) ++L.ncol;
                L.ix = stI.data() + sp; L.dv = stV.data() + sp;
                sp += lnnz[l];
                L.active = L.ncol > 0;
                if (L.active) {
                    L.cid = order[pos + l]; L.b = bk[L.cid]; L.i1 = 0; L.x0 = 0;
                    L.start_ix = L.ix; off[L.cid] = (u64)(L.ix - stI.data());
                }
            }
            p = reinterpret_cast<const u8*>(wp);
            Timer tl; const double lc0 = thread_cycles();
            decode_block_scalar(md, lane, LG, order, pos, bk, N, kcol.data(), off.data(), stI.data(), PBITS);
            lanes_ms += tl.ms(); lanes_cyc += thread_cycles() - lc0;
            stpos = sp;
            pos = e;
        }
        if (cls == 0 && P.theta_x16 > 0) { Timer tr; md.refit(); refit_ms += tr.ms(); md.export_prior(prior); }
    }
    g_rep.dec_refit_ms = std::min(g_rep.dec_refit_ms, refit_ms);
    g_rep.dec_lanes_ms = std::min(g_rep.dec_lanes_ms, lanes_ms);
    g_rep.dec_lanes_cyc = std::min(g_rep.dec_lanes_cyc, lanes_cyc);
    Timer t_g;
    M.indptr.assign(n + 1, 0);
    for (u32 j = 0; j < n; ++j) M.indptr[j + 1] = M.indptr[j] + kcol[j];
    if (M.indptr[n] != M.nnz) throw std::runtime_error("e2: nnz mismatch after decode");
    M.indices.resize(M.nnz); M.data.resize(M.nnz);
    for (u32 j = 0; j < n; ++j) {
        u32 k = kcol[j];
        if (!k) continue;
        std::memcpy(M.indices.data() + M.indptr[j], stI.data() + off[j], (size_t)k * 4);
        std::memcpy(M.data.data() + M.indptr[j], stV.data() + off[j], (size_t)k * 4);
    }
    g_rep.dec_gather_ms = std::min(g_rep.dec_gather_ms, t_g.ms());
    g_rep.dec_payload_ms = std::min(g_rep.dec_payload_ms, t_pay.ms());
    g_rep.dec_payload_cyc = std::min(g_rep.dec_payload_cyc, thread_cycles() - cyc0);
    {
        const u8* q = names_p;
        M.rownames = split_names(get_stream(q), M.m);
        M.colnames = split_names(get_stream(q), M.n);
        u32 nkv = get_u32(q);
        auto kv = get_stream(q);
        auto parts = split_names(kv, 2 * (size_t)nkv);
        for (u32 i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
    }
    g_rep.dec_total_ms = std::min(g_rep.dec_total_ms, t_all.ms());
    return M;
}

inline Codec make_e2(const std::string& name, Params P, bool from_env = true) {
    Codec c;
    c.name = name;
    c.encode = [P, from_env](const CanonicalMatrix& M) { return encode_e2(M, from_env ? params_from_env(P) : P); };
    c.decode = [](const std::vector<u8>& b) { return decode_e2(b); };
    return c;
}

}  // namespace e2

inline void register_e2(std::vector<Codec>& v) {
    v.push_back(e2::make_e2("e2_b1024", e2::Params()));
}
}  // namespace cs
