// Track B: chunk decoder (native rank-space output) and canonicalisation (original row ids, ascending).
#pragma once
#include "track_b_lane.h"
#include "track_b_avx512.h"
#ifdef TB_PROFILE
#include <x86intrin.h>
#define TB_TS(v) uint64_t v = __rdtsc()
#else
#define TB_TS(v)
#endif

namespace cs {
namespace tb {

inline void decode_flat(const uint8_t* p, size_t nw, const uint32_t* tab, int pb, const uint32_t* ctxb, size_t N, uint32_t* out) {
#ifdef TB_HAVE_AVX512
    if (use_simd()) { decode_flat_avx512(p, tab, pb, ctxb, N, out); return; }
#endif
    decode_flat_scalar(p, nw, tab, pb, ctxb, N, out);
}

inline void decode_open(const uint8_t*& p, DecState& D, uint32_t n, uint32_t chunk_cols) {
    D.n = n; D.chunk_cols = chunk_cols;
    D.cfg = Cfg::get(p);
    {  // counts
        DecTables CT = parse_tables(p, 12, 1);
        std::vector<uint32_t> k(static_cast<size_t>(n) + 16);
        size_t nw = leb_get(p);
        decode_flat(p, nw, CT.tab.get(), 12, nullptr, n, k.data());
        p += 4 * kLanes + nw * 2;
        D.ptr.assign(static_cast<size_t>(n) + 1, 0);
        for (uint32_t j = 0; j < n; ++j) D.ptr[j + 1] = D.ptr[j] + k[j];
    }
    D.gap = parse_tables(p, D.cfg.pb, D.cfg.nctx());
    D.res = parse_tables(p, D.cfg.pbv, kResCtx);
    const uint32_t nchunks = (n + chunk_cols - 1) / chunk_cols;
    std::vector<size_t> sz(nchunks);
    for (auto& s : sz) s = leb_get(p);
    D.blob.resize(nchunks);
    for (uint32_t c = 0; c < nchunks; ++c) { D.blob[c] = p; p += sz[c]; }
}

// Decode chunk c: out_rk / out_val point at the chunk's first element (global position D.ptr[c0]).
inline void decode_chunk_native(const DecState& D, uint32_t c, uint32_t* out_rk, uint32_t* out_val, Scratch& sc) {
    const uint32_t c0 = c * D.chunk_cols, c1 = std::min(D.n, c0 + D.chunk_cols), nc = c1 - c0;
    const uint32_t* kptr = D.ptr.data() + c0;
    const uint32_t base = D.ptr[c0];
    const uint32_t cn = D.ptr[c1] - base;

    uint32_t bnd[kLanes + 1];
    {
        sc.kk.resize(nc);
        for (uint32_t j = 0; j < nc; ++j) sc.kk[j] = kptr[j + 1] - kptr[j];
        lane_partition(sc.kk.data(), nc, kLanes, bnd);
    }
    sc.kl.clear();
    MainIn in;
    for (int l = 0; l < kLanes; ++l) {
        in.kl_start[l] = static_cast<uint32_t>(sc.kl.size());
        in.left[l] = 0;
        in.pos0[l] = D.ptr[c0 + bnd[l]] - base;
        for (uint32_t j = bnd[l]; j < bnd[l + 1]; ++j) {
            uint32_t k = kptr[j + 1] - kptr[j];
            if (k) { sc.kl.push_back(k); in.left[l] += k; }
        }
    }
    sc.kl.push_back(0);
    in.kl = sc.kl.data();

    const uint8_t* q = D.blob[c];
    size_t nwm = leb_get(q), nwr = leb_get(q);
    uint32_t stm[kLanes];
    for (int l = 0; l < kLanes; ++l) stm[l] = *reinterpret_cast<const u32u*>(q + 4 * l);
    const uint8_t* mw = q + 4 * kLanes;
    const uint8_t* rs = mw + nwm * 2;

#ifdef TB_HAVE_AVX512
    const bool simd = use_simd();
#else
    const bool simd = false;
#endif
    sc.fl.resize(cn + 64);
    sc.flr.resize(cn + 64);
    FlagList fl;
    fl.item = sc.fl.data();
    fl.rank = sc.flr.data();
    TB_TS(t0);
    if (simd) {
#ifdef TB_HAVE_AVX512
        main_pass_avx512(D, mw, in, stm, out_rk, fl);
#endif
    } else {
        main_pass_scalar(D, mw, in, stm, out_rk, fl);
    }
    TB_TS(t1);
    std::fill(out_val, out_val + cn, 1u);
    TB_TS(t2);
    const uint32_t nf = fl.nf;
    if (nf) {
        sc.ctxb.resize(nf + 64);
        sc.vals.resize(nf + 64);
        const int pbv = D.cfg.pbv;
        uint32_t* cb = sc.ctxb.data();
#ifdef TB_HAVE_AVX512
        if (simd) {
            res_ctx_avx512(fl.item, fl.rank, nf, cb, pbv);
        } else
#endif
        {
            for (uint32_t j = 0; j < nf; ++j) cb[j] = ((posb_raw(fl.rank[j]) / 2) * 4 + (fl.item[j] >> 28)) << pbv;
        }
        TB_TS(t3);
        decode_flat(rs, nwr, D.res.tab.get(), pbv, cb, nf, sc.vals.data());
        TB_TS(t4);
#ifdef TB_HAVE_AVX512
        if (simd) {
            scatter_vals_avx512(fl.item, nf, sc.vals.data(), out_val);
        } else
#endif
        {
            for (uint32_t j = 0; j < nf; ++j) out_val[fl.item[j] & 0x0FFFFFFFu] = 2 + sc.vals[j];
        }
#ifdef TB_PROFILE
        sc.prof[2] += t3 - t2; sc.prof[3] += t4 - t3; sc.prof[4] += __rdtsc() - t4;
#endif
    }
#ifdef TB_PROFILE
    sc.prof[0] += t1 - t0; sc.prof[1] += t2 - t1;
#endif
}

// ------------------------------------------------------------------------------------------------
// Canonicalisation: rank ids -> original row ids, ascending, values carried along. In place per column.
// Small columns: insertion sort of packed (orig,value) keys. Large columns: two-level bitmap over orig ids.
// ------------------------------------------------------------------------------------------------
struct Canon {
    const uint32_t* perm;
    uint32_t m;
    std::vector<uint64_t> l0, l1;
    std::vector<uint32_t> vt;
    Canon(const uint32_t* perm_, uint32_t m_) : perm(perm_), m(m_) {
        l0.assign(((m + 63) / 64) + 8, 0);
        l1.assign(l0.size() / 64 + 8, 0);
        vt.assign(static_cast<size_t>(m) + 64, 0);
    }
    void column(uint32_t* idx, uint32_t* val, uint32_t k) {
        if (k == 0) return;
        if (k == 1) { idx[0] = perm[idx[0]]; return; }
        if (k <= 24) {
            uint64_t key[24];
            for (uint32_t i = 0; i < k; ++i) key[i] = (static_cast<uint64_t>(perm[idx[i]]) << 32) | val[i];
            for (uint32_t i = 1; i < k; ++i) {
                uint64_t v = key[i];
                uint32_t j = i;
                while (j > 0 && key[j - 1] > v) { key[j] = key[j - 1]; --j; }
                key[j] = v;
            }
            for (uint32_t i = 0; i < k; ++i) { idx[i] = static_cast<uint32_t>(key[i] >> 32); val[i] = static_cast<uint32_t>(key[i]); }
            return;
        }
        uint64_t* a0 = l0.data();
        uint64_t* a1 = l1.data();
        uint32_t* v = vt.data();
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t o = perm[idx[i]];
            a0[o >> 6] |= 1ull << (o & 63);
            a1[o >> 12] |= 1ull << ((o >> 6) & 63);
            v[o] = val[i];
        }
        uint32_t p = 0;
        const size_t n1 = (static_cast<size_t>(m) >> 12) + 1;
        for (size_t i1 = 0; i1 < n1; ++i1) {
            uint64_t b1 = a1[i1];
            if (!b1) continue;
            a1[i1] = 0;
            do {
                uint32_t i0 = static_cast<uint32_t>(i1 << 6) | static_cast<uint32_t>(__builtin_ctzll(b1));
                b1 &= b1 - 1;
                uint64_t b0 = a0[i0];
                a0[i0] = 0;
                do {
                    uint32_t o = (i0 << 6) | static_cast<uint32_t>(__builtin_ctzll(b0));
                    b0 &= b0 - 1;
                    idx[p] = o;
                    val[p] = v[o];
                    ++p;
                } while (b0);
            } while (b1);
        }
    }
};

}  // namespace tb
}  // namespace cs
