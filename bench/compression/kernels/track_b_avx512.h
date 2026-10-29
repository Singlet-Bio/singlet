// Track B: AVX-512 (F/BW/DQ/CD/VBMI2 + BMI2) kernels for the lock-step rANS format in track_b_lane.h.
// kLanes lanes are processed as NV = kLanes/16 independent zmm vectors to hide gather / expand latency.
// Every kernel is bit-exact with its scalar counterpart (checked by harness/track_b_bench.cpp via use_simd()).
#pragma once
#include "track_b_lane.h"

#ifdef TB_HAVE_AVX512
namespace cs {
namespace tb {

constexpr int kNV = kLanes / 16;

// bucket index (2 per octave) of position p, vector form of posb_raw
static inline __m512i posb_vec(__m512i p) {
    const __m512i one = _mm512_set1_epi32(1);
    __m512i q = _mm512_add_epi32(p, _mm512_set1_epi32(2));
    __m512i lg = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(q));
    __m512i nxt = _mm512_and_si512(_mm512_srlv_epi32(q, _mm512_sub_epi32(lg, one)), one);
    return _mm512_sub_epi32(_mm512_add_epi32(_mm512_slli_epi32(lg, 1), nxt), _mm512_set1_epi32(2));
}

// renorm: lanes in `m` read one u16 word from the shared stream (lane order).
static inline __m512i renorm_vec(__m512i X, __mmask16 m, const uint8_t*& wp) {
#ifdef TB_PROBE_NOEXPAND  // timing probe only: wrong output
    (void)wp;
    return _mm512_mask_mov_epi32(X, m, _mm512_or_si512(_mm512_slli_epi32(X, 16), _mm512_set1_epi32(0x8000)));
#elif defined(TB_EXPAND_MEM)
    __m512i words = _mm512_maskz_expandloadu_epi16(_pdep_u32(static_cast<uint32_t>(m), 0x55555555u), wp);
    wp += 2 * _mm_popcnt_u32(static_cast<uint32_t>(m));
    return _mm512_mask_mov_epi32(X, m, _mm512_or_si512(_mm512_slli_epi32(X, 16), words));
#else
    // plain 64-byte load + register-form expand: the memory-form vpexpandw is microcoded and dominated the main
    // pass (probe: 18 of 42 thread-cycles/nnz). Needs 64 readable bytes past the stream (payload has 64 B padding).
    __m512i raw = _mm512_loadu_si512(wp);
    __m512i words = _mm512_maskz_expand_epi16(_pdep_u32(static_cast<uint32_t>(m), 0x55555555u), raw);
    wp += 2 * _mm_popcnt_u32(static_cast<uint32_t>(m));
    return _mm512_mask_mov_epi32(X, m, _mm512_or_si512(_mm512_slli_epi32(X, 16), words));
#endif
}

// class -> (base, nb) arithmetic
static inline void cls_decode_vec(__m512i cls, __m512i& base, __m512i& nb) {
    const __mmask16 big = _mm512_cmpge_epu32_mask(cls, _mm512_set1_epi32(8));
    __m512i t = _mm512_sub_epi32(cls, _mm512_set1_epi32(8));
    nb = _mm512_maskz_add_epi32(big, _mm512_srli_epi32(t, 2), _mm512_set1_epi32(1));
    __m512i gb = _mm512_sllv_epi32(_mm512_add_epi32(_mm512_set1_epi32(4), _mm512_and_si512(t, _mm512_set1_epi32(3))), nb);
    base = _mm512_mask_blend_epi32(big, cls, gb);
}

inline void main_pass_avx512(const DecState& D, const uint8_t* wp, const MainIn& in, const uint32_t* st, uint32_t* out_rk, FlagList& fl) {
    const Cfg& cfg = D.cfg;
    const uint32_t pb = cfg.pb;
    const uint32_t* tab = D.gap.tab.get();
    const __m512i vmask = _mm512_set1_epi32(static_cast<int>((1u << pb) - 1));
    const __m512i vzero = _mm512_setzero_si512(), vone = _mm512_set1_epi32(1);
    const __m512i vnkpb = _mm512_set1_epi32(static_cast<int>(static_cast<uint32_t>(cfg.nk) << pb));
    const __m512i vcap = _mm512_set1_epi32(cfg.cap), vkcap = _mm512_set1_epi32(cfg.kcap);
    const __m128i csh = _mm_cvtsi32_si128(cfg.sh), cks = _mm_cvtsi32_si128(cfg.ks), cpb = _mm_cvtsi32_si128(pb);
    const __m512i v4095 = _mm512_set1_epi32(0xFFF), v3 = _mm512_set1_epi32(3), v11 = _mm512_set1_epi32(11);
    const __m512i v65536 = _mm512_set1_epi32(65536), vff = _mm512_set1_epi32(0xFF);

    __m512i X[kNV], LEFT[kNV], OUT[kNV], KP[kNV], PP[kNV], REM[kNV], KBP[kNV], DK[kNV];
    uint32_t steps = 0;
    for (int l = 0; l < kLanes; ++l) steps = std::max(steps, in.left[l]);
    for (int v = 0; v < kNV; ++v) {
        X[v] = _mm512_loadu_si512(st + 16 * v);
        LEFT[v] = _mm512_loadu_si512(in.left + 16 * v);
        OUT[v] = _mm512_loadu_si512(in.pos0 + 16 * v);
        KP[v] = _mm512_loadu_si512(in.kl_start + 16 * v);
        PP[v] = REM[v] = KBP[v] = DK[v] = vzero;
    }

    auto colinit = [&](int v, __mmask16 m) {
        __m512i k = _mm512_mask_i32gather_epi32(vzero, m, KP[v], in.kl, 4);
        __m512i kraw = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(_mm512_add_epi32(k, vone)));
        __m512i kc = _mm512_min_epu32(_mm512_srl_epi32(kraw, cks), vkcap);
        KBP[v] = _mm512_mask_mov_epi32(KBP[v], m, _mm512_sll_epi32(kc, cpb));
        DK[v] = _mm512_mask_mov_epi32(DK[v], m, _mm512_min_epu32(_mm512_srli_epi32(_mm512_mullo_epi32(kraw, v11), 5), v3));
        REM[v] = _mm512_mask_mov_epi32(REM[v], m, k);
        PP[v] = _mm512_mask_mov_epi32(PP[v], m, vzero);
    };
    for (int v = 0; v < kNV; ++v) colinit(v, _mm512_test_epi32_mask(LEFT[v], LEFT[v]));

    uint32_t nf = 0;
    uint32_t* const fl_item = fl.item;
    uint32_t* const fl_rank = fl.rank;
    __m512i E[kNV];
    __mmask16 act[kNV];
    for (uint32_t s = 0; s < steps; ++s) {
        // ---- phase A: symbol decode (+ renorm), lanes ascending ----
        for (int v = 0; v < kNV; ++v) {
            act[v] = _mm512_test_epi32_mask(LEFT[v], LEFT[v]);
            __m512i b = posb_vec(PP[v]);
            __m512i pc = _mm512_min_epu32(_mm512_srl_epi32(b, csh), vcap);
            __m512i idx = _mm512_add_epi32(_mm512_add_epi32(_mm512_mullo_epi32(pc, vnkpb), KBP[v]), _mm512_and_si512(X[v], vmask));
#ifdef TB_PROBE_NOGATHER  // timing probe only: wrong output
            E[v] = _mm512_or_si512(_mm512_and_si512(idx, _mm512_set1_epi32(0x3F)), _mm512_set1_epi32(512 << 8));
#else
            E[v] = _mm512_mask_i32gather_epi32(vzero, act[v], idx, tab, 4);
#endif
            __m512i f = _mm512_and_si512(_mm512_srli_epi32(E[v], 8), v4095);
            __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(X[v], cpb)), _mm512_srli_epi32(E[v], 20));
            X[v] = _mm512_mask_mov_epi32(X[v], act[v], Xn);
            X[v] = renorm_vec(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
        }
        // ---- phase B: extra bits (+ renorm), output, column bookkeeping ----
        for (int v = 0; v < kNV; ++v) {
            __m512i sym = _mm512_and_si512(E[v], vff);
            __m512i cls = _mm512_srli_epi32(sym, 1);
            __m512i gbase, nb;
            cls_decode_vec(cls, gbase, nb);
            __m512i ex = _mm512_and_si512(X[v], _mm512_sub_epi32(_mm512_sllv_epi32(vone, nb), vone));
            X[v] = _mm512_mask_mov_epi32(X[v], act[v], _mm512_srlv_epi32(X[v], nb));
            X[v] = renorm_vec(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
            __m512i rank = _mm512_add_epi32(_mm512_add_epi32(PP[v], gbase), ex);
#ifdef TB_PROBE_NOSCATTER  // timing probe only: wrong output
            { static thread_local uint32_t dummy[16 * 1024 + 16]; _mm512_storeu_si512(dummy + 16 * (s & 1023), rank); }
#else
            _mm512_mask_i32scatter_epi32(out_rk, act[v], OUT[v], rank, 4);
#endif
            __mmask16 fm = act[v] & _mm512_test_epi32_mask(sym, vone);
            if (fm) {
                _mm512_storeu_si512(fl_item + nf, _mm512_maskz_compress_epi32(fm, _mm512_or_si512(OUT[v], _mm512_slli_epi32(DK[v], 28))));
                _mm512_storeu_si512(fl_rank + nf, _mm512_maskz_compress_epi32(fm, rank));
                nf += static_cast<uint32_t>(_mm_popcnt_u32(fm));
            }
            PP[v] = _mm512_mask_add_epi32(PP[v], act[v], rank, vone);
            OUT[v] = _mm512_mask_add_epi32(OUT[v], act[v], OUT[v], vone);
            LEFT[v] = _mm512_mask_sub_epi32(LEFT[v], act[v], LEFT[v], vone);
            REM[v] = _mm512_mask_sub_epi32(REM[v], act[v], REM[v], vone);
            __mmask16 ce = act[v] & _mm512_testn_epi32_mask(REM[v], REM[v]) & _mm512_test_epi32_mask(LEFT[v], LEFT[v]);
            if (ce) {
                KP[v] = _mm512_mask_add_epi32(KP[v], ce, KP[v], vone);
                colinit(v, ce);
            }
        }
    }
    fl.nf = nf;
}

// Flat decode (counts / residuals). N elements in lanes j % kLanes, optional per-element table offsets.
inline void decode_flat_avx512(const uint8_t* p, const uint32_t* tab, int pb, const uint32_t* ctxb, size_t N, uint32_t* out) {
    const __m512i vmask = _mm512_set1_epi32(static_cast<int>((1u << pb) - 1));
    const __m512i vzero = _mm512_setzero_si512(), vone = _mm512_set1_epi32(1);
    const __m512i v4095 = _mm512_set1_epi32(0xFFF), v65536 = _mm512_set1_epi32(65536), vff = _mm512_set1_epi32(0xFF);
    const __m128i cpb = _mm_cvtsi32_si128(pb);
    __m512i X[kNV], E[kNV];
    __mmask16 act[kNV];
    for (int v = 0; v < kNV; ++v) X[v] = _mm512_loadu_si512(p + 64 * v);
    const uint8_t* wp = p + 4 * kLanes;
    for (size_t j0 = 0; j0 < N; j0 += kLanes) {
        for (int v = 0; v < kNV; ++v) {
            size_t base = j0 + 16 * v;
            size_t na = base < N ? std::min<size_t>(16, N - base) : 0;
            act[v] = static_cast<__mmask16>((1u << na) - 1);
            __m512i cv = ctxb ? _mm512_maskz_loadu_epi32(act[v], ctxb + base) : vzero;
            __m512i idx = _mm512_add_epi32(cv, _mm512_and_si512(X[v], vmask));
            E[v] = _mm512_mask_i32gather_epi32(vzero, act[v], idx, tab, 4);
            __m512i f = _mm512_and_si512(_mm512_srli_epi32(E[v], 8), v4095);
            __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(X[v], cpb)), _mm512_srli_epi32(E[v], 20));
            X[v] = _mm512_mask_mov_epi32(X[v], act[v], Xn);
            X[v] = renorm_vec(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
        }
        for (int v = 0; v < kNV; ++v) {
            size_t base = j0 + 16 * v;
            __m512i cls = _mm512_and_si512(E[v], vff);
            __m512i gbase, nb;
            cls_decode_vec(cls, gbase, nb);
            __m512i ex = _mm512_and_si512(X[v], _mm512_sub_epi32(_mm512_sllv_epi32(vone, nb), vone));
            X[v] = _mm512_mask_mov_epi32(X[v], act[v], _mm512_srlv_epi32(X[v], nb));
            X[v] = renorm_vec(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
            if (act[v]) _mm512_mask_storeu_epi32(out + base, act[v], _mm512_add_epi32(gbase, ex));
        }
    }
}

// Pass 2b helpers: residual table offsets from the flagged list, and value scatter.
inline void res_ctx_avx512(const uint32_t* item, const uint32_t* rank, uint32_t nf, uint32_t* cb, int pbv) {
    const __m128i cpb = _mm_cvtsi32_si128(pbv);
    for (uint32_t j = 0; j < nf; j += 16) {
        uint32_t na = std::min<uint32_t>(16, nf - j);
        __mmask16 m = static_cast<__mmask16>((1u << na) - 1);
        __m512i dk = _mm512_srli_epi32(_mm512_maskz_loadu_epi32(m, item + j), 28);
        __m512i r = _mm512_maskz_loadu_epi32(m, rank + j);
        __m512i c = _mm512_add_epi32(_mm512_slli_epi32(_mm512_srli_epi32(posb_vec(r), 1), 2), dk);
        _mm512_mask_storeu_epi32(cb + j, m, _mm512_sll_epi32(c, cpb));
    }
}
inline void scatter_vals_avx512(const uint32_t* item, uint32_t nf, const uint32_t* vals, uint32_t* out_val) {
    for (uint32_t j = 0; j < nf; j += 16) {
        uint32_t na = std::min<uint32_t>(16, nf - j);
        __mmask16 m = static_cast<__mmask16>((1u << na) - 1);
        __m512i pos = _mm512_and_si512(_mm512_maskz_loadu_epi32(m, item + j), _mm512_set1_epi32(0x0FFFFFFF));
        __m512i v = _mm512_add_epi32(_mm512_maskz_loadu_epi32(m, vals + j), _mm512_set1_epi32(2));
        _mm512_mask_i32scatter_epi32(out_val, m, pos, v, 4);
    }
}

}  // namespace tb
}  // namespace cs
#endif  // TB_HAVE_AVX512
