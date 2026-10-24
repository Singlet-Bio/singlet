// SPDX-License-Identifier: MIT
// w3_fast AVX-512 (F/BW/DQ/VL/VBMI2 + BMI2) decoder. Bit-exact with decode_chunk_scalar (w3_fast_dec.h).
// NV = L/16 zmm vectors of 16 lanes each are kept in flight. Integer ops only (the float convert is an exact int->float of a
// value < 2^24, used to read the position bucket out of the exponent and top mantissa bit).
// v1 output path: scatter of ids and values into the caller's final arrays (uint32 values; other value types go through a
// temporary and a narrowing loop).
#pragma once
#include "w3_fast_dec.h"
#include <immintrin.h>

namespace cs {
namespace wf {

static inline __m512i wf_renorm(__m512i X, __mmask16 m, const uint8_t*& wp) {
    __m512i raw = _mm512_loadu_si512(wp);
    __m512i words = _mm512_maskz_expand_epi16(_pdep_u32(static_cast<uint32_t>(m), 0x55555555u), raw);
    wp += 2 * _mm_popcnt_u32(static_cast<uint32_t>(m));
    return _mm512_mask_mov_epi32(X, m, _mm512_or_si512(_mm512_slli_epi32(X, 16), words));
}

// position bucket posb_raw(p) = (bits(float(p+2)) >> 22) - 256
static inline __m512i wf_posb(__m512i p) {
    __m512i q = _mm512_add_epi32(p, _mm512_set1_epi32(2));
    __m512i bits = _mm512_castps_si512(_mm512_cvtepu32_ps(q));
    return _mm512_sub_epi32(_mm512_srli_epi32(bits, 22), _mm512_set1_epi32(256));
}

// class -> (base, nb)
static inline void wf_cls(__m512i cls, __m512i& base, __m512i& nb) {
    const __m512i one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512();
    __m512i e = _mm512_srli_epi32(cls, 2);
    __m512i mm = _mm512_and_si512(cls, _mm512_set1_epi32(3));
    nb = _mm512_max_epi32(_mm512_sub_epi32(e, one), zero);
    __m512i hi = _mm512_slli_epi32(_mm512_min_epu32(e, one), 2);
    base = _mm512_sllv_epi32(_mm512_or_si512(mm, hi), nb);
}

// 16x16 transpose of 32-bit elements: in r[i] = row i; out r[c] = column c.
static inline void wf_transpose16(__m512i r[16]) {
    __m512i t[16], u[16];
    for (int i = 0; i < 8; ++i) { t[2 * i] = _mm512_unpacklo_epi32(r[2 * i], r[2 * i + 1]); t[2 * i + 1] = _mm512_unpackhi_epi32(r[2 * i], r[2 * i + 1]); }
    for (int i = 0; i < 4; ++i) {
        u[4 * i + 0] = _mm512_unpacklo_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 1] = _mm512_unpackhi_epi64(t[4 * i], t[4 * i + 2]);
        u[4 * i + 2] = _mm512_unpacklo_epi64(t[4 * i + 1], t[4 * i + 3]);
        u[4 * i + 3] = _mm512_unpackhi_epi64(t[4 * i + 1], t[4 * i + 3]);
    }
    for (int j = 0; j < 4; ++j) {
        __m512i a = _mm512_shuffle_i32x4(u[j], u[4 + j], 0x88), b = _mm512_shuffle_i32x4(u[j], u[4 + j], 0xDD);
        __m512i c = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0x88), d = _mm512_shuffle_i32x4(u[8 + j], u[12 + j], 0xDD);
        r[0 + j] = _mm512_shuffle_i32x4(a, c, 0x88);
        r[8 + j] = _mm512_shuffle_i32x4(a, c, 0xDD);
        r[4 + j] = _mm512_shuffle_i32x4(b, d, 0x88);
        r[12 + j] = _mm512_shuffle_i32x4(b, d, 0xDD);
    }
}

template <int L>
inline void chunk_avx512_u32(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    constexpr int NV = L / 16;
    const Cfg& cfg = P.h.cfg;
    const int pb = cfg.pb, pbv = cfg.pbv;
    const uint32_t* gt = P.gap.tab.get();
    const uint32_t* rt = P.res.tab.get();
    const uint32_t* kl = su.kl.data();
    const __m512i vmask = _mm512_set1_epi32(static_cast<int>((1u << pb) - 1));
    const __m512i vmaskv = _mm512_set1_epi32(static_cast<int>((1u << pbv) - 1));
    const __m512i vzero = _mm512_setzero_si512(), vone = _mm512_set1_epi32(1);
    const __m512i v4095 = _mm512_set1_epi32(0xFFF), v65536 = _mm512_set1_epi32(65536);
    const __m512i vcap = _mm512_set1_epi32(cfg.cap), vkcap = _mm512_set1_epi32(cfg.kcap);
    const __m512i vnpc = _mm512_set1_epi32(static_cast<int>(cfg.npc()));
    const __m512i vremmask = _mm512_set1_epi32(0x0FFFFFFF);
    const __m512i v11 = _mm512_set1_epi32(11), v3 = _mm512_set1_epi32(3);
    const __m128i csh = _mm_cvtsi32_si128(cfg.sh), cks = _mm_cvtsi32_si128(cfg.ks), cpb = _mm_cvtsi32_si128(pb), cpbv = _mm_cvtsi32_si128(pbv);

    __m512i X[NV], PP[NV], KB[NV], REM[NV], OUT[NV], KP[NV];
    for (int v = 0; v < NV; ++v) {
        X[v] = _mm512_loadu_si512(su.states + 64 * v);
        PP[v] = _mm512_loadu_si512(su.pp0 + 16 * v);
        KB[v] = _mm512_loadu_si512(su.kb0 + 16 * v);
        REM[v] = _mm512_or_si512(_mm512_loadu_si512(su.rem0 + 16 * v), _mm512_slli_epi32(_mm512_loadu_si512(su.dk0 + 16 * v), 28));
        OUT[v] = _mm512_loadu_si512(su.e0 + 16 * v);
        KP[v] = _mm512_loadu_si512(su.kp0 + 16 * v);
    }
    const uint8_t* wp = su.words;
#ifndef WF_SCATTER
    alignas(64) uint32_t tmpI[16][L], tmpV[16][L];
    auto flush = [&](uint32_t s0, uint32_t ns) {
        for (int v = 0; v < NV; ++v) {
            for (int which = 0; which < 2; ++which) {
                __m512i r[16];
                for (uint32_t i = 0; i < 16; ++i) r[i] = i < ns ? _mm512_load_si512(which ? &tmpV[i][16 * v] : &tmpI[i][16 * v]) : _mm512_setzero_si512();
                wf_transpose16(r);
                for (int i = 0; i < 16; ++i) {
                    const int lane = 16 * v + i;
                    uint32_t valid = su.cnt[lane] > s0 ? std::min<uint32_t>(ns, su.cnt[lane] - s0) : 0;
                    __mmask16 mk = static_cast<__mmask16>((1u << valid) - 1u);
                    if (which) _mm512_mask_storeu_epi32(val_out + su.e0[lane] + s0, mk, r[i]);
                    else _mm512_mask_storeu_epi32(idx_out + su.e0[lane] + s0, mk, r[i]);
                }
            }
        }
    };
#else
    for (uint32_t i = 0; i < su.N; ++i) val_out[i] = 1;
#endif

    auto step = [&]<bool Partial>(const __mmask16* actp, uint32_t bs) {
        (void)bs;
        __m512i E[NV], RK[NV], VAL[NV];
        __mmask16 act[NV], fm[NV];
        for (int v = 0; v < NV; ++v) {
            act[v] = Partial ? actp[v] : static_cast<__mmask16>(0xFFFF);
            __m512i b = wf_posb(PP[v]);
            __m512i pc = _mm512_min_epu32(_mm512_srl_epi32(b, csh), vcap);
            __m512i idx = _mm512_add_epi32(_mm512_add_epi32(KB[v], _mm512_sll_epi32(pc, cpb)), _mm512_and_si512(X[v], vmask));
            E[v] = Partial ? _mm512_mask_i32gather_epi32(vzero, act[v], idx, gt, 4) : _mm512_i32gather_epi32(idx, gt, 4);
            __m512i f = _mm512_and_si512(_mm512_srli_epi32(E[v], 12), v4095);
            __m512i bias = _mm512_and_si512(E[v], v4095);
            __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(X[v], cpb)), bias);
            X[v] = Partial ? _mm512_mask_mov_epi32(X[v], act[v], Xn) : Xn;
            X[v] = wf_renorm(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
        }
        for (int v = 0; v < NV; ++v) {
            __m512i sym = _mm512_srli_epi32(E[v], 24);
            __m512i base, nb;
            wf_cls(_mm512_srli_epi32(sym, 1), base, nb);
            __m512i ex = _mm512_and_si512(X[v], _mm512_sub_epi32(_mm512_sllv_epi32(vone, nb), vone));
            __m512i Xn = _mm512_srlv_epi32(X[v], nb);
            X[v] = Partial ? _mm512_mask_mov_epi32(X[v], act[v], Xn) : Xn;
            X[v] = wf_renorm(X[v], act[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
            RK[v] = _mm512_add_epi32(_mm512_add_epi32(PP[v], base), ex);
            fm[v] = act[v] & _mm512_test_epi32_mask(sym, vone);
            VAL[v] = vone;
        }
        __m512i E2[NV];
        for (int v = 0; v < NV; ++v) {
            if (!fm[v]) continue;
            __m512i pcr = _mm512_srli_epi32(wf_posb(RK[v]), 1);
            __m512i dk = _mm512_srli_epi32(REM[v], 28);
            __m512i cx = _mm512_sll_epi32(_mm512_add_epi32(_mm512_slli_epi32(pcr, 2), dk), cpbv);
            __m512i idx = _mm512_add_epi32(cx, _mm512_and_si512(X[v], vmaskv));
            E2[v] = _mm512_mask_i32gather_epi32(vzero, fm[v], idx, rt, 4);
            __m512i f = _mm512_and_si512(_mm512_srli_epi32(E2[v], 12), v4095);
            __m512i bias = _mm512_and_si512(E2[v], v4095);
            __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(X[v], cpbv)), bias);
            X[v] = _mm512_mask_mov_epi32(X[v], fm[v], Xn);
            X[v] = wf_renorm(X[v], fm[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
        }
        for (int v = 0; v < NV; ++v) {
            if (!fm[v]) continue;
            __m512i base, nb;
            wf_cls(_mm512_srli_epi32(E2[v], 24), base, nb);
            __m512i ex = _mm512_and_si512(X[v], _mm512_sub_epi32(_mm512_sllv_epi32(vone, nb), vone));
            X[v] = _mm512_mask_mov_epi32(X[v], fm[v], _mm512_srlv_epi32(X[v], nb));
            X[v] = wf_renorm(X[v], fm[v] & _mm512_cmplt_epu32_mask(X[v], v65536), wp);
            VAL[v] = _mm512_mask_add_epi32(vone, fm[v], _mm512_add_epi32(base, ex), _mm512_set1_epi32(2));
        }
        for (int v = 0; v < NV; ++v) {
#ifdef WF_SCATTER
            if (Partial) _mm512_mask_i32scatter_epi32(idx_out, act[v], OUT[v], RK[v], 4);
            else _mm512_i32scatter_epi32(idx_out, OUT[v], RK[v], 4);
            if (fm[v]) _mm512_mask_i32scatter_epi32(val_out, fm[v], OUT[v], VAL[v], 4);
#else
            _mm512_store_si512(&tmpI[bs][16 * v], RK[v]);
            _mm512_store_si512(&tmpV[bs][16 * v], VAL[v]);
#endif   // val_out is pre-filled with 1
            PP[v] = _mm512_add_epi32(RK[v], vone);
            OUT[v] = _mm512_add_epi32(OUT[v], vone);
            REM[v] = _mm512_sub_epi32(REM[v], vone);
            __mmask16 ce = act[v] & _mm512_testn_epi32_mask(REM[v], vremmask);
            if (ce) {
                __m512i k = _mm512_mask_i32gather_epi32(vzero, ce, KP[v], kl, 4);
                __m512i kraw = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(_mm512_add_epi32(k, vone)));
                __m512i kc = _mm512_min_epu32(_mm512_srl_epi32(kraw, cks), vkcap);
                __m512i kb = _mm512_sll_epi32(_mm512_mullo_epi32(kc, vnpc), cpb);
                __m512i dk = _mm512_min_epu32(_mm512_srli_epi32(_mm512_mullo_epi32(kraw, v11), 5), v3);
                KB[v] = _mm512_mask_mov_epi32(KB[v], ce, kb);
                REM[v] = _mm512_mask_mov_epi32(REM[v], ce, _mm512_or_si512(k, _mm512_slli_epi32(dk, 28)));
                PP[v] = _mm512_mask_mov_epi32(PP[v], ce, vzero);
                KP[v] = _mm512_mask_add_epi32(KP[v], ce, KP[v], vone);
            }
        }
    };

    for (uint32_t s = 0; s < su.S; ++s) {
        step.template operator()<false>(nullptr, s & 15);
#ifndef WF_SCATTER
        if ((s & 15) == 15) flush(s - 15, 16);
#endif
    }
    if (su.r) {
        __mmask16 act[NV];
        for (int v = 0; v < NV; ++v) {
            int a = static_cast<int>(su.r) - 16 * v;
            act[v] = a <= 0 ? 0 : a >= 16 ? 0xFFFF : static_cast<__mmask16>((1u << a) - 1);
        }
        step.template operator()<true>(act, su.S & 15);
    }
#ifndef WF_SCATTER
    {
        uint32_t s0 = su.S & ~15u, ns = su.steps - s0;
        if (ns) flush(s0, ns);
    }
#endif
}

template <class OT>
inline void decode_chunk_avx512(const Prep& P, const ChunkInfo& ci, int32_t* idx_out, OT* val_out, Setup& su) {
    setup_chunk(P, ci, su);
    thread_local std::vector<uint32_t> tmp;
    uint32_t* v32;
    if constexpr (std::is_same_v<OT, uint32_t>) v32 = val_out;
    else { if (tmp.size() < su.N + 64) tmp.resize(su.N + 64); v32 = tmp.data(); }
    switch (su.L) {
        case 16: chunk_avx512_u32<16>(P, su, idx_out, v32); break;
        case 32: chunk_avx512_u32<32>(P, su, idx_out, v32); break;
        default: chunk_avx512_u32<64>(P, su, idx_out, v32); break;
    }
    if constexpr (!std::is_same_v<OT, uint32_t>)
        for (uint32_t i = 0; i < su.N; ++i) val_out[i] = static_cast<OT>(v32[i]);
}

}  // namespace wf
}  // namespace cs
