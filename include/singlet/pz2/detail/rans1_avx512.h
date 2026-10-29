// SPDX-License-Identifier: MIT
// pz2 codec 1: AVX-512 (F/BW/DQ/VL/CD/VBMI2 + BMI2) chunk decoder.  Bit-exact with decode_chunk_scalar (rans1_dec.h).
// NV = L/16 zmm vectors of 16 lanes are kept in flight; integer ops only (the int->float convert reads the position bucket out of
// the exponent and top mantissa bit and is exact for values < 2^24).  Output goes through a [16 steps][L] scratch that is
// flushed with a 16x16 transpose into each lane's contiguous output slice, so there is no scatter.
#pragma once
#ifdef PZ2_COMPILED_AVX512

namespace singlet {
namespace pz2 {
namespace detail {
namespace r1 {

PZ2_BEGIN_AVX512

PZ2_AINLINE __m512i k512_renorm(__m512i X, __mmask16 m, const uint8_t*& wp) {
    __m512i raw = _mm512_loadu_si512(static_cast<const void*>(wp));
    __m512i words = _mm512_maskz_expand_epi16(_pdep_u32(static_cast<uint32_t>(m), 0x55555555u), raw);
    wp += 2 * _mm_popcnt_u32(static_cast<uint32_t>(m));
    return _mm512_mask_mov_epi32(X, m, _mm512_or_si512(_mm512_slli_epi32(X, 16), words));
}
PZ2_AINLINE __m512i k512_posb(__m512i p) {   // posb_raw(p) = (bits(float(p + 2)) >> 22) - 256
    __m512i q = _mm512_add_epi32(p, _mm512_set1_epi32(2));
    return _mm512_sub_epi32(_mm512_srli_epi32(_mm512_castps_si512(_mm512_cvtepu32_ps(q)), 22), _mm512_set1_epi32(256));
}
PZ2_AINLINE void k512_cls(__m512i cls, __m512i& base, __m512i& nb) {
    const __m512i one = _mm512_set1_epi32(1), zero = _mm512_setzero_si512();
    __m512i e = _mm512_srli_epi32(cls, 2);
    __m512i mm = _mm512_and_si512(cls, _mm512_set1_epi32(3));
    nb = _mm512_max_epi32(_mm512_sub_epi32(e, one), zero);
    __m512i hi = _mm512_slli_epi32(_mm512_min_epu32(e, one), 2);
    base = _mm512_sllv_epi32(_mm512_or_si512(mm, hi), nb);
}
PZ2_AINLINE __m512i k512_lowmask(__m512i bits) {   // (1 << bits) - 1
    return _mm512_sub_epi32(_mm512_sllv_epi32(_mm512_set1_epi32(1), bits), _mm512_set1_epi32(1));
}
// 16x16 transpose of 32-bit elements: in r[i] = row i; out r[c] = column c.
PZ2_AINLINE void k512_transpose16(__m512i r[16]) {
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

template <int NV>
struct K512 {
    __m512i X[NV], PP[NV], KB[NV], REM[NV], KP[NV];
    const uint8_t* wp;
};
struct C512 {
    __m512i vmask, vmaskv, vzero, vone, vtwo, v16, v20, vcap, vkcap, vnpc, vremmask, v11, v3, v65536;
    __m128i csh, cks, cpb, cpbv;
    const uint32_t *gt, *rt, *kl;
};

template <int NV, bool Partial>
PZ2_AINLINE void k512_step(K512<NV>& s, const C512& c, const __mmask16* actp, uint32_t* rowI, uint32_t* rowV) {
    __m512i E[NV], RK[NV], NBv[NV], EXv[NV], BASEv[NV], E2[NV], VAL[NV];
    __mmask16 act[NV], fm[NV];
    for (int v = 0; v < NV; ++v) {   // A
        act[v] = Partial ? actp[v] : static_cast<__mmask16>(0xFFFF);
        __m512i b = k512_posb(s.PP[v]);
        __m512i pc = _mm512_min_epu32(_mm512_srl_epi32(b, c.csh), c.vcap);
        __m512i idx = _mm512_add_epi32(_mm512_add_epi32(s.KB[v], _mm512_sll_epi32(pc, c.cpb)), _mm512_and_si512(s.X[v], c.vmask));
        E[v] = Partial ? _mm512_mask_i32gather_epi32(c.vzero, act[v], idx, c.gt, 4) : _mm512_i32gather_epi32(idx, c.gt, 4);
        __m512i f = _mm512_and_si512(_mm512_srli_epi32(E[v], 12), _mm512_set1_epi32(0xFFF));
        __m512i bias = _mm512_and_si512(E[v], _mm512_set1_epi32(0xFFF));
        __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(s.X[v], c.cpb)), bias);
        s.X[v] = Partial ? _mm512_mask_mov_epi32(s.X[v], act[v], Xn) : Xn;
        s.X[v] = k512_renorm(s.X[v], act[v] & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
    }
    for (int v = 0; v < NV; ++v) {   // B
        __m512i sym = _mm512_srli_epi32(E[v], 24);
        k512_cls(_mm512_srli_epi32(sym, 1), BASEv[v], NBv[v]);
        __m512i lo = _mm512_min_epu32(NBv[v], c.v16);
        EXv[v] = _mm512_and_si512(s.X[v], k512_lowmask(lo));
        __m512i Xn = _mm512_srlv_epi32(s.X[v], lo);
        s.X[v] = Partial ? _mm512_mask_mov_epi32(s.X[v], act[v], Xn) : Xn;
        s.X[v] = k512_renorm(s.X[v], act[v] & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
        fm[v] = act[v] & _mm512_test_epi32_mask(sym, c.vone);
    }
    for (int v = 0; v < NV; ++v) {   // B2 (rare: gap extra wider than 16 bits)
        __mmask16 m2 = act[v] & _mm512_cmpgt_epu32_mask(NBv[v], c.v16);
        if (m2) {
            __m512i hi = _mm512_sub_epi32(NBv[v], c.v16);
            __m512i exh = _mm512_and_si512(s.X[v], k512_lowmask(hi));
            s.X[v] = _mm512_mask_mov_epi32(s.X[v], m2, _mm512_srlv_epi32(s.X[v], hi));
            s.X[v] = k512_renorm(s.X[v], m2 & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
            EXv[v] = _mm512_mask_or_epi32(EXv[v], m2, EXv[v], _mm512_slli_epi32(exh, 16));
        }
        RK[v] = _mm512_add_epi32(_mm512_add_epi32(s.PP[v], BASEv[v]), EXv[v]);
    }
    for (int v = 0; v < NV; ++v) {   // C1
        E2[v] = c.vzero;
        if (!fm[v]) continue;
        __m512i pcr = _mm512_min_epu32(_mm512_srli_epi32(k512_posb(RK[v]), 1), c.v20);
        __m512i dk = _mm512_srli_epi32(s.REM[v], 28);
        __m512i cx = _mm512_sll_epi32(_mm512_add_epi32(_mm512_slli_epi32(pcr, 2), dk), c.cpbv);
        __m512i idx = _mm512_add_epi32(cx, _mm512_and_si512(s.X[v], c.vmaskv));
        E2[v] = _mm512_mask_i32gather_epi32(c.vzero, fm[v], idx, c.rt, 4);
        __m512i f = _mm512_and_si512(_mm512_srli_epi32(E2[v], 12), _mm512_set1_epi32(0xFFF));
        __m512i bias = _mm512_and_si512(E2[v], _mm512_set1_epi32(0xFFF));
        __m512i Xn = _mm512_add_epi32(_mm512_mullo_epi32(f, _mm512_srl_epi32(s.X[v], c.cpbv)), bias);
        s.X[v] = _mm512_mask_mov_epi32(s.X[v], fm[v], Xn);
        s.X[v] = k512_renorm(s.X[v], fm[v] & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
    }
    for (int v = 0; v < NV; ++v) {   // C2
        VAL[v] = c.vone;
        if (!fm[v]) continue;
        __m512i base2, nb2;
        k512_cls(_mm512_srli_epi32(E2[v], 24), base2, nb2);
        __m512i lo = _mm512_min_epu32(nb2, c.v16);
        __m512i ex2 = _mm512_and_si512(s.X[v], k512_lowmask(lo));
        s.X[v] = _mm512_mask_mov_epi32(s.X[v], fm[v], _mm512_srlv_epi32(s.X[v], lo));
        s.X[v] = k512_renorm(s.X[v], fm[v] & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
        NBv[v] = nb2; EXv[v] = ex2; BASEv[v] = base2;   // reuse the registers for the residual
    }
    for (int v = 0; v < NV; ++v) {   // C3 (rare: residual extra wider than 16 bits) and value assembly
        if (!fm[v]) continue;
        __mmask16 m3 = fm[v] & _mm512_cmpgt_epu32_mask(NBv[v], c.v16);
        if (m3) {
            __m512i hi = _mm512_sub_epi32(NBv[v], c.v16);
            __m512i exh = _mm512_and_si512(s.X[v], k512_lowmask(hi));
            s.X[v] = _mm512_mask_mov_epi32(s.X[v], m3, _mm512_srlv_epi32(s.X[v], hi));
            s.X[v] = k512_renorm(s.X[v], m3 & _mm512_cmplt_epu32_mask(s.X[v], c.v65536), s.wp);
            EXv[v] = _mm512_mask_or_epi32(EXv[v], m3, EXv[v], _mm512_slli_epi32(exh, 16));
        }
        {   // r = base + ex ; value = r ? r + 1 : 0 ; unflagged lanes keep 1
            __m512i rr = _mm512_add_epi32(BASEv[v], EXv[v]);
            __m512i vv = _mm512_maskz_add_epi32(_mm512_test_epi32_mask(rr, rr), rr, c.vone);
            VAL[v] = _mm512_mask_mov_epi32(c.vone, fm[v], vv);
        }
    }
    for (int v = 0; v < NV; ++v) {   // emit + advance
        _mm512_store_si512(static_cast<void*>(rowI + 16 * v), RK[v]);
        _mm512_store_si512(static_cast<void*>(rowV + 16 * v), VAL[v]);
        s.PP[v] = _mm512_add_epi32(RK[v], c.vone);
        s.REM[v] = _mm512_sub_epi32(s.REM[v], c.vone);
        __mmask16 ce = act[v] & _mm512_testn_epi32_mask(s.REM[v], c.vremmask);
        if (ce) {
            __m512i k = _mm512_mask_i32gather_epi32(c.vzero, ce, s.KP[v], c.kl, 4);
            __m512i kraw = _mm512_sub_epi32(_mm512_set1_epi32(31), _mm512_lzcnt_epi32(_mm512_add_epi32(k, c.vone)));
            __m512i kc = _mm512_min_epu32(_mm512_srl_epi32(kraw, c.cks), c.vkcap);
            __m512i kb = _mm512_sll_epi32(_mm512_mullo_epi32(kc, c.vnpc), c.cpb);
            __m512i dk = _mm512_min_epu32(_mm512_srli_epi32(_mm512_mullo_epi32(kraw, c.v11), 5), c.v3);
            s.KB[v] = _mm512_mask_mov_epi32(s.KB[v], ce, kb);
            s.REM[v] = _mm512_mask_mov_epi32(s.REM[v], ce, _mm512_or_si512(k, _mm512_slli_epi32(dk, 28)));
            s.PP[v] = _mm512_mask_mov_epi32(s.PP[v], ce, c.vzero);
            s.KP[v] = _mm512_mask_add_epi32(s.KP[v], ce, s.KP[v], c.vone);
        }
    }
}

template <int NV>
PZ2_AINLINE void k512_flush(const Setup& su, const uint32_t* tmpI, const uint32_t* tmpV, int32_t* idx_out, uint32_t* val_out, uint32_t s0, uint32_t ns) {
    constexpr int L = NV * 16;
    for (int v = 0; v < NV; ++v) {
        for (int which = 0; which < 2; ++which) {
            __m512i r[16];
            const uint32_t* tmp = which ? tmpV : tmpI;
            for (uint32_t i = 0; i < 16; ++i) r[i] = i < ns ? _mm512_load_si512(static_cast<const void*>(tmp + i * L + 16 * v)) : _mm512_setzero_si512();
            k512_transpose16(r);
            for (int i = 0; i < 16; ++i) {
                const int lane = 16 * v + i;
                uint32_t valid = su.cnt[lane] > s0 ? (su.cnt[lane] - s0 < ns ? su.cnt[lane] - s0 : ns) : 0;
                __mmask16 mk = static_cast<__mmask16>((1u << valid) - 1u);
                if (which) _mm512_mask_storeu_epi32(val_out + su.e0[lane] + s0, mk, r[i]);
                else _mm512_mask_storeu_epi32(idx_out + su.e0[lane] + s0, mk, r[i]);
            }
        }
    }
}

template <int L>
inline bool chunk_avx512(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    constexpr int NV = L / 16;
    const Cfg& cfg = P.cfg;
    C512 c;
    c.vmask = _mm512_set1_epi32(int((1u << cfg.pb) - 1));
    c.vmaskv = _mm512_set1_epi32(int((1u << cfg.pbv) - 1));
    c.vzero = _mm512_setzero_si512(); c.vone = _mm512_set1_epi32(1); c.vtwo = _mm512_set1_epi32(2);
    c.v16 = _mm512_set1_epi32(16); c.v20 = _mm512_set1_epi32(kResPos - 1);
    c.vcap = _mm512_set1_epi32(cfg.cap); c.vkcap = _mm512_set1_epi32(cfg.kcap);
    c.vnpc = _mm512_set1_epi32(int(cfg.npc()));
    c.vremmask = _mm512_set1_epi32(0x0FFFFFFF);
    c.v11 = _mm512_set1_epi32(11); c.v3 = _mm512_set1_epi32(3); c.v65536 = _mm512_set1_epi32(65536);
    c.csh = _mm_cvtsi32_si128(cfg.sh); c.cks = _mm_cvtsi32_si128(cfg.ks);
    c.cpb = _mm_cvtsi32_si128(cfg.pb); c.cpbv = _mm_cvtsi32_si128(cfg.pbv);
    c.gt = P.gap.data(); c.rt = P.res.data(); c.kl = su.kl.data();

    K512<NV> s;
    for (int v = 0; v < NV; ++v) {
        s.X[v] = _mm512_loadu_si512(static_cast<const void*>(su.states + 64 * v));
        s.PP[v] = _mm512_load_si512(static_cast<const void*>(su.pp0 + 16 * v));
        s.KB[v] = _mm512_load_si512(static_cast<const void*>(su.kb0 + 16 * v));
        s.REM[v] = _mm512_or_si512(_mm512_load_si512(static_cast<const void*>(su.rem0 + 16 * v)),
                                   _mm512_slli_epi32(_mm512_load_si512(static_cast<const void*>(su.dk0 + 16 * v)), 28));
        s.KP[v] = _mm512_load_si512(static_cast<const void*>(su.kp0 + 16 * v));
    }
    s.wp = su.words;
    alignas(64) uint32_t tmpI[16 * L], tmpV[16 * L];
    for (uint32_t st = 0; st < su.S; ++st) {
        const uint32_t slot = st & 15;
        k512_step<NV, false>(s, c, nullptr, tmpI + slot * L, tmpV + slot * L);
        if (slot == 15) k512_flush<NV>(su, tmpI, tmpV, idx_out, val_out, st - 15, 16);
        if (s.wp > su.words_end) return false;
    }
    if (su.r) {
        __mmask16 act[NV];
        for (int v = 0; v < NV; ++v) {
            int a = static_cast<int>(su.r) - 16 * v;
            act[v] = a <= 0 ? static_cast<__mmask16>(0) : a >= 16 ? static_cast<__mmask16>(0xFFFF) : static_cast<__mmask16>((1u << a) - 1);
        }
        const uint32_t slot = su.S & 15;
        k512_step<NV, true>(s, c, act, tmpI + slot * L, tmpV + slot * L);
    }
    {
        uint32_t s0 = su.S & ~15u, ns = su.steps - s0;
        if (ns) k512_flush<NV>(su, tmpI, tmpV, idx_out, val_out, s0, ns);
    }
    if (s.wp != su.words_end) return false;
    for (int v = 0; v < NV; ++v)
        if (_mm512_cmpneq_epi32_mask(s.X[v], c.v65536)) return false;
    return true;
}

inline bool decode_chunk_avx512(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    switch (su.L) {
        case 16: return chunk_avx512<16>(P, su, idx_out, val_out);
        case 32: return chunk_avx512<32>(P, su, idx_out, val_out);
        default: return chunk_avx512<64>(P, su, idx_out, val_out);
    }
}

PZ2_END_TARGET

}  // namespace r1
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
#endif  // PZ2_COMPILED_AVX512
