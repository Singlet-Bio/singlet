// SPDX-License-Identifier: MIT
// pz2 codec 1: AVX2 (+BMI2, POPCNT) chunk decoder.  Bit-exact with decode_chunk_scalar (rans1_dec.h).
// NV = L/8 ymm vectors of 8 lanes are kept in flight.  Differences from the AVX-512 kernel: refill = vpermd through a 256-entry
// LUT indexed by the refill mask (no expand), unsigned compares via min_epu32, ilog2 via the exact int->float exponent trick
// (values < 2^24), 8x8 transposes for the output flush, vector masks instead of k-registers.
#pragma once
#ifdef PZ2_COMPILED_AVX2

namespace singlet {
namespace pz2 {
namespace detail {
namespace r1 {

// LUT[mask][i] = index of the word lane i takes when the lanes in `mask` refill (popcount of the lower mask bits).
struct RefillLut {
    alignas(32) uint32_t idx[256 * 8];
    RefillLut() {
        for (uint32_t m = 0; m < 256; ++m) {
            uint32_t c = 0;
            for (uint32_t i = 0; i < 8; ++i) { idx[m * 8 + i] = c; if ((m >> i) & 1) ++c; }
        }
    }
};
inline const uint32_t* refill_lut() { static const RefillLut t; return t.idx; }

PZ2_BEGIN_AVX2

PZ2_AINLINE __m256i k2_renorm(__m256i X, __m256i need, const uint8_t*& wp, const uint32_t* lut) {
    const int mask = _mm256_movemask_ps(_mm256_castsi256_ps(need));
    __m256i w32 = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(wp)));
    __m256i perm = _mm256_load_si256(reinterpret_cast<const __m256i*>(lut + 8 * mask));
    __m256i words = _mm256_permutevar8x32_epi32(w32, perm);
    wp += 2 * _mm_popcnt_u32(static_cast<uint32_t>(mask));
    return _mm256_blendv_epi8(X, _mm256_or_si256(_mm256_slli_epi32(X, 16), words), need);
}
PZ2_AINLINE __m256i k2_lt65536(__m256i X, __m256i v65535) {   // lanes with X < 65536 (unsigned)
    return _mm256_cmpeq_epi32(_mm256_min_epu32(X, v65535), X);
}
PZ2_AINLINE __m256i k2_posb(__m256i p) {   // posb_raw(p) = (bits(float(p + 2)) >> 22) - 256
    __m256i q = _mm256_add_epi32(p, _mm256_set1_epi32(2));
    return _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(q)), 22), _mm256_set1_epi32(256));
}
PZ2_AINLINE void k2_cls(__m256i cls, __m256i& base, __m256i& nb) {
    const __m256i one = _mm256_set1_epi32(1), zero = _mm256_setzero_si256();
    __m256i e = _mm256_srli_epi32(cls, 2);
    __m256i mm = _mm256_and_si256(cls, _mm256_set1_epi32(3));
    nb = _mm256_max_epi32(_mm256_sub_epi32(e, one), zero);
    __m256i hi = _mm256_slli_epi32(_mm256_min_epu32(e, one), 2);
    base = _mm256_sllv_epi32(_mm256_or_si256(mm, hi), nb);
}
PZ2_AINLINE __m256i k2_lowmask(__m256i bits) {
    return _mm256_sub_epi32(_mm256_sllv_epi32(_mm256_set1_epi32(1), bits), _mm256_set1_epi32(1));
}
PZ2_AINLINE void k2_transpose8(__m256i r[8]) {
    __m256i t0 = _mm256_unpacklo_epi32(r[0], r[1]), t1 = _mm256_unpackhi_epi32(r[0], r[1]);
    __m256i t2 = _mm256_unpacklo_epi32(r[2], r[3]), t3 = _mm256_unpackhi_epi32(r[2], r[3]);
    __m256i t4 = _mm256_unpacklo_epi32(r[4], r[5]), t5 = _mm256_unpackhi_epi32(r[4], r[5]);
    __m256i t6 = _mm256_unpacklo_epi32(r[6], r[7]), t7 = _mm256_unpackhi_epi32(r[6], r[7]);
    __m256i u0 = _mm256_unpacklo_epi64(t0, t2), u1 = _mm256_unpackhi_epi64(t0, t2);
    __m256i u2 = _mm256_unpacklo_epi64(t1, t3), u3 = _mm256_unpackhi_epi64(t1, t3);
    __m256i u4 = _mm256_unpacklo_epi64(t4, t6), u5 = _mm256_unpackhi_epi64(t4, t6);
    __m256i u6 = _mm256_unpacklo_epi64(t5, t7), u7 = _mm256_unpackhi_epi64(t5, t7);
    r[0] = _mm256_permute2x128_si256(u0, u4, 0x20); r[4] = _mm256_permute2x128_si256(u0, u4, 0x31);
    r[1] = _mm256_permute2x128_si256(u1, u5, 0x20); r[5] = _mm256_permute2x128_si256(u1, u5, 0x31);
    r[2] = _mm256_permute2x128_si256(u2, u6, 0x20); r[6] = _mm256_permute2x128_si256(u2, u6, 0x31);
    r[3] = _mm256_permute2x128_si256(u3, u7, 0x20); r[7] = _mm256_permute2x128_si256(u3, u7, 0x31);
}

template <int NV>
struct K2 {
    __m256i X[NV], PP[NV], KB[NV], REM[NV], KP[NV];
    const uint8_t* wp;
};
struct C2 {
    __m256i vmask, vmaskv, vzero, vone, vtwo, v16, v20, vcap, vkcap, vnpc, vremmask, v11, v3, v65535, vall;
    __m128i csh, cks, cpb, cpbv;
    const uint32_t *gt, *rt, *kl, *lut;
};

template <int NV, bool Partial>
PZ2_AINLINE void k2_step(K2<NV>& s, const C2& c, const __m256i* actp, uint32_t* rowI, uint32_t* rowV) {
    __m256i E[NV], RK[NV], NBv[NV], EXv[NV], BASEv[NV], E2[NV], VAL[NV], act[NV], fm[NV];
    const __m256i f4095 = _mm256_set1_epi32(0xFFF);
    for (int v = 0; v < NV; ++v) {   // A
        act[v] = Partial ? actp[v] : c.vall;
        __m256i b = k2_posb(s.PP[v]);
        __m256i pc = _mm256_min_epu32(_mm256_srl_epi32(b, c.csh), c.vcap);
        __m256i idx = _mm256_add_epi32(_mm256_add_epi32(s.KB[v], _mm256_sll_epi32(pc, c.cpb)), _mm256_and_si256(s.X[v], c.vmask));
        E[v] = Partial ? _mm256_mask_i32gather_epi32(c.vzero, reinterpret_cast<const int*>(c.gt), idx, act[v], 4)
                       : _mm256_i32gather_epi32(reinterpret_cast<const int*>(c.gt), idx, 4);
        __m256i f = _mm256_and_si256(_mm256_srli_epi32(E[v], 12), f4095);
        __m256i bias = _mm256_and_si256(E[v], f4095);
        __m256i Xn = _mm256_add_epi32(_mm256_mullo_epi32(f, _mm256_srl_epi32(s.X[v], c.cpb)), bias);
        s.X[v] = Partial ? _mm256_blendv_epi8(s.X[v], Xn, act[v]) : Xn;
        __m256i need = k2_lt65536(s.X[v], c.v65535);
        if (Partial) need = _mm256_and_si256(need, act[v]);
        s.X[v] = k2_renorm(s.X[v], need, s.wp, c.lut);
    }
    for (int v = 0; v < NV; ++v) {   // B
        __m256i sym = _mm256_srli_epi32(E[v], 24);
        k2_cls(_mm256_srli_epi32(sym, 1), BASEv[v], NBv[v]);
        __m256i lo = _mm256_min_epu32(NBv[v], c.v16);
        EXv[v] = _mm256_and_si256(s.X[v], k2_lowmask(lo));
        __m256i Xn = _mm256_srlv_epi32(s.X[v], lo);
        s.X[v] = Partial ? _mm256_blendv_epi8(s.X[v], Xn, act[v]) : Xn;
        __m256i need = k2_lt65536(s.X[v], c.v65535);
        if (Partial) need = _mm256_and_si256(need, act[v]);
        s.X[v] = k2_renorm(s.X[v], need, s.wp, c.lut);
        __m256i flag = _mm256_cmpeq_epi32(_mm256_and_si256(sym, c.vone), c.vone);
        fm[v] = Partial ? _mm256_and_si256(flag, act[v]) : flag;
    }
    for (int v = 0; v < NV; ++v) {   // B2 (rare)
        __m256i m2 = _mm256_cmpgt_epi32(NBv[v], c.v16);
        if (Partial) m2 = _mm256_and_si256(m2, act[v]);
        if (_mm256_movemask_ps(_mm256_castsi256_ps(m2))) {
            __m256i hi = _mm256_and_si256(_mm256_sub_epi32(NBv[v], c.v16), m2);
            __m256i exh = _mm256_and_si256(s.X[v], k2_lowmask(hi));
            s.X[v] = _mm256_srlv_epi32(s.X[v], hi);
            s.X[v] = k2_renorm(s.X[v], _mm256_and_si256(k2_lt65536(s.X[v], c.v65535), m2), s.wp, c.lut);
            EXv[v] = _mm256_or_si256(EXv[v], _mm256_slli_epi32(exh, 16));
        }
        RK[v] = _mm256_add_epi32(_mm256_add_epi32(s.PP[v], BASEv[v]), EXv[v]);
    }
    for (int v = 0; v < NV; ++v) {   // C1
        E2[v] = c.vzero;
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(fm[v]))) continue;
        __m256i pcr = _mm256_min_epu32(_mm256_srli_epi32(k2_posb(RK[v]), 1), c.v20);
        __m256i dk = _mm256_srli_epi32(s.REM[v], 28);
        __m256i cx = _mm256_sll_epi32(_mm256_add_epi32(_mm256_slli_epi32(pcr, 2), dk), c.cpbv);
        __m256i idx = _mm256_add_epi32(cx, _mm256_and_si256(s.X[v], c.vmaskv));
        E2[v] = _mm256_mask_i32gather_epi32(c.vzero, reinterpret_cast<const int*>(c.rt), idx, fm[v], 4);
        __m256i f = _mm256_and_si256(_mm256_srli_epi32(E2[v], 12), f4095);
        __m256i bias = _mm256_and_si256(E2[v], f4095);
        __m256i Xn = _mm256_add_epi32(_mm256_mullo_epi32(f, _mm256_srl_epi32(s.X[v], c.cpbv)), bias);
        s.X[v] = _mm256_blendv_epi8(s.X[v], Xn, fm[v]);
        s.X[v] = k2_renorm(s.X[v], _mm256_and_si256(k2_lt65536(s.X[v], c.v65535), fm[v]), s.wp, c.lut);
    }
    for (int v = 0; v < NV; ++v) {   // C2
        VAL[v] = c.vone;
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(fm[v]))) continue;
        __m256i base2, nb2;
        k2_cls(_mm256_srli_epi32(E2[v], 24), base2, nb2);   // unflagged lanes: E2 == 0 -> nb2 == 0 -> X unchanged
        __m256i lo = _mm256_min_epu32(nb2, c.v16);
        __m256i ex2 = _mm256_and_si256(s.X[v], k2_lowmask(lo));
        s.X[v] = _mm256_srlv_epi32(s.X[v], lo);
        s.X[v] = k2_renorm(s.X[v], _mm256_and_si256(k2_lt65536(s.X[v], c.v65535), fm[v]), s.wp, c.lut);
        NBv[v] = nb2; EXv[v] = ex2; BASEv[v] = base2;
    }
    for (int v = 0; v < NV; ++v) {   // C3 (rare) and value assembly
        if (!_mm256_movemask_ps(_mm256_castsi256_ps(fm[v]))) continue;
        __m256i m3 = _mm256_and_si256(_mm256_cmpgt_epi32(NBv[v], c.v16), fm[v]);
        if (_mm256_movemask_ps(_mm256_castsi256_ps(m3))) {
            __m256i hi = _mm256_and_si256(_mm256_sub_epi32(NBv[v], c.v16), m3);
            __m256i exh = _mm256_and_si256(s.X[v], k2_lowmask(hi));
            s.X[v] = _mm256_srlv_epi32(s.X[v], hi);
            s.X[v] = k2_renorm(s.X[v], _mm256_and_si256(k2_lt65536(s.X[v], c.v65535), m3), s.wp, c.lut);
            EXv[v] = _mm256_or_si256(EXv[v], _mm256_slli_epi32(exh, 16));
        }
        {   // r = base + ex ; value = r ? r + 1 : 0 ; unflagged lanes keep 1
            __m256i rr = _mm256_add_epi32(BASEv[v], EXv[v]);
            __m256i vv = _mm256_andnot_si256(_mm256_cmpeq_epi32(rr, c.vzero), _mm256_add_epi32(rr, c.vone));
            VAL[v] = _mm256_blendv_epi8(c.vone, vv, fm[v]);
        }
    }
    for (int v = 0; v < NV; ++v) {   // emit + advance
        _mm256_store_si256(reinterpret_cast<__m256i*>(rowI + 8 * v), RK[v]);
        _mm256_store_si256(reinterpret_cast<__m256i*>(rowV + 8 * v), VAL[v]);
        s.PP[v] = _mm256_add_epi32(RK[v], c.vone);
        s.REM[v] = _mm256_sub_epi32(s.REM[v], c.vone);
        __m256i ce = _mm256_cmpeq_epi32(_mm256_and_si256(s.REM[v], c.vremmask), c.vzero);
        if (Partial) ce = _mm256_and_si256(ce, act[v]);
        if (_mm256_movemask_ps(_mm256_castsi256_ps(ce))) {
            __m256i k = _mm256_mask_i32gather_epi32(c.vzero, reinterpret_cast<const int*>(c.kl), s.KP[v], ce, 4);
            __m256i k1 = _mm256_add_epi32(k, c.vone);
            // ilog2(k + 1) via the float exponent (exact for k + 1 < 2^24)
            __m256i kraw = _mm256_sub_epi32(_mm256_srli_epi32(_mm256_castps_si256(_mm256_cvtepi32_ps(k1)), 23), _mm256_set1_epi32(127));
            __m256i kc = _mm256_min_epu32(_mm256_srl_epi32(kraw, c.cks), c.vkcap);
            __m256i kb = _mm256_sll_epi32(_mm256_mullo_epi32(kc, c.vnpc), c.cpb);
            __m256i dk = _mm256_min_epu32(_mm256_srli_epi32(_mm256_mullo_epi32(kraw, c.v11), 5), c.v3);
            s.KB[v] = _mm256_blendv_epi8(s.KB[v], kb, ce);
            s.REM[v] = _mm256_blendv_epi8(s.REM[v], _mm256_or_si256(k, _mm256_slli_epi32(dk, 28)), ce);
            s.PP[v] = _mm256_blendv_epi8(s.PP[v], c.vzero, ce);
            s.KP[v] = _mm256_sub_epi32(s.KP[v], ce);   // ce lanes are -1: +1
        }
    }
}

template <int NV>
PZ2_AINLINE void k2_flush(const Setup& su, const uint32_t* tmpI, const uint32_t* tmpV, int32_t* idx_out, uint32_t* val_out, uint32_t s0, uint32_t ns) {
    constexpr int L = NV * 8;
    const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    for (int v = 0; v < NV; ++v) {
        for (int which = 0; which < 2; ++which) {
            __m256i r[8];
            const uint32_t* tmp = which ? tmpV : tmpI;
            for (uint32_t i = 0; i < 8; ++i) r[i] = i < ns ? _mm256_load_si256(reinterpret_cast<const __m256i*>(tmp + i * L + 8 * v)) : _mm256_setzero_si256();
            k2_transpose8(r);
            for (int i = 0; i < 8; ++i) {
                const int lane = 8 * v + i;
                uint32_t valid = su.cnt[lane] > s0 ? (su.cnt[lane] - s0 < ns ? su.cnt[lane] - s0 : ns) : 0;
                int32_t* dst = which ? reinterpret_cast<int32_t*>(val_out + su.e0[lane] + s0) : idx_out + su.e0[lane] + s0;
                if (valid == 8) _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), r[i]);
                else if (valid) _mm256_maskstore_epi32(dst, _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(valid)), iota), r[i]);
            }
        }
    }
}

template <int L>
inline bool chunk_avx2(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    constexpr int NV = L / 8;
    const Cfg& cfg = P.cfg;
    C2 c;
    c.vmask = _mm256_set1_epi32(int((1u << cfg.pb) - 1));
    c.vmaskv = _mm256_set1_epi32(int((1u << cfg.pbv) - 1));
    c.vzero = _mm256_setzero_si256(); c.vone = _mm256_set1_epi32(1); c.vtwo = _mm256_set1_epi32(2);
    c.v16 = _mm256_set1_epi32(16); c.v20 = _mm256_set1_epi32(kResPos - 1);
    c.vcap = _mm256_set1_epi32(cfg.cap); c.vkcap = _mm256_set1_epi32(cfg.kcap);
    c.vnpc = _mm256_set1_epi32(int(cfg.npc()));
    c.vremmask = _mm256_set1_epi32(0x0FFFFFFF);
    c.v11 = _mm256_set1_epi32(11); c.v3 = _mm256_set1_epi32(3); c.v65535 = _mm256_set1_epi32(65535);
    c.vall = _mm256_set1_epi32(-1);
    c.csh = _mm_cvtsi32_si128(cfg.sh); c.cks = _mm_cvtsi32_si128(cfg.ks);
    c.cpb = _mm_cvtsi32_si128(cfg.pb); c.cpbv = _mm_cvtsi32_si128(cfg.pbv);
    c.gt = P.gap.data(); c.rt = P.res.data(); c.kl = su.kl.data(); c.lut = refill_lut();

    K2<NV> s;
    for (int v = 0; v < NV; ++v) {
        s.X[v] = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(su.states + 32 * v));
        s.PP[v] = _mm256_load_si256(reinterpret_cast<const __m256i*>(su.pp0 + 8 * v));
        s.KB[v] = _mm256_load_si256(reinterpret_cast<const __m256i*>(su.kb0 + 8 * v));
        s.REM[v] = _mm256_or_si256(_mm256_load_si256(reinterpret_cast<const __m256i*>(su.rem0 + 8 * v)),
                                   _mm256_slli_epi32(_mm256_load_si256(reinterpret_cast<const __m256i*>(su.dk0 + 8 * v)), 28));
        s.KP[v] = _mm256_load_si256(reinterpret_cast<const __m256i*>(su.kp0 + 8 * v));
    }
    s.wp = su.words;
    alignas(32) uint32_t tmpI[8 * L], tmpV[8 * L];
    for (uint32_t st = 0; st < su.S; ++st) {
        const uint32_t slot = st & 7;
        k2_step<NV, false>(s, c, nullptr, tmpI + slot * L, tmpV + slot * L);
        if (slot == 7) k2_flush<NV>(su, tmpI, tmpV, idx_out, val_out, st - 7, 8);
        if (s.wp > su.words_end) return false;
    }
    if (su.r) {
        __m256i act[NV];
        const __m256i iota = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
        for (int v = 0; v < NV; ++v) act[v] = _mm256_cmpgt_epi32(_mm256_set1_epi32(static_cast<int>(su.r) - 8 * v), iota);
        const uint32_t slot = su.S & 7;
        k2_step<NV, true>(s, c, act, tmpI + slot * L, tmpV + slot * L);
    }
    {
        uint32_t s0 = su.S & ~7u, ns = su.steps - s0;
        if (ns) k2_flush<NV>(su, tmpI, tmpV, idx_out, val_out, s0, ns);
    }
    if (s.wp != su.words_end) return false;
    for (int v = 0; v < NV; ++v)
        if (_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(s.X[v], _mm256_set1_epi32(65536)))) != 0xFF) return false;
    return true;
}

inline bool decode_chunk_avx2(const Prep& P, const Setup& su, int32_t* idx_out, uint32_t* val_out) {
    switch (su.L) {
        case 16: return chunk_avx2<16>(P, su, idx_out, val_out);
        case 32: return chunk_avx2<32>(P, su, idx_out, val_out);
        default: return chunk_avx2<64>(P, su, idx_out, val_out);
    }
}

PZ2_END_TARGET

}  // namespace r1
}  // namespace detail
}  // namespace pz2
}  // namespace singlet
#endif  // PZ2_COMPILED_AVX2
