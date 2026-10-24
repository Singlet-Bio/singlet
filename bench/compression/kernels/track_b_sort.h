// Track B: AVX-512 canonicalisation by vectorised merge sort.
//
// A column holds ranks r_1<...<r_k (frequency-order row ids) and values. Canonical output wants the original row
// ids o_i = perm[r_i] in ascending order with values carried along. We sort 32-bit keys (o_i << 13) | i
// (k <= 8192, m < 2^19), then gather the values by the low 13 bits.  In-register bitonic sort of 16 keys, then
// streaming merges of runs with a 16+16 bitonic merge kernel (Inoue / AA-sort style).
#pragma once
#include "track_b_lane.h"
#include "track_b_decode.h"

#ifdef TB_HAVE_AVX512
namespace cs {
namespace tb {

namespace sortk {

constexpr uint16_t smask(int k, int j) {  // lanes that take the max in bitonic stage (k, j)
    uint16_t m = 0;
    for (int i = 0; i < 16; ++i)
        if ((((i & j) != 0) ? 1 : 0) ^ (((i & k) != 0) ? 1 : 0)) m |= static_cast<uint16_t>(1u << i);
    return m;
}
alignas(64) static const int32_t kXor1[16] = {1, 0, 3, 2, 5, 4, 7, 6, 9, 8, 11, 10, 13, 12, 15, 14};
alignas(64) static const int32_t kXor2[16] = {2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13};
alignas(64) static const int32_t kXor4[16] = {4, 5, 6, 7, 0, 1, 2, 3, 12, 13, 14, 15, 8, 9, 10, 11};
alignas(64) static const int32_t kXor8[16] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
alignas(64) static const int32_t kRev[16] = {15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};

template <int J>
static inline __m512i xorperm(__m512i v) {
    const __m512i idx = _mm512_load_si512(J == 1 ? kXor1 : J == 2 ? kXor2 : J == 4 ? kXor4 : kXor8);
    return _mm512_permutexvar_epi32(idx, v);
}
template <int K, int J>
static inline __m512i bstage(__m512i v) {
    __m512i t = xorperm<J>(v);
    return _mm512_mask_blend_epi32(smask(K, J), _mm512_min_epu32(v, t), _mm512_max_epu32(v, t));
}
// sort 16 u32 ascending
static inline __m512i sort16(__m512i v) {
    v = bstage<2, 1>(v);
    v = bstage<4, 2>(v); v = bstage<4, 1>(v);
    v = bstage<8, 4>(v); v = bstage<8, 2>(v); v = bstage<8, 1>(v);
    v = bstage<16, 8>(v); v = bstage<16, 4>(v); v = bstage<16, 2>(v); v = bstage<16, 1>(v);
    return v;
}
// merge a bitonic 16-sequence into ascending order
static inline __m512i bmerge16(__m512i v) {
    const __mmask16 m8 = 0xFF00, m4 = 0xF0F0, m2 = 0xCCCC, m1 = 0xAAAA;
    __m512i t = xorperm<8>(v);
    v = _mm512_mask_blend_epi32(m8, _mm512_min_epu32(v, t), _mm512_max_epu32(v, t));
    t = xorperm<4>(v);
    v = _mm512_mask_blend_epi32(m4, _mm512_min_epu32(v, t), _mm512_max_epu32(v, t));
    t = xorperm<2>(v);
    v = _mm512_mask_blend_epi32(m2, _mm512_min_epu32(v, t), _mm512_max_epu32(v, t));
    t = xorperm<1>(v);
    return _mm512_mask_blend_epi32(m1, _mm512_min_epu32(v, t), _mm512_max_epu32(v, t));
}
// merge two ascending 16-vectors: lo = smallest 16, hi = largest 16, both ascending
static inline void merge16(__m512i a, __m512i b, __m512i& lo, __m512i& hi) {
    __m512i br = _mm512_permutexvar_epi32(_mm512_load_si512(kRev), b);
    lo = bmerge16(_mm512_min_epu32(a, br));
    hi = bmerge16(_mm512_max_epu32(a, br));
}
// merge ascending runs A[na], B[nb] (multiples of 16) into out
static inline void merge_runs(const uint32_t* A, size_t na, const uint32_t* B, size_t nb, uint32_t* out) {
    __m512i va = _mm512_load_si512(A), vb = _mm512_load_si512(B);
    size_t ia = 16, ib = 16;
    __m512i lo, carry;
    merge16(va, vb, lo, carry);
    _mm512_store_si512(out, lo);
    out += 16;
    while (ia < na || ib < nb) {
        const bool takeA = (ia < na) && (ib >= nb || A[ia] <= B[ib]);
        const uint32_t* src = takeA ? A + ia : B + ib;
        ia += takeA ? 16 : 0;
        ib += takeA ? 0 : 16;
        merge16(carry, _mm512_load_si512(src), lo, carry);
        _mm512_store_si512(out, lo);
        out += 16;
    }
    _mm512_store_si512(out, carry);
}

}  // namespace sortk

struct CanonVec {
    const uint32_t* perm;
    uint32_t m;
    Canon fallback;
    alignas(64) uint32_t A[8192 + 64];
    alignas(64) uint32_t B[8192 + 64];
    alignas(64) uint32_t V[8192 + 64];
    CanonVec(const uint32_t* perm_, uint32_t m_) : perm(perm_), m(m_), fallback(perm_, m_) {}

    void column(uint32_t* idx, uint32_t* val, uint32_t k) {
        if (k <= 1) { if (k) idx[0] = perm[idx[0]]; return; }
        if (k <= 8 || k > 8192 || m >= (1u << 19)) { fallback.column(idx, val, k); return; }
        const uint32_t kp = (k + 15) & ~15u;
        const __m512i iota = _mm512_set_epi32(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
        const __m512i vmax = _mm512_set1_epi32(-1);
        // keys
        for (uint32_t i = 0; i < kp; i += 16) {
            const uint32_t na = std::min<uint32_t>(16, k - std::min(k, i));
            const __mmask16 mk = static_cast<__mmask16>((1u << na) - 1);
            __m512i r = _mm512_maskz_loadu_epi32(mk, idx + i);
            __m512i o = _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mk, r, perm, 4);
            __m512i key = _mm512_or_si512(_mm512_slli_epi32(o, 13), _mm512_add_epi32(iota, _mm512_set1_epi32(static_cast<int>(i))));
            key = _mm512_mask_blend_epi32(mk, vmax, key);
            _mm512_store_si512(A + i, sortk::sort16(key));
        }
        std::memcpy(V, val, static_cast<size_t>(k) * 4);
        uint32_t* src = A;
        uint32_t* dst = B;
        for (uint32_t w = 16; w < kp; w *= 2) {
            for (uint32_t s = 0; s < kp; s += 2 * w) {
                if (s + w >= kp) {
                    std::memcpy(dst + s, src + s, (kp - s) * 4);
                } else {
                    const uint32_t nb = std::min(w, kp - s - w);
                    sortk::merge_runs(src + s, w, src + s + w, nb, dst + s);
                }
            }
            std::swap(src, dst);
        }
        const __m512i m13 = _mm512_set1_epi32(0x1FFF);
        for (uint32_t i = 0; i < k; i += 16) {
            const uint32_t na = std::min<uint32_t>(16, k - i);
            const __mmask16 mk = static_cast<__mmask16>((1u << na) - 1);
            __m512i key = _mm512_load_si512(src + i);
            _mm512_mask_storeu_epi32(idx + i, mk, _mm512_srli_epi32(key, 13));
            _mm512_mask_storeu_epi32(val + i, mk, _mm512_mask_i32gather_epi32(_mm512_setzero_si512(), mk, _mm512_and_si512(key, m13), V, 4));
        }
    }
};

}  // namespace tb
}  // namespace cs
#endif  // TB_HAVE_AVX512
