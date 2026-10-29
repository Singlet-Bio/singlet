// SPDX-License-Identifier: MIT
// pz2 internals: SHA-256 (FIPS 180-4), streaming.  Portable scalar block function plus an x86 SHA-NI block function chosen at run
// time (never for PZ2_ISA=scalar), bit-identical by construction and by test.
#pragma once

namespace singlet {
namespace pz2 {
namespace detail {

alignas(16) inline constexpr uint32_t kSha256K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be,
    0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa,
    0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85,
    0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f,
    0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

inline void sha256_blocks_scalar(uint32_t h_[8], const uint8_t* p, size_t nblocks) {
    auto rotr = [](uint32_t x, int n) { return (x >> n) | (x << (32 - n)); };
    for (; nblocks; --nblocks, p += 64) {
        uint32_t w[64];
        for (int i = 0; i < 16; ++i) w[i] = (uint32_t(p[4 * i]) << 24) | (uint32_t(p[4 * i + 1]) << 16) | (uint32_t(p[4 * i + 2]) << 8) | p[4 * i + 3];
        for (int i = 16; i < 64; ++i) {
            uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }
        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3], e = h_[4], f = h_[5], g = h_[6], h = h_[7];
        for (int i = 0; i < 64; ++i) {
            uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            uint32_t ch = (e & f) ^ (~e & g);
            uint32_t t1 = h + S1 + ch + kSha256K[i] + w[i];
            uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint32_t t2 = S0 + mj;
            h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
        }
        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d; h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
}

#ifdef PZ2_COMPILED_SHANI
#if defined(_MSC_VER) && !defined(__clang__)
#define PZ2_BEGIN_SHA
#define PZ2_END_SHA
#elif defined(__clang__)
#define PZ2_BEGIN_SHA _Pragma("clang attribute push (__attribute__((target(\"sha,sse4.1,ssse3\"))), apply_to=function)")
#define PZ2_END_SHA _Pragma("clang attribute pop")
#else
#define PZ2_BEGIN_SHA _Pragma("GCC push_options") _Pragma("GCC target(\"sha,sse4.1,ssse3\")")
#define PZ2_END_SHA _Pragma("GCC pop_options")
#endif

PZ2_BEGIN_SHA
inline void sha256_blocks_ni(uint32_t state[8], const uint8_t* data, size_t nblocks) {
    const __m128i MASK = _mm_set_epi64x(0x0c0d0e0f08090a0bULL, 0x0405060700010203ULL);
    __m128i TMP = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[0]));
    __m128i STATE1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&state[4]));
    TMP = _mm_shuffle_epi32(TMP, 0xB1);            // CDAB
    STATE1 = _mm_shuffle_epi32(STATE1, 0x1B);      // EFGH
    __m128i STATE0 = _mm_alignr_epi8(TMP, STATE1, 8);   // ABEF
    STATE1 = _mm_blend_epi16(STATE1, TMP, 0xF0);        // CDGH
    for (; nblocks; --nblocks, data += 64) {
        const __m128i ABEF_SAVE = STATE0, CDGH_SAVE = STATE1;
        __m128i w[4];
        for (int i = 0; i < 4; ++i) w[i] = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(data + 16 * i)), MASK);
        for (int i = 0; i < 16; ++i) {   // four rounds per iteration; w[i & 3] holds message group i
            __m128i msg = _mm_add_epi32(w[i & 3], _mm_load_si128(reinterpret_cast<const __m128i*>(&kSha256K[4 * i])));
            STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, msg);
            msg = _mm_shuffle_epi32(msg, 0x0E);
            STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, msg);
            if (i < 12) {   // schedule group i + 4 into the slot group i just vacated
                __m128i t1 = _mm_sha256msg1_epu32(w[i & 3], w[(i + 1) & 3]);
                __m128i t2 = _mm_alignr_epi8(w[(i + 3) & 3], w[(i + 2) & 3], 4);
                w[i & 3] = _mm_sha256msg2_epu32(_mm_add_epi32(t1, t2), w[(i + 3) & 3]);
            }
        }
        STATE0 = _mm_add_epi32(STATE0, ABEF_SAVE);
        STATE1 = _mm_add_epi32(STATE1, CDGH_SAVE);
    }
    TMP = _mm_shuffle_epi32(STATE0, 0x1B);         // FEBA
    STATE1 = _mm_shuffle_epi32(STATE1, 0xB1);      // DCHG
    STATE0 = _mm_blend_epi16(TMP, STATE1, 0xF0);   // DCBA
    STATE1 = _mm_alignr_epi8(STATE1, TMP, 8);      // HGFE
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[0]), STATE0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&state[4]), STATE1);
}
PZ2_END_SHA
#endif

class Sha256 {
public:
    /// use_ni: true = SHA-NI when the CPU has it, false = portable code.  The default follows PZ2_ISA (scalar forces the portable path).
    explicit Sha256(int use_ni = -1) {
        if (use_ni < 0) use_ni = (selected_isa() != Isa::Scalar) ? 1 : 0;
        ni_ = use_ni != 0 && sha_ni_available();
        reset();
    }
    static bool sha_ni_available() {
#ifdef PZ2_COMPILED_SHANI
        return cpu().sha;
#else
        return false;
#endif
    }
    bool uses_ni() const { return ni_; }
    void reset() {
        static const uint32_t iv[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au, 0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        std::memcpy(h_, iv, sizeof(h_));
        len_ = 0; fill_ = 0;
    }
    void update(const void* data, size_t n) {
        const uint8_t* p = static_cast<const uint8_t*>(data);
        len_ += n;
        if (fill_) {
            size_t take = std::min(n, size_t(64) - fill_);
            std::memcpy(buf_ + fill_, p, take);
            fill_ += take; p += take; n -= take;
            if (fill_ == 64) { blocks(buf_, 1); fill_ = 0; }
        }
        if (n >= 64) { size_t nb = n / 64; blocks(p, nb); p += nb * 64; n -= nb * 64; }
        if (n) { std::memcpy(buf_, p, n); fill_ = n; }
    }
    void update_u32(uint32_t v) { uint8_t b[4]; st_u32(b, v); update(b, 4); }
    void update_u64(uint64_t v) { uint8_t b[8]; st_u64(b, v); update(b, 8); }
    void final(uint8_t out[32]) {
        uint64_t bits = len_ * 8;
        uint8_t pad[72] = {0x80};
        size_t padlen = (fill_ < 56) ? 56 - fill_ : 120 - fill_;
        uint8_t lenb[8];
        for (int i = 0; i < 8; ++i) lenb[i] = uint8_t(bits >> (56 - 8 * i));   // big-endian bit length
        update(pad, padlen);
        update(lenb, 8);
        for (int i = 0; i < 8; ++i) { out[4 * i] = uint8_t(h_[i] >> 24); out[4 * i + 1] = uint8_t(h_[i] >> 16); out[4 * i + 2] = uint8_t(h_[i] >> 8); out[4 * i + 3] = uint8_t(h_[i]); }
    }

private:
    uint32_t h_[8];
    uint64_t len_;
    uint8_t buf_[64];
    size_t fill_;
    bool ni_ = false;
    void blocks(const uint8_t* p, size_t nb) {
#ifdef PZ2_COMPILED_SHANI
        if (ni_) { sha256_blocks_ni(h_, p, nb); return; }
#endif
        sha256_blocks_scalar(h_, p, nb);
    }
};

}  // namespace detail
}  // namespace pz2
}  // namespace singlet
