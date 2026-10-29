// Track C core: integer-only range coder, fixed-point math, hazard (run-skipping) Bernoulli coder.
// Everything on the decode path is integer arithmetic (no float/double), so encoder and decoder agree
// bit for bit on every x86-64 CPU. Float is used only in encoder-side statistics (guarded) and offline training.
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>
#include <algorithm>
#include "track_c_tables.h"

namespace cs { namespace tc {

using u8 = uint8_t; using u16 = uint16_t; using u32 = uint32_t; using u64 = uint64_t;
using u128 = unsigned __int128;

// ---------------------------------------------------------------------------------------------
// Binary range coder (LZMA style carry handling). Probabilities are P(bit==1) in 16 bits, 1..65535.
// ---------------------------------------------------------------------------------------------
struct RcEnc {
    std::vector<u8> out;
    u64 low = 0; u32 range = 0xFFFFFFFFu; u8 cache = 0; u64 cache_size = 1;
    void shift_low() {
        if ((u32)low < 0xFF000000u || (low >> 32) != 0) {
            u8 carry = (u8)(low >> 32);
            u8 temp = cache;
            do { out.push_back((u8)(temp + carry)); temp = 0xFF; } while (--cache_size != 0);
            cache = (u8)((low >> 24) & 0xFF);
        }
        cache_size++;
        low = (low & 0x00FFFFFFu) << 8;
    }
    inline int codebit(u32 p1, int bit) {
        u32 bound = (u32)(((u64)range * p1) >> 16);
        if (bit) range = bound; else { low += bound; range -= bound; }
        while (range < (1u << 24)) { range <<= 8; shift_low(); }
        return bit;
    }
    inline int bypass(int bit) { return codebit(32768, bit); }
    void finish() { for (int i = 0; i < 5; ++i) shift_low(); }
};

struct RcDec {
    const u8* p; const u8* end; u32 range = 0xFFFFFFFFu; u32 code = 0;
    RcDec(const u8* b, const u8* e) : p(b), end(e) { for (int i = 0; i < 5; ++i) code = (code << 8) | next(); }
    inline u8 next() { return p < end ? *p++ : 0; }
    inline int codebit(u32 p1, int /*unused*/) {
        u32 bound = (u32)(((u64)range * p1) >> 16);
        int bit;
        if (code < bound) { range = bound; bit = 1; } else { code -= bound; range -= bound; bit = 0; }
        while (range < (1u << 24)) { range <<= 8; code = (code << 8) | next(); }
        return bit;
    }
    inline int bypass(int) { return codebit(32768, 0); }
};

// Encoder-side ideal-cost accounting (bits). Only the encoder touches this; it is never read by the decoder.
struct Cost {
    double side = 0, bucket = 0, pres = 0, val = 0;
    double pb[40] = {0}, vb[40] = {0}; double nb[40] = {0};   // per depth bucket: presence bits, value bits, nnz
};

// ---------------------------------------------------------------------------------------------
// Fixed point math. Q40 = 40 fractional bits.
// ---------------------------------------------------------------------------------------------
static const u64 ONE40 = 1ull << 40;
static const u64 LN2Q = TC_LN2_Q40;

// 1 - exp(-x), x in Q40 (u64), result in Q40.
inline u64 E40(u64 x) {
    if (x == 0) return 0;
    if (x >= (40ull << 40)) return ONE40;
    if (x < (1ull << 32)) {                       // x < 2^-8: Taylor series
        u64 x2 = (u64)(((u128)x * x) >> 40), x3 = (u64)(((u128)x2 * x) >> 40);
        u64 x4 = (u64)(((u128)x3 * x) >> 40), x5 = (u64)(((u128)x4 * x) >> 40);
        return x - x2 / 2 + x3 / 6 - x4 / 24 + x5 / 120;
    }
    u64 k = x / LN2Q;                             // constant divisor -> multiply/shift
    u64 r = x - k * LN2Q;
    u64 idx = r >> TC_STEP_SHIFT, rem = r & ((1ull << TC_STEP_SHIFT) - 1);
    u64 a = TC_EXP[idx], b = TC_EXP[idx + 1];     // e^{-r} in Q48, decreasing
    u64 en = a - (u64)(((u128)(a - b) * rem) >> TC_STEP_SHIFT);
    en >>= 8;                                     // Q40
    en >>= k;
    return ONE40 - en;
}

// -ln(y) for y in (0,1], y in Q48 (u64), result Q40.
inline u64 neg_ln_q48(u64 y) {
    if (y >= (1ull << 48)) return 0;
    if (y == 0) y = 1;
    int lz = __builtin_clzll(y);
    u64 yn = y << lz;                             // in [2^63, 2^64)
    int e = 63 - lz - 48;                         // y = (yn/2^63) * 2^e
    u64 frac = yn - (1ull << 63);                 // 63 bits
    u64 idx = frac >> 53;                         // top 10 bits
    u64 rem = frac & ((1ull << 53) - 1);
    u64 a = TC_LN[idx], b = TC_LN[idx + 1];
    u64 lnm = a + (u64)(((u128)(b - a) * rem) >> 53);          // ln(m) Q40, >= 0
    // ln y = e*ln2 + lnm  (e <= 0)  -> -ln y = (-e)*ln2 - lnm
    return (u64)(-(long long)e) * LN2Q - lnm;
}

// hazard h = -ln(1-p), p in Q48. Series for small p keeps relative precision.
inline u64 hazard_from_p48(u64 p) {
    if (p < (1ull << 38)) {                       // p < 2^-10
        u64 p2 = (u64)(((u128)p * p) >> 48), p3 = (u64)(((u128)p2 * p) >> 48), p4 = (u64)(((u128)p3 * p) >> 48);
        u64 h48 = p + p2 / 2 + p3 / 3 + p4 / 4;
        return h48 >> 8;
    }
    return neg_ln_q48((1ull << 48) - p);
}

// 2^(-v/2048) in Q48, v is a u16 log-domain code.
inline u64 p48_from_code(u32 v) {
    u32 ip = v >> 11, fr = v & 2047;
    u64 t = TC_EXP2[fr];
    return ip >= 48 ? 0 : (t >> ip);
}

inline u32 clamp16(u64 v) { return v < 1 ? 1u : (v > 65535 ? 65535u : (u32)v); }

// ---------------------------------------------------------------------------------------------
// Adaptive binary probability with count-based learning rate.
// ---------------------------------------------------------------------------------------------
struct ABit {
    u16 p = 32768; u8 n = 0;
    inline u32 P() const { return p; }
    inline void update(int bit, int limit) {
        static const int RT[256] = {40960, 25206, 18204, 14246, 11702, 9929, 8623, 7620, 6826, 6182, 5649, 5201, 4818, 4488, 4201, 3947, 3723, 3523, 3343, 3181, 3034, 2899, 2776, 2664, 2560, 2463, 2374, 2291, 2214, 2141, 2073, 2010, 1950, 1894, 1840, 1790, 1742, 1697, 1654, 1614, 1575, 1538, 1503, 1469, 1437, 1406, 1376, 1348, 1321, 1295, 1270, 1245, 1222, 1200, 1178, 1157, 1137, 1118, 1099, 1081, 1063, 1046, 1030, 1014, 999, 984, 969, 955, 941, 928, 915, 902, 890, 878, 866, 855, 844, 833, 823, 813, 803, 793, 783, 774, 765, 756, 748, 739, 731, 723, 715, 707, 700, 692, 685, 678, 671, 664, 657, 651, 645, 638, 632, 626, 620, 614, 609, 603, 597, 592, 587, 582, 576, 571, 566, 562, 557, 552, 547, 543, 538, 534, 530, 525, 521, 517, 513, 509, 505, 501, 497, 494, 490, 486, 483, 479, 476, 472, 469, 466, 462, 459, 456, 453, 450, 447, 444, 441, 438, 435, 432, 429, 426, 423, 421, 418, 415, 413, 410, 408, 405, 403, 400, 398, 395, 393, 391, 388, 386, 384, 381, 379, 377, 375, 373, 371, 369, 366, 364, 362, 360, 358, 356, 355, 353, 351, 349, 347, 345, 343, 342, 340, 338, 336, 335, 333, 331, 329, 328, 326, 325, 323, 321, 320, 318, 317, 315, 314, 312, 311, 309, 308, 306, 305, 303, 302, 301, 299, 298, 297, 295, 294, 293, 291, 290, 289, 287, 286, 285, 284, 282, 281, 280, 279, 278, 276, 275, 274, 273, 272, 271, 270, 269, 267, 266, 265, 264, 263, 262, 261, 260, 259, 258, 257, 256, 255};   // 65536/(n+1.6)
        int t = bit ? 65535 : 0;
        int d = (int)(((long long)(t - (int)p) * RT[n]) >> 16);
        int np = (int)p + d;
        if (np < 48) np = 48; if (np > 65535 - 48) np = 65535 - 48;
        p = (u16)np;
        if (n < limit) n++;
    }
};

}}  // namespace cs::tc
