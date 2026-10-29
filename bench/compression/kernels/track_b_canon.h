// Track B: canonicalisation variants (rank-space columns -> ascending original row ids), micro-benchmarked in
// harness/track_b_canon.cpp.  The production path (track_b_decode.h Canon) uses the best variant.
#pragma once
#include "track_b_lane.h"
#include <algorithm>

namespace cs {
namespace tb {

// V2: plain std::sort of packed 64-bit keys (baseline)
struct CanonSort {
    const uint32_t* perm;
    std::vector<uint64_t> key;
    CanonSort(const uint32_t* p, uint32_t) : perm(p) {}
    void column(uint32_t* __restrict idx, uint32_t* __restrict val, uint32_t k) {
        if (key.size() < k) key.resize(k);
        for (uint32_t i = 0; i < k; ++i) key[i] = (static_cast<uint64_t>(perm[idx[i]]) << 32) | val[i];
        std::sort(key.begin(), key.begin() + k);
        for (uint32_t i = 0; i < k; ++i) { idx[i] = static_cast<uint32_t>(key[i] >> 32); val[i] = static_cast<uint32_t>(key[i]); }
    }
};

// V1: two-level bitmap, split loops, u16 value table
struct CanonBm {
    const uint32_t* perm;
    uint32_t m;
    std::vector<uint64_t> l0, l1;
    std::vector<uint16_t> vt;
    std::vector<uint32_t> ob;
    CanonBm(const uint32_t* perm_, uint32_t m_) : perm(perm_), m(m_) {
        l0.assign(((m + 63) / 64) + 8, 0);
        l1.assign(l0.size() / 64 + 8, 0);
        vt.assign(static_cast<size_t>(m) + 64, 0);
    }
    void column(uint32_t* __restrict idx, uint32_t* __restrict val, uint32_t k) {
        if (ob.size() < k) ob.resize(k);
        uint32_t* __restrict o = ob.data();
        uint64_t* __restrict a0 = l0.data();
        uint64_t* __restrict a1 = l1.data();
        uint16_t* __restrict v = vt.data();
        for (uint32_t i = 0; i < k; ++i) o[i] = perm[idx[i]];
        for (uint32_t i = 0; i < k; ++i) {
            uint32_t x = o[i];
            a0[x >> 6] |= 1ull << (x & 63);
            a1[x >> 12] |= 1ull << ((x >> 6) & 63);
            v[x] = static_cast<uint16_t>(val[i]);
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
                    uint32_t x = (i0 << 6) | static_cast<uint32_t>(__builtin_ctzll(b0));
                    b0 &= b0 - 1;
                    idx[p] = x;
                    val[p] = v[x];
                    ++p;
                } while (b0);
            } while (b1);
        }
    }
};

}  // namespace tb
}  // namespace cs
