// Track B shared helpers: bit utilities, LEB128, rank-space preparation, side-info coding, size breakdown.
#pragma once
#include "common.h"
#include "ref_zstd_csc.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <zstd.h>

namespace cs {
namespace tb {

// ---------------------------------------------------------------------------------------------
// Size breakdown of the last encode() on this thread (read by harness/track_b_bench.cpp).
// payload = col counts + coder tables + chunk streams. side = row-nnz counts (derive the frequency permutation).
// names = rownames + colnames + user_kv + headers.
// ---------------------------------------------------------------------------------------------
struct Breakdown {
    size_t payload = 0, side = 0, names = 0, tables = 0, counts = 0, chunks = 0, index_bytes = 0, value_bytes = 0;
    void clear() { *this = Breakdown(); }
};
inline Breakdown& last_breakdown() {
    static thread_local Breakdown b;
    return b;
}

// ---------------------------------------------------------------------------------------------
inline int ilog2u(uint32_t x) { return 31 - __builtin_clz(x | 1); }  // floor(log2(max(x,1)))

inline void leb_put(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 128) { o.push_back(static_cast<uint8_t>(v | 128)); v >>= 7; }
    o.push_back(static_cast<uint8_t>(v));
}
inline uint64_t leb_get(const uint8_t*& p) {
    uint64_t v = 0;
    int s = 0;
    while (true) {
        uint8_t b = *p++;
        v |= static_cast<uint64_t>(b & 127) << s;
        if (!(b & 128)) break;
        s += 7;
    }
    return v;
}

// Log-linear gap classes: g<8 direct, then 4 sub-buckets per octave. nb extra raw bits.
constexpr int kNumCls = 124;
inline uint32_t g_cls(uint32_t g, uint32_t& ex, uint32_t& nb) {
    if (g < 8) { ex = 0; nb = 0; return g; }
    int b = ilog2u(g);
    nb = static_cast<uint32_t>(b - 2);
    ex = g & ((1u << nb) - 1);
    return 8 + static_cast<uint32_t>(b - 3) * 4 + ((g >> nb) & 3);
}
// per class: base value and number of extra bits, packed (base << 5 | nb); base < 2^31 fits since cls<124 base<2^32 -> store separately
struct ClsTab {
    uint32_t base[kNumCls];
    uint8_t nb[kNumCls];
    ClsTab() {
        for (int c = 0; c < 8; ++c) { base[c] = c; nb[c] = 0; }
        for (int c = 8; c < kNumCls; ++c) {
            uint32_t b = 3 + (c - 8) / 4;
            uint32_t n = b - 2;
            base[c] = (4u + static_cast<uint32_t>((c - 8) & 3)) << n;
            nb[c] = static_cast<uint8_t>(n);
        }
    }
};
inline const ClsTab& cls_tab() {
    static ClsTab t;
    return t;
}

// ---------------------------------------------------------------------------------------------
// Forward bit stream (LSB first). Reader needs 8 bytes of tail padding.
// ---------------------------------------------------------------------------------------------
struct BitW {
    std::vector<uint8_t> buf;
    uint64_t acc = 0;
    int nacc = 0;
    void put(uint32_t v, int nb) {  // nb <= 32
        if (!nb) return;
        acc |= static_cast<uint64_t>(v) << nacc;
        nacc += nb;
        while (nacc >= 8) { buf.push_back(static_cast<uint8_t>(acc)); acc >>= 8; nacc -= 8; }
    }
    void finish() {
        if (nacc > 0) { buf.push_back(static_cast<uint8_t>(acc)); acc = 0; nacc = 0; }
    }
};
inline uint32_t bit_get(const uint8_t* base, uint64_t& bitpos, int nb) {
    uint64_t w;
    std::memcpy(&w, base + (bitpos >> 3), 8);
    uint32_t v = static_cast<uint32_t>((w >> (bitpos & 7)) & ((1ull << nb) - 1));
    bitpos += static_cast<uint64_t>(nb);
    return v;
}

// ---------------------------------------------------------------------------------------------
// Frequency permutation (identical to the production writer: descending row nnz, ties by row id).
// ---------------------------------------------------------------------------------------------
struct RowPerm {
    std::vector<uint32_t> rnnz;  // per original row
    std::vector<uint32_t> perm;  // rank -> original row
    std::vector<uint32_t> inv;   // original row -> rank
};
inline void perm_from_counts(RowPerm& P) {
    const uint32_t m = static_cast<uint32_t>(P.rnnz.size());
    uint32_t mx = 0;
    for (auto c : P.rnnz) mx = std::max(mx, c);
    std::vector<uint32_t> start(static_cast<size_t>(mx) + 2, 0);
    for (auto c : P.rnnz) start[c]++;
    // descending: bucket mx first
    uint32_t pos = 0;
    for (int64_t b = mx; b >= 0; --b) { uint32_t cnt = start[b]; start[b] = pos; pos += cnt; }
    P.perm.assign(m, 0);
    P.inv.assign(m, 0);
    for (uint32_t i = 0; i < m; ++i) {
        uint32_t r = start[P.rnnz[i]]++;
        P.perm[r] = i;
        P.inv[i] = r;
    }
}
inline RowPerm make_perm(const CanonicalMatrix& M) {
    RowPerm P;
    P.rnnz.assign(M.m, 0);
    for (auto r : M.indices) P.rnnz[r]++;
    perm_from_counts(P);
    return P;
}

// Matrix in rank space: row ids replaced by frequency rank, ascending within each column.
struct RankMat {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    std::vector<uint32_t> ptr, rk, val;
};
inline RankMat to_rank_space(const CanonicalMatrix& M, const RowPerm& P) {
    RankMat R;
    R.m = M.m; R.n = M.n; R.nnz = M.nnz; R.ptr = M.indptr;
    R.rk.resize(M.nnz);
    R.val.resize(M.nnz);
    std::vector<uint64_t> tmp;
    for (uint32_t j = 0; j < M.n; ++j) {
        tmp.clear();
        for (uint32_t k = M.indptr[j]; k < M.indptr[j + 1]; ++k)
            tmp.push_back((static_cast<uint64_t>(P.inv[M.indices[k]]) << 32) | M.data[k]);
        std::sort(tmp.begin(), tmp.end());
        for (size_t t = 0; t < tmp.size(); ++t) {
            R.rk[M.indptr[j] + t] = static_cast<uint32_t>(tmp[t] >> 32);
            R.val[M.indptr[j] + t] = static_cast<uint32_t>(tmp[t]);
        }
    }
    return R;
}

// ---------------------------------------------------------------------------------------------
// Side-info coding shared by all Track B codecs: names / kv with the cheapest passing coder (zstd-3 streams)
// and the row-nnz counts (LEB128 + zstd-3) from which the frequency permutation is re-derived.
// ---------------------------------------------------------------------------------------------
inline void put_header(std::vector<uint8_t>& o, const CanonicalMatrix& M) {
    detail::put_u32(o, M.m);
    detail::put_u32(o, M.n);
    detail::put_u64(o, M.nnz);
    o.push_back(M.vt_code);
}
inline void put_names(std::vector<uint8_t>& o, const CanonicalMatrix& M) {
    using namespace detail;
    auto rn = join_names(M.rownames), cn = join_names(M.colnames);
    put_stream(o, rn.data(), rn.size(), 3);
    put_stream(o, cn.data(), cn.size(), 3);
    std::vector<uint8_t> kv;
    for (auto& [k, v] : M.user_kv) {
        kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
        kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
    }
    put_u32(o, static_cast<uint32_t>(M.user_kv.size()));
    put_stream(o, kv.data(), kv.size(), 3);
}
inline void get_names(const uint8_t*& p, CanonicalMatrix& M) {
    using namespace detail;
    M.rownames = split_names(get_stream(p), M.m);
    M.colnames = split_names(get_stream(p), M.n);
    uint32_t nkv = get_u32(p);
    auto kv = get_stream(p);
    auto parts = split_names(kv, 2 * static_cast<size_t>(nkv));
    for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
}
inline void put_rowcounts(std::vector<uint8_t>& o, const RowPerm& P) {
    std::vector<uint8_t> b;
    for (auto c : P.rnnz) leb_put(b, c);
    detail::put_stream(o, b.data(), b.size(), 3);
}
inline void get_rowcounts(const uint8_t*& p, RowPerm& P, uint32_t m) {
    auto b = detail::get_stream(p);
    const uint8_t* q = b.data();
    P.rnnz.resize(m);
    for (uint32_t i = 0; i < m; ++i) P.rnnz[i] = static_cast<uint32_t>(leb_get(q));
    perm_from_counts(P);
}

// Cycles consumed by this thread (excludes time the OS scheduled us out; SMT sharing still inflates). Windows:
// QueryThreadCycleTime (TSC ticks while running, ~3.0 GHz nominal here); elsewhere steady_clock x 3 GHz.
#ifdef _WIN32
extern "C" __declspec(dllimport) int __stdcall QueryThreadCycleTime(void* ThreadHandle, unsigned long long* CycleTime);
extern "C" __declspec(dllimport) void* __stdcall GetCurrentThread(void);
inline uint64_t thread_cycles() {
    unsigned long long c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return c;
}
#else
inline uint64_t thread_cycles() {
    return static_cast<uint64_t>(std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count() * 3.0e9);
}
#endif
constexpr double kNominalGHz = 3.0;  // i7-1185G7 base clock = TSC rate
struct CycTimer {
    uint64_t c0 = thread_cycles();
    double cycles() const { return static_cast<double>(thread_cycles() - c0); }
    double ms_at_nominal() const { return cycles() / (kNominalGHz * 1e6); }
};

}  // namespace tb
}  // namespace cs
