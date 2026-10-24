// SPDX-License-Identifier: MIT
// pz2 internals: byte helpers, CRC-32, LEB128, zstd framing, names.  Included by pz2.h only.
#pragma once
#include <zstd.h>

#include <algorithm>
#include <cmath>

namespace singlet {
namespace pz2 {
namespace detail {

// ---- little-endian scalar access (portable, alignment safe) --------------------------------------------------
inline void put_u16(std::vector<uint8_t>& o, uint16_t v) { o.push_back(uint8_t(v)); o.push_back(uint8_t(v >> 8)); }
inline void put_u32(std::vector<uint8_t>& o, uint32_t v) { for (int i = 0; i < 4; ++i) o.push_back(uint8_t(v >> (8 * i))); }
inline void put_u64(std::vector<uint8_t>& o, uint64_t v) { for (int i = 0; i < 8; ++i) o.push_back(uint8_t(v >> (8 * i))); }
inline void st_u32(uint8_t* p, uint32_t v) { for (int i = 0; i < 4; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline void st_u64(uint8_t* p, uint64_t v) { for (int i = 0; i < 8; ++i) p[i] = uint8_t(v >> (8 * i)); }
inline uint16_t ld_u16(const uint8_t* p) { return uint16_t(p[0] | (p[1] << 8)); }
inline uint32_t ld_u32(const uint8_t* p) { return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24); }
inline uint64_t ld_u64(const uint8_t* p) { return uint64_t(ld_u32(p)) | (uint64_t(ld_u32(p + 4)) << 32); }

// ---- CRC-32 (ISO 3309 / zlib polynomial 0xEDB88320) -------------------------------------------------------------
inline uint32_t crc32(const uint8_t* p, size_t n, uint32_t crc = 0) {
    static const std::array<uint32_t, 256> tab = [] {
        std::array<uint32_t, 256> t{};
        for (uint32_t i = 0; i < 256; ++i) {
            uint32_t c = i;
            for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            t[i] = c;
        }
        return t;
    }();
    crc = ~crc;
    for (size_t i = 0; i < n; ++i) crc = tab[(crc ^ p[i]) & 255] ^ (crc >> 8);
    return ~crc;
}

// ---- LEB128 --------------------------------------------------------------------------------------------------
inline void leb_put(std::vector<uint8_t>& o, uint64_t v) {
    while (v >= 128) { o.push_back(uint8_t(v | 128)); v >>= 7; }
    o.push_back(uint8_t(v));
}
/// Bounds-checked LEB128 reader over [p, end).
inline uint64_t leb_get(const uint8_t*& p, const uint8_t* end) {
    uint64_t v = 0;
    int s = 0;
    for (;;) {
        if (p >= end) throw Error(Errc::Corrupt, "LEB128 runs past the end of its stream");
        uint8_t b = *p++;
        if (s > 63) throw Error(Errc::Corrupt, "LEB128 too long");
        v |= uint64_t(b & 127) << s;
        if (!(b & 128)) return v;
        s += 7;
    }
}

// ---- integer helpers (no compiler builtins, so MSVC builds) -------------------------------------------------
inline int ilog2u(uint32_t x) { int r = 0; x |= 1; while (x >>= 1) ++r; return r; }   // slow generic form
#if defined(__GNUC__) || defined(__clang__)
inline int ilog2f(uint32_t x) { return 31 - __builtin_clz(x | 1); }
inline int popcnt32(uint32_t x) { return __builtin_popcount(x); }
inline int ctz64(uint64_t x) { return __builtin_ctzll(x); }
#else
inline int ilog2f(uint32_t x) { return ilog2u(x); }
inline int popcnt32(uint32_t x) { int c = 0; while (x) { x &= x - 1; ++c; } return c; }
inline int ctz64(uint64_t x) { int c = 0; while (!(x & 1)) { x >>= 1; ++c; } return c; }
#endif

// ---- zstd framing: [u32 raw_len][u32 zstd_len][zstd frame]; raw_len == 0 stores zstd_len 0 and no frame ---------------------
inline std::vector<uint8_t> zstd_pack(const uint8_t* d, size_t n, int level) {
    std::vector<uint8_t> z(ZSTD_compressBound(n));
    size_t r = n ? ZSTD_compress(z.data(), z.size(), d, n, level) : 0;
    if (n && ZSTD_isError(r)) throw Error(Errc::Corrupt, std::string("zstd compress: ") + ZSTD_getErrorName(r));
    z.resize(r);
    return z;
}
inline void put_stream(std::vector<uint8_t>& o, const uint8_t* d, size_t n, int level) {
    if (n > 0xFFFFFFFFull) throw Error(Errc::BadArgument, "side stream larger than 4 GiB");
    auto z = zstd_pack(d, n, level);
    put_u32(o, uint32_t(n));
    put_u32(o, uint32_t(z.size()));
    o.insert(o.end(), z.begin(), z.end());
}
/// Decode a framed stream starting at p (within [p, end)); advances p.  `max_raw` bounds the allocation.
inline std::vector<uint8_t> get_stream(const uint8_t*& p, const uint8_t* end, uint64_t max_raw) {
    if (end - p < 8) throw Error(Errc::Truncated, "stream header");
    uint64_t n = ld_u32(p), zn = ld_u32(p + 4);
    p += 8;
    if (zn > uint64_t(end - p)) throw Error(Errc::Truncated, "stream body");
    if (n > max_raw) throw Error(Errc::Corrupt, "stream raw length exceeds bound");
    if ((n == 0) != (zn == 0)) throw Error(Errc::Corrupt, "stream length fields inconsistent");
    std::vector<uint8_t> out(static_cast<size_t>(n));
    if (n) {
        size_t r = ZSTD_decompress(out.data(), out.size(), p, size_t(zn));
        if (ZSTD_isError(r) || r != n) throw Error(Errc::Corrupt, "zstd decompress failed");
    }
    p += zn;
    return out;
}

// ---- names -----------------------------------------------------------------------------------------------------
inline uint64_t blob_count(const std::vector<uint8_t>& b) {
    uint64_t c = 0;
    for (uint8_t x : b) c += (x == 0);
    return c;
}

}  // namespace detail

inline std::vector<uint8_t> join_names(const std::vector<std::string>& names) {
    std::vector<uint8_t> o;
    size_t tot = 0;
    for (auto& s : names) tot += s.size() + 1;
    o.reserve(tot);
    for (auto& s : names) { o.insert(o.end(), s.begin(), s.end()); o.push_back(0); }
    return o;
}
inline std::vector<std::string> split_names(const std::vector<uint8_t>& b, size_t count) {
    std::vector<std::string> v;
    v.reserve(count);
    size_t i = 0;
    for (size_t k = 0; k < count; ++k) {
        size_t j = i;
        while (j < b.size() && b[j] != 0) ++j;
        v.emplace_back(reinterpret_cast<const char*>(b.data() + std::min(i, b.size())), j - std::min(i, b.size()));
        i = j + 1;
    }
    return v;
}

inline uint32_t default_chunk_cols(uint32_t n, uint64_t nnz, uint32_t target_nnz) {
    // About `target_nnz` non-zeros per chunk, as a power of two in [128, 4096] columns (rounded at the geometric midpoint).
    if (target_nnz == 0) target_nnz = 49152;
    if (n == 0 || nnz == 0) return 1024;
    double mean_k = double(nnz) / n;
    double want = double(target_nnz) / std::max(mean_k, 1.0);
    uint32_t c = 128;
    while (c < 4096 && c * 1.414 < want) c *= 2;
    return c;
}

}  // namespace pz2
}  // namespace singlet
