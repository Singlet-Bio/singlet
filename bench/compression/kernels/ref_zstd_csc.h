// Reference codecs.
//  ref_1pz_v1_L3   : re-emits the production .1pz (zstd level 3, 1024 cols/chunk); the yardstick.
//  ref_zstd_csc_L  : naive baseline. Raw little-endian CSC arrays (indptr u32, indices u32,
//                    data u16 or u32) and NUL-joined names, each stream zstd level L. Shows how much
//                    the current VOCSC transform earns over plain zstd on the arrays.
#pragma once
#include "common.h"
#include <zstd.h>

namespace cs {

namespace detail {
inline void put_u32(std::vector<uint8_t>& o, uint32_t v) {
    for (int i = 0; i < 4; ++i) o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline void put_u64(std::vector<uint8_t>& o, uint64_t v) {
    for (int i = 0; i < 8; ++i) o.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}
inline uint32_t get_u32(const uint8_t*& p) { uint32_t v; std::memcpy(&v, p, 4); p += 4; return v; }
inline uint64_t get_u64(const uint8_t*& p) { uint64_t v; std::memcpy(&v, p, 8); p += 8; return v; }

inline std::vector<uint8_t> zstd_c(const void* d, size_t n, int level) {
    std::vector<uint8_t> o(ZSTD_compressBound(n));
    size_t z = ZSTD_compress(o.data(), o.size(), d, n, level);
    if (ZSTD_isError(z)) throw std::runtime_error(ZSTD_getErrorName(z));
    o.resize(z);
    return o;
}
inline std::vector<uint8_t> zstd_d(const uint8_t* d, size_t n, size_t out_n) {
    std::vector<uint8_t> o(out_n);
    size_t z = ZSTD_decompress(o.data(), out_n, d, n);
    if (ZSTD_isError(z)) throw std::runtime_error(ZSTD_getErrorName(z));
    return o;
}
// Stream framing: [u64 raw_len][u64 z_len][z bytes]
inline void put_stream(std::vector<uint8_t>& o, const void* d, size_t n, int level) {
    auto z = zstd_c(d, n, level);
    put_u64(o, n);
    put_u64(o, z.size());
    o.insert(o.end(), z.begin(), z.end());
}
inline std::vector<uint8_t> get_stream(const uint8_t*& p) {
    uint64_t n = get_u64(p), zn = get_u64(p);
    auto o = zstd_d(p, static_cast<size_t>(zn), static_cast<size_t>(n));
    p += zn;
    return o;
}
inline std::vector<uint8_t> join_names(const std::vector<std::string>& v) {
    std::vector<uint8_t> o;
    for (auto& s : v) { o.insert(o.end(), s.begin(), s.end()); o.push_back(0); }
    return o;
}
inline std::vector<std::string> split_names(const std::vector<uint8_t>& b, size_t count) {
    std::vector<std::string> v;
    v.reserve(count);
    size_t i = 0;
    for (size_t k = 0; k < count; ++k) {
        size_t j = i;
        while (j < b.size() && b[j] != 0) ++j;
        v.emplace_back(reinterpret_cast<const char*>(b.data() + i), j - i);
        i = j + 1;
    }
    return v;
}
}  // namespace detail

inline Codec make_ref_identity_1pz() {
    Codec c;
    c.name = "ref_1pz_v1_L3";
    c.encode = [](const CanonicalMatrix& M) { return rewrite_1pz(M, "results/.ref_identity.tmp.1pz"); };
    c.decode = [](const std::vector<uint8_t>& b) {
        auto r = singlet::pz::read_1pz_buffer(b.data(), b.size());
        CanonicalMatrix M;
        M.m = r.m; M.n = r.n; M.nnz = r.nnz; M.vt_code = r.vt_code;
        M.indptr = std::move(r.indptr); M.indices = std::move(r.indices); M.data = std::move(r.data);
        M.rownames = std::move(r.rownames); M.colnames = std::move(r.colnames); M.user_kv = std::move(r.user_kv);
        return M;
    };
    return c;
}

inline Codec make_ref_zstd_csc(int level) {
    Codec c;
    c.name = "ref_zstd_csc_L" + std::to_string(level);
    c.encode = [level](const CanonicalMatrix& M) {
        using namespace detail;
        std::vector<uint8_t> o;
        put_u32(o, M.m); put_u32(o, M.n); put_u64(o, M.nnz); o.push_back(M.vt_code);
        put_stream(o, M.indptr.data(), M.indptr.size() * 4, level);
        put_stream(o, M.indices.data(), M.indices.size() * 4, level);
        uint32_t mx = 0;
        for (auto v : M.data) mx = std::max(mx, v);
        uint8_t w = (mx <= 65535) ? 2 : 4;
        o.push_back(w);
        if (w == 2) {
            std::vector<uint16_t> d(M.data.begin(), M.data.end());
            put_stream(o, d.data(), d.size() * 2, level);
        } else {
            put_stream(o, M.data.data(), M.data.size() * 4, level);
        }
        auto rn = join_names(M.rownames), cn = join_names(M.colnames);
        put_stream(o, rn.data(), rn.size(), level);
        put_stream(o, cn.data(), cn.size(), level);
        std::vector<uint8_t> kv;
        for (auto& [k, v] : M.user_kv) {
            kv.insert(kv.end(), k.begin(), k.end()); kv.push_back(0);
            kv.insert(kv.end(), v.begin(), v.end()); kv.push_back(0);
        }
        put_u32(o, static_cast<uint32_t>(M.user_kv.size()));
        put_stream(o, kv.data(), kv.size(), level);
        return o;
    };
    c.decode = [](const std::vector<uint8_t>& b) {
        using namespace detail;
        const uint8_t* p = b.data();
        CanonicalMatrix M;
        M.m = get_u32(p); M.n = get_u32(p); M.nnz = get_u64(p); M.vt_code = *p++;
        auto ip = get_stream(p);
        M.indptr.resize(ip.size() / 4);
        std::memcpy(M.indptr.data(), ip.data(), ip.size());
        auto ix = get_stream(p);
        M.indices.resize(ix.size() / 4);
        std::memcpy(M.indices.data(), ix.data(), ix.size());
        uint8_t w = *p++;
        auto dd = get_stream(p);
        M.data.resize(dd.size() / w);
        if (w == 2) {
            for (size_t k = 0; k < M.data.size(); ++k) { uint16_t v; std::memcpy(&v, dd.data() + 2 * k, 2); M.data[k] = v; }
        } else {
            std::memcpy(M.data.data(), dd.data(), dd.size());
        }
        M.rownames = split_names(get_stream(p), M.m);
        M.colnames = split_names(get_stream(p), M.n);
        uint32_t nkv = get_u32(p);
        auto kv = get_stream(p);
        auto parts = split_names(kv, 2 * static_cast<size_t>(nkv));
        for (uint32_t i = 0; i < nkv; ++i) M.user_kv[parts[2 * i]] = parts[2 * i + 1];
        return M;
    };
    return c;
}

}  // namespace cs
