// SPDX-License-Identifier: MIT
// Compression study harness: shared types, .1pz loading, canonical form, hashing, timing.
//
// Every experimental codec in this study operates on a CanonicalMatrix: the exact CSC
// triple (indptr, indices, data) plus names that singlet::pz::read_1pz() returns for a
// .1pz member. Two levels of "bit identical" are checked:
//   1. matrix roundtrip:  decode(encode(M)) == M   (indptr/indices/data/names byte-equal)
//   2. file roundtrip:    write_1pz(decode(encode(M)), level=3, chunk_cols=1024) == original bytes
// Level 2 holds whenever level 1 holds, because write_1pz is deterministic; the harness still
// checks it so the claim "bit-identical reconstruction of the original .1pz file" is measured,
// not assumed.
#pragma once
#include <singlet/pileup/pz_reader.h>   // also brings in pz_writer.h, zstd
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace cs {

struct CanonicalMatrix {
    uint32_t m = 0, n = 0;
    uint64_t nnz = 0;
    uint8_t vt_code = 0;
    std::vector<uint32_t> indptr;     // n+1
    std::vector<uint32_t> indices;    // nnz, row (gene) ids, ascending within a column
    std::vector<uint32_t> data;       // nnz, UMI counts (>0)
    std::vector<std::string> rownames, colnames;
    std::map<std::string, std::string> user_kv;

    bool operator==(const CanonicalMatrix& o) const {
        return m == o.m && n == o.n && nnz == o.nnz && indptr == o.indptr &&
               indices == o.indices && data == o.data && rownames == o.rownames &&
               colnames == o.colnames && user_kv == o.user_kv;
    }
};

inline std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.seekg(0, std::ios::end);
    std::vector<uint8_t> buf(static_cast<size_t>(f.tellg()));
    f.seekg(0);
    f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size()));
    return buf;
}

inline void write_file(const std::string& path, const std::vector<uint8_t>& b) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(b.data()), static_cast<std::streamsize>(b.size()));
}

inline CanonicalMatrix load_1pz(const std::string& path) {
    auto r = singlet::pz::read_1pz(path);
    CanonicalMatrix M;
    M.m = r.m; M.n = r.n; M.nnz = r.nnz; M.vt_code = r.vt_code;
    M.indptr = std::move(r.indptr); M.indices = std::move(r.indices); M.data = std::move(r.data);
    M.rownames = std::move(r.rownames); M.colnames = std::move(r.colnames); M.user_kv = std::move(r.user_kv);
    return M;
}

// Re-emit a .1pz with the production writer's defaults (level 3, 1024 cols/chunk). Used for the
// file-level roundtrip check. Returns the file bytes.
inline std::vector<uint8_t> rewrite_1pz(const CanonicalMatrix& M, const std::string& tmp_path,
                                        int level = 3, int chunk_cols = 1024) {
    std::vector<int32_t> indptr(M.indptr.begin(), M.indptr.end());
    std::vector<int32_t> indices(M.indices.begin(), M.indices.end());
    uint32_t mx = 0;
    for (auto v : M.data) mx = std::max(mx, v);
    if (mx <= 255) {
        std::vector<uint8_t> d(M.data.begin(), M.data.end());
        singlet::pz::write_1pz<uint8_t>(tmp_path, M.m, M.n, indptr, indices, d, M.rownames, M.colnames, level, chunk_cols, 1, M.user_kv);
    } else if (mx <= 65535) {
        std::vector<uint16_t> d(M.data.begin(), M.data.end());
        singlet::pz::write_1pz<uint16_t>(tmp_path, M.m, M.n, indptr, indices, d, M.rownames, M.colnames, level, chunk_cols, 1, M.user_kv);
    } else {
        std::vector<uint32_t> d(M.data.begin(), M.data.end());
        singlet::pz::write_1pz<uint32_t>(tmp_path, M.m, M.n, indptr, indices, d, M.rownames, M.colnames, level, chunk_cols, 1, M.user_kv);
    }
    return read_file(tmp_path);
}

// FNV-1a 64 over the canonical arrays — cheap identity check for logs (not a security hash).
inline uint64_t fnv64(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
inline uint64_t hash_matrix(const CanonicalMatrix& M) {
    uint64_t h = fnv64(M.indptr.data(), M.indptr.size() * 4);
    h = fnv64(M.indices.data(), M.indices.size() * 4, h);
    h = fnv64(M.data.data(), M.data.size() * 4, h);
    for (auto& s : M.rownames) h = fnv64(s.data(), s.size(), h);
    for (auto& s : M.colnames) h = fnv64(s.data(), s.size(), h);
    return h;
}

struct Timer {
    std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
    double ms() const {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
};

// A codec under test. encode() returns the full self-describing byte stream (everything a
// decoder needs except the model, which must be a compile-time or separately-loaded artefact
// that fits comfortably in L3 (<= 8 MiB on this machine)). decode() must reproduce the matrix.
struct Codec {
    std::string name;
    std::function<std::vector<uint8_t>(const CanonicalMatrix&)> encode;
    std::function<CanonicalMatrix(const std::vector<uint8_t>&)> decode;
    size_t model_bytes = 0;   // size of any out-of-band model the decoder needs (reported, not counted)
};

// Split the encoded size into named components so results can say *where* the bytes go.
struct SizeBreakdown { std::vector<std::pair<std::string, size_t>> parts; };

}  // namespace cs
