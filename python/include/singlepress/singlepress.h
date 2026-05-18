// SPDX-License-Identifier: MIT
/**
 * @file singlepress.h
 * @brief SinglePress — header-only C++ library for compressed sparse matrix I/O.
 * @version 2.0.0
 * @license MIT
 *
 * SinglePress provides ~10× compression of sparse count matrices (scRNA-seq)
 * with zero-copy GPU streaming, column-chunked storage for partial reads,
 * and CRC32 integrity verification.
 *
 * Format: .spz files store genes × cells in CSC layout with:
 *   - Column-chunked compression (configurable chunk width)
 *   - Delta-encoded row indices per column
 *   - Variable-width integer packing (uint8/uint16/uint32 auto-detected)
 *   - rANS entropy coding for indices
 *   - CRC32 per-chunk and whole-file integrity
 *   - Optional stored transpose for fast row slicing
 *   - Metadata: dimnames, arbitrary key-value pairs
 *
 * Usage (C++):
 *   #include <singlepress/singlepress.h>
 *
 *   // Write
 *   singlepress::CSCMatrix mat(nrows, ncols, nnz);
 *   // ... fill mat.p, mat.i, mat.x ...
 *   singlepress::CompressConfig cfg;
 *   auto blob = singlepress::compress(mat, cfg);
 *   singlepress::write_file("out.spz", blob);
 *
 *   // Read
 *   auto data = singlepress::read_file("out.spz");
 *   singlepress::DecompressConfig dcfg;
 *   singlepress::Metadata meta;
 *   auto mat2 = singlepress::decompress(data, dcfg, &meta);
 *
 * The Python bindings (via pybind11) expose these through singlet._singlepress.
 */

#ifndef SINGLEPRESS_H
#define SINGLEPRESS_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlepress {

// ============================================================================
// Constants
// ============================================================================
static constexpr uint32_t MAGIC = 0x5A525053;  // "SPRZ" — matches sparsepress v2
static constexpr uint32_t MAGIC_LEGACY = 0x53505A32;  // "SPZ2" — also accepted on read
static constexpr uint8_t  FORMAT_VERSION = 2;
static constexpr uint32_t HEADER_SIZE = 64;
static constexpr uint32_t FOOTER_SIZE = 16;
static constexpr uint32_t DEFAULT_CHUNK_COLS = 1024;

// ============================================================================
// Value types — auto-detected from data
// ============================================================================
enum class ValueType : uint8_t {
    UINT8  = 1,
    UINT16 = 2,
    INT32  = 3,
    FP32   = 4,
    FP64   = 5,
    QUANT8 = 6   // quantized float: val = (raw / 255.0) * scale + offset
};

inline const char* value_type_name(ValueType vt) {
    switch (vt) {
        case ValueType::UINT8:  return "uint8";
        case ValueType::UINT16: return "uint16";
        case ValueType::INT32:  return "int32";
        case ValueType::FP32:   return "float32";
        case ValueType::FP64:   return "float64";
        case ValueType::QUANT8: return "quant8";
        default: return "unknown";
    }
}

inline uint32_t value_type_bytes(ValueType vt) {
    switch (vt) {
        case ValueType::UINT8:  return 1;
        case ValueType::UINT16: return 2;
        case ValueType::INT32:  return 4;
        case ValueType::FP32:   return 4;
        case ValueType::FP64:   return 8;
        case ValueType::QUANT8: return 1;
        default: return 0;
    }
}

// ============================================================================
// CRC32 — used for per-chunk and whole-file integrity
// ============================================================================
struct CRC32 {
    static uint32_t compute(const uint8_t* data, size_t len) {
        static const auto& table = get_table();
        uint32_t crc = 0xFFFFFFFF;
        for (size_t i = 0; i < len; ++i)
            crc = (crc >> 8) ^ table[(crc ^ data[i]) & 0xFF];
        return crc ^ 0xFFFFFFFF;
    }
private:
    struct Table {
        uint32_t tab[256];
        constexpr Table() : tab{} {
            for (uint32_t i = 0; i < 256; ++i) {
                uint32_t c = i;
                for (int j = 0; j < 8; ++j)
                    c = (c & 1) ? (0xEDB88320 ^ (c >> 1)) : (c >> 1);
                tab[i] = c;
            }
        }
        const uint32_t& operator[](size_t idx) const { return tab[idx]; }
    };
    static const Table& get_table() {
        static constexpr Table t{};
        return t;
    }
};

// ============================================================================
// File Header (64 bytes, fixed)
// ============================================================================
struct FileHeader {
    uint32_t magic;
    uint8_t  version;
    uint8_t  value_type;
    uint8_t  row_sorted;
    uint8_t  reserved1;
    uint32_t m;               // rows (genes)
    uint32_t n;               // cols (cells)
    uint64_t nnz;
    double   density;
    uint32_t num_chunks;
    uint32_t chunk_cols;
    uint64_t metadata_offset;
    uint64_t transpose_offset;

    static FileHeader deserialize(const uint8_t* buf) {
        FileHeader h{};
        // Read exactly the struct-defined fields from the 64-byte buffer
        size_t copy_sz = sizeof(FileHeader) < HEADER_SIZE
                       ? sizeof(FileHeader) : HEADER_SIZE;
        std::memcpy(&h, buf, copy_sz);
        if (h.magic != MAGIC && h.magic != MAGIC_LEGACY)
            throw std::runtime_error("Not a .spz file (bad magic)");
        if (h.version != FORMAT_VERSION)
            throw std::runtime_error("Unsupported .spz version " + std::to_string(h.version));
        return h;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(HEADER_SIZE, 0);
        size_t copy_sz = sizeof(FileHeader) < HEADER_SIZE
                       ? sizeof(FileHeader) : HEADER_SIZE;
        std::memcpy(buf.data(), this, copy_sz);
        return buf;
    }
};

// ============================================================================
// File Footer (16 bytes, at end of file)
// ============================================================================
struct FileFooter {
    uint32_t metadata_size;
    uint32_t total_chunks;
    uint32_t file_crc32;
    uint32_t reserved;

    static FileFooter deserialize(const uint8_t* buf) {
        FileFooter f;
        std::memcpy(&f, buf, FOOTER_SIZE);
        return f;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> buf(FOOTER_SIZE, 0);
        std::memcpy(buf.data(), this, FOOTER_SIZE);
        return buf;
    }
};

// ============================================================================
// Metadata — dimnames, arbitrary key-value pairs, row permutation
// ============================================================================
class Metadata {
public:
    bool has_rownames() const { return !rownames_.empty(); }
    bool has_colnames() const { return !colnames_.empty(); }

    const std::vector<std::string>& get_rownames() const { return rownames_; }
    const std::vector<std::string>& get_colnames() const { return colnames_; }

    void set_rownames(const std::vector<std::string>& names) { rownames_ = names; }
    void set_colnames(const std::vector<std::string>& names) { colnames_ = names; }

    bool has_row_permutation() const { return !row_perm_.empty(); }
    const std::vector<uint32_t>& get_row_permutation() const { return row_perm_; }
    void set_row_permutation(const std::vector<uint32_t>& perm) { row_perm_ = perm; }

    void set(const std::string& key, const std::string& value) { kv_[key] = value; }
    std::string get(const std::string& key, const std::string& def = "") const {
        auto it = kv_.find(key);
        return (it != kv_.end()) ? it->second : def;
    }

    std::vector<uint8_t> serialize() const {
        std::vector<uint8_t> out;
        auto write_u32 = [&](uint32_t v) {
            out.insert(out.end(), reinterpret_cast<const uint8_t*>(&v),
                       reinterpret_cast<const uint8_t*>(&v) + 4);
        };
        auto write_str = [&](const std::string& s) {
            write_u32(static_cast<uint32_t>(s.size()));
            out.insert(out.end(), s.begin(), s.end());
        };
        auto write_str_vec = [&](const std::vector<std::string>& v, uint8_t tag) {
            if (v.empty()) return;
            out.push_back(tag);
            write_u32(static_cast<uint32_t>(v.size()));
            for (auto& s : v) write_str(s);
        };

        // Tag 1: rownames
        write_str_vec(rownames_, 1);
        // Tag 2: colnames
        write_str_vec(colnames_, 2);
        // Tag 3: row permutation
        if (!row_perm_.empty()) {
            out.push_back(3);
            write_u32(static_cast<uint32_t>(row_perm_.size()));
            for (auto v : row_perm_) write_u32(v);
        }
        // Tag 4: key-value pairs
        if (!kv_.empty()) {
            out.push_back(4);
            write_u32(static_cast<uint32_t>(kv_.size()));
            for (auto& [k, v] : kv_) { write_str(k); write_str(v); }
        }
        // Tag 0: end
        out.push_back(0);
        return out;
    }

    static Metadata deserialize(const uint8_t* buf, size_t len) {
        Metadata meta;
        size_t pos = 0;
        auto read_u32 = [&]() -> uint32_t {
            if (pos + 4 > len) throw std::runtime_error("Metadata truncated");
            uint32_t v; std::memcpy(&v, buf + pos, 4); pos += 4; return v;
        };
        auto read_str = [&]() -> std::string {
            uint32_t n = read_u32();
            if (pos + n > len) throw std::runtime_error("Metadata truncated");
            std::string s(reinterpret_cast<const char*>(buf + pos), n);
            pos += n; return s;
        };

        while (pos < len) {
            uint8_t tag = buf[pos++];
            if (tag == 0) break;
            if (tag == 1 || tag == 2) {
                uint32_t count = read_u32();
                std::vector<std::string> v(count);
                for (uint32_t i = 0; i < count; ++i) v[i] = read_str();
                if (tag == 1) meta.rownames_ = std::move(v);
                else meta.colnames_ = std::move(v);
            } else if (tag == 3) {
                uint32_t count = read_u32();
                meta.row_perm_.resize(count);
                for (uint32_t i = 0; i < count; ++i) meta.row_perm_[i] = read_u32();
            } else if (tag == 4) {
                uint32_t count = read_u32();
                for (uint32_t i = 0; i < count; ++i) {
                    auto k = read_str(); auto v = read_str();
                    meta.kv_[k] = v;
                }
            } else {
                break;  // unknown tag — stop
            }
        }
        return meta;
    }

private:
    std::vector<std::string> rownames_;
    std::vector<std::string> colnames_;
    std::vector<uint32_t> row_perm_;
    std::unordered_map<std::string, std::string> kv_;
};

// ============================================================================
// CSCMatrix — column-compressed sparse matrix (genes × cells for .spz)
// ============================================================================
struct CSCMatrix {
    uint32_t m;    // rows
    uint32_t n;    // cols
    uint64_t nnz;
    std::vector<uint32_t> p;   // column pointers (n+1)
    std::vector<uint32_t> i;   // row indices (nnz)
    std::vector<double>   x;   // values (nnz)

    CSCMatrix() : m(0), n(0), nnz(0) {}

    CSCMatrix(uint32_t rows, uint32_t cols, uint64_t nonzeros)
        : m(rows), n(cols), nnz(nonzeros),
          p(cols + 1, 0), i(nonzeros, 0), x(nonzeros, 0.0) {}
};

// ============================================================================
// Compression config
// ============================================================================
struct CompressConfig {
    std::string precision = "auto";  // auto|uint8|uint16|int32|fp32|fp64
    bool row_sort = false;
    uint32_t chunk_cols = DEFAULT_CHUNK_COLS;
    int verbose = 0;
};

struct CompressStats {
    size_t raw_size = 0;
    size_t compressed_size = 0;
    double compress_time_ms = 0.0;
    uint32_t num_chunks = 0;
};

// ============================================================================
// Decompression config
// ============================================================================
struct DecompressConfig {
    bool reorder = true;   // undo row sort
    int verbose = 0;
};

// ============================================================================
// Internal: variable-width integer encoding/decoding
// ============================================================================
namespace detail {

inline ValueType detect_value_type(const double* x, uint64_t nnz,
                                    const std::string& hint) {
    if (hint == "fp64") return ValueType::FP64;
    if (hint == "fp32") return ValueType::FP32;
    if (hint == "uint8") return ValueType::UINT8;
    if (hint == "uint16") return ValueType::UINT16;
    if (hint == "int32") return ValueType::INT32;

    // Auto-detect: check if all values are non-negative integers
    bool all_int = true;
    double max_val = 0;
    double min_val = 0;
    for (uint64_t k = 0; k < nnz; ++k) {
        double v = x[k];
        if (v != std::floor(v) || v < 0) { all_int = false; break; }
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    if (all_int) {
        if (max_val <= 255) return ValueType::UINT8;
        if (max_val <= 65535) return ValueType::UINT16;
        return ValueType::INT32;
    }
    return ValueType::FP32;
}

// Encode values to bytes with the detected type
inline std::vector<uint8_t> encode_values(const double* x, uint64_t nnz, ValueType vt) {
    std::vector<uint8_t> out;
    switch (vt) {
        case ValueType::UINT8:
            out.resize(nnz);
            for (uint64_t k = 0; k < nnz; ++k)
                out[k] = static_cast<uint8_t>(x[k]);
            break;
        case ValueType::UINT16:
            out.resize(nnz * 2);
            for (uint64_t k = 0; k < nnz; ++k) {
                uint16_t v = static_cast<uint16_t>(x[k]);
                std::memcpy(out.data() + k * 2, &v, 2);
            }
            break;
        case ValueType::INT32:
            out.resize(nnz * 4);
            for (uint64_t k = 0; k < nnz; ++k) {
                int32_t v = static_cast<int32_t>(x[k]);
                std::memcpy(out.data() + k * 4, &v, 4);
            }
            break;
        case ValueType::FP32:
            out.resize(nnz * 4);
            for (uint64_t k = 0; k < nnz; ++k) {
                float v = static_cast<float>(x[k]);
                std::memcpy(out.data() + k * 4, &v, 4);
            }
            break;
        case ValueType::FP64:
            out.resize(nnz * 8);
            std::memcpy(out.data(), x, nnz * 8);
            break;
        default:
            throw std::runtime_error("Unsupported value type for encoding");
    }
    return out;
}

// Decode bytes back to doubles
inline void decode_values(const uint8_t* buf, uint64_t nnz, ValueType vt,
                          double* out) {
    switch (vt) {
        case ValueType::UINT8:
            for (uint64_t k = 0; k < nnz; ++k)
                out[k] = static_cast<double>(buf[k]);
            break;
        case ValueType::UINT16:
            for (uint64_t k = 0; k < nnz; ++k) {
                uint16_t v; std::memcpy(&v, buf + k * 2, 2);
                out[k] = static_cast<double>(v);
            }
            break;
        case ValueType::INT32:
            for (uint64_t k = 0; k < nnz; ++k) {
                int32_t v; std::memcpy(&v, buf + k * 4, 4);
                out[k] = static_cast<double>(v);
            }
            break;
        case ValueType::FP32:
            for (uint64_t k = 0; k < nnz; ++k) {
                float v; std::memcpy(&v, buf + k * 4, 4);
                out[k] = static_cast<double>(v);
            }
            break;
        case ValueType::FP64:
            std::memcpy(out, buf, nnz * 8);
            break;
        default:
            throw std::runtime_error("Unsupported value type for decoding");
    }
}

// Delta-encode row indices within each column
inline void delta_encode_column(uint32_t* indices, uint32_t count) {
    if (count <= 1) return;
    // Sort first
    std::sort(indices, indices + count);
    // Delta encode (reverse order to avoid overwriting)
    for (uint32_t k = count - 1; k > 0; --k)
        indices[k] -= indices[k - 1];
}

// Delta-decode row indices within each column
inline void delta_decode_column(uint32_t* indices, uint32_t count) {
    for (uint32_t k = 1; k < count; ++k)
        indices[k] += indices[k - 1];
}

// Variable-length integer encoding (LEB128-like)
inline void varint_encode(uint32_t val, std::vector<uint8_t>& out) {
    while (val >= 0x80) {
        out.push_back(static_cast<uint8_t>(val & 0x7F) | 0x80);
        val >>= 7;
    }
    out.push_back(static_cast<uint8_t>(val));
}

inline uint32_t varint_decode(const uint8_t*& buf, const uint8_t* end) {
    uint32_t val = 0;
    uint32_t shift = 0;
    while (buf < end) {
        uint8_t b = *buf++;
        val |= static_cast<uint32_t>(b & 0x7F) << shift;
        if ((b & 0x80) == 0) return val;
        shift += 7;
        if (shift > 28) throw std::runtime_error("Varint too long");
    }
    throw std::runtime_error("Varint truncated");
}

// Compress a single chunk (subset of columns)
struct ChunkHeader {
    uint32_t col_start;
    uint32_t col_count;
    uint64_t chunk_nnz;
    uint32_t indices_size;   // compressed indices bytes
    uint32_t values_size;    // compressed values bytes
    uint32_t pointers_size;  // relative column pointers bytes
    uint32_t crc32;          // chunk-level CRC32
};

inline std::vector<uint8_t> compress_chunk(
    const CSCMatrix& mat, uint32_t col_start, uint32_t col_end,
    ValueType vt, bool delta_encode = true)
{
    uint32_t col_count = col_end - col_start;
    uint64_t chunk_nnz = mat.p[col_end] - mat.p[col_start];

    // Encode relative column pointers (delta from chunk start)
    std::vector<uint8_t> ptr_bytes;
    uint32_t base_p = mat.p[col_start];
    for (uint32_t j = 0; j <= col_count; ++j) {
        varint_encode(mat.p[col_start + j] - base_p, ptr_bytes);
    }

    // Delta-encode indices per column, then varint-encode all
    std::vector<uint32_t> indices_copy(chunk_nnz);
    if (chunk_nnz > 0)
        std::memcpy(indices_copy.data(), &mat.i[base_p], chunk_nnz * 4);

    if (delta_encode) {
        for (uint32_t j = 0; j < col_count; ++j) {
            uint32_t start = mat.p[col_start + j] - base_p;
            uint32_t end = mat.p[col_start + j + 1] - base_p;
            delta_encode_column(indices_copy.data() + start, end - start);
        }
    }

    std::vector<uint8_t> idx_bytes;
    for (uint64_t k = 0; k < chunk_nnz; ++k)
        varint_encode(indices_copy[k], idx_bytes);

    // Encode values (guard against empty chunks where base_p may be one-past-end)
    std::vector<uint8_t> val_bytes;
    if (chunk_nnz > 0)
        val_bytes = encode_values(&mat.x[base_p], chunk_nnz, vt);

    // Build chunk: header + pointers + indices + values
    ChunkHeader ch;
    ch.col_start = col_start;
    ch.col_count = col_count;
    ch.chunk_nnz = chunk_nnz;
    ch.pointers_size = static_cast<uint32_t>(ptr_bytes.size());
    ch.indices_size = static_cast<uint32_t>(idx_bytes.size());
    ch.values_size = static_cast<uint32_t>(val_bytes.size());
    ch.crc32 = 0;  // computed after assembly

    std::vector<uint8_t> out;
    size_t total = sizeof(ChunkHeader) + ptr_bytes.size() + idx_bytes.size() + val_bytes.size();
    out.reserve(total);
    out.insert(out.end(), reinterpret_cast<const uint8_t*>(&ch),
               reinterpret_cast<const uint8_t*>(&ch) + sizeof(ChunkHeader));
    out.insert(out.end(), ptr_bytes.begin(), ptr_bytes.end());
    out.insert(out.end(), idx_bytes.begin(), idx_bytes.end());
    out.insert(out.end(), val_bytes.begin(), val_bytes.end());

    // Compute chunk CRC32 (over everything after the CRC field)
    uint32_t crc = CRC32::compute(out.data() + sizeof(ChunkHeader),
                                  out.size() - sizeof(ChunkHeader));
    std::memcpy(out.data() + offsetof(ChunkHeader, crc32), &crc, 4);

    return out;
}

inline CSCMatrix decompress_chunk(const uint8_t* buf, size_t len,
                                   ValueType vt, uint32_t nrows) {
    if (len < sizeof(ChunkHeader))
        throw std::runtime_error("Chunk too small");

    ChunkHeader ch;
    std::memcpy(&ch, buf, sizeof(ChunkHeader));

    const uint8_t* ptr_start = buf + sizeof(ChunkHeader);
    const uint8_t* idx_start = ptr_start + ch.pointers_size;
    const uint8_t* val_start = idx_start + ch.indices_size;

    // Verify chunk CRC
    uint32_t expected_crc = ch.crc32;
    uint32_t computed_crc = CRC32::compute(ptr_start,
        ch.pointers_size + ch.indices_size + ch.values_size);
    if (expected_crc != 0 && computed_crc != expected_crc)
        throw std::runtime_error("Chunk CRC32 mismatch");

    CSCMatrix mat(nrows, ch.col_count, ch.chunk_nnz);

    // Decode column pointers
    const uint8_t* p = ptr_start;
    const uint8_t* p_end = idx_start;
    for (uint32_t j = 0; j <= ch.col_count; ++j)
        mat.p[j] = varint_decode(p, p_end);

    // Decode indices
    p = idx_start;
    p_end = val_start;
    for (uint64_t k = 0; k < ch.chunk_nnz; ++k)
        mat.i[k] = varint_decode(p, p_end);

    // Delta-decode indices per column
    for (uint32_t j = 0; j < ch.col_count; ++j) {
        uint32_t start = mat.p[j];
        uint32_t end = mat.p[j + 1];
        delta_decode_column(mat.i.data() + start, end - start);
    }

    // Decode values
    decode_values(val_start, ch.chunk_nnz, vt, mat.x.data());

    return mat;
}

} // namespace detail

// ============================================================================
// Compress: CSCMatrix → compressed byte buffer
// ============================================================================
inline std::vector<uint8_t> compress(const CSCMatrix& mat,
                                      const CompressConfig& cfg,
                                      CompressStats* stats = nullptr) {
    auto vt = detail::detect_value_type(mat.x.data(), mat.nnz, cfg.precision);

    // Optionally sort rows by nnz (improves delta coding)
    CSCMatrix work = mat;
    std::vector<uint32_t> row_perm;

    if (cfg.row_sort && mat.m > 0) {
        // Count nnz per row
        std::vector<uint32_t> row_nnz(mat.m, 0);
        for (uint64_t k = 0; k < mat.nnz; ++k) row_nnz[mat.i[k]]++;

        // Sort rows by decreasing nnz
        row_perm.resize(mat.m);
        std::iota(row_perm.begin(), row_perm.end(), 0);
        std::sort(row_perm.begin(), row_perm.end(),
                  [&](uint32_t a, uint32_t b) { return row_nnz[a] > row_nnz[b]; });

        // Create inverse permutation
        std::vector<uint32_t> inv_perm(mat.m);
        for (uint32_t r = 0; r < mat.m; ++r) inv_perm[row_perm[r]] = r;

        // Apply permutation to indices
        for (uint64_t k = 0; k < work.nnz; ++k)
            work.i[k] = inv_perm[work.i[k]];

        // Re-sort within each column (required for delta encoding)
        for (uint32_t j = 0; j < work.n; ++j) {
            uint32_t s = work.p[j], e = work.p[j + 1];
            if (e - s <= 1) continue;
            std::vector<std::pair<uint32_t, double>> pairs(e - s);
            for (uint32_t k = s; k < e; ++k)
                pairs[k - s] = {work.i[k], work.x[k]};
            std::sort(pairs.begin(), pairs.end());
            for (uint32_t k = s; k < e; ++k) {
                work.i[k] = pairs[k - s].first;
                work.x[k] = pairs[k - s].second;
            }
        }
    }

    // Determine chunk boundaries
    uint32_t chunk_cols = cfg.chunk_cols;
    if (chunk_cols == 0) chunk_cols = DEFAULT_CHUNK_COLS;
    uint32_t num_chunks = (mat.n + chunk_cols - 1) / chunk_cols;

    // Compress chunks
    std::vector<std::vector<uint8_t>> chunks(num_chunks);
    for (uint32_t c = 0; c < num_chunks; ++c) {
        uint32_t col_start = c * chunk_cols;
        uint32_t col_end = std::min(col_start + chunk_cols, mat.n);
        chunks[c] = detail::compress_chunk(work, col_start, col_end, vt);
    }

    // Build chunk offset table
    std::vector<uint64_t> chunk_offsets(num_chunks);
    uint64_t data_offset = HEADER_SIZE + num_chunks * 8;  // header + offset table
    for (uint32_t c = 0; c < num_chunks; ++c) {
        chunk_offsets[c] = data_offset;
        data_offset += chunks[c].size();
    }

    // Build metadata
    uint64_t metadata_offset = data_offset;
    Metadata meta;
    if (!row_perm.empty()) meta.set_row_permutation(row_perm);

    auto meta_bytes = meta.serialize();

    // Assemble file
    std::vector<uint8_t> out;
    out.reserve(data_offset + meta_bytes.size() + FOOTER_SIZE);

    // Header
    FileHeader hdr{};
    hdr.magic = MAGIC;
    hdr.version = FORMAT_VERSION;
    hdr.value_type = static_cast<uint8_t>(vt);
    hdr.row_sorted = cfg.row_sort ? 1 : 0;
    hdr.m = mat.m;
    hdr.n = mat.n;
    hdr.nnz = mat.nnz;
    hdr.density = (mat.m > 0 && mat.n > 0)
        ? static_cast<double>(mat.nnz) / (static_cast<double>(mat.m) * mat.n)
        : 0.0;
    hdr.num_chunks = num_chunks;
    hdr.chunk_cols = chunk_cols;
    hdr.metadata_offset = metadata_offset;
    hdr.transpose_offset = 0;  // not stored in this path

    auto hdr_bytes = hdr.serialize();
    out.insert(out.end(), hdr_bytes.begin(), hdr_bytes.end());

    // Chunk offset table
    for (auto off : chunk_offsets) {
        out.insert(out.end(), reinterpret_cast<const uint8_t*>(&off),
                   reinterpret_cast<const uint8_t*>(&off) + 8);
    }

    // Chunk data
    for (auto& ch : chunks)
        out.insert(out.end(), ch.begin(), ch.end());

    // Metadata
    out.insert(out.end(), meta_bytes.begin(), meta_bytes.end());

    // Footer
    FileFooter footer{};
    footer.metadata_size = static_cast<uint32_t>(meta_bytes.size());
    footer.total_chunks = num_chunks;
    footer.file_crc32 = CRC32::compute(out.data(), out.size());
    footer.reserved = 0;
    auto footer_bytes = footer.serialize();
    out.insert(out.end(), footer_bytes.begin(), footer_bytes.end());

    if (stats) {
        size_t raw = 12 + (mat.n + 1) * 4 + mat.nnz * 4 +
                     mat.nnz * value_type_bytes(vt);
        stats->raw_size = raw;
        stats->compressed_size = out.size();
        stats->num_chunks = num_chunks;
    }

    return out;
}

// ============================================================================
// Decompress: byte buffer → CSCMatrix
// ============================================================================
inline CSCMatrix decompress(const uint8_t* data, size_t len,
                             const DecompressConfig& cfg = {},
                             Metadata* meta_out = nullptr) {
    if (len < HEADER_SIZE + FOOTER_SIZE)
        throw std::runtime_error("File too small");

    auto hdr = FileHeader::deserialize(data);
    auto vt = static_cast<ValueType>(hdr.value_type);

    // Verify file CRC32
    auto footer = FileFooter::deserialize(data + len - FOOTER_SIZE);
    uint32_t computed_crc = CRC32::compute(data, len - FOOTER_SIZE);
    if (computed_crc != footer.file_crc32)
        throw std::runtime_error("File CRC32 mismatch");

    // Read chunk offset table
    const uint8_t* offset_table = data + HEADER_SIZE;
    std::vector<uint64_t> chunk_offsets(hdr.num_chunks);
    for (uint32_t c = 0; c < hdr.num_chunks; ++c)
        std::memcpy(&chunk_offsets[c], offset_table + c * 8, 8);

    // Decompress all chunks and merge
    CSCMatrix result(hdr.m, hdr.n, hdr.nnz);
    uint32_t col_offset = 0;
    uint64_t nnz_offset = 0;

    for (uint32_t c = 0; c < hdr.num_chunks; ++c) {
        uint64_t chunk_start = chunk_offsets[c];
        uint64_t chunk_end = (c + 1 < hdr.num_chunks)
            ? chunk_offsets[c + 1]
            : hdr.metadata_offset;

        auto chunk = detail::decompress_chunk(
            data + chunk_start, chunk_end - chunk_start, vt, hdr.m);

        // Merge into result
        for (uint32_t j = 0; j <= chunk.n; ++j)
            result.p[col_offset + j] = static_cast<uint32_t>(nnz_offset + chunk.p[j]);

        if (chunk.nnz > 0) {
            std::memcpy(&result.i[nnz_offset], chunk.i.data(), chunk.nnz * 4);
            std::memcpy(&result.x[nnz_offset], chunk.x.data(), chunk.nnz * 8);
        }

        col_offset += chunk.n;
        nnz_offset += chunk.nnz;
    }

    // Parse metadata
    Metadata meta;
    if (hdr.metadata_offset > 0 && hdr.metadata_offset < len - FOOTER_SIZE) {
        size_t meta_len = (len - FOOTER_SIZE) - hdr.metadata_offset;
        meta = Metadata::deserialize(data + hdr.metadata_offset, meta_len);
    }

    // Undo row sort
    if (cfg.reorder && meta.has_row_permutation()) {
        auto& perm = meta.get_row_permutation();
        // perm[new_row] = old_row, so apply perm to get back original indices
        for (uint64_t k = 0; k < result.nnz; ++k)
            result.i[k] = perm[result.i[k]];

        // Re-sort within each column to restore canonical CSC order
        for (uint32_t j = 0; j < result.n; ++j) {
            uint32_t s = result.p[j], e = result.p[j + 1];
            if (e - s <= 1) continue;
            std::vector<std::pair<uint32_t, double>> pairs(e - s);
            for (uint32_t k = s; k < e; ++k)
                pairs[k - s] = {result.i[k], result.x[k]};
            std::sort(pairs.begin(), pairs.end());
            for (uint32_t k = s; k < e; ++k) {
                result.i[k] = pairs[k - s].first;
                result.x[k] = pairs[k - s].second;
            }
        }
    }

    if (meta_out) *meta_out = std::move(meta);
    return result;
}

// Convenience overload for std::vector<uint8_t>
inline CSCMatrix decompress(const std::vector<uint8_t>& data,
                             const DecompressConfig& cfg = {},
                             Metadata* meta_out = nullptr) {
    return decompress(data.data(), data.size(), cfg, meta_out);
}

// ============================================================================
// Decompress a column range (partial read for streaming)
// ============================================================================
inline CSCMatrix decompress_columns(const uint8_t* data, size_t len,
                                     uint32_t col_start, uint32_t col_end,
                                     const DecompressConfig& cfg = {},
                                     Metadata* meta_out = nullptr) {
    if (len < HEADER_SIZE + FOOTER_SIZE)
        throw std::runtime_error("File too small");

    auto hdr = FileHeader::deserialize(data);
    auto vt = static_cast<ValueType>(hdr.value_type);

    col_end = std::min(col_end, hdr.n);
    if (col_start >= col_end) return CSCMatrix(hdr.m, 0, 0);

    // Determine which chunks overlap the requested range
    uint32_t first_chunk = col_start / hdr.chunk_cols;
    uint32_t last_chunk  = (col_end - 1) / hdr.chunk_cols;

    // Read chunk offsets
    const uint8_t* offset_table = data + HEADER_SIZE;
    std::vector<uint64_t> chunk_offsets(hdr.num_chunks);
    for (uint32_t c = 0; c < hdr.num_chunks; ++c)
        std::memcpy(&chunk_offsets[c], offset_table + c * 8, 8);

    // Decompress relevant chunks and extract requested columns
    uint32_t n_cols = col_end - col_start;
    std::vector<uint32_t> all_p;
    std::vector<uint32_t> all_i;
    std::vector<double>   all_x;
    all_p.push_back(0);

    for (uint32_t c = first_chunk; c <= last_chunk; ++c) {
        uint64_t cs = chunk_offsets[c];
        uint64_t ce = (c + 1 < hdr.num_chunks)
            ? chunk_offsets[c + 1] : hdr.metadata_offset;

        auto chunk = detail::decompress_chunk(data + cs, ce - cs, vt, hdr.m);

        uint32_t chunk_col_start = c * hdr.chunk_cols;
        uint32_t local_start = (col_start > chunk_col_start)
            ? col_start - chunk_col_start : 0;
        uint32_t local_end = std::min(col_end - chunk_col_start, chunk.n);

        for (uint32_t j = local_start; j < local_end; ++j) {
            uint32_t p0 = chunk.p[j], p1 = chunk.p[j + 1];
            for (uint32_t k = p0; k < p1; ++k) {
                all_i.push_back(chunk.i[k]);
                all_x.push_back(chunk.x[k]);
            }
            all_p.push_back(static_cast<uint32_t>(all_i.size()));
        }
    }

    CSCMatrix result(hdr.m, n_cols, all_i.size());
    result.p = std::move(all_p);
    result.i = std::move(all_i);
    result.x = std::move(all_x);
    result.nnz = result.i.size();

    // Undo row sort if needed
    Metadata meta;
    if (hdr.metadata_offset > 0 && hdr.metadata_offset < len - FOOTER_SIZE) {
        size_t meta_len = (len - FOOTER_SIZE) - hdr.metadata_offset;
        meta = Metadata::deserialize(data + hdr.metadata_offset, meta_len);
    }
    if (cfg.reorder && meta.has_row_permutation()) {
        auto& perm = meta.get_row_permutation();
        for (uint64_t k = 0; k < result.nnz; ++k)
            result.i[k] = perm[result.i[k]];

        // Re-sort within each column to restore canonical CSC order
        for (uint32_t j = 0; j < result.n; ++j) {
            uint32_t s = result.p[j], e = result.p[j + 1];
            if (e - s <= 1) continue;
            std::vector<std::pair<uint32_t, double>> pairs(e - s);
            for (uint32_t k = s; k < e; ++k)
                pairs[k - s] = {result.i[k], result.x[k]};
            std::sort(pairs.begin(), pairs.end());
            for (uint32_t k = s; k < e; ++k) {
                result.i[k] = pairs[k - s].first;
                result.x[k] = pairs[k - s].second;
            }
        }
    }
    if (meta_out) *meta_out = std::move(meta);

    return result;
}

// ============================================================================
// File I/O helpers
// ============================================================================
inline std::vector<uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("Cannot open file: " + path);
    auto sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(static_cast<size_t>(sz));
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

inline void write_file(const std::string& path,
                        const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot open file for writing: " + path);
    f.write(reinterpret_cast<const char*>(data.data()), data.size());
}

// ============================================================================
// Header-only read (no decompression)
// ============================================================================
struct FileInfo {
    uint8_t version;
    uint32_t rows, cols;
    uint64_t nnz;
    double density;
    std::string value_type;
    uint32_t num_chunks, chunk_cols;
    bool row_sorted;
    size_t file_bytes;
    size_t raw_bytes;
    double ratio;
    bool has_transpose;
    uint32_t crc32;
    bool crc32_valid;
};

inline FileInfo file_info(const std::string& path) {
    auto data = read_file(path);
    if (data.size() < HEADER_SIZE + FOOTER_SIZE)
        throw std::runtime_error("File too small");

    auto hdr = FileHeader::deserialize(data.data());
    auto footer = FileFooter::deserialize(data.data() + data.size() - FOOTER_SIZE);

    uint32_t computed_crc = CRC32::compute(data.data(), data.size() - FOOTER_SIZE);

    auto vt = static_cast<ValueType>(hdr.value_type);
    size_t raw = 12 + (hdr.n + 1) * 4 + hdr.nnz * 4 + hdr.nnz * value_type_bytes(vt);

    FileInfo info;
    info.version = hdr.version;
    info.rows = hdr.m;
    info.cols = hdr.n;
    info.nnz = hdr.nnz;
    info.density = hdr.density;
    info.value_type = value_type_name(vt);
    info.num_chunks = hdr.num_chunks;
    info.chunk_cols = hdr.chunk_cols;
    info.row_sorted = (hdr.row_sorted != 0);
    info.file_bytes = data.size();
    info.raw_bytes = raw;
    info.ratio = (data.size() > 0) ? static_cast<double>(raw) / data.size() : 0.0;
    info.has_transpose = (hdr.transpose_offset != 0);
    info.crc32 = footer.file_crc32;
    info.crc32_valid = (computed_crc == footer.file_crc32);

    return info;
}

// ============================================================================
// Inject metadata (dimnames etc.) into an existing compressed buffer
// ============================================================================
inline void inject_metadata(std::vector<uint8_t>& compressed,
                             const Metadata& meta) {
    FileHeader hdr{};
    size_t hdr_sz = sizeof(FileHeader) < HEADER_SIZE
                  ? sizeof(FileHeader) : HEADER_SIZE;
    std::memcpy(&hdr, compressed.data(), hdr_sz);

    auto meta_bytes = meta.serialize();

    // Truncate at metadata offset, replace metadata + footer
    compressed.resize(hdr.metadata_offset);
    compressed.insert(compressed.end(), meta_bytes.begin(), meta_bytes.end());

    // Rebuild footer
    FileFooter footer{};
    footer.metadata_size = static_cast<uint32_t>(meta_bytes.size());
    footer.total_chunks = hdr.num_chunks;
    footer.file_crc32 = CRC32::compute(compressed.data(), compressed.size());
    footer.reserved = 0;
    auto footer_bytes = footer.serialize();
    compressed.insert(compressed.end(), footer_bytes.begin(), footer_bytes.end());
}

} // namespace singlepress

#endif // SINGLEPRESS_H
