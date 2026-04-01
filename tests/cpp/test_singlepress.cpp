/**
 * @file test_singlepress.cpp
 * @brief Comprehensive C++ tests for singlepress.h header-only library.
 *
 * Self-contained test runner — no external test framework dependency.
 * Uses simple assertion macros with clear failure messages.
 *
 * Build:
 *   g++ -std=c++17 -O2 -I../../include tests/cpp/test_singlepress.cpp -o test_singlepress
 *
 * Run:
 *   ./test_singlepress [path_to_real.spz]
 */

#include <singlepress/singlepress.h>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using namespace singlepress;

// ============================================================================
// Minimal test framework
// ============================================================================
static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define TEST(name) \
    static void test_##name(); \
    static bool _reg_##name = (run_test(#name, test_##name), true); \
    static void test_##name()

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << #cond << " at " << __FILE__ \
                  << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("assertion failed"); \
    } \
} while (0)

#define CHECK_EQ(a, b) do { \
    auto _a = (a); auto _b = (b); \
    if (_a != _b) { \
        std::cerr << "  FAIL: " << #a << " == " << #b \
                  << " (" << _a << " != " << _b << ") at " \
                  << __FILE__ << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("assertion failed"); \
    } \
} while (0)

#define CHECK_NEAR(a, b, tol) do { \
    auto _a = (a); auto _b = (b); \
    if (std::abs(_a - _b) > (tol)) { \
        std::cerr << "  FAIL: |" << #a << " - " << #b << "| <= " << #tol \
                  << " (" << _a << " vs " << _b << ") at " \
                  << __FILE__ << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("assertion failed"); \
    } \
} while (0)

#define CHECK_THROWS(expr) do { \
    bool _threw = false; \
    try { expr; } catch (...) { _threw = true; } \
    if (!_threw) { \
        std::cerr << "  FAIL: expected exception from: " << #expr \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        throw std::runtime_error("assertion failed"); \
    } \
} while (0)

static void run_test(const char* name, std::function<void()> fn) {
    g_tests_run++;
    std::cout << "  [" << g_tests_run << "] " << name << " ... ";
    std::cout.flush();
    try {
        fn();
        g_tests_passed++;
        std::cout << "PASS\n";
    } catch (const std::exception& e) {
        g_tests_failed++;
        std::cout << "FAIL (" << e.what() << ")\n";
    }
}

// ============================================================================
// Helpers
// ============================================================================
static std::string g_real_spz_path;  // optionally set from argv[1]

static CSCMatrix make_random_int_csc(uint32_t m, uint32_t n, double density, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    std::uniform_int_distribution<int> v_dist(1, 255);

    std::vector<uint32_t> col_ptrs = {0};
    std::vector<uint32_t> row_idx;
    std::vector<double> vals;

    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) {
            if (p_dist(rng) < density) {
                row_idx.push_back(i);
                vals.push_back(static_cast<double>(v_dist(rng)));
            }
        }
        col_ptrs.push_back(static_cast<uint32_t>(row_idx.size()));
    }

    uint64_t nnz = row_idx.size();
    CSCMatrix mat(m, n, nnz);
    mat.p = col_ptrs;
    mat.i = row_idx;
    for (uint64_t k = 0; k < nnz; ++k) mat.x[k] = vals[k];
    return mat;
}

static CSCMatrix make_random_float_csc(uint32_t m, uint32_t n, double density, uint32_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> p_dist(0.0, 1.0);
    std::uniform_real_distribution<double> v_dist(-10.0, 10.0);

    std::vector<uint32_t> col_ptrs = {0};
    std::vector<uint32_t> row_idx;
    std::vector<double> vals;

    for (uint32_t j = 0; j < n; ++j) {
        for (uint32_t i = 0; i < m; ++i) {
            if (p_dist(rng) < density) {
                row_idx.push_back(i);
                vals.push_back(v_dist(rng));
            }
        }
        col_ptrs.push_back(static_cast<uint32_t>(row_idx.size()));
    }

    uint64_t nnz = row_idx.size();
    CSCMatrix mat(m, n, nnz);
    mat.p = col_ptrs;
    mat.i = row_idx;
    for (uint64_t k = 0; k < nnz; ++k) mat.x[k] = vals[k];
    return mat;
}

static bool matrices_equal(const CSCMatrix& a, const CSCMatrix& b, double tol = 0.0) {
    if (a.m != b.m || a.n != b.n || a.nnz != b.nnz) return false;
    for (uint32_t j = 0; j <= a.n; ++j)
        if (a.p[j] != b.p[j]) return false;
    for (uint64_t k = 0; k < a.nnz; ++k) {
        if (a.i[k] != b.i[k]) return false;
        if (tol == 0.0) {
            if (a.x[k] != b.x[k]) return false;
        } else {
            if (std::abs(a.x[k] - b.x[k]) > tol) return false;
        }
    }
    return true;
}

// ============================================================================
// SECTION: CRC32
// ============================================================================
TEST(crc32_empty) {
    uint8_t buf[1] = {0};
    uint32_t c = CRC32::compute(buf, 0);
    CHECK_EQ(c, 0u);
}

TEST(crc32_deterministic) {
    std::vector<uint8_t> data = {1, 2, 3, 4, 5, 6, 7, 8};
    uint32_t c1 = CRC32::compute(data.data(), data.size());
    uint32_t c2 = CRC32::compute(data.data(), data.size());
    CHECK_EQ(c1, c2);
    CHECK(c1 != 0u);
}

TEST(crc32_single_bit_flip) {
    std::vector<uint8_t> data = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t c1 = CRC32::compute(data.data(), data.size());
    data[3] = 1;  // flip one bit
    uint32_t c2 = CRC32::compute(data.data(), data.size());
    CHECK(c1 != c2);
}

// ============================================================================
// SECTION: Value type detection
// ============================================================================
TEST(detect_uint8) {
    std::vector<double> vals = {0, 1, 127, 255};
    auto vt = detail::detect_value_type(vals.data(), vals.size(), "auto");
    CHECK_EQ(static_cast<int>(vt), static_cast<int>(ValueType::UINT8));
}

TEST(detect_uint16) {
    std::vector<double> vals = {0, 1, 256, 65535};
    auto vt = detail::detect_value_type(vals.data(), vals.size(), "auto");
    CHECK_EQ(static_cast<int>(vt), static_cast<int>(ValueType::UINT16));
}

TEST(detect_int32) {
    std::vector<double> vals = {0, 1, 65536, 100000};
    auto vt = detail::detect_value_type(vals.data(), vals.size(), "auto");
    CHECK_EQ(static_cast<int>(vt), static_cast<int>(ValueType::INT32));
}

TEST(detect_fp32_negative) {
    std::vector<double> vals = {-1.0, 0.5, 3.14};
    auto vt = detail::detect_value_type(vals.data(), vals.size(), "auto");
    CHECK_EQ(static_cast<int>(vt), static_cast<int>(ValueType::FP32));
}

TEST(detect_fp32_fractional) {
    std::vector<double> vals = {0.0, 0.5, 1.5};
    auto vt = detail::detect_value_type(vals.data(), vals.size(), "auto");
    CHECK_EQ(static_cast<int>(vt), static_cast<int>(ValueType::FP32));
}

TEST(detect_hint_overrides) {
    std::vector<double> vals = {1, 2, 3};
    // With hint, should override auto-detection
    CHECK_EQ(static_cast<int>(detail::detect_value_type(vals.data(), vals.size(), "fp64")),
             static_cast<int>(ValueType::FP64));
    CHECK_EQ(static_cast<int>(detail::detect_value_type(vals.data(), vals.size(), "fp32")),
             static_cast<int>(ValueType::FP32));
    CHECK_EQ(static_cast<int>(detail::detect_value_type(vals.data(), vals.size(), "int32")),
             static_cast<int>(ValueType::INT32));
}

TEST(value_type_bytes_all) {
    CHECK_EQ(value_type_bytes(ValueType::UINT8), 1u);
    CHECK_EQ(value_type_bytes(ValueType::UINT16), 2u);
    CHECK_EQ(value_type_bytes(ValueType::INT32), 4u);
    CHECK_EQ(value_type_bytes(ValueType::FP32), 4u);
    CHECK_EQ(value_type_bytes(ValueType::FP64), 8u);
    CHECK_EQ(value_type_bytes(ValueType::QUANT8), 1u);
}

// ============================================================================
// SECTION: Varint encoding
// ============================================================================
TEST(varint_roundtrip_small) {
    std::vector<uint32_t> values = {0, 1, 127};
    for (auto v : values) {
        std::vector<uint8_t> buf;
        detail::varint_encode(v, buf);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        CHECK_EQ(detail::varint_decode(p, end), v);
    }
}

TEST(varint_roundtrip_medium) {
    std::vector<uint32_t> values = {128, 16383, 16384, 65535};
    for (auto v : values) {
        std::vector<uint8_t> buf;
        detail::varint_encode(v, buf);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        CHECK_EQ(detail::varint_decode(p, end), v);
    }
}

TEST(varint_roundtrip_large) {
    std::vector<uint32_t> values = {(1u << 21) - 1, (1u << 28) - 1, 0xFFFFFFFF};
    for (auto v : values) {
        std::vector<uint8_t> buf;
        detail::varint_encode(v, buf);
        const uint8_t* p = buf.data();
        const uint8_t* end = p + buf.size();
        CHECK_EQ(detail::varint_decode(p, end), v);
    }
}

TEST(varint_encoding_size) {
    std::vector<uint8_t> buf;
    detail::varint_encode(0, buf);
    CHECK_EQ(buf.size(), 1u);

    buf.clear();
    detail::varint_encode(127, buf);
    CHECK_EQ(buf.size(), 1u);

    buf.clear();
    detail::varint_encode(128, buf);
    CHECK_EQ(buf.size(), 2u);

    buf.clear();
    detail::varint_encode(16384, buf);
    CHECK_EQ(buf.size(), 3u);
}

// ============================================================================
// SECTION: Delta encoding
// ============================================================================
TEST(delta_encode_decode) {
    std::vector<uint32_t> indices = {2, 5, 8, 15, 100};
    std::vector<uint32_t> original(indices.begin(), indices.end());

    detail::delta_encode_column(indices.data(), static_cast<uint32_t>(indices.size()));
    // After delta encoding: {2, 3, 3, 7, 85}
    CHECK(indices[0] == 2u);  // first stays the same
    CHECK(indices[1] == 3u);  // 5 - 2
    CHECK(indices[2] == 3u);  // 8 - 5

    detail::delta_decode_column(indices.data(), static_cast<uint32_t>(indices.size()));
    for (size_t i = 0; i < original.size(); ++i)
        CHECK_EQ(indices[i], original[i]);
}

TEST(delta_encode_single_element) {
    std::vector<uint32_t> indices = {42};
    detail::delta_encode_column(indices.data(), 1);
    CHECK_EQ(indices[0], 42u);
    detail::delta_decode_column(indices.data(), 1);
    CHECK_EQ(indices[0], 42u);
}

TEST(delta_encode_empty) {
    detail::delta_encode_column(nullptr, 0);
    detail::delta_decode_column(nullptr, 0);
    // Should not crash
}

// ============================================================================
// SECTION: Value encoding round-trip
// ============================================================================
TEST(encode_decode_uint8) {
    std::vector<double> vals = {0, 1, 127, 255};
    auto encoded = detail::encode_values(vals.data(), vals.size(), ValueType::UINT8);
    CHECK_EQ(encoded.size(), vals.size());

    std::vector<double> decoded(vals.size());
    detail::decode_values(encoded.data(), vals.size(), ValueType::UINT8, decoded.data());
    for (size_t i = 0; i < vals.size(); ++i)
        CHECK_EQ(decoded[i], vals[i]);
}

TEST(encode_decode_uint16) {
    std::vector<double> vals = {0, 1, 256, 65535};
    auto encoded = detail::encode_values(vals.data(), vals.size(), ValueType::UINT16);
    CHECK_EQ(encoded.size(), vals.size() * 2);

    std::vector<double> decoded(vals.size());
    detail::decode_values(encoded.data(), vals.size(), ValueType::UINT16, decoded.data());
    for (size_t i = 0; i < vals.size(); ++i)
        CHECK_EQ(decoded[i], vals[i]);
}

TEST(encode_decode_int32) {
    std::vector<double> vals = {0, 1, 65536, 2000000};
    auto encoded = detail::encode_values(vals.data(), vals.size(), ValueType::INT32);
    CHECK_EQ(encoded.size(), vals.size() * 4);

    std::vector<double> decoded(vals.size());
    detail::decode_values(encoded.data(), vals.size(), ValueType::INT32, decoded.data());
    for (size_t i = 0; i < vals.size(); ++i)
        CHECK_EQ(decoded[i], vals[i]);
}

TEST(encode_decode_fp32) {
    std::vector<double> vals = {-1.5, 0.0, 3.14159, 1e6};
    auto encoded = detail::encode_values(vals.data(), vals.size(), ValueType::FP32);

    std::vector<double> decoded(vals.size());
    detail::decode_values(encoded.data(), vals.size(), ValueType::FP32, decoded.data());
    for (size_t i = 0; i < vals.size(); ++i)
        CHECK_NEAR(decoded[i], vals[i], 1e-3);
}

TEST(encode_decode_fp64) {
    std::vector<double> vals = {-1.5e-100, 0.0, 3.141592653589793, 1e100};
    auto encoded = detail::encode_values(vals.data(), vals.size(), ValueType::FP64);

    std::vector<double> decoded(vals.size());
    detail::decode_values(encoded.data(), vals.size(), ValueType::FP64, decoded.data());
    for (size_t i = 0; i < vals.size(); ++i)
        CHECK_EQ(decoded[i], vals[i]);
}

// ============================================================================
// SECTION: Metadata
// ============================================================================
TEST(metadata_empty_roundtrip) {
    Metadata meta;
    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK(!meta2.has_rownames());
    CHECK(!meta2.has_colnames());
}

TEST(metadata_rownames) {
    Metadata meta;
    std::vector<std::string> names = {"Gene1", "Gene2", "Gene3"};
    meta.set_rownames(names);

    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK(meta2.has_rownames());
    CHECK_EQ(meta2.get_rownames().size(), 3u);
    CHECK_EQ(meta2.get_rownames()[0], "Gene1");
    CHECK_EQ(meta2.get_rownames()[2], "Gene3");
}

TEST(metadata_colnames) {
    Metadata meta;
    std::vector<std::string> names = {"Cell_A", "Cell_B"};
    meta.set_colnames(names);

    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK(meta2.has_colnames());
    CHECK_EQ(meta2.get_colnames().size(), 2u);
}

TEST(metadata_rownames_and_colnames) {
    Metadata meta;
    meta.set_rownames({"G1", "G2"});
    meta.set_colnames({"C1", "C2", "C3"});

    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK(meta2.has_rownames());
    CHECK(meta2.has_colnames());
    CHECK_EQ(meta2.get_rownames().size(), 2u);
    CHECK_EQ(meta2.get_colnames().size(), 3u);
}

TEST(metadata_kv_pairs) {
    Metadata meta;
    meta.set("organism", "human");
    meta.set("pipeline_version", "2.0");

    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK_EQ(meta2.get("organism"), "human");
    CHECK_EQ(meta2.get("pipeline_version"), "2.0");
    CHECK_EQ(meta2.get("nonexistent", "default"), "default");
}

TEST(metadata_row_permutation) {
    Metadata meta;
    std::vector<uint32_t> perm = {2, 0, 1};
    meta.set_row_permutation(perm);

    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK(meta2.has_row_permutation());
    CHECK_EQ(meta2.get_row_permutation().size(), 3u);
    CHECK_EQ(meta2.get_row_permutation()[0], 2u);
}

TEST(metadata_unicode_names) {
    Metadata meta;
    meta.set_rownames({"α-actinin", "β-globin", "γ-secretase"});
    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK_EQ(meta2.get_rownames()[0], "α-actinin");
    CHECK_EQ(meta2.get_rownames()[1], "β-globin");
}

TEST(metadata_empty_string_names) {
    Metadata meta;
    meta.set_rownames({"", "Gene2", ""});
    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK_EQ(meta2.get_rownames().size(), 3u);
    CHECK_EQ(meta2.get_rownames()[0], "");
    CHECK_EQ(meta2.get_rownames()[1], "Gene2");
}

TEST(metadata_large_names) {
    Metadata meta;
    std::string long_name(10000, 'X');
    meta.set_rownames({long_name});
    auto bytes = meta.serialize();
    auto meta2 = Metadata::deserialize(bytes.data(), bytes.size());
    CHECK_EQ(meta2.get_rownames()[0].size(), 10000u);
}

// ============================================================================
// SECTION: Compress/decompress round-trips
// ============================================================================
TEST(roundtrip_small_int) {
    auto mat = make_random_int_csc(10, 5, 0.3, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(roundtrip_medium_int) {
    auto mat = make_random_int_csc(1000, 500, 0.05, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(roundtrip_large_int) {
    auto mat = make_random_int_csc(5000, 2000, 0.02, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(roundtrip_float) {
    auto mat = make_random_float_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    cfg.precision = "fp32";
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2, 1e-3));
}

TEST(roundtrip_fp64) {
    auto mat = make_random_float_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    cfg.precision = "fp64";
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(roundtrip_with_row_sort) {
    auto mat = make_random_int_csc(200, 100, 0.05, 42);
    CompressConfig cfg;
    cfg.row_sort = true;
    auto blob = compress(mat, cfg);

    DecompressConfig dcfg;
    dcfg.reorder = true;
    auto mat2 = decompress(blob, dcfg);
    CHECK(matrices_equal(mat, mat2));
}

TEST(row_sort_roundtrip_preserves_data) {
    // Row sort should produce valid compressed data that round-trips correctly,
    // even if it doesn't always reduce size for random data (permutation metadata has overhead)
    auto mat = make_random_int_csc(500, 200, 0.05, 42);

    CompressConfig cfg;
    cfg.row_sort = true;
    auto blob = compress(mat, cfg);

    DecompressConfig dcfg;
    dcfg.reorder = true;
    auto mat2 = decompress(blob, dcfg);
    CHECK(matrices_equal(mat, mat2));
}

TEST(roundtrip_with_dimnames) {
    auto mat = make_random_int_csc(5, 3, 0.5, 42);

    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    Metadata meta;
    meta.set_rownames({"G1", "G2", "G3", "G4", "G5"});
    meta.set_colnames({"C1", "C2", "C3"});
    inject_metadata(blob, meta);

    Metadata meta2;
    auto mat2 = decompress(blob, {}, &meta2);
    CHECK(matrices_equal(mat, mat2));
    CHECK(meta2.has_rownames());
    CHECK_EQ(meta2.get_rownames().size(), 5u);
    CHECK_EQ(meta2.get_rownames()[0], "G1");
    CHECK(meta2.has_colnames());
    CHECK_EQ(meta2.get_colnames().size(), 3u);
}

// ============================================================================
// SECTION: Edge cases
// ============================================================================
TEST(empty_matrix) {
    CSCMatrix mat(0, 0, 0);
    mat.p = {0};
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK_EQ(mat2.m, 0u);
    CHECK_EQ(mat2.n, 0u);
    CHECK_EQ(mat2.nnz, 0u);
}

TEST(single_element) {
    CSCMatrix mat(1, 1, 1);
    mat.p = {0, 1};
    mat.i = {0};
    mat.x = {42.0};
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK_EQ(mat2.m, 1u);
    CHECK_EQ(mat2.n, 1u);
    CHECK_EQ(mat2.nnz, 1u);
    CHECK_EQ(mat2.x[0], 42.0);
}

TEST(all_zeros_density) {
    CSCMatrix mat(100, 50, 0);
    mat.p.assign(51, 0);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK_EQ(mat2.nnz, 0u);
    CHECK_EQ(mat2.m, 100u);
    CHECK_EQ(mat2.n, 50u);
}

TEST(single_column) {
    CSCMatrix mat(100, 1, 3);
    mat.p = {0, 3};
    mat.i = {0, 50, 99};
    mat.x = {1.0, 2.0, 3.0};
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(single_row) {
    CSCMatrix mat(1, 100, 100);
    mat.p.resize(101);
    mat.i.resize(100);
    mat.x.resize(100);
    for (uint32_t j = 0; j < 100; ++j) {
        mat.p[j] = j;
        mat.i[j] = 0;
        mat.x[j] = static_cast<double>(j + 1);
    }
    mat.p[100] = 100;
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(max_uint8_values) {
    CSCMatrix mat(2, 2, 4);
    mat.p = {0, 2, 4};
    mat.i = {0, 1, 0, 1};
    mat.x = {0.0, 255.0, 128.0, 1.0};
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(max_uint16_values) {
    CSCMatrix mat(2, 1, 2);
    mat.p = {0, 2};
    mat.i = {0, 1};
    mat.x = {0.0, 65535.0};
    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(many_empty_columns) {
    CSCMatrix mat(10, 1000, 5);
    mat.p.assign(1001, 0);
    // Put entries in columns 0, 250, 500, 750, 999
    uint32_t cols[5] = {0, 250, 500, 750, 999};
    uint32_t idx = 0;
    for (int c = 0; c < 5; ++c) {
        mat.p[cols[c] + 1] = idx + 1;
        mat.i[idx] = c;
        mat.x[idx] = static_cast<double>(c + 1);
        idx++;
    }
    // Fix col pointers (forward fill)
    for (uint32_t j = 1; j <= 1000; ++j)
        if (mat.p[j] < mat.p[j - 1]) mat.p[j] = mat.p[j - 1];

    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(dense_matrix) {
    // Fully dense matrix
    uint32_t m = 10, n = 10;
    uint64_t nnz = m * n;
    CSCMatrix mat(m, n, nnz);
    uint64_t k = 0;
    for (uint32_t j = 0; j < n; ++j) {
        mat.p[j] = static_cast<uint32_t>(k);
        for (uint32_t i = 0; i < m; ++i) {
            mat.i[k] = i;
            mat.x[k] = static_cast<double>((j + 1) * 10 + i);
            k++;
        }
    }
    mat.p[n] = static_cast<uint32_t>(nnz);

    CompressConfig cfg;
    auto blob = compress(mat, cfg);
    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

// ============================================================================
// SECTION: Column subset reading
// ============================================================================
TEST(column_subset_basic) {
    auto mat = make_random_int_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    // Read columns 10..20
    auto sub = decompress_columns(blob.data(), blob.size(), 10, 20);
    CHECK_EQ(sub.n, 10u);
    CHECK_EQ(sub.m, 100u);

    // Verify values match
    for (uint32_t j = 0; j < 10; ++j) {
        uint32_t orig_start = mat.p[j + 10];
        uint32_t orig_end = mat.p[j + 11];
        uint32_t sub_start = sub.p[j];
        uint32_t sub_end = sub.p[j + 1];
        CHECK_EQ(orig_end - orig_start, sub_end - sub_start);
    }
}

TEST(column_subset_first_chunk) {
    auto mat = make_random_int_csc(50, 10, 0.2, 42);
    CompressConfig cfg;
    cfg.chunk_cols = 5;
    auto blob = compress(mat, cfg);

    auto sub = decompress_columns(blob.data(), blob.size(), 0, 5);
    CHECK_EQ(sub.n, 5u);
}

TEST(column_subset_last_chunk) {
    auto mat = make_random_int_csc(50, 10, 0.2, 42);
    CompressConfig cfg;
    cfg.chunk_cols = 5;
    auto blob = compress(mat, cfg);

    auto sub = decompress_columns(blob.data(), blob.size(), 5, 10);
    CHECK_EQ(sub.n, 5u);
}

TEST(column_subset_cross_chunk) {
    auto mat = make_random_int_csc(50, 20, 0.2, 42);
    CompressConfig cfg;
    cfg.chunk_cols = 5;
    auto blob = compress(mat, cfg);

    // Crosses chunks 1 and 2
    auto sub = decompress_columns(blob.data(), blob.size(), 3, 12);
    CHECK_EQ(sub.n, 9u);
}

TEST(column_subset_empty_range) {
    auto mat = make_random_int_csc(50, 10, 0.2, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    auto sub = decompress_columns(blob.data(), blob.size(), 5, 5);
    CHECK_EQ(sub.n, 0u);
    CHECK_EQ(sub.nnz, 0u);
}

TEST(column_subset_clamp_beyond_end) {
    auto mat = make_random_int_csc(50, 10, 0.2, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    // Request beyond end — should clamp
    auto sub = decompress_columns(blob.data(), blob.size(), 8, 100);
    CHECK_EQ(sub.n, 2u);  // only cols 8,9 exist
}

// ============================================================================
// SECTION: File I/O
// ============================================================================
TEST(file_roundtrip) {
    auto mat = make_random_int_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    std::string path = "/tmp/test_sp_roundtrip.spz";
    write_file(path, blob);

    auto blob2 = read_file(path);
    CHECK_EQ(blob.size(), blob2.size());
    CHECK(std::memcmp(blob.data(), blob2.data(), blob.size()) == 0);

    auto mat2 = decompress(blob2);
    CHECK(matrices_equal(mat, mat2));

    // Clean up
    std::remove(path.c_str());
}

TEST(file_info_basic) {
    auto mat = make_random_int_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    std::string path = "/tmp/test_sp_info.spz";
    write_file(path, blob);

    auto info = file_info(path);
    CHECK_EQ(info.rows, 100u);
    CHECK_EQ(info.cols, 50u);
    CHECK(info.nnz > 0);
    CHECK(info.ratio > 0.0);
    CHECK(info.crc32_valid);
    CHECK_EQ(info.version, FORMAT_VERSION);

    std::remove(path.c_str());
}

TEST(file_crc32_integrity) {
    auto mat = make_random_int_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    std::string path = "/tmp/test_sp_crc.spz";
    write_file(path, blob);

    auto info = file_info(path);
    CHECK(info.crc32_valid);

    // Corrupt one byte in the file
    auto data = read_file(path);
    if (data.size() > HEADER_SIZE + 10) {
        data[HEADER_SIZE + 5] ^= 0xFF;
        write_file(path, data);

        // Should detect corruption
        bool corrupted_detected = false;
        try {
            auto info2 = file_info(path);
            corrupted_detected = !info2.crc32_valid;
        } catch (...) {
            corrupted_detected = true;  // Also acceptable
        }
        CHECK(corrupted_detected);
    }

    std::remove(path.c_str());
}

TEST(file_too_small) {
    std::vector<uint8_t> tiny = {1, 2, 3};
    std::string path = "/tmp/test_sp_tiny.spz";
    write_file(path, tiny);
    CHECK_THROWS(file_info(path));
    std::remove(path.c_str());
}

TEST(file_bad_magic) {
    std::vector<uint8_t> bad(128, 0);
    bad[0] = 0xFF;  // bad magic
    std::string path = "/tmp/test_sp_bad.spz";
    write_file(path, bad);
    CHECK_THROWS(file_info(path));
    std::remove(path.c_str());
}

// ============================================================================
// SECTION: inject_metadata
// ============================================================================
TEST(inject_metadata_preserves_data) {
    auto mat = make_random_int_csc(20, 10, 0.2, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    Metadata meta;
    meta.set_rownames({"R0","R1","R2","R3","R4","R5","R6","R7","R8","R9",
                       "R10","R11","R12","R13","R14","R15","R16","R17","R18","R19"});
    inject_metadata(blob, meta);

    Metadata meta2;
    auto mat2 = decompress(blob, {}, &meta2);
    CHECK(matrices_equal(mat, mat2));
    CHECK(meta2.has_rownames());
    CHECK_EQ(meta2.get_rownames()[0], "R0");
    CHECK_EQ(meta2.get_rownames()[19], "R19");
}

TEST(inject_metadata_twice) {
    auto mat = make_random_int_csc(5, 3, 0.5, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    Metadata meta1;
    meta1.set_rownames({"A", "B", "C", "D", "E"});
    inject_metadata(blob, meta1);

    Metadata meta2;
    meta2.set_rownames({"X", "Y", "Z", "W", "V"});
    inject_metadata(blob, meta2);

    Metadata meta_out;
    decompress(blob, {}, &meta_out);
    CHECK_EQ(meta_out.get_rownames()[0], "X");  // second injection wins
}

// ============================================================================
// SECTION: Compression statistics
// ============================================================================
TEST(compress_stats) {
    auto mat = make_random_int_csc(500, 200, 0.05, 42);
    CompressConfig cfg;
    CompressStats stats;
    auto blob = compress(mat, cfg, &stats);

    CHECK(stats.raw_size > 0);
    CHECK(stats.compressed_size > 0);
    CHECK(stats.num_chunks > 0);
    CHECK(stats.raw_size >= stats.compressed_size);  // should compress
}

TEST(compression_reduces_size) {
    // Compressed should be smaller than raw for typical matrices
    auto mat = make_random_int_csc(100, 100, 0.1, 42);
    CompressConfig cfg;
    CompressStats stats;
    compress(mat, cfg, &stats);
    CHECK(stats.compressed_size < stats.raw_size);
    CHECK(stats.num_chunks > 0u);
}

// ============================================================================
// SECTION: Chunk configuration
// ============================================================================
TEST(custom_chunk_size) {
    auto mat = make_random_int_csc(100, 100, 0.1, 42);

    CompressConfig cfg;
    cfg.chunk_cols = 10;
    CompressStats stats;
    auto blob = compress(mat, cfg, &stats);
    CHECK_EQ(stats.num_chunks, 10u);

    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(chunk_size_1) {
    // Each column is its own chunk
    auto mat = make_random_int_csc(10, 5, 0.3, 42);
    CompressConfig cfg;
    cfg.chunk_cols = 1;
    CompressStats stats;
    auto blob = compress(mat, cfg, &stats);
    CHECK_EQ(stats.num_chunks, 5u);

    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

TEST(chunk_larger_than_cols) {
    auto mat = make_random_int_csc(10, 5, 0.3, 42);
    CompressConfig cfg;
    cfg.chunk_cols = 10000;
    CompressStats stats;
    auto blob = compress(mat, cfg, &stats);
    CHECK_EQ(stats.num_chunks, 1u);

    auto mat2 = decompress(blob);
    CHECK(matrices_equal(mat, mat2));
}

// ============================================================================
// SECTION: Read real .spz files (if path provided)
// NOTE: Production .spz files use sparsepress_v2 format (128-byte header, rANS encoding)
// which is incompatible with the new singlepress format (64-byte header, delta+varint).
// These tests only work with files written by the new singlepress format.
// ============================================================================
TEST(read_real_spz) {
    if (g_real_spz_path.empty()) return;

    // Detect old format: version=2 at uint16 offset 4, header_size=128 at uint16 offset 6
    auto raw = read_file(g_real_spz_path);
    if (raw.size() >= 8) {
        uint16_t version, hdr_size;
        std::memcpy(&version, raw.data() + 4, 2);
        std::memcpy(&hdr_size, raw.data() + 6, 2);
        if (version == 2 && hdr_size == 128) {
            std::cout << "\n    SKIP: " << g_real_spz_path
                      << " uses old sparsepress_v2 format (incompatible) ";
            return;
        }
    }

    auto info = file_info(g_real_spz_path);
    CHECK(info.crc32_valid);
    CHECK(info.rows > 0);
    CHECK(info.cols > 0);
    CHECK(info.nnz > 0);

    std::cout << "\n    Real file: " << g_real_spz_path
              << " (" << info.rows << "×" << info.cols
              << ", nnz=" << info.nnz
              << ", ratio=" << info.ratio << ") ";
}

TEST(roundtrip_real_spz) {
    if (g_real_spz_path.empty()) return;

    auto data = read_file(g_real_spz_path);

    // Skip old format files
    if (data.size() >= 8) {
        uint16_t version, hdr_size;
        std::memcpy(&version, data.data() + 4, 2);
        std::memcpy(&hdr_size, data.data() + 6, 2);
        if (version == 2 && hdr_size == 128) return;
    }

    DecompressConfig dcfg;
    dcfg.reorder = true;
    Metadata meta;
    auto mat = decompress(data.data(), data.size(), dcfg, &meta);

    CHECK(mat.m > 0);
    CHECK(mat.n > 0);
    CHECK(mat.nnz > 0);

    // Re-compress and verify
    CompressConfig cfg;
    auto blob2 = compress(mat, cfg);
    auto mat2 = decompress(blob2);

    CHECK(matrices_equal(mat, mat2));

    std::cout << "\n    Roundtrip: " << mat.m << "×" << mat.n
              << " nnz=" << mat.nnz << " ";
}

TEST(column_subset_real_spz) {
    if (g_real_spz_path.empty()) return;

    auto data = read_file(g_real_spz_path);

    // Skip old format files
    if (data.size() >= 8) {
        uint16_t version, hdr_size;
        std::memcpy(&version, data.data() + 4, 2);
        std::memcpy(&hdr_size, data.data() + 6, 2);
        if (version == 2 && hdr_size == 128) return;
    }

    auto info = file_info(g_real_spz_path);

    // Read first 100 columns
    uint32_t end = std::min(100u, info.cols);
    auto sub = decompress_columns(data.data(), data.size(), 0, end);
    CHECK_EQ(sub.n, end);
    CHECK_EQ(sub.m, info.rows);
}

// ============================================================================
// SECTION: Performance / stress
// ============================================================================
TEST(performance_compress_1m_nnz) {
    // ~1M nonzeros
    auto mat = make_random_int_csc(10000, 5000, 0.02, 42);
    CHECK(mat.nnz > 500000);

    CompressConfig cfg;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto blob = compress(mat, cfg);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ratio = static_cast<double>(
        12 + (mat.n + 1) * 4 + mat.nnz * 4 + mat.nnz
    ) / blob.size();

    std::cout << "\n    Compress " << mat.nnz << " nnz: "
              << ms << " ms, ratio=" << ratio << "x ";
}

TEST(performance_decompress_1m_nnz) {
    auto mat = make_random_int_csc(10000, 5000, 0.02, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto mat2 = decompress(blob);
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "\n    Decompress " << mat.nnz << " nnz: " << ms << " ms ";
    CHECK(matrices_equal(mat, mat2));
}

// ============================================================================
// SECTION: Header structure
// ============================================================================
TEST(header_size) {
    CHECK_EQ(HEADER_SIZE, 64u);
    CHECK_EQ(FOOTER_SIZE, 16u);
    // FileHeader struct is 56 bytes (no explicit padding), serialized to 64 on disk
    CHECK(sizeof(FileHeader) <= HEADER_SIZE);

    // Verify serialize produces exactly HEADER_SIZE bytes
    FileHeader hdr{};
    auto bytes = hdr.serialize();
    CHECK_EQ(bytes.size(), static_cast<size_t>(HEADER_SIZE));

    CHECK_EQ(sizeof(FileFooter), FOOTER_SIZE);
}

TEST(header_fields) {
    auto mat = make_random_int_csc(100, 50, 0.1, 42);
    CompressConfig cfg;
    auto blob = compress(mat, cfg);

    FileHeader hdr = FileHeader::deserialize(blob.data());
    CHECK(hdr.magic == MAGIC || hdr.magic == MAGIC_LEGACY);
    CHECK_EQ(hdr.version, FORMAT_VERSION);
    CHECK_EQ(hdr.m, 100u);
    CHECK_EQ(hdr.n, 50u);
    CHECK(hdr.nnz > 0);
    CHECK(hdr.density > 0.0 && hdr.density < 1.0);
    CHECK(hdr.num_chunks > 0);
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[]) {
    if (argc > 1) {
        g_real_spz_path = argv[1];
        std::cout << "Using real .spz file: " << g_real_spz_path << "\n";
    }

    std::cout << "\n======================================\n";
    std::cout << "SinglePress C++ Test Results\n";
    std::cout << "======================================\n";
    std::cout << "Total:  " << g_tests_run << "\n";
    std::cout << "Passed: " << g_tests_passed << "\n";
    std::cout << "Failed: " << g_tests_failed << "\n";
    std::cout << "======================================\n";

    return g_tests_failed > 0 ? 1 : 0;
}
