// test_pz_reader.cpp — round-trip test for pz_writer.h ↔ pz_reader.h
//
// CPP-1: verifies header parsing, CRC32 validation, permutation decode,
//        column-count ptr decode, metadata TLV decode, and nnz consistency.
// CPP-2: adds full CSC data round-trip (values, indices, per-column layout).

#include "singlet/pileup/pz_reader.h"
#include "singlet/pileup/pz_writer.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace pz = singlet::pz;

// Tiny CSC builder for test matrices.
//
// Input: (rows, cols, list of (row, col, value)) triples.
// Output: int32 indptr/indices and uint16 data ready for write_1pz<uint16_t>.
struct TinyCSC {
    uint32_t m, n;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
};

static TinyCSC build_csc(
    uint32_t m, uint32_t n,
    const std::vector<std::tuple<int32_t, int32_t, uint16_t>>& triples)
{
    // Bucket triples by column
    std::vector<std::vector<std::pair<int32_t, uint16_t>>> cols(n);
    for (auto& t : triples) {
        int32_t r = std::get<0>(t);
        int32_t c = std::get<1>(t);
        uint16_t v = std::get<2>(t);
        cols[c].push_back({r, v});
    }
    TinyCSC out;
    out.m = m;
    out.n = n;
    out.indptr.push_back(0);
    for (uint32_t j = 0; j < n; ++j) {
        auto& col = cols[j];
        std::sort(col.begin(), col.end(),
                  [](auto& a, auto& b){ return a.first < b.first; });
        for (auto& p : col) {
            out.indices.push_back(p.first);
            out.data.push_back(p.second);
        }
        out.indptr.push_back(static_cast<int32_t>(out.indices.size()));
    }
    return out;
}

static void test_roundtrip_small_uint16() {
    // 5x3 matrix
    //   col 0:  row 0 -> 7, row 3 -> 2
    //   col 1:  row 1 -> 5
    //   col 2:  row 0 -> 1, row 2 -> 9, row 4 -> 3
    auto csc = build_csc(5, 3, {
        {0, 0, 7}, {3, 0, 2},
        {1, 1, 5},
        {0, 2, 1}, {2, 2, 9}, {4, 2, 3},
    });
    assert(csc.data.size() == 6);

    std::vector<std::string> rownames = {
        "ENSG00000000003", "ENSG00000000005", "ENSG00000000419",
        "ENSG00000000457", "ENSG00000000460"};
    std::vector<std::string> colnames = {"AACGTAGCT-1", "AACCTTGCT-1", "CCGTAGCTA-1"};
    std::map<std::string, std::string> user_kv = {
        {"gsm_id", "GSM_TEST_0001"},
        {"gse_id", "GSE_TEST_0001"},
        {"organism", "Homo sapiens"},
        {"protocol", "dropseq"},
        {"singlet_version", "0.3.0"},
        {"pipeline_date", "2026-04-13"},
    };

    std::string path = "/tmp/test_pz_reader_small.1pz";
    bool ok = pz::write_1pz<uint16_t>(
        path, csc.m, csc.n, csc.indptr, csc.indices, csc.data,
        rownames, colnames, 3, 1024, 1, user_kv);
    (void)ok;
    assert(ok && "write_1pz failed");

    pz::ReadResult r = pz::read_1pz(path);

    // Header fields
    assert(r.m == 5);
    assert(r.n == 3);
    assert(r.nnz == 6);

    // indptr reconstructed from column counts
    assert(r.indptr.size() == 4);
    assert(r.indptr[0] == 0);
    assert(r.indptr[1] == 2);     // col 0 has 2 nnz
    assert(r.indptr[2] == 3);     // col 1 has 1 nnz
    assert(r.indptr[3] == 6);     // col 2 has 3 nnz

    // Rownames + colnames round-trip
    assert(r.rownames.size() == rownames.size());
    for (size_t i = 0; i < rownames.size(); ++i) {
        assert(r.rownames[i] == rownames[i]);
    }
    assert(r.colnames.size() == colnames.size());
    for (size_t i = 0; i < colnames.size(); ++i) {
        assert(r.colnames[i] == colnames[i]);
    }

    // User KV round-trip
    assert(r.user_kv.size() == user_kv.size());
    for (auto& kv : user_kv) {
        auto it = r.user_kv.find(kv.first);
        (void)it;
        assert(it != r.user_kv.end());
        assert(it->second == kv.second);
    }

    // CPP-2: full CSC round-trip
    assert(r.indices.size() == 6);
    assert(r.data.size() == 6);

    // Rebuild "expected" CSC from the original triples, sorted per column by row
    std::vector<uint32_t> exp_idx = {
        0, 3,       // col 0: rows 0, 3
        1,          // col 1: row 1
        0, 2, 4,    // col 2: rows 0, 2, 4
    };
    std::vector<uint32_t> exp_val = {
        7, 2,       // col 0 values
        5,          // col 1 values
        1, 9, 3,    // col 2 values
    };
    for (size_t k = 0; k < exp_idx.size(); ++k) {
        if (r.indices[k] != exp_idx[k]) {
            std::fprintf(stderr,
                "indices[%zu] mismatch: got %u, expected %u\n",
                k, r.indices[k], exp_idx[k]);
            std::abort();
        }
        if (r.data[k] != exp_val[k]) {
            std::fprintf(stderr,
                "data[%zu] mismatch: got %u, expected %u\n",
                k, r.data[k], exp_val[k]);
            std::abort();
        }
    }

    std::printf("test_roundtrip_small_uint16: PASS "
                "(header+metadata+indptr+CSC data round-trip OK)\n");
    std::remove(path.c_str());
}

// Bigger test: 40x20 with ~100 non-zeros spanning multiple value groups and
// exercising both 10xv3-style small values and a few larger ones. This is the
// most important correctness test — it stresses the byte-split / bit-plane /
// bitmap / varint pipeline on a non-trivial matrix.
static void test_roundtrip_medium_uint16() {
    const uint32_t m = 40, n = 20;
    std::vector<std::tuple<int32_t, int32_t, uint16_t>> triples;

    // Deterministic sparse pattern: ~5 nnz per column, spread across rows,
    // values 1..8 cycling so each column has multiple value groups.
    for (uint32_t j = 0; j < n; ++j) {
        for (int k = 0; k < 5; ++k) {
            int32_t row = (j * 7 + k * 11) % m;
            uint16_t val = static_cast<uint16_t>(1 + ((j + k) % 8));
            triples.push_back({row, static_cast<int32_t>(j), val});
        }
    }

    // Deduplicate same-column duplicates by taking the first one
    std::sort(triples.begin(), triples.end(),
              [](auto& a, auto& b) {
                  if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
                  return std::get<0>(a) < std::get<0>(b);
              });
    std::vector<std::tuple<int32_t, int32_t, uint16_t>> dedup;
    for (auto& t : triples) {
        if (!dedup.empty() &&
            std::get<0>(dedup.back()) == std::get<0>(t) &&
            std::get<1>(dedup.back()) == std::get<1>(t)) continue;
        dedup.push_back(t);
    }

    auto csc = build_csc(m, n, dedup);
    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "G" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "C" + std::to_string(j);
    std::map<std::string, std::string> user_kv = {{"dataset", "med_test"}};

    std::string path = "/tmp/test_pz_reader_medium.1pz";
    bool ok = pz::write_1pz<uint16_t>(
        path, m, n, csc.indptr, csc.indices, csc.data,
        rownames, colnames, 3, 1024, 1, user_kv);
    (void)ok;
    assert(ok);

    pz::ReadResult r = pz::read_1pz(path);
    assert(r.m == m && r.n == n && r.nnz == csc.data.size());

    // Column-by-column: verify the sorted original pairs match.
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t start = csc.indptr[j];
        uint32_t end = csc.indptr[j + 1];
        uint32_t rstart = r.indptr[j];
        uint32_t rend = r.indptr[j + 1];
        if ((end - start) != (rend - rstart)) {
            std::fprintf(stderr, "col %u: length mismatch %u vs %u\n",
                         j, end - start, rend - rstart);
            std::abort();
        }
        for (uint32_t k = 0; k < end - start; ++k) {
            if (static_cast<int32_t>(r.indices[rstart + k]) != csc.indices[start + k]) {
                std::fprintf(stderr,
                    "col %u idx[%u] mismatch: got %u expected %d\n",
                    j, k, r.indices[rstart + k], csc.indices[start + k]);
                std::abort();
            }
            if (static_cast<uint16_t>(r.data[rstart + k]) != csc.data[start + k]) {
                std::fprintf(stderr,
                    "col %u data[%u] mismatch: got %u expected %u\n",
                    j, k, r.data[rstart + k], csc.data[start + k]);
                std::abort();
            }
        }
    }

    std::printf("test_roundtrip_medium_uint16: PASS (%u×%u, nnz=%zu)\n",
                m, n, csc.data.size());
    std::remove(path.c_str());
}

// uint8 coverage — smallest values, vt_code=1
static void test_roundtrip_uint8() {
    const uint32_t m = 20, n = 10;
    std::vector<int32_t> indptr = {0};
    std::vector<int32_t> indices;
    std::vector<uint8_t> data;

    // Each column has 3 non-zeros with values 1..255
    for (uint32_t j = 0; j < n; ++j) {
        for (int k = 0; k < 3; ++k) {
            indices.push_back(static_cast<int32_t>((j * 3 + k * 7) % m));
            data.push_back(static_cast<uint8_t>(1 + ((j * 37 + k * 13) % 255)));
        }
        // must be sorted within column for write_1pz
        auto start = indices.end() - 3;
        auto dstart = data.end() - 3;
        // simple insertion sort over 3 elements
        for (int a = 0; a < 3; ++a) {
            for (int b = a + 1; b < 3; ++b) {
                if (*(start + b) < *(start + a)) {
                    std::swap(*(start + a), *(start + b));
                    std::swap(*(dstart + a), *(dstart + b));
                }
            }
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    }

    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "G" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "C" + std::to_string(j);

    std::string path = "/tmp/test_pz_reader_u8.1pz";
    bool ok = pz::write_1pz<uint8_t>(
        path, m, n, indptr, indices, data,
        rownames, colnames, 3, 1024, 1, {});
    (void)ok;
    assert(ok);

    pz::ReadResult r = pz::read_1pz(path);
    assert(r.vt_code == 1 && "uint8 should be vt_code=1");
    assert(r.m == m && r.n == n);
    assert(r.nnz == data.size());
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t exp_len = indptr[j + 1] - indptr[j];
        uint32_t got_len = r.indptr[j + 1] - r.indptr[j];
        (void)got_len;
        assert(exp_len == got_len);
        for (uint32_t k = 0; k < exp_len; ++k) {
            assert(static_cast<int32_t>(r.indices[r.indptr[j] + k]) == indices[indptr[j] + k]);
            assert(static_cast<uint8_t>(r.data[r.indptr[j] + k]) == data[indptr[j] + k]);
        }
    }

    std::printf("test_roundtrip_uint8: PASS (vt_code=%u, m=%u n=%u nnz=%zu)\n",
                r.vt_code, m, n, data.size());
    std::remove(path.c_str());
}

// uint32 coverage — realistic SNP-depth-style values. vt_code=3.
//
// IMPORTANT: write_1pz's VOCSC encoder allocates a value-indexed table
// (val_rows[max_val+1]), so very large values cause runaway memory usage.
// Production usage (SNP allele depth, SNP total depth) tops out around a
// few thousand, so we test with that realistic range. Large-value support
// would require a redesign of vocsc_encode_chunk to bucket sparsely.
static void test_roundtrip_uint32() {
    const uint32_t m = 30, n = 12;
    std::vector<int32_t> indptr = {0};
    std::vector<int32_t> indices;
    std::vector<uint32_t> data;

    // SNP-depth-style values: 1..2000 (realistic for high-coverage samples).
    // Exercises vt_code=3 (uint32) and value-group grouping logic.
    auto col_push = [&](std::initializer_list<std::pair<int32_t, uint32_t>> items) {
        std::vector<std::pair<int32_t, uint32_t>> sorted(items);
        std::sort(sorted.begin(), sorted.end());
        for (auto& p : sorted) {
            indices.push_back(p.first);
            data.push_back(p.second);
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    };

    // To force vt_code=3 (uint32) we include ONE value >65535 (but modest
    // enough that val_rows allocation stays under a few MB). 100,000 is used.
    col_push({{0, 5}, {10, 100000}, {28, 1}});             // col 0 forces vt_code=3
    col_push({{3, 999}, {7, 999}});                        // col 1
    col_push({});                                           // col 2 empty
    col_push({{1, 1}, {5, 1}, {29, 1}});                   // col 3
    col_push({{0, 80000}});                                // col 4
    col_push({{2, 42}, {8, 42}, {15, 100}, {20, 42}});     // col 5 value-group
    col_push({});                                           // col 6 empty
    col_push({{11, 500}, {12, 501}, {13, 502}});           // col 7 dense block
    col_push({{0, 7}, {15, 8}, {29, 9}});                  // col 8
    col_push({{1, 100}, {28, 100}});                       // col 9
    col_push({{5, 90000}});                                // col 10
    col_push({{6, 12}, {7, 13}, {8, 14}, {9, 15}});        // col 11

    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "F" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "B" + std::to_string(j);

    std::string path = "/tmp/test_pz_reader_u32.1pz";
    bool ok = pz::write_1pz<uint32_t>(
        path, m, n, indptr, indices, data,
        rownames, colnames, 3, 1024, 1, {});
    (void)ok;
    assert(ok);

    pz::ReadResult r = pz::read_1pz(path);
    assert(r.vt_code == 3 && "uint32 should be vt_code=3");
    assert(r.m == m && r.n == n);
    assert(r.nnz == data.size());
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t exp_len = indptr[j + 1] - indptr[j];
        uint32_t got_len = r.indptr[j + 1] - r.indptr[j];
        if (exp_len != got_len) {
            std::fprintf(stderr, "col %u len %u vs %u\n", j, exp_len, got_len);
            std::abort();
        }
        for (uint32_t k = 0; k < exp_len; ++k) {
            if (static_cast<int32_t>(r.indices[r.indptr[j] + k]) != indices[indptr[j] + k] ||
                r.data[r.indptr[j] + k] != data[indptr[j] + k]) {
                std::fprintf(stderr,
                    "col %u row[%u]: got (%u,%u) expected (%d,%u)\n",
                    j, k, r.indices[r.indptr[j] + k], r.data[r.indptr[j] + k],
                    indices[indptr[j] + k], data[indptr[j] + k]);
                std::abort();
            }
        }
    }

    std::printf("test_roundtrip_uint32: PASS (vt_code=%u, empty+dense+large-vals)\n",
                r.vt_code);
    std::remove(path.c_str());
}

// Multi-chunk: write a matrix wider than chunk_cols=256 so the chunk table
// has multiple blobs. Tests that the chunk_cursor and gap-stream resets
// correctly across chunk boundaries.
static void test_roundtrip_multi_chunk() {
    const uint32_t m = 50, n = 600;
    const int CHUNK_COLS = 128;  // force 5 chunks (ceil(600/128))

    std::vector<int32_t> indptr = {0};
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;

    // Per-column: deterministic scatter with 2-4 non-zeros, values cycling 1..20
    for (uint32_t j = 0; j < n; ++j) {
        std::vector<std::pair<int32_t, uint16_t>> col;
        int nnz_j = 2 + static_cast<int>((j * 13) % 3);
        for (int k = 0; k < nnz_j; ++k) {
            int32_t row = static_cast<int32_t>((j * 17 + k * 23) % m);
            uint16_t val = static_cast<uint16_t>(1 + ((j + k) % 20));
            col.push_back({row, val});
        }
        std::sort(col.begin(), col.end());
        // dedup by row
        col.erase(std::unique(col.begin(), col.end(),
                              [](auto& a, auto& b){ return a.first == b.first; }),
                  col.end());
        for (auto& p : col) {
            indices.push_back(p.first);
            data.push_back(p.second);
        }
        indptr.push_back(static_cast<int32_t>(indices.size()));
    }

    std::vector<std::string> rownames(m), colnames(n);
    for (uint32_t i = 0; i < m; ++i) rownames[i] = "r" + std::to_string(i);
    for (uint32_t j = 0; j < n; ++j) colnames[j] = "c" + std::to_string(j);

    std::string path = "/tmp/test_pz_reader_multi.1pz";
    bool ok = pz::write_1pz<uint16_t>(
        path, m, n, indptr, indices, data,
        rownames, colnames, 3, CHUNK_COLS, 1, {});
    (void)ok;
    assert(ok);

    pz::ReadResult r = pz::read_1pz(path);
    assert(r.m == m && r.n == n && r.nnz == data.size());

    for (uint32_t j = 0; j < n; ++j) {
        uint32_t exp_len = indptr[j + 1] - indptr[j];
        uint32_t got_len = r.indptr[j + 1] - r.indptr[j];
        (void)got_len;
        assert(exp_len == got_len);
        for (uint32_t k = 0; k < exp_len; ++k) {
            assert(static_cast<int32_t>(r.indices[r.indptr[j] + k]) == indices[indptr[j] + k]);
            assert(static_cast<uint16_t>(r.data[r.indptr[j] + k]) == data[indptr[j] + k]);
        }
    }

    const int expected_chunks = (600 + CHUNK_COLS - 1) / CHUNK_COLS;
    std::printf("test_roundtrip_multi_chunk: PASS (%u×%u, nnz=%zu, %d chunks)\n",
                m, n, data.size(), expected_chunks);
    std::remove(path.c_str());
}

int main() {
    try {
        test_roundtrip_small_uint16();
        test_roundtrip_medium_uint16();
        test_roundtrip_uint8();
        test_roundtrip_uint32();
        test_roundtrip_multi_chunk();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
    std::printf("pz_reader CPP-1+CPP-2+CPP-3 tests PASSED\n");
    return 0;
}
