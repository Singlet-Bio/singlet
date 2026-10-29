// Track B robustness self-test: synthetic and edge-case matrices through every Track B codec (decode(encode(M)) == M on all
// fields). Covers empty matrices, single entries, fewer columns than lanes, huge columns (> 8192 nnz, exercising the sorter
// fallback), ties in row counts, big values, row ids at the format limits, and the documented unsupported cases (which must
// fail loudly with an exception, never silently).
//   bin/track_b_selftest.exe [seed]
#include "common.h"
#include "track_b.h"
#include <algorithm>
#include <random>

using namespace cs;

struct Case {
    std::string name;
    CanonicalMatrix M;
    bool expect_supported = true;
};

static CanonicalMatrix build(uint32_t m, uint32_t n, const std::vector<std::vector<std::pair<uint32_t, uint32_t>>>& cols) {
    CanonicalMatrix M;
    M.m = m; M.n = n; M.vt_code = 5;
    M.indptr.assign(n + 1, 0);
    for (uint32_t j = 0; j < n; ++j) {
        auto c = j < cols.size() ? cols[j] : std::vector<std::pair<uint32_t, uint32_t>>();
        std::sort(c.begin(), c.end());
        for (auto& [r, v] : c) { M.indices.push_back(r); M.data.push_back(v); }
        M.indptr[j + 1] = static_cast<uint32_t>(M.indices.size());
    }
    M.nnz = M.indices.size();
    for (uint32_t i = 0; i < m; ++i) M.rownames.push_back("ENSG" + std::to_string(i) + "_G" + std::to_string(i % 977) + "_chr1:" + std::to_string(i * 7) + "-" + std::to_string(i * 7 + 5));
    for (uint32_t j = 0; j < n; ++j) M.colnames.push_back("AAAC" + std::to_string(j) + "-1");
    M.user_kv["k"] = "v";
    return M;
}

// random matrix: column j has k_j distinct rows drawn with a power-law row preference
static CanonicalMatrix random_matrix(std::mt19937_64& rng, uint32_t m, uint32_t n, double mean_k, double p_big, uint32_t max_val, uint32_t forced_big_col = ~0u, uint32_t big_k = 0) {
    std::vector<std::vector<std::pair<uint32_t, uint32_t>>> cols(n);
    std::exponential_distribution<double> ek(1.0 / std::max(mean_k, 0.01));
    std::uniform_real_distribution<double> u(0, 1);
    for (uint32_t j = 0; j < n; ++j) {
        uint32_t k = static_cast<uint32_t>(ek(rng));
        if (j == forced_big_col) k = big_k;
        k = std::min(k, m);
        std::vector<uint8_t> seen(m, 0);
        for (uint32_t t = 0; t < k; ++t) {
            uint32_t r = static_cast<uint32_t>(std::pow(u(rng), 3.0) * m);
            if (r >= m) r = m - 1;
            for (int tries = 0; tries < 20 && seen[r]; ++tries) r = (r + 1 + static_cast<uint32_t>(rng() % 7)) % m;
            if (seen[r]) continue;
            seen[r] = 1;
            uint32_t v = 1;
            if (u(rng) < p_big) v = 2 + static_cast<uint32_t>(std::pow(u(rng), 4.0) * (max_val - 2));
            cols[j].push_back({r, v});
        }
    }
    return build(m, n, cols);
}

int main(int argc, char** argv) {
    uint64_t seed = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 12345;
    std::mt19937_64 rng(seed);
    std::vector<Case> cases;
    cases.push_back({"empty_matrix_5x100", build(100, 5, {})});
    cases.push_back({"zero_columns", build(100, 0, {})});
    cases.push_back({"zero_rows_zero_cols", build(0, 0, {})});
    cases.push_back({"single_entry", build(1000, 3, {{}, {{500, 7}}, {}})});
    cases.push_back({"one_column_dense_head", build(64, 1, {[] { std::vector<std::pair<uint32_t, uint32_t>> c; for (uint32_t i = 0; i < 64; ++i) c.push_back({i, 1 + i % 5}); return c; }()})});
    cases.push_back({"fewer_cols_than_lanes", random_matrix(rng, 5000, 7, 300, 0.3, 40)});
    cases.push_back({"tiny_m_ties", random_matrix(rng, 40, 300, 12, 0.2, 10)});
    cases.push_back({"all_columns_k1", [&] { std::vector<std::vector<std::pair<uint32_t, uint32_t>>> c(500); for (uint32_t j = 0; j < 500; ++j) c[j].push_back({static_cast<uint32_t>(rng() % 3000), 1u}); return build(3000, 500, c); }()});
    cases.push_back({"mixed_sparse", random_matrix(rng, 20000, 3000, 25, 0.25, 500)});
    cases.push_back({"chunk_boundary_1025_cols", random_matrix(rng, 4000, 1025, 20, 0.2, 60)});
    cases.push_back({"huge_column_20000", random_matrix(rng, 60000, 200, 40, 0.15, 90, 77, 20000)});
    cases.push_back({"column_8192_8193", [&] {
        auto M = random_matrix(rng, 30000, 4, 5, 0.1, 30, 1, 8192);
        return M;
    }()});
    for (uint32_t kk : {8191u, 8192u, 8193u, 16000u}) {   // exactly around the sorter's 8192-element limit (column fully dense)
        std::vector<std::vector<std::pair<uint32_t, uint32_t>>> c(3);
        for (uint32_t i = 0; i < kk; ++i) c[1].push_back({(i * 7919u) % 16001u, 1 + i % 3});   // 7919 prime: distinct rows < 16001
        c[0].push_back({5, 2});
        c[2].push_back({16000, 1});
        cases.push_back({"dense_column_" + std::to_string(kk), build(16001, 3, c)});
    }
    cases.push_back({"values_up_to_2e5", random_matrix(rng, 5000, 400, 60, 0.4, 200000)});
    cases.push_back({"m_524287", random_matrix(rng, 524287, 120, 80, 0.2, 100)});
    {   // unsupported by the lane codec: m > 2^19, or value-2 >= 2^19
        Case c{"m_600000_unsupported_by_lane_codec", random_matrix(rng, 600000, 60, 50, 0.2, 100), false};
        cases.push_back(std::move(c));
        Case d{"value_1e6_unsupported_by_lane_codec", build(2000, 3, {{{5, 1000000}}, {{7, 3}, {9, 1}}, {}}), false};
        cases.push_back(std::move(d));
    }

    std::vector<Codec> codecs;
    register_track_b(codecs);
    int fails = 0, runs = 0, unsupported = 0;
    for (auto& cs_ : cases) {
        std::printf("case %-40s m=%u n=%u nnz=%llu\n", cs_.name.c_str(), cs_.M.m, cs_.M.n, (unsigned long long)cs_.M.nnz);
        for (auto& c : codecs) {
            if (c.name != "b_rans16" && c.name != "b_rans16_scalar" && c.name != "b_leb_raw" && c.name != "b_leb_z3" && c.name != "b_rans16_c256" && c.name != "b_rans16_c4096" && c.name.rfind("b_rans16_k", 0) != 0) continue;
            ++runs;
            try {
                auto enc = c.encode(cs_.M);
                auto dec = c.decode(enc);
                bool ok = (dec == cs_.M);
                if (!ok) { ++fails; std::printf("   FAIL  %-22s roundtrip mismatch\n", c.name.c_str()); }
                else if (!cs_.expect_supported && c.name.rfind("b_rans16", 0) == 0) { std::printf("   (note) %-22s accepted a case documented as unsupported\n", c.name.c_str()); }
            } catch (const std::exception& e) {
                if (cs_.expect_supported && !(c.name.rfind("b_rans16", 0) != 0 && false)) {
                    ++fails;
                    std::printf("   FAIL  %-22s threw: %s\n", c.name.c_str(), e.what());
                } else {
                    ++unsupported;
                    std::printf("   ok    %-22s rejected as documented: %s\n", c.name.c_str(), e.what());
                }
            }
        }
    }
    std::printf("selftest: %d codec runs, %d failures, %d documented rejections\n", runs, fails, unsupported);
    return fails ? 1 : 0;
}
