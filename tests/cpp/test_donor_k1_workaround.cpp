// test/test_donor_k1_workaround.cpp — B-G5-3 unit test
// Verifies that make_single_donor_result() produces a valid DemuxResult for K=1,
// and that write_donor_assignments() emits all cells as donor0 with the correct schema.
//
// Synthetic 100-cell test: no real data or files required.

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "singlet/pileup/donor_demux.h"

using namespace singlet;
namespace fs = std::filesystem;

// ─── Minimal test framework ─────────────────────────────────────────────────
static int n_pass = 0, n_fail = 0;
#define CHECK(cond, msg) \
    do { \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++n_pass; } \
        else      { std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; ++n_fail; } \
    } while(0)

// ─── Synthetic SNP/cell data ─────────────────────────────────────────────────
// Generate minimal CSC AD/DP matrices:
//   n_cells x n_snps, each cell covers 5 random SNPs with random AD/DP.
struct SyntheticSNPData {
    int n_snps, n_cells;
    std::vector<int32_t> ad_indptr, ad_indices, dp_indptr, dp_indices;
    std::vector<uint8_t> ad_data,  dp_data;
};

static SyntheticSNPData make_synthetic(int n_cells, int n_snps, uint64_t seed = 42) {
    SyntheticSNPData d;
    d.n_cells = n_cells;
    d.n_snps  = n_snps;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> snp_dist(0, n_snps - 1);
    std::uniform_int_distribution<uint8_t> dp_dist(1, 30);
    std::uniform_int_distribution<uint8_t> ad_frac(0, 1);

    d.ad_indptr.push_back(0);
    d.dp_indptr.push_back(0);

    for (int c = 0; c < n_cells; ++c) {
        // 5 unique random SNPs per cell
        std::vector<int32_t> snps;
        while ((int)snps.size() < 5) {
            int s = snp_dist(rng);
            bool dup = false;
            for (auto x : snps) if (x == s) { dup = true; break; }
            if (!dup) snps.push_back(s);
        }
        std::sort(snps.begin(), snps.end());
        for (auto s : snps) {
            uint8_t dp = dp_dist(rng);
            uint8_t ad = static_cast<uint8_t>(dp * ad_frac(rng));
            d.dp_indices.push_back(s);
            d.dp_data.push_back(dp);
            if (ad > 0) {
                d.ad_indices.push_back(s);
                d.ad_data.push_back(ad);
            }
        }
        d.dp_indptr.push_back(static_cast<int32_t>(d.dp_indices.size()));
        d.ad_indptr.push_back(static_cast<int32_t>(d.ad_indices.size()));
    }
    return d;
}

// ─── Tests ──────────────────────────────────────────────────────────────────

void test_k1_all_cells_donor0() {
    auto d = make_synthetic(100, 50);
    auto result = make_single_donor_result(
        static_cast<uint32_t>(d.n_snps), static_cast<uint32_t>(d.n_cells),
        d.ad_indptr.data(), d.ad_indices.data(), d.ad_data.data(),
        d.dp_indptr.data(), d.dp_indices.data(), d.dp_data.data());

    CHECK(result.n_donors_k == 1, "n_donors_k == 1");
    CHECK(result.assignments.size() == 100u, "assignments.size() == 100");
    bool all_donor0 = true;
    bool all_prob1  = true;
    for (const auto& a : result.assignments) {
        if (a.donor_id != 0 || a.label != "donor0") all_donor0 = false;
        if (a.prob_max != 1.0f || a.prob_doublet != 0.0f) all_prob1 = false;
    }
    CHECK(all_donor0, "all cells assigned to donor0");
    CHECK(all_prob1,  "prob_max=1.0 and prob_doublet=0.0 for all cells");
    CHECK(!result.covered_to_original.empty(), "covered_to_original populated");
}

void test_k1_write_donor_assignments() {
    auto d = make_synthetic(100, 50, 123);
    auto result = make_single_donor_result(
        static_cast<uint32_t>(d.n_snps), static_cast<uint32_t>(d.n_cells),
        d.ad_indptr.data(), d.ad_indices.data(), d.ad_data.data(),
        d.dp_indptr.data(), d.dp_indices.data(), d.dp_data.data());

    // Build synthetic barcode list
    std::vector<std::string> barcodes;
    for (int i = 0; i < d.n_cells; ++i)
        barcodes.push_back("CELL" + std::to_string(i) + "-1");

    // Write to a temp file
    std::string tmpfile = "/tmp/test_donor_assignments_k1.tsv";
    bool ok = write_donor_assignments(tmpfile, result.assignments, barcodes);
    CHECK(ok, "write_donor_assignments returns true");

    // Read and validate
    std::ifstream f(tmpfile);
    CHECK(f.is_open(), "donor_assignments.tsv opened for reading");

    std::string header_line;
    std::getline(f, header_line);
    CHECK(header_line.find("cell") != std::string::npos &&
          header_line.find("donor_id") != std::string::npos,
          "header contains 'cell' and 'donor_id' columns");

    int n_rows = 0;
    bool all_donor0_tsv = true;
    std::string line;
    while (std::getline(f, line)) {
        ++n_rows;
        if (line.find("donor0") == std::string::npos) all_donor0_tsv = false;
    }
    CHECK(n_rows == 100, "donor_assignments.tsv has 100 data rows");
    CHECK(all_donor0_tsv, "all rows contain 'donor0'");

    std::remove(tmpfile.c_str());
}

void test_k1_snp_ad_dp_independence() {
    // snp_ad / snp_dp matrices are produced independently of demux;
    // verify covered_to_original correctly indexes into the n_snps SNP space.
    auto d = make_synthetic(50, 20, 999);
    auto result = make_single_donor_result(
        static_cast<uint32_t>(d.n_snps), static_cast<uint32_t>(d.n_cells),
        d.ad_indptr.data(), d.ad_indices.data(), d.ad_data.data(),
        d.dp_indptr.data(), d.dp_indices.data(), d.dp_data.data());

    bool valid_indices = true;
    for (int32_t orig : result.covered_to_original)
        if (orig < 0 || orig >= d.n_snps) valid_indices = false;
    CHECK(valid_indices, "covered_to_original indices in [0, n_snps)");
}

// ─── Main ────────────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== B-G5-3 donor K=1 workaround unit tests ===\n";
    test_k1_all_cells_donor0();
    test_k1_write_donor_assignments();
    test_k1_snp_ad_dp_independence();

    std::cout << "\n" << n_pass << " passed, " << n_fail << " failed\n";
    return n_fail > 0 ? 1 : 0;
}
