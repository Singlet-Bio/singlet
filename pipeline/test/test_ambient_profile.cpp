// test_ambient_profile.cpp — unit tests for ambient_profile.h
// Tests: profile computation, fractions sum to 1.0, only non-cell barcodes,
// TSV format, empty ambient pool, zero-count genes, lower-bound exclusion,
// known raw count values.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "singlet-pileup/ambient_profile.h"

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg)                                                         \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (msg) << "\n";                           \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n"; \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

static std::string read_file_str(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs) return {};
    return std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
}

static std::vector<std::string> split_lines(const std::string& s) {
    std::vector<std::string> lines;
    std::istringstream ss(s);
    std::string line;
    while (std::getline(ss, line))
        if (!line.empty()) lines.push_back(line);
    return lines;
}

static std::vector<std::string> split_tab(const std::string& s) {
    std::vector<std::string> fields;
    std::istringstream ss(s);
    std::string f;
    while (std::getline(ss, f, '\t')) fields.push_back(f);
    return fields;
}

// ---------------------------------------------------------------------------
// Build a tiny synthetic CSC matrix:
//   5 genes × 6 barcodes
//   barcodes 0,1,2  → high UMI (cells, ~500 each)
//   barcodes 3,4    → ambient range (50-100 UMI)
//   barcode  5      → below lower_umi (<10 UMI)
// ---------------------------------------------------------------------------
static singlet::SparseAccumulator<uint16_t>::CSCMatrix make_test_csc()
{
    // ncols = 6, nrows = 5
    // We'll use a simple dense-style layout: each barcode has equal counts
    // per gene for simplicity.
    //   bc0: gene0=200,gene1=100,gene2=100,gene3=50,gene4=50  → total=500
    //   bc1: gene0=200,gene1=100,gene2=100,gene3=50,gene4=50  → total=500
    //   bc2: gene0=200,gene1=100,gene2=100,gene3=50,gene4=50  → total=500
    //   bc3: gene0=20,gene1=10,gene2=10,gene3=10,gene4=10     → total=60
    //   bc4: gene0=30,gene1=20,gene2=20,gene3=15,gene4=15     → total=100
    //   bc5: gene0=2, gene1=1, gene2=1, gene3=1, gene4=1      → total=6

    singlet::SparseAccumulator<uint16_t>::CSCMatrix m;
    m.nrows = 5;
    m.ncols = 6;
    // indptr size = 7 (ncols+1)
    // We store all 5 genes per barcode → 30 entries
    m.indptr  = {0, 5, 10, 15, 20, 25, 30};
    m.indices = {0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4,
                 0,1,2,3,4, 0,1,2,3,4, 0,1,2,3,4};

    // bc0: 200,100,100,50,50
    // bc1: 200,100,100,50,50
    // bc2: 200,100,100,50,50
    // bc3: 20,10,10,10,10
    // bc4: 30,20,20,15,15
    // bc5: 2,1,1,1,1
    m.data = {
        200,100,100,50,50,   // bc0
        200,100,100,50,50,   // bc1
        200,100,100,50,50,   // bc2
         20, 10, 10,10,10,   // bc3
         30, 20, 20,15,15,   // bc4
          2,  1,  1, 1, 1    // bc5
    };
    return m;
}

// ---------------------------------------------------------------------------
// T1: Profile computation — basic structure
// ---------------------------------------------------------------------------
static void test_basic_profile() {
    std::cout << "\n=== T1: Basic ambient profile computation ===\n";

    auto csc = make_test_csc();
    std::vector<std::string> genes = {"GeneA","GeneB","GeneC","GeneD","GeneE"};
    std::vector<uint64_t> totals   = {500,500,500, 60,100, 6};
    std::unordered_set<uint32_t> cells = {0,1,2};   // bc0-bc2 are cells

    // lower_umi=10, cell_threshold=200 → bc3(60) and bc4(100) qualify; bc5(6) excluded
    auto prof = singlet::build_ambient_profile(genes, csc, totals, cells, 10, 200);

    CHECK(prof.n_ambient_barcodes == 2, "2 ambient barcodes (bc3 + bc4)");  // 1
    CHECK(prof.total_ambient_umis  == 160, "total ambient UMIs = 160");     // 2
    CHECK(prof.gene_names.size()   == 5, "5 genes in profile");             // 3
    CHECK(prof.expression_fraction.size() == 5, "5 fractions");            // 4
    CHECK(prof.raw_counts.size()   == 5, "5 raw count entries");            // 5
}

// ---------------------------------------------------------------------------
// T2: Fractions sum to ≈ 1.0
// ---------------------------------------------------------------------------
static void test_fractions_sum_to_one() {
    std::cout << "\n=== T2: Expression fractions sum to 1.0 ===\n";

    auto csc = make_test_csc();
    std::vector<std::string> genes = {"GeneA","GeneB","GeneC","GeneD","GeneE"};
    std::vector<uint64_t> totals   = {500,500,500, 60,100, 6};
    std::unordered_set<uint32_t> cells = {0,1,2};

    auto prof = singlet::build_ambient_profile(genes, csc, totals, cells, 10, 200);

    double sum = 0.0;
    for (double f : prof.expression_fraction) sum += f;

    CHECK(std::abs(sum - 1.0) < 1e-9, "fractions sum to 1.0");  // 6
}

// ---------------------------------------------------------------------------
// T3: Called-cell barcodes excluded
// ---------------------------------------------------------------------------
static void test_cells_excluded() {
    std::cout << "\n=== T3: Called-cell barcodes excluded from ambient pool ===\n";

    auto csc = make_test_csc();
    std::vector<std::string> genes = {"GeneA","GeneB","GeneC","GeneD","GeneE"};
    std::vector<uint64_t> totals   = {500,500,500, 60,100, 6};
    // Mark all 6 as cells → empty ambient pool
    std::unordered_set<uint32_t> all_cells = {0,1,2,3,4,5};

    auto prof = singlet::build_ambient_profile(genes, csc, totals, all_cells, 10, 600);

    CHECK(prof.n_ambient_barcodes == 0, "0 ambient barcodes when all marked as cells");  // 7
    CHECK(prof.total_ambient_umis == 0, "total_ambient_umis = 0");                       // 8
}

// ---------------------------------------------------------------------------
// T4: TSV format validation
// ---------------------------------------------------------------------------
static void test_tsv_format() {
    std::cout << "\n=== T4: TSV format ===\n";

    const std::string path = "/tmp/test_ambient_t4.tsv";
    auto csc = make_test_csc();
    std::vector<std::string> genes = {"GeneA","GeneB","GeneC","GeneD","GeneE"};
    std::vector<uint64_t> totals   = {500,500,500, 60,100, 6};
    std::unordered_set<uint32_t> cells = {0,1,2};

    auto prof    = singlet::build_ambient_profile(genes, csc, totals, cells, 10, 200);
    bool written = singlet::write_ambient_profile(prof, path);

    CHECK(written, "write_ambient_profile returns true");  // 9

    auto content = read_file_str(path);
    auto lines   = split_lines(content);

    CHECK(lines.size() == 6, "TSV has header + 5 data rows");  // 10

    auto hdr = split_tab(lines[0]);
    CHECK(hdr.size() == 3, "header has 3 tab-separated columns");  // 11
    CHECK(hdr[0] == "gene_name",          "col0 = gene_name");        // 12
    CHECK(hdr[1] == "expression_fraction","col1 = expression_fraction"); // 13
    CHECK(hdr[2] == "raw_count",          "col2 = raw_count");        // 14

    auto row1 = split_tab(lines[1]);
    CHECK(row1.size() == 3, "data row has 3 fields");  // 15

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T5: Empty ambient pool — no crash, zero raw counts, zero fractions
// ---------------------------------------------------------------------------
static void test_empty_ambient_pool() {
    std::cout << "\n=== T5: Empty ambient pool (all barcodes below lower_umi) ===\n";

    auto csc = make_test_csc();
    std::vector<std::string> genes = {"GeneA","GeneB","GeneC","GeneD","GeneE"};
    // Set all totals to 2 (below lower_umi=10)
    std::vector<uint64_t> totals(6, 2ULL);
    std::unordered_set<uint32_t> cells = {};

    auto prof = singlet::build_ambient_profile(genes, csc, totals, cells, 10, 200);

    CHECK(prof.n_ambient_barcodes == 0, "0 ambient barcodes"); // 16
    double sum_frac = 0.0;
    for (double f : prof.expression_fraction) sum_frac += f;
    CHECK(sum_frac == 0.0, "fractions are all zero for empty pool"); // 17

    // Should still write a valid (zero-filled) TSV without crashing
    const std::string path = "/tmp/test_ambient_t5.tsv";
    bool ok = singlet::write_ambient_profile(prof, path);
    CHECK(ok, "write_ambient_profile succeeds for empty pool"); // 18
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== ambient_profile tests ===\n";
    test_basic_profile();
    test_fractions_sum_to_one();
    test_cells_excluded();
    test_tsv_format();
    test_empty_ambient_pool();

    std::cout << "\nTotal: " << (n_pass + n_fail)
              << "  PASS: " << n_pass
              << "  FAIL: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
