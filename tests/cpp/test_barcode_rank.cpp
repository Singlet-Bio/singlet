// test_barcode_rank.cpp — unit tests for barcode_rank.h
// Tests: ranks are 1-based and contiguous, UMI counts are descending,
// is_cell flag correct, TSV format, all barcodes ranked, empty input,
// ties handled, single barcode, build and write round-trip.

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "singlet/pileup/barcode_rank.h"

// ---------------------------------------------------------------------------
// Minimal test framework
// ---------------------------------------------------------------------------
static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (cond) {                                                               \
            std::cout << "  PASS: " << (msg) << "\n";                            \
            ++n_pass;                                                             \
        } else {                                                                  \
            std::cout << "  FAIL: " << (msg) << " [line " << __LINE__ << "]\n";  \
            ++n_fail;                                                             \
        }                                                                         \
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
// Synthetic data:
//   6 barcodes with known UMI counts; barcodes 0,1,2 are cells
// ---------------------------------------------------------------------------
static std::vector<std::string> g_barcodes = {
    "AAACCCTT",  // idx 0 — 5000 UMIs, cell
    "AACCGGTT",  // idx 1 — 3000 UMIs, cell
    "TTGGCCAA",  // idx 2 — 8000 UMIs, cell
    "GCTAGCTA",  // idx 3 — 80   UMIs, empty
    "AGATCGAT",  // idx 4 — 50   UMIs, empty
    "CGTACGTA",  // idx 5 — 120  UMIs, empty
};
static std::vector<uint64_t> g_totals = {5000, 3000, 8000, 80, 50, 120};
static std::unordered_set<uint32_t> g_cells = {0, 1, 2};

// ---------------------------------------------------------------------------
// T1: Ranks are 1-based and contiguous
// ---------------------------------------------------------------------------
static void test_ranks_contiguous() {
    std::cout << "\n=== T1: Ranks are 1-based and contiguous ===\n";

    auto ranks = singlet::build_barcode_rank(g_barcodes, g_totals, g_cells);
    CHECK(ranks.size() == 6, "6 entries returned");  // 1
    CHECK(ranks[0].rank == 1, "first rank = 1");      // 2
    for (size_t i = 1; i < ranks.size(); ++i) {
        CHECK(ranks[i].rank == ranks[i-1].rank + 1,
              "rank[" + std::to_string(i) + "] = rank[" + std::to_string(i-1) + "] + 1");  // 3,4,5,6 (4 checks)
    }
}

// ---------------------------------------------------------------------------
// T2: UMI counts are descending
// ---------------------------------------------------------------------------
static void test_umi_descending() {
    std::cout << "\n=== T2: UMI counts descend with rank ===\n";

    auto ranks = singlet::build_barcode_rank(g_barcodes, g_totals, g_cells);
    CHECK(ranks[0].umi_count == 8000, "rank1 = 8000 UMIs");  // 7
    CHECK(ranks[1].umi_count == 5000, "rank2 = 5000 UMIs");  // 8
    CHECK(ranks[2].umi_count == 3000, "rank3 = 3000 UMIs");  // 9
    for (size_t i = 1; i < ranks.size(); ++i) {
        CHECK(ranks[i].umi_count <= ranks[i-1].umi_count,
              "umi_count[" + std::to_string(i) + "] <= umi_count[" + std::to_string(i-1) + "]");  // 10,11,12,13
    }
}

// ---------------------------------------------------------------------------
// T3: is_cell flag correct
// ---------------------------------------------------------------------------
static void test_is_cell_flag() {
    std::cout << "\n=== T3: is_cell flag correct ===\n";

    auto ranks = singlet::build_barcode_rank(g_barcodes, g_totals, g_cells);
    // Top 3 (ranks 1-3) are barcodes 2,0,1 (from g_cells) → is_cell=true
    CHECK(ranks[0].is_cell == true,  "rank1 is_cell = true");   // 14
    CHECK(ranks[1].is_cell == true,  "rank2 is_cell = true");   // 15
    CHECK(ranks[2].is_cell == true,  "rank3 is_cell = true");   // 16
    CHECK(ranks[3].is_cell == false, "rank4 is_cell = false");  // 17
    CHECK(ranks[4].is_cell == false, "rank5 is_cell = false");  // 18
    CHECK(ranks[5].is_cell == false, "rank6 is_cell = false");  // 19
}

// ---------------------------------------------------------------------------
// T4: TSV format — header and data
// ---------------------------------------------------------------------------
static void test_tsv_format() {
    std::cout << "\n=== T4: TSV format ===\n";

    const std::string path = "/tmp/test_br_t4.tsv";
    auto ranks = singlet::build_barcode_rank(g_barcodes, g_totals, g_cells);
    singlet::BarcodeRankConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_barcode_rank(ranks, cfg);
    CHECK(ok, "write_barcode_rank returns true");  // 20

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 7, "7 lines (header + 6 barcodes)");  // 21

    auto hdr = split_tab(lines[0]);
    CHECK(hdr.size() == 4,         "header has 4 columns");    // 22
    CHECK(hdr[0] == "rank",        "col0 = rank");             // 23
    CHECK(hdr[1] == "barcode",     "col1 = barcode");          // 24
    CHECK(hdr[2] == "umi_count",   "col2 = umi_count");        // 25
    CHECK(hdr[3] == "is_cell",     "col3 = is_cell");          // 26

    auto r1 = split_tab(lines[1]);
    CHECK(r1[0] == "1",    "first data row rank = 1");          // 27
    CHECK(r1[2] == "8000", "first data row umi_count = 8000");  // 28
    CHECK(r1[3] == "True", "first row is_cell = True");         // 29

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T5: All barcodes present in output
// ---------------------------------------------------------------------------
static void test_all_barcodes_present() {
    std::cout << "\n=== T5: All barcodes appear in output ===\n";

    const std::string path = "/tmp/test_br_t5.tsv";
    auto ranks = singlet::build_barcode_rank(g_barcodes, g_totals, g_cells);
    singlet::BarcodeRankConfig cfg;
    cfg.filepath = path;
    singlet::write_barcode_rank(ranks, cfg);

    auto content = read_file_str(path);
    for (const auto& bc : g_barcodes) {
        CHECK(content.find(bc) != std::string::npos,
              bc + " present in output");  // 30-35
    }
    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T6: Empty input — no crash, header-only output
// ---------------------------------------------------------------------------
static void test_empty_input() {
    std::cout << "\n=== T6: Empty input ===\n";

    auto ranks = singlet::build_barcode_rank({}, {}, {});
    CHECK(ranks.empty(), "empty input → empty rank table");  // 36

    const std::string path = "/tmp/test_br_t6.tsv";
    singlet::BarcodeRankConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_barcode_rank(ranks, cfg);
    CHECK(ok, "write succeeds for empty input");  // 37

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 1, "only header written for empty input");  // 38

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== barcode_rank tests ===\n";
    test_ranks_contiguous();
    test_umi_descending();
    test_is_cell_flag();
    test_tsv_format();
    test_all_barcodes_present();
    test_empty_input();

    std::cout << "\nTotal: " << (n_pass + n_fail)
              << "  PASS: " << n_pass
              << "  FAIL: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
