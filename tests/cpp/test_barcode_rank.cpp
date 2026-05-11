// test_barcode_rank.cpp — ground-truth unit tests for barcode_rank.h
//
// Every assertion compares against pre-computed expected values rather than
// checking properties of the function's own output.

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
// T_RANK_EXACT: 10 barcodes with known UMI counts
//   Verify exact descending rank order, 1-indexed consecutive ranks.
//
//   Input (index order):
//     BC_A:1000  BC_B:500  BC_C:500  BC_D:200  BC_E:100
//     BC_F:50    BC_G:50   BC_H:50   BC_I:10   BC_J:1
//
//   Expected sort: descending UMI, alphabetical tie-break
//     rank 1: BC_A  1000
//     rank 2: BC_B   500   (tie with BC_C; B < C)
//     rank 3: BC_C   500
//     rank 4: BC_D   200
//     rank 5: BC_E   100
//     rank 6: BC_F    50   (tie with BC_G, BC_H; F < G < H)
//     rank 7: BC_G    50
//     rank 8: BC_H    50
//     rank 9: BC_I    10
//     rank10: BC_J     1
// ---------------------------------------------------------------------------
static void test_rank_exact() {
    std::cout << "\n=== T_RANK_EXACT ===\n";

    std::vector<std::string> barcodes = {
        "BC_A", "BC_B", "BC_C", "BC_D", "BC_E",
        "BC_F", "BC_G", "BC_H", "BC_I", "BC_J"
    };
    std::vector<uint64_t> totals = {1000, 500, 500, 200, 100, 50, 50, 50, 10, 1};
    std::unordered_set<uint32_t> cells; // none

    auto ranks = singlet::build_barcode_rank(barcodes, totals, cells);

    CHECK(ranks.size() == 10, "10 entries returned");

    // Pre-computed ground truth
    const std::string expected_bc[] = {
        "BC_A", "BC_B", "BC_C", "BC_D", "BC_E",
        "BC_F", "BC_G", "BC_H", "BC_I", "BC_J"
    };
    const uint64_t expected_umi[] = {1000, 500, 500, 200, 100, 50, 50, 50, 10, 1};

    for (size_t i = 0; i < 10; ++i) {
        CHECK(ranks[i].rank == static_cast<uint32_t>(i + 1),
              "rank[" + std::to_string(i) + "] = " + std::to_string(i + 1));
        CHECK(ranks[i].barcode == expected_bc[i],
              "rank " + std::to_string(i + 1) + " barcode = " + expected_bc[i]);
        CHECK(ranks[i].umi_count == expected_umi[i],
              "rank " + std::to_string(i + 1) + " umi = " + std::to_string(expected_umi[i]));
    }
}

// ---------------------------------------------------------------------------
// T_RANK_TIES: 3 barcodes with identical UMI count (500 each)
//   Verify stable alphabetical secondary sort: BC_X < BC_Y < BC_Z
//   Verify deterministic ordering across two runs.
// ---------------------------------------------------------------------------
static void test_rank_ties() {
    std::cout << "\n=== T_RANK_TIES ===\n";

    std::vector<std::string> barcodes = {"BC_Z", "BC_X", "BC_Y"};
    std::vector<uint64_t> totals = {500, 500, 500};
    std::unordered_set<uint32_t> cells;

    // Run 1
    auto r1 = singlet::build_barcode_rank(barcodes, totals, cells);
    CHECK(r1.size() == 3, "3 entries returned");
    CHECK(r1[0].barcode == "BC_X", "tie rank 1 = BC_X (alphabetical)");
    CHECK(r1[1].barcode == "BC_Y", "tie rank 2 = BC_Y");
    CHECK(r1[2].barcode == "BC_Z", "tie rank 3 = BC_Z");
    CHECK(r1[0].umi_count == 500, "all UMI = 500 (rank 1)");
    CHECK(r1[1].umi_count == 500, "all UMI = 500 (rank 2)");
    CHECK(r1[2].umi_count == 500, "all UMI = 500 (rank 3)");

    // Run 2 — determinism
    auto r2 = singlet::build_barcode_rank(barcodes, totals, cells);
    CHECK(r2[0].barcode == r1[0].barcode, "deterministic: run2[0] == run1[0]");
    CHECK(r2[1].barcode == r1[1].barcode, "deterministic: run2[1] == run1[1]");
    CHECK(r2[2].barcode == r1[2].barcode, "deterministic: run2[2] == run1[2]");
}

// ---------------------------------------------------------------------------
// T_TSV_FORMAT: Write barcode rank TSV, read it back, verify format.
//   - Correct column headers (tab-separated)
//   - Tab separation between columns
//   - Newline termination of each row
//   - Correct number of data rows
// ---------------------------------------------------------------------------
static void test_tsv_format() {
    std::cout << "\n=== T_TSV_FORMAT ===\n";

    std::vector<std::string> barcodes = {"BC_A", "BC_B", "BC_C"};
    std::vector<uint64_t> totals = {1000, 500, 100};
    std::unordered_set<uint32_t> cells = {0};

    auto ranks = singlet::build_barcode_rank(barcodes, totals, cells);

    const std::string path = "test_br_tsv_format.tsv";
    singlet::BarcodeRankConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_barcode_rank(ranks, cfg);
    CHECK(ok, "write_barcode_rank returns true");

    std::string raw = read_file_str(path);

    // Verify newline termination: file should end with '\n'
    CHECK(!raw.empty() && raw.back() == '\n', "file ends with newline");

    auto lines = split_lines(raw);
    CHECK(lines.size() == 4, "4 lines (1 header + 3 data rows)");

    // Header
    auto hdr = split_tab(lines[0]);
    CHECK(hdr.size() == 4,       "header has 4 columns");
    CHECK(hdr[0] == "rank",      "header col0 = rank");
    CHECK(hdr[1] == "barcode",   "header col1 = barcode");
    CHECK(hdr[2] == "umi_count", "header col2 = umi_count");
    CHECK(hdr[3] == "is_cell",   "header col3 = is_cell");

    // Verify tab separation in raw line (3 tabs per data row)
    for (size_t i = 1; i < lines.size(); ++i) {
        size_t tab_count = 0;
        for (char c : lines[i]) if (c == '\t') ++tab_count;
        CHECK(tab_count == 3,
              "data row " + std::to_string(i) + " has 3 tabs");
    }

    // Spot-check first data row against ground truth
    auto r1 = split_tab(lines[1]);
    CHECK(r1[0] == "1",      "row1 rank = 1");
    CHECK(r1[1] == "BC_A",   "row1 barcode = BC_A");
    CHECK(r1[2] == "1000",   "row1 umi_count = 1000");
    CHECK(r1[3] == "True",   "row1 is_cell = True");

    // Spot-check last data row
    auto r3 = split_tab(lines[3]);
    CHECK(r3[0] == "3",      "row3 rank = 3");
    CHECK(r3[1] == "BC_C",   "row3 barcode = BC_C");
    CHECK(r3[2] == "100",    "row3 umi_count = 100");
    CHECK(r3[3] == "False",  "row3 is_cell = False");

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T_IS_CELL_ACCURACY: 10 barcodes, 4 are called cells (indices 0,1,2,3).
//   Verify exactly those 4 have is_cell=true, the other 6 false.
//
//   Input indices 0..9, cells = {0,1,2,3}
//   After sorting by UMI desc the mapping is:
//     rank 1: idx 0 (10000) → cell
//     rank 2: idx 1 (8000)  → cell
//     rank 3: idx 2 (6000)  → cell
//     rank 4: idx 3 (4000)  → cell
//     rank 5: idx 4 (2000)  → NOT cell
//     rank 6: idx 5 (1000)  → NOT cell
//     rank 7: idx 6 (500)   → NOT cell
//     rank 8: idx 7 (100)   → NOT cell
//     rank 9: idx 8 (50)    → NOT cell
//     rank10: idx 9 (10)    → NOT cell
// ---------------------------------------------------------------------------
static void test_is_cell_accuracy() {
    std::cout << "\n=== T_IS_CELL_ACCURACY ===\n";

    std::vector<std::string> barcodes = {
        "BC_01", "BC_02", "BC_03", "BC_04", "BC_05",
        "BC_06", "BC_07", "BC_08", "BC_09", "BC_10"
    };
    std::vector<uint64_t> totals = {10000, 8000, 6000, 4000, 2000,
                                     1000,  500,  100,   50,   10};
    std::unordered_set<uint32_t> cells = {0, 1, 2, 3};

    auto ranks = singlet::build_barcode_rank(barcodes, totals, cells);
    CHECK(ranks.size() == 10, "10 entries returned");

    // Ground truth: ranks 1-4 are cells, 5-10 are not
    const bool expected_cell[] = {
        true, true, true, true, false, false, false, false, false, false
    };

    int cell_count = 0;
    int non_cell_count = 0;
    for (size_t i = 0; i < 10; ++i) {
        CHECK(ranks[i].is_cell == expected_cell[i],
              "rank " + std::to_string(i + 1) + " is_cell = " +
              (expected_cell[i] ? "true" : "false"));
        if (ranks[i].is_cell) ++cell_count;
        else ++non_cell_count;
    }
    CHECK(cell_count == 4,     "exactly 4 cells");
    CHECK(non_cell_count == 6, "exactly 6 non-cells");
}

// ---------------------------------------------------------------------------
// T_EMPTY_INPUT: Zero barcodes → no crash, empty output
// ---------------------------------------------------------------------------
static void test_empty_input() {
    std::cout << "\n=== T_EMPTY_INPUT ===\n";

    auto ranks = singlet::build_barcode_rank({}, {}, {});
    CHECK(ranks.empty(), "empty input produces empty rank table");

    const std::string path = "test_br_empty.tsv";
    singlet::BarcodeRankConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_barcode_rank(ranks, cfg);
    CHECK(ok, "write succeeds for empty input");

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 1, "only header line for empty input");

    auto hdr = split_tab(lines[0]);
    CHECK(hdr.size() == 4, "header still has 4 columns");

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== barcode_rank ground-truth tests ===\n";
    test_rank_exact();
    test_rank_ties();
    test_tsv_format();
    test_is_cell_accuracy();
    test_empty_input();

    std::cout << "\nTotal: " << (n_pass + n_fail)
              << "  PASS: " << n_pass
              << "  FAIL: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
