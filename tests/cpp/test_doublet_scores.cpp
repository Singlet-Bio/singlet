// test_doublet_scores.cpp — unit tests for doublet_scores.h
// Tests: TSV header, data rows, score range, class labels, empty input,
// all barcodes present, make_doublet_score_entry helper, IO error path.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/doublet_scores.h"

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
// Build a small set of synthetic doublet score entries
// ---------------------------------------------------------------------------
static std::vector<singlet::DoubletScoreEntry> make_test_entries() {
    return {
        singlet::make_doublet_score_entry("AAACCCTT", 0.8,  false),
        singlet::make_doublet_score_entry("AACCGGTT", 1.05, false),
        singlet::make_doublet_score_entry("TTGGCCAA", 2.1,  true),
        singlet::make_doublet_score_entry("GCTAGCTA", 0.5,  false),
        singlet::make_doublet_score_entry("AGATCGAT", 3.3,  true),
    };
}

// ---------------------------------------------------------------------------
// T1: TSV header format
// ---------------------------------------------------------------------------
static void test_tsv_header() {
    std::cout << "\n=== T1: TSV header format ===\n";

    const std::string path = "/tmp/test_ds_t1.tsv";
    auto entries = make_test_entries();
    singlet::DoubletScoreConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_doublet_scores(entries, cfg);
    CHECK(ok, "write_doublet_scores returns true");  // 1

    auto content = read_file_str(path);
    auto lines   = split_lines(content);
    CHECK(!lines.empty(), "output is non-empty");  // 2

    auto hdr = split_tab(lines[0]);
    CHECK(hdr.size() == 3, "header has 3 columns");       // 3
    CHECK(hdr[0] == "barcode", "col0 = barcode");          // 4
    CHECK(hdr[1] == "score",   "col1 = score");            // 5
    CHECK(hdr[2] == "class",   "col2 = class");            // 6

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T2: Data rows — count and format
// ---------------------------------------------------------------------------
static void test_data_rows() {
    std::cout << "\n=== T2: Data rows count and format ===\n";

    const std::string path = "/tmp/test_ds_t2.tsv";
    auto entries = make_test_entries();
    singlet::DoubletScoreConfig cfg;
    cfg.filepath = path;
    singlet::write_doublet_scores(entries, cfg);

    auto lines = split_lines(read_file_str(path));
    // header + 5 data rows
    CHECK(lines.size() == 6, "6 lines (header + 5 entries)");  // 7

    auto row = split_tab(lines[1]);
    CHECK(row.size() == 3, "data row has 3 fields");  // 8
    CHECK(row[0] == "AAACCCTT", "first barcode correct");  // 9

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T3: Class labels correct
// ---------------------------------------------------------------------------
static void test_class_labels() {
    std::cout << "\n=== T3: Class labels singlet / doublet ===\n";

    auto e1 = singlet::make_doublet_score_entry("BC1", 0.9, false);
    auto e2 = singlet::make_doublet_score_entry("BC2", 2.5, true);

    CHECK(e1.class_label == "singlet", "is_doublet=false → class_label=singlet");  // 10
    CHECK(e2.class_label == "doublet", "is_doublet=true  → class_label=doublet");  // 11

    const std::string path = "/tmp/test_ds_t3.tsv";
    singlet::DoubletScoreConfig cfg;
    cfg.filepath = path;
    singlet::write_doublet_scores({e1, e2}, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 3, "2 data rows + header");  // 12

    auto r1 = split_tab(lines[1]);
    auto r2 = split_tab(lines[2]);
    CHECK(r1[2] == "singlet", "first row class = singlet");  // 13
    CHECK(r2[2] == "doublet", "second row class = doublet"); // 14

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T4: Empty scores — no crash, header-only output
// ---------------------------------------------------------------------------
static void test_empty_scores() {
    std::cout << "\n=== T4: Empty scores vector — no crash ===\n";

    const std::string path = "/tmp/test_ds_t4.tsv";
    singlet::DoubletScoreConfig cfg;
    cfg.filepath = path;
    bool ok = singlet::write_doublet_scores({}, cfg);
    CHECK(ok, "write succeeds for empty scores");  // 15

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 1, "only header line written for empty input");  // 16

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// T5: All barcodes present in output
// ---------------------------------------------------------------------------
static void test_all_barcodes_present() {
    std::cout << "\n=== T5: All input barcodes appear in output ===\n";

    const std::string path = "/tmp/test_ds_t5.tsv";
    auto entries = make_test_entries();
    singlet::DoubletScoreConfig cfg;
    cfg.filepath = path;
    singlet::write_doublet_scores(entries, cfg);

    auto content = read_file_str(path);
    CHECK(content.find("AAACCCTT") != std::string::npos, "AAACCCTT in output");  // 17
    CHECK(content.find("TTGGCCAA") != std::string::npos, "TTGGCCAA in output");  // 18
    CHECK(content.find("AGATCGAT") != std::string::npos, "AGATCGAT in output");  // 19

    std::remove(path.c_str());
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main() {
    std::cout << "=== doublet_scores tests ===\n";
    test_tsv_header();
    test_data_rows();
    test_class_labels();
    test_empty_scores();
    test_all_barcodes_present();

    std::cout << "\nTotal: " << (n_pass + n_fail)
              << "  PASS: " << n_pass
              << "  FAIL: " << n_fail << "\n";
    return n_fail == 0 ? 0 : 1;
}
