// test_barcode_metrics.cpp — unit tests for barcode_metrics.h
// Tests: TSV format, header row, metric values, cells-only filter, empty metrics,
// build output, saturation calculation, mito fraction.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include "singlet-pileup/barcode_metrics.h"

// ---------------------------------------------------------------------------
// Test framework
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
static std::string read_file_str(const std::string& path) {
    std::ifstream ifs(path);
    if (!ifs.is_open()) return {};
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

static void rm(const std::string& path) { std::remove(path.c_str()); }

// ---------------------------------------------------------------------------
// T1: Output file is created and non-empty
// ---------------------------------------------------------------------------
static void test_file_created() {
    std::cout << "\n=== T1: File creation ===\n";
    const std::string path = "/tmp/test_bm_t1.tsv";

    singlet::BarcodeMetric m;
    m.barcode     = "AACCGGTT-1";
    m.total_reads = 100;
    m.n_umi       = 80;
    m.n_genes     = 50;
    m.is_cell     = true;

    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;

    bool ok = singlet::write_barcode_metrics({m}, cfg);
    CHECK(ok, "write_barcode_metrics returns true");  // 1

    std::FILE* fp = std::fopen(path.c_str(), "r");
    CHECK(fp != nullptr, "output file created");  // 2
    if (fp) std::fclose(fp);

    rm(path);
}

// ---------------------------------------------------------------------------
// T2: Header row fields
// ---------------------------------------------------------------------------
static void test_header_row() {
    std::cout << "\n=== T2: Header row ===\n";
    const std::string path = "/tmp/test_bm_t2.tsv";

    singlet::BarcodeMetric m;
    m.barcode = "BC1-1";
    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    singlet::write_barcode_metrics({m}, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() >= 2, "header + data row");  // 3
    if (!lines.empty()) {
        auto hdr = split_tab(lines[0]);
        CHECK(!hdr.empty() && hdr[0] == "barcode",     "header[0]=barcode");      // 4
        CHECK(hdr.size() > 8 && hdr[8] == "n_umi",     "header[8]=n_umi");        // 5
        CHECK(hdr.size() > 9 && hdr[9] == "n_genes",   "header[9]=n_genes");      // 6
        CHECK(hdr.size() > 10 && hdr[10] == "saturation", "header[10]=saturation"); // 7
        CHECK(hdr.size() > 12 && hdr[12] == "is_cell", "header[12]=is_cell");     // 8
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T3: header=false — first field is a barcode, not "barcode"
// ---------------------------------------------------------------------------
static void test_no_header() {
    std::cout << "\n=== T3: No header ===\n";
    const std::string path = "/tmp/test_bm_t3.tsv";

    singlet::BarcodeMetric m;
    m.barcode = "BC1-1";
    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = false;
    singlet::write_barcode_metrics({m}, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(!lines.empty(), "at least one line");  // 9
    if (!lines.empty()) {
        auto fields = split_tab(lines[0]);
        CHECK(!fields.empty() && fields[0] == "BC1-1", "first field is barcode, not header");  // 10
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T4: Metric values round-trip correctly
// ---------------------------------------------------------------------------
static void test_metric_values() {
    std::cout << "\n=== T4: Metric values ===\n";
    const std::string path = "/tmp/test_bm_t4.tsv";

    singlet::BarcodeMetric m;
    m.barcode          = "AAACGGGT-1";
    m.total_reads      = 10000;
    m.mapped_reads     = 9000;
    m.unique_reads     = 8000;
    m.exonic_reads     = 6000;
    m.intronic_reads   = 1500;
    m.intergenic_reads = 500;
    m.antisense_reads  = 200;
    m.n_umi            = 7500;
    m.n_genes          = 3200;
    m.saturation       = 0.25;
    m.mito_fraction    = 0.05;
    m.is_cell          = true;

    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    singlet::write_barcode_metrics({m}, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 2, "header + 1 data row");  // 11
    if (lines.size() >= 2) {
        auto fields = split_tab(lines[1]);
        CHECK(fields.size() == 13, "13 columns");         // 12
        CHECK(fields[0] == "AAACGGGT-1", "barcode");      // 13
        CHECK(fields[1] == "10000",      "total_reads");  // 14
        CHECK(fields[7] == "200",        "antisense");    // 15
        CHECK(fields[8] == "7500",       "n_umi");        // 16
        CHECK(fields[9] == "3200",       "n_genes");      // 17
        CHECK(fields[12] == "1",         "is_cell=1");    // 18
        double sat = std::stod(fields[10]);
        CHECK(std::fabs(sat - 0.25) < 1e-4, "saturation~0.25");  // 19
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T5: cells_only filter
// ---------------------------------------------------------------------------
static void test_cells_only_filter() {
    std::cout << "\n=== T5: cells_only filter ===\n";
    const std::string path = "/tmp/test_bm_t5.tsv";

    std::vector<singlet::BarcodeMetric> metrics(4);
    metrics[0].barcode = "CELL1-1"; metrics[0].is_cell = true;
    metrics[1].barcode = "BACK1-1"; metrics[1].is_cell = false;
    metrics[2].barcode = "CELL2-1"; metrics[2].is_cell = true;
    metrics[3].barcode = "BACK2-1"; metrics[3].is_cell = false;

    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath   = path;
    cfg.header     = false;
    cfg.cells_only = true;
    singlet::write_barcode_metrics(metrics, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 2, "2 cell rows (non-cells filtered)");         // 20
    if (lines.size() >= 2) {
        CHECK(split_tab(lines[0])[0] == "CELL1-1", "first row is CELL1"); // 21
        CHECK(split_tab(lines[1])[0] == "CELL2-1", "second row is CELL2");// 22
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T6: Empty metrics — only header
// ---------------------------------------------------------------------------
static void test_empty_metrics() {
    std::cout << "\n=== T6: Empty metrics ===\n";
    const std::string path = "/tmp/test_bm_t6.tsv";

    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    bool ok = singlet::write_barcode_metrics({}, cfg);
    CHECK(ok, "write empty metrics returns true");  // 23

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 1, "only header row");                 // 24
    CHECK(!lines.empty() && split_tab(lines[0])[0] == "barcode",
          "header present");  // 25
    rm(path);
}

// ---------------------------------------------------------------------------
// T7: build_barcode_metrics output
// ---------------------------------------------------------------------------
static void test_build_barcode_metrics() {
    std::cout << "\n=== T7: build_barcode_metrics ===\n";
    std::vector<std::string> barcodes = {"BC1-1", "BC2-1", "BC3-1"};
    std::vector<uint64_t>    umis     = {1000, 200, 5000};
    std::vector<uint32_t>    genes    = {800, 150, 3200};
    std::unordered_set<uint32_t> cells = {0, 2};

    auto m = singlet::build_barcode_metrics(barcodes, umis, genes, cells);
    CHECK(m.size() == 3,           "3 metrics built");            // 26
    CHECK(m[0].barcode == "BC1-1", "m[0].barcode");               // 27
    CHECK(m[0].n_umi   == 1000,    "m[0].n_umi==1000");           // 28
    CHECK(m[0].n_genes == 800,     "m[0].n_genes==800");          // 29
    CHECK(m[0].is_cell == true,    "m[0].is_cell==true");         // 30
    CHECK(m[1].is_cell == false,   "m[1].is_cell==false");        // 31
    CHECK(m[2].n_umi   == 5000,    "m[2].n_umi==5000");           // 32
    CHECK(m[2].is_cell == true,    "m[2].is_cell==true");         // 33
}

// ---------------------------------------------------------------------------
// T8: Saturation value from build
// ---------------------------------------------------------------------------
static void test_saturation_build() {
    std::cout << "\n=== T8: Saturation from build ===\n";
    std::vector<std::string> barcodes = {"BC-1"};
    std::vector<uint64_t>    umis     = {500};
    std::vector<uint32_t>    genes    = {300};
    std::unordered_set<uint32_t> cells = {};

    auto m = singlet::build_barcode_metrics(barcodes, umis, genes, cells);
    // stub: total_reads == n_umi → saturation = 0
    CHECK(std::fabs(m[0].saturation - 0.0) < 1e-9, "saturation==0 when reads==umis");  // 34
}

// ---------------------------------------------------------------------------
// T9: mito_fraction round-trips
// ---------------------------------------------------------------------------
static void test_mito_fraction() {
    std::cout << "\n=== T9: mito_fraction round-trip ===\n";
    const std::string path = "/tmp/test_bm_t9.tsv";

    singlet::BarcodeMetric m;
    m.barcode       = "MT_HEAVY-1";
    m.mito_fraction = 0.42;
    m.is_cell       = false;

    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    singlet::write_barcode_metrics({m}, cfg);

    auto lines = split_lines(read_file_str(path));
    if (lines.size() >= 2) {
        auto fields = split_tab(lines[1]);
        double mf = std::stod(fields[11]);
        CHECK(std::fabs(mf - 0.42) < 1e-4, "mito_fraction~0.42");  // 35
    } else {
        CHECK(false, "expected 2 lines (header + data)");
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T10: is_cell=false written as 0
// ---------------------------------------------------------------------------
static void test_cell_flag_false() {
    std::cout << "\n=== T10: is_cell=false → 0 ===\n";
    const std::string path = "/tmp/test_bm_t10.tsv";

    singlet::BarcodeMetric m;
    m.barcode = "EMPTY-1";
    m.is_cell = false;
    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    singlet::write_barcode_metrics({m}, cfg);

    auto lines = split_lines(read_file_str(path));
    if (lines.size() >= 2) {
        auto fields = split_tab(lines[1]);
        CHECK(fields.size() > 12 && fields[12] == "0", "is_cell=false → field 12 == 0");  // 36
    } else {
        CHECK(false, "expected 2 lines");
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T11: Multiple rows all readable
// ---------------------------------------------------------------------------
static void test_multiple_rows() {
    std::cout << "\n=== T11: Multiple rows ===\n";
    const std::string path = "/tmp/test_bm_t11.tsv";

    std::vector<singlet::BarcodeMetric> metrics(5);
    for (int i = 0; i < 5; ++i) {
        metrics[i].barcode = "BARCODE" + std::to_string(i) + "-1";
        metrics[i].n_umi   = static_cast<uint32_t>(i * 100);
        metrics[i].n_genes = static_cast<uint32_t>(i * 50);
        metrics[i].is_cell = (i % 2 == 0);
    }
    singlet::BarcodeMetricsConfig cfg;
    cfg.filepath = path;
    cfg.header   = true;
    singlet::write_barcode_metrics(metrics, cfg);

    auto lines = split_lines(read_file_str(path));
    CHECK(lines.size() == 6, "header + 5 rows");  // 37
    if (lines.size() >= 4) {
        auto fields = split_tab(lines[3]);  // data row index 2 (0-based metrics)
        CHECK(!fields.empty() && fields[0] == "BARCODE2-1", "row3 barcode==BARCODE2-1");  // 38
        CHECK(fields.size() > 8 && fields[8] == "200", "row3 n_umi==200");  // 39
    }
    rm(path);
}

// ---------------------------------------------------------------------------
// T12: Bad path returns false
// ---------------------------------------------------------------------------
static void test_write_bad_path() {
    std::cout << "\n=== T12: Bad path returns false ===\n";
    singlet::BarcodeMetric m;
    m.barcode = "BC-1";
    bool ok = singlet::write_barcode_metrics({m}, {"/nonexistent_dir/x.tsv"});
    CHECK(!ok, "write_barcode_metrics returns false for bad path");  // 40
}

// ---------------------------------------------------------------------------
// T13: build with empty called_cells → all is_cell = false
// ---------------------------------------------------------------------------
static void test_build_no_cells() {
    std::cout << "\n=== T13: build with no called cells ===\n";
    std::vector<std::string> barcodes = {"A-1", "B-1"};
    std::vector<uint64_t>    umis     = {100, 200};
    std::vector<uint32_t>    genes    = {80, 160};
    std::unordered_set<uint32_t> cells = {};

    auto m = singlet::build_barcode_metrics(barcodes, umis, genes, cells);
    CHECK(!m.empty(), "2 metrics built");              // 41
    CHECK(!m[0].is_cell, "m[0].is_cell == false");    // 42
    CHECK(!m[1].is_cell, "m[1].is_cell == false");    // 43
}

int main() {
    test_file_created();
    test_header_row();
    test_no_header();
    test_metric_values();
    test_cells_only_filter();
    test_empty_metrics();
    test_build_barcode_metrics();
    test_saturation_build();
    test_mito_fraction();
    test_cell_flag_false();
    test_multiple_rows();
    test_write_bad_path();
    test_build_no_cells();

    std::cout << "\n──────────────────────────\n";
    std::cout << "PASSED: " << n_pass << "\n";
    std::cout << "FAILED: " << n_fail << "\n";
    return n_fail ? 1 : 0;
}
