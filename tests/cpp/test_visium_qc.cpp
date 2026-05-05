// test_visium_qc.cpp — Unit tests for visium_qc.h (V3)
//
// Tests:
//   1. build_spot_qc: barcode join with positions (in_tissue / not found)
//   2. compute_visium_summary: tissue_coverage, median, mean, std
//   3. write_visium_qc_tsv: column count and header format
//   4. Empty parser (no positions file): all in_tissue=false, coverage=0
//   5. All in-tissue spots below threshold: coverage=0
//   6. Median computation for even/odd count

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/visium_qc.h"
#include "singlet/pileup/visium_spatial.h"

using namespace singlet;

// ── Helpers ───────────────────────────────────────────────────────────────────

static std::string make_positions_csv(const std::vector<std::tuple<std::string, int, int, int, float, float>>& rows) {
    std::ostringstream ss;
    ss << "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n";
    for (const auto& [bc, it, ar, ac, pr, pc] : rows) {
        ss << bc << "," << it << "," << ar << "," << ac << ","
           << pr << "," << pc << "\n";
    }
    return ss.str();
}

static VisiumSpatialParser load_from_string(const std::string& csv) {
    // Write to a temp file and load
    const char* tmp = "/tmp/test_visium_positions.csv";
    std::ofstream f(tmp);
    f << csv;
    f.close();
    VisiumSpatialParser p;
    p.load_positions(tmp);
    return p;
}

// ── Test 1: build_spot_qc basic join ──────────────────────────────────────────

static void test_build_spot_qc_join() {
    auto csv = make_positions_csv({
        {"AAACAAGTATCTCCCA", 1, 10, 20, 1000.f, 2000.f},
        {"BBBBBBBBBBBBBBBB", 0, 11, 21, 500.f, 600.f},
    });
    auto parser = load_from_string(csv);

    std::vector<std::string> barcodes = {"AAACAAGTATCTCCCA", "BBBBBBBBBBBBBBBB", "CCCCCCCCCCCCCCCC"};
    std::vector<uint32_t> umis = {500, 200, 50};
    std::vector<uint32_t> genes = {300, 150, 30};
    std::vector<float> mt = {3.f, 5.f, 1.f};

    auto records = build_spot_qc(barcodes, umis, genes, mt, parser);

    assert(records.size() == 3);

    // First barcode: in tissue, found in positions
    assert(records[0].barcode == "AAACAAGTATCTCCCA");
    assert(records[0].total_umis == 500);
    assert(records[0].genes_detected == 300);
    assert(records[0].in_tissue == true);
    assert(records[0].in_positions == true);
    assert(records[0].array_row == 10);
    assert(records[0].array_col == 20);

    // Second barcode: not in tissue (in_tissue=0)
    assert(records[1].in_tissue == false);
    assert(records[1].in_positions == true);

    // Third barcode: not in positions at all
    assert(records[2].in_tissue == false);
    assert(records[2].in_positions == false);
}

// ── Test 2: compute_visium_summary correctness ────────────────────────────────

static void test_compute_summary() {
    // 3 in-tissue spots with UMIs 100, 200, 300; threshold=150
    auto csv = make_positions_csv({
        {"BC1", 1, 0, 0, 0.f, 0.f},
        {"BC2", 1, 1, 0, 0.f, 0.f},
        {"BC3", 1, 2, 0, 0.f, 0.f},
        {"BC4", 0, 3, 0, 0.f, 0.f},  // not in tissue
    });
    auto parser = load_from_string(csv);

    std::vector<std::string> barcodes = {"BC1", "BC2", "BC3", "BC4"};
    std::vector<uint32_t> umis = {100, 200, 300, 999};
    std::vector<uint32_t> genes = {50, 100, 150, 500};
    std::vector<float> mt;

    auto records = build_spot_qc(barcodes, umis, genes, mt, parser);
    auto s = compute_visium_summary(records, /*min_umi=*/150);
    (void)s;

    assert(s.n_barcodes == 4);
    assert(s.n_in_tissue == 3);
    assert(s.n_detected_spots == 2);  // BC2 (200≥150) and BC3 (300≥150)
    assert(std::fabs(s.tissue_coverage - 2.0 / 3.0) < 1e-6);

    // Mean = (100+200+300)/3 = 200; median = 200
    assert(std::fabs(s.mean_umis - 200.0) < 1e-9);
    assert(std::fabs(s.median_umis - 200.0) < 1e-9);

    // Std-dev of [100,200,300] = sqrt(((100-200)^2+(200-200)^2+(300-200)^2)/2) = sqrt(10000)=100
    assert(std::fabs(s.std_umis - 100.0) < 1e-6);
}

// ── Test 3: write_visium_qc_tsv ───────────────────────────────────────────────

static void test_write_tsv() {
    std::vector<SpotQCRecord> records(2);
    records[0].barcode = "AAAAAAAAAAAA";
    records[0].total_umis = 500;
    records[0].genes_detected = 200;
    records[0].mt_pct = 2.5f;
    records[0].array_row = 5;
    records[0].array_col = 10;
    records[0].in_tissue = true;
    records[0].in_positions = true;

    records[1].barcode = "BBBBBBBBBBBB";
    records[1].total_umis = 0;
    records[1].in_tissue = false;
    records[1].in_positions = false;

    const char* tmp = "/tmp/test_visium_qc_out.tsv";
    bool ok = write_visium_qc_tsv(tmp, records);
    (void)ok;
    assert(ok);

    std::ifstream f(tmp);
    std::string header;
    std::getline(f, header);
    // Count tabs: 9 tab-separated columns → 9 tabs
    size_t ntabs = std::count(header.begin(), header.end(), '\t');
    (void)ntabs;
    assert(ntabs == 9);

    // Check header contains expected columns
    assert(header.find("barcode") != std::string::npos);
    assert(header.find("total_umis") != std::string::npos);
    assert(header.find("in_tissue") != std::string::npos);

    // Two data rows
    std::string r1, r2, extra;
    std::getline(f, r1);
    std::getline(f, r2);
    std::getline(f, extra);
    assert(!r1.empty());
    assert(!r2.empty());
    assert(extra.empty());  // no extra lines

    std::remove(tmp);
}

// ── Test 4: empty parser (no positions loaded) ────────────────────────────────

static void test_empty_parser() {
    VisiumSpatialParser parser;  // no load_positions called

    std::vector<std::string> barcodes = {"BC1", "BC2"};
    std::vector<uint32_t> umis = {500, 300};
    std::vector<uint32_t> genes = {100, 80};
    std::vector<float> mt;

    auto records = build_spot_qc(barcodes, umis, genes, mt, parser);
    assert(records[0].in_tissue == false);
    assert(records[0].in_positions == false);

    auto s = compute_visium_summary(records, 100);
    (void)s;
    assert(s.n_in_tissue == 0);
    assert(s.tissue_coverage == 0.0);
    assert(s.median_umis == 0.0);
}

// ── Test 5: all spots below threshold ────────────────────────────────────────

static void test_coverage_zero_below_threshold() {
    auto csv = make_positions_csv({
        {"BC1", 1, 0, 0, 0.f, 0.f},
        {"BC2", 1, 1, 0, 0.f, 0.f},
    });
    auto parser = load_from_string(csv);

    std::vector<std::string> barcodes = {"BC1", "BC2"};
    std::vector<uint32_t> umis = {50, 80};
    std::vector<uint32_t> genes = {10, 20};
    std::vector<float> mt;

    auto records = build_spot_qc(barcodes, umis, genes, mt, parser);
    auto s = compute_visium_summary(records, 100);
    (void)s;
    assert(s.n_in_tissue == 2);
    assert(s.n_detected_spots == 0);
    assert(s.tissue_coverage == 0.0);
}

// ── Test 6: median with even count ───────────────────────────────────────────

static void test_median_even() {
    auto csv = make_positions_csv({
        {"BC1", 1, 0, 0, 0.f, 0.f},
        {"BC2", 1, 1, 0, 0.f, 0.f},
        {"BC3", 1, 2, 0, 0.f, 0.f},
        {"BC4", 1, 3, 0, 0.f, 0.f},
    });
    auto parser = load_from_string(csv);

    std::vector<std::string> barcodes = {"BC1", "BC2", "BC3", "BC4"};
    std::vector<uint32_t> umis = {100, 200, 300, 400};
    std::vector<uint32_t> genes = {10, 20, 30, 40};
    std::vector<float> mt;

    auto records = build_spot_qc(barcodes, umis, genes, mt, parser);
    auto s = compute_visium_summary(records, 1);
    (void)s;
    // Sorted: [100,200,300,400] → median = (200+300)/2 = 250
    assert(std::fabs(s.median_umis - 250.0) < 1e-9);
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_build_spot_qc_join();
    test_compute_summary();
    test_write_tsv();
    test_empty_parser();
    test_coverage_zero_below_threshold();
    test_median_even();
    return 0;
}
