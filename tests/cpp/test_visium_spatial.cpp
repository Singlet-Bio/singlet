// test_visium_spatial.cpp — unit tests for V1 Visium spatial coordinate parser
// Tests VisiumSpatialParser: CSV loading, barcode lookup, in_tissue filter,
// coordinate values, missing barcode fallback, and TSV export.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/visium_spatial.h"

using namespace singlet;

// ── Helpers ──────────────────────────────────────────────────────────────────

static std::string write_temp_csv(const char* content) {
    std::string tmpl = "/tmp/test_visium_XXXXXX.csv";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    int fd = mkstemps(buf.data(), 4);  // 4 = strlen(".csv")
    if (fd < 0) {
        perror("mkstemps");
        abort();
    }
    FILE* f = fdopen(fd, "w");
    fputs(content, f);
    fclose(f);
    return std::string(buf.data());
}

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << "\n";
        std::exit(1);
    }
    std::cerr << "PASS: " << msg << "\n";
}

// ── Test 1: Load headerless CSV + lookup + stats ──────────────────────────────

static void test_load_and_lookup() {
    // Space Ranger ≤1.x format: no header
    const char* csv =
        "AAACAAGTATCTCCCA-1,1,0,0,2348,1456\n"
        "AAACACCAATAACTGC-1,1,0,1,2348,1584\n"
        "AAACAGAGCGACTCCT-1,0,0,2,2348,1712\n"  // not in tissue
        "AAACAGGGTCTATGCG-1,1,1,0,2476,1456\n";

    auto path = write_temp_csv(csv);

    VisiumSpatialParser p;
    bool ok = p.load_positions(path);
    check(ok, "load_positions returns true");
    check(p.total_spots() == 4, "total_spots == 4");
    check(p.in_tissue() == 3, "in_tissue == 3");

    // Known-good coordinate
    const SpotCoordinate* c = p.get_coordinate("AAACAAGTATCTCCCA-1");
    check(c != nullptr, "first barcode found");
    check(c->in_tissue == true, "first spot in_tissue=true");
    check(c->array_row == 0, "array_row == 0");
    check(c->array_col == 0, "array_col == 0");
    check(std::abs(c->pixel_row - 2348.0f) < 0.5f, "pixel_row ~ 2348");
    check(std::abs(c->pixel_col - 1456.0f) < 0.5f, "pixel_col ~ 1456");

    // Not-in-tissue spot
    const SpotCoordinate* c2 = p.get_coordinate("AAACAGAGCGACTCCT-1");
    check(c2 != nullptr, "not-in-tissue barcode found");
    check(c2->in_tissue == false, "that spot in_tissue=false");

    // Missing barcode returns nullptr
    const SpotCoordinate* cnull = p.get_coordinate("DOESNOTEXIST");
    check(cnull == nullptr, "missing barcode returns nullptr");

    std::remove(path.c_str());
}

// ── Test 2: Load CSV with header (Space Ranger ≥2.0) ────────────────────────

static void test_load_with_header() {
    const char* csv =
        "barcode,in_tissue,array_row,array_col,pxl_row_in_fullres,pxl_col_in_fullres\n"
        "AAACCTGAGAAACCAT-1,1,5,10,3100,2200\n"
        "AAACCTGAGAAACCGG-1,0,5,11,3100,2328\n";

    auto path = write_temp_csv(csv);

    VisiumSpatialParser p;
    bool ok = p.load_positions(path);
    check(ok, "load_with_header: load_positions returns true");
    check(p.total_spots() == 2, "load_with_header: total_spots == 2");
    check(p.in_tissue() == 1, "load_with_header: in_tissue == 1");

    const SpotCoordinate* c = p.get_coordinate("AAACCTGAGAAACCAT-1");
    check(c != nullptr, "load_with_header: barcode found");
    check(c->array_row == 5 && c->array_col == 10, "load_with_header: correct grid coords");

    std::remove(path.c_str());
}

// ── Test 3: in_tissue filter — only tissue spots in provided list ─────────────

static void test_in_tissue_filter() {
    const char* csv =
        "BC_A,1,0,0,100,200\n"
        "BC_B,0,0,1,100,330\n"
        "BC_C,1,1,0,230,200\n"
        "BC_D,0,1,1,230,330\n"
        "BC_E,1,2,0,360,200\n";

    auto path = write_temp_csv(csv);

    VisiumSpatialParser p;
    p.load_positions(path);
    check(p.in_tissue() == 3, "in_tissue_filter: 3 tissue spots");

    // Verify each spot's tissue flag
    size_t tissue_count = 0;
    for (const auto& bc : std::vector<std::string>{"BC_A", "BC_B", "BC_C", "BC_D", "BC_E"}) {
        const SpotCoordinate* c = p.get_coordinate(bc);
        if (c && c->in_tissue) ++tissue_count;
    }
    check(tissue_count == 3, "in_tissue_filter: manual count == 3");

    std::remove(path.c_str());
}

// ── Test 4: write_coordinates_tsv ────────────────────────────────────────────

static void test_write_tsv() {
    const char* csv =
        "SPOT1,1,3,7,1500,2000\n"
        "SPOT2,0,4,8,1628,2128\n";

    auto csv_path = write_temp_csv(csv);

    VisiumSpatialParser p;
    p.load_positions(csv_path);

    std::string out_path = "/tmp/visium_test_out.tsv";
    std::vector<std::string> barcodes = {"SPOT1", "SPOT2", "MISSING"};
    p.write_coordinates_tsv(out_path, barcodes);

    // Read and validate TSV
    std::ifstream in(out_path);
    std::string line;
    std::getline(in, line);  // header
    check(line.find("barcode") != std::string::npos, "tsv: header present");

    std::getline(in, line);  // SPOT1
    check(line.find("SPOT1") != std::string::npos, "tsv: SPOT1 row present");
    check(line.find("1500") != std::string::npos, "tsv: pixel_row 1500 present");

    std::getline(in, line);  // SPOT2
    check(line.find("SPOT2") != std::string::npos, "tsv: SPOT2 row present");

    std::getline(in, line);  // MISSING — should have zeros
    check(line.find("MISSING") != std::string::npos, "tsv: MISSING row present");
    check(line.find("\t0\t0\t0\t0\t0") != std::string::npos, "tsv: MISSING has zero coords");

    std::remove(csv_path.c_str());
    std::remove(out_path.c_str());
}

// ── Test 5: empty / non-existent file ────────────────────────────────────────

static void test_bad_file() {
    VisiumSpatialParser p;
    bool ok = p.load_positions("/tmp/this_file_does_not_exist_visium_test.csv");
    check(!ok, "bad_file: load_positions returns false for missing file");
    check(p.total_spots() == 0, "bad_file: total_spots == 0");
}

// ── Main ─────────────────────────────────────────────────────────────────────

int main() {
    test_load_and_lookup();
    test_load_with_header();
    test_in_tissue_filter();
    test_write_tsv();
    test_bad_file();
    std::cerr << "All visium_spatial tests PASSED\n";
    return 0;
}
