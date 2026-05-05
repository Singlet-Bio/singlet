// test_spatial_multiome.cpp — unit tests for spatial_multiome.h
// G-SPATIAL-MULTIOME: Spatial Multiome Router (Visium HD + spatial RNA/ATAC)

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet/pileup/spatial_multiome.h"

using namespace singlet;

// ── helpers ───────────────────────────────────────────────────────────────────

static void check(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "FAIL: " << msg << '\n';
        std::exit(1);
    }
}

static bool float_eq(float a, float b, float eps = 0.01f) {
    return std::fabs(a - b) < eps;
}

// ── detect_spatial_config tests ───────────────────────────────────────────────

static void test_detect_visium_hd() {
    SpatialConfig cfg = detect_spatial_config("visium-hd");
    check(cfg.resolution == SpatialResolution::VISIUM_HD_8, "visium-hd → HD_8");
    check(cfg.bin_size_um == 8, "visium-hd → 8µm");
    check(!cfg.has_atac, "visium-hd → no ATAC");
}

static void test_detect_visium_hd_16() {
    SpatialConfig cfg = detect_spatial_config("visium-hd-16");
    check(cfg.resolution == SpatialResolution::VISIUM_HD_16, "visium-hd-16 → HD_16");
    check(cfg.bin_size_um == 16, "visium-hd-16 → 16µm");
}

static void test_detect_visium_hd_2() {
    SpatialConfig cfg = detect_spatial_config("visium-hd-2");
    check(cfg.resolution == SpatialResolution::VISIUM_HD_2, "visium-hd-2 → HD_2");
    check(cfg.bin_size_um == 2, "visium-hd-2 → 2µm");
}

static void test_detect_visium_classic() {
    SpatialConfig cfg = detect_spatial_config("visium");
    check(cfg.resolution == SpatialResolution::VISIUM_V1, "visium → V1");
    check(cfg.bin_size_um == 55, "visium → 55µm");
    check(!cfg.has_atac, "visium → no ATAC");
}

static void test_detect_spatial_multiome() {
    SpatialConfig cfg = detect_spatial_config("spatial-multiome");
    check(cfg.has_atac, "spatial-multiome → has_atac");
    check(!cfg.has_protein, "spatial-multiome → no protein");
}

static void test_detect_visium_atac() {
    SpatialConfig cfg = detect_spatial_config("visium-atac");
    check(cfg.has_atac, "visium-atac → has_atac");
    check(cfg.resolution == SpatialResolution::VISIUM_V1, "visium-atac → V1");
}

static void test_detect_cytassist_protein() {
    SpatialConfig cfg = detect_spatial_config("visium-cytassist-protein");
    check(cfg.has_protein, "visium-cytassist-protein → has_protein");
}

// ── is_spatial_experiment tests ──────────────────────────────────────────────

static void test_is_spatial() {
    check(is_spatial_experiment("visium-hd"), "visium-hd is spatial");
    check(is_spatial_experiment("visium"), "visium is spatial");
    check(is_spatial_experiment("spatial-rna"), "spatial-rna is spatial");
    check(is_spatial_experiment("visium-v3"), "visium-v3 is spatial");
    check(!is_spatial_experiment("10x-3p-v3"), "10x-3p-v3 is not spatial");
    check(!is_spatial_experiment("10x-arc-gex"), "10x-arc-gex is not spatial");
    check(!is_spatial_experiment("dropseq"), "dropseq is not spatial");
    check(!is_spatial_experiment(""), "empty is not spatial");
}

// ── is_visium_hd tests ────────────────────────────────────────────────────────

static void test_is_visium_hd() {
    check(is_visium_hd("visium-hd"), "visium-hd is HD");
    check(is_visium_hd("visium-hd-8"), "visium-hd-8 is HD");
    check(is_visium_hd("visium-hd-16"), "visium-hd-16 is HD");
    check(is_visium_hd("visiumhd"), "visiumhd (no dash) is HD");
    check(!is_visium_hd("visium"), "plain visium is not HD");
    check(!is_visium_hd("10x-3p-v3"), "10x-3p-v3 is not HD");
}

// ── parse_hd_barcode tests ────────────────────────────────────────────────────

static void test_parse_hd_barcode() {
    // Standard format: "ROWs_COLs"
    {
        SpatialCoord c = parse_hd_barcode("0001s_0002s", 8);
        check(c.row == 1, "hd barcode row=1");
        check(c.col == 2, "hd barcode col=2");
        check(float_eq(c.y_um, 8.0f), "hd barcode y_um=8");
        check(float_eq(c.x_um, 16.0f), "hd barcode x_um=16");
    }
    // No trailing 's'
    {
        SpatialCoord c = parse_hd_barcode("0010_0020", 8);
        check(c.row == 10, "hd barcode row=10");
        check(c.col == 20, "hd barcode col=20");
        check(float_eq(c.y_um, 80.0f), "hd barcode y_um=80");
        check(float_eq(c.x_um, 160.0f), "hd barcode x_um=160");
    }
    // Origin
    {
        SpatialCoord c = parse_hd_barcode("0000_0000", 8);
        check(c.row == 0 && c.col == 0, "hd barcode origin");
        check(float_eq(c.x_um, 0.0f) && float_eq(c.y_um, 0.0f), "origin coords=0");
    }
    // Malformed — should not crash, return 0,0
    {
        SpatialCoord c = parse_hd_barcode("ACGTACGT", 8);
        check(c.row == 0 && c.col == 0, "malformed barcode → 0,0");
    }
}

// ── write_spatial_metadata tests ─────────────────────────────────────────────

static void test_write_spatial_metadata() {
    const std::string path = "/tmp/test_spatial_meta.tsv";

    std::vector<std::string> barcodes = {
        "AAACCCAAGAAACACT",
        "AAACGGGCATGCATGC",
        "TTTGGGAAATTTCCAT"};
    std::vector<SpatialCoord> coords = {
        {0.0f, 0.0f, 0, 0},
        {8.0f, 0.0f, 0, 1},
        {16.0f, 8.0f, 1, 2}};
    SpatialConfig cfg;
    cfg.resolution = SpatialResolution::VISIUM_HD_8;
    cfg.bin_size_um = 8;

    bool ok = write_spatial_metadata(path, coords, barcodes, cfg);
    check(ok, "write_spatial_metadata returns true");

    // Verify header and 3 data rows
    std::ifstream f(path);
    check(f.is_open(), "output file opened");

    std::string line;
    std::getline(f, line);
    check(line == "barcode\trow\tcol\tx_um\ty_um", "header line correct");

    int data_rows = 0;
    while (std::getline(f, line)) {
        if (!line.empty()) ++data_rows;
    }
    check(data_rows == 3, "3 data rows written");

    std::remove(path.c_str());
}

static void test_write_spatial_metadata_mismatch() {
    // Mismatched vectors → returns false
    std::vector<std::string> barcodes = {"AAACCCAAGAAACACT"};
    std::vector<SpatialCoord> coords = {{0, 0, 0, 0}, {8, 0, 0, 1}};
    SpatialConfig cfg;
    bool ok = write_spatial_metadata("/tmp/no_file.tsv", coords, barcodes, cfg);
    check(!ok, "mismatched vectors returns false");
}

// ── resolution bin-size consistency ──────────────────────────────────────────

static void test_resolution_bin_sizes() {
    check(detect_spatial_config("visium").bin_size_um == 55, "V1 = 55µm");
    check(detect_spatial_config("visium-hd").bin_size_um == 8, "HD_8 = 8µm");
    check(detect_spatial_config("visium-hd-16").bin_size_um == 16, "HD_16 = 16µm");
    check(detect_spatial_config("visium-hd-2").bin_size_um == 2, "HD_2 = 2µm");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_detect_visium_hd();
    test_detect_visium_hd_16();
    test_detect_visium_hd_2();
    test_detect_visium_classic();
    test_detect_spatial_multiome();
    test_detect_visium_atac();
    test_detect_cytassist_protein();

    test_is_spatial();
    test_is_visium_hd();

    test_parse_hd_barcode();

    test_write_spatial_metadata();
    test_write_spatial_metadata_mismatch();

    test_resolution_bin_sizes();

    std::cout << "All spatial_multiome tests passed.\n";
    return 0;
}
