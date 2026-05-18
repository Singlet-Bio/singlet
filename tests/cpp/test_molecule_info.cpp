// SPDX-License-Identifier: MIT
// test/test_molecule_info.cpp
// Unit tests for G-MOLINFO: Molecule info HDF5 (molecule_info.h)
// Run: ./test_molecule_info
// Requires: HDF5 C library linked.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <hdf5.h>
#include "singlet/pileup/bus_writer.h"
#include "singlet/pileup/molecule_info.h"

using namespace singlet;
using namespace singlet::mol_info_detail;
namespace fs = std::filesystem;

// ── Counters ─────────────────────────────────────────────────────────────────

#include "test_harness.h"  // CHECK(cond, msg) + n_pass / n_fail

// ── helpers ───────────────────────────────────────────────────────────────────

static std::string tmp_path(const char* name) {
    return std::string("/tmp/") + name;
}

static bool ds_exists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

// ── Test suite ────────────────────────────────────────────────────────────────

// T1: file creation
static void test_file_creation() {
    std::cout << "\n=== T1: File creation ===\n";
    const std::string path = tmp_path("test_molinfo_create.h5");
    fs::remove(path);

    MoleculeInfoWriter w;
    bool opened = w.open({path, 16, 12});
    CHECK(opened, "open() returns true");

    std::vector<std::string> gnames = {"GeneA", "GeneB"};
    std::vector<std::string> gids   = {"ENSG00000000001", "ENSG00000000002"};
    std::vector<std::string> bcs    = {"ACGTACGTACGTACGT", "TTTTTTTTTTTTTTTT"};

    bool ok = w.finalize(gnames, gids, bcs);
    CHECK(ok, "finalize() returns true (empty molecules)");
    CHECK(fs::exists(path), "HDF5 file created on disk");

    fs::remove(path);
}

// T2: HDF5 structure — required datasets exist
static void test_hdf5_structure() {
    std::cout << "\n=== T2: HDF5 structure ===\n";
    const std::string path = tmp_path("test_molinfo_struct.h5");
    fs::remove(path);

    MoleculeInfoWriter w;
    w.open({path, 16, 12});

    // Add one molecule so arrays are non-empty
    MoleculeRecord r;
    r.barcode     = 0xAB00000000000000ULL;
    r.umi         = 0x1200000000000000ULL;
    r.feature_idx = 0;
    r.count       = 1;
    w.add_molecule(r);

    std::vector<std::string> gnames = {"GeneA"};
    std::vector<std::string> gids   = {"ENSG00000000001"};
    std::vector<std::string> bcs    = {"ACGTACGTACGTACGT"};
    w.finalize(gnames, gids, bcs, "{\"total\":1}");

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "HDF5 file opens for reading");

    CHECK(ds_exists(fid, "barcode"),      "/barcode dataset exists");
    CHECK(ds_exists(fid, "barcode_idx"),  "/barcode_idx dataset exists");
    CHECK(ds_exists(fid, "count"),        "/count dataset exists");
    CHECK(ds_exists(fid, "feature_idx"),  "/feature_idx dataset exists");
    CHECK(ds_exists(fid, "gem_group"),    "/gem_group dataset exists");
    CHECK(ds_exists(fid, "library_idx"),  "/library_idx dataset exists");
    CHECK(ds_exists(fid, "umi"),          "/umi dataset exists");
    CHECK(ds_exists(fid, "umi_type"),     "/umi_type dataset exists");
    CHECK(ds_exists(fid, "metrics_json"), "/metrics_json dataset exists");
    CHECK(ds_exists(fid, "barcodes"),     "/barcodes dataset exists");
    CHECK(ds_exists(fid, "feature_ref"),  "/feature_ref group exists");

    // feature_ref children
    hid_t fg = H5Gopen2(fid, "feature_ref", H5P_DEFAULT);
    CHECK(fg >= 0, "/feature_ref group opens");
    if (fg >= 0) {
        CHECK(ds_exists(fg, "name"),         "/feature_ref/name exists");
        CHECK(ds_exists(fg, "id"),           "/feature_ref/id exists");
        CHECK(ds_exists(fg, "feature_type"), "/feature_ref/feature_type exists");
        H5Gclose(fg);
    }

    H5Fclose(fid);
    fs::remove(path);
}

// T3: molecule data round-trip
static void test_molecule_roundtrip() {
    std::cout << "\n=== T3: Molecule data round-trip ===\n";
    const std::string path = tmp_path("test_molinfo_rt.h5");
    fs::remove(path);

    // Two distinct barcodes "ACGTACGTACGTACGT" and "TTTTTTTTTTTTTTTT"
    // Use the encode_bus_barcode helper from bus_writer.h
    const std::string bc0_str = "ACGTACGTACGTACGT";
    const std::string bc1_str = "TTTTTTTTTTTTTTTT";
    uint64_t bc0 = encode_bus_barcode(bc0_str);
    uint64_t bc1 = encode_bus_barcode(bc1_str);

    MoleculeInfoWriter w;
    w.open({path, 16, 12});

    MoleculeRecord r0; r0.barcode = bc0; r0.feature_idx = 0; r0.umi = 1; r0.count = 2;
    MoleculeRecord r1; r1.barcode = bc1; r1.feature_idx = 1; r1.umi = 2; r1.count = 3;
    // Add in reverse-barcode order to test sorting
    w.add_molecule(r1);
    w.add_molecule(r0);

    std::vector<std::string> gnames = {"GeneA", "GeneB"};
    std::vector<std::string> gids   = {"ENSG00000000001", "ENSG00000000002"};
    std::vector<std::string> bcs    = {bc0_str, bc1_str};
    w.finalize(gnames, gids, bcs);

    CHECK(w.molecules_written() == 2, "molecules_written() == 2 (before finalize wipes)");

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "file opens");

    auto barcode_ds  = read_int64_ds(fid, "barcode");
    auto feat_ds     = read_int32_ds(fid, "feature_idx");
    auto count_ds    = read_int32_ds(fid, "count");
    auto gem_ds      = read_int32_ds(fid, "gem_group");
    auto lib_ds      = read_int32_ds(fid, "library_idx");

    CHECK(barcode_ds.size() == 2, "barcode dataset length == 2");
    CHECK(feat_ds.size()    == 2, "feature_idx dataset length == 2");
    CHECK(count_ds.size()   == 2, "count dataset length == 2");

    // After sorting by uint64 barcode value, smaller barcode comes first.
    // Read back as int64 — cast to uint64 for comparison since TTTT has MSB=1.
    CHECK(static_cast<uint64_t>(barcode_ds[0]) <= static_cast<uint64_t>(barcode_ds[1]),
          "barcodes sorted ascending (uint64 comparison)");

    bool bc0_first = (static_cast<uint64_t>(barcode_ds[0]) == bc0);
    CHECK(bc0_first, "ACGT... barcode first (smaller 2-bit value)");

    CHECK(gem_ds[0] == 1 && gem_ds[1] == 1,   "gem_group == 1 for all");
    CHECK(lib_ds[0] == 0 && lib_ds[1] == 0,   "library_idx == 0 for all");

    H5Fclose(fid);
    fs::remove(path);
}

// T4: feature reference round-trip
static void test_feature_ref() {
    std::cout << "\n=== T4: Feature reference ===\n";
    const std::string path = tmp_path("test_molinfo_feat.h5");
    fs::remove(path);

    MoleculeInfoWriter w;
    w.open({path, 16, 12});

    std::vector<std::string> gnames = {"TP53", "BRCA1", "ACTB"};
    std::vector<std::string> gids   = {"ENSG00000141510", "ENSG00000012048", "ENSG00000075624"};
    std::vector<std::string> bcs    = {"ACGTACGTACGTACGT"};
    w.finalize(gnames, gids, bcs);

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "file opens");

    hid_t fg  = H5Gopen2(fid, "feature_ref", H5P_DEFAULT);
    CHECK(fg >= 0, "/feature_ref opens");
    if (fg >= 0) {
        auto names = read_str_ds(fg, "name");
        auto ids   = read_str_ds(fg, "id");
        auto types = read_str_ds(fg, "feature_type");

        CHECK(names.size() == 3, "feature_ref/name has 3 entries");
        if (names.size() == 3) {
            CHECK(names[0] == "TP53",  "name[0] == TP53");
            CHECK(names[1] == "BRCA1", "name[1] == BRCA1");
            CHECK(names[2] == "ACTB",  "name[2] == ACTB");
        }
        CHECK(ids.size() == 3, "feature_ref/id has 3 entries");
        if (ids.size() == 3) {
            CHECK(ids[0] == "ENSG00000141510", "id[0] == ENSG00000141510");
        }
        CHECK(types.size() == 3, "feature_ref/feature_type has 3 entries");
        if (!types.empty())
            CHECK(types[0] == "Gene Expression", "feature_type[0] == 'Gene Expression'");

        H5Gclose(fg);
    }

    H5Fclose(fid);
    fs::remove(path);
}

// T5: barcodes dataset
static void test_barcodes_dataset() {
    std::cout << "\n=== T5: Barcodes dataset ===\n";
    const std::string path = tmp_path("test_molinfo_bc.h5");
    fs::remove(path);

    std::vector<std::string> bcs = {
        "ACGTACGTACGTACGT",
        "CCCCCCCCCCCCCCCC",
        "TTTTTTTTTTTTTTTT"
    };

    MoleculeInfoWriter w;
    w.open({path, 16, 12});
    w.finalize({}, {}, bcs);

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "file opens");

    auto read_bcs = read_str_ds(fid, "barcodes");
    CHECK(read_bcs.size() == 3, "/barcodes has 3 entries");
    if (read_bcs.size() == 3) {
        CHECK(read_bcs[0] == bcs[0], "barcodes[0] round-trip");
        CHECK(read_bcs[1] == bcs[1], "barcodes[1] round-trip");
        CHECK(read_bcs[2] == bcs[2], "barcodes[2] round-trip");
    }

    H5Fclose(fid);
    fs::remove(path);
}

// T6: metrics_json dataset
static void test_metrics_json() {
    std::cout << "\n=== T6: metrics_json ===\n";
    const std::string path = tmp_path("test_molinfo_meta.h5");
    fs::remove(path);

    const std::string json_str = R"({"total_reads":1000,"mapped":950})";

    MoleculeInfoWriter w;
    w.open({path, 16, 12});
    w.finalize({}, {}, {}, json_str);

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "file opens");

    std::string rb = read_scalar_str_ds(fid, "metrics_json");
    CHECK(rb == json_str, "metrics_json round-trips correctly");

    H5Fclose(fid);
    fs::remove(path);
}

// T7: empty molecules (no records written)
static void test_empty_molecules() {
    std::cout << "\n=== T7: Empty molecules ===\n";
    const std::string path = tmp_path("test_molinfo_empty.h5");
    fs::remove(path);

    MoleculeInfoWriter w;
    w.open({path, 16, 12});
    CHECK(w.molecules_written() == 0, "molecules_written() == 0 before finalize");
    bool ok = w.finalize({"GeneA"}, {"ENSG00000000001"}, {"ACGTACGTACGTACGT"});
    CHECK(ok, "finalize() succeeds with 0 molecules");
    CHECK(fs::exists(path), "file created even with 0 molecules");

    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(fid >= 0, "empty-molecule file opens");

    auto bds = read_int64_ds(fid, "barcode");
    CHECK(bds.size() == 0, "/barcode is empty (length 0)");

    H5Fclose(fid);
    fs::remove(path);
}

// T8: molecules_written tracking
static void test_molecule_count() {
    std::cout << "\n=== T8: molecules_written() tracking ===\n";

    MoleculeInfoWriter w;
    w.open({"/tmp/test_molinfo_cnt.h5", 16, 12});

    CHECK(w.molecules_written() == 0, "initial count == 0");

    MoleculeRecord r{};
    w.add_molecule(r); w.add_molecule(r); w.add_molecule(r);
    CHECK(w.molecules_written() == 3, "count == 3 after 3 add_molecule calls");

    w.close();
    fs::remove("/tmp/test_molinfo_cnt.h5");
}

// ── main ──────────────────────────────────────────────────────────────────────

int main() {
    test_file_creation();
    test_hdf5_structure();
    test_molecule_roundtrip();
    test_feature_ref();
    test_barcodes_dataset();
    test_metrics_json();
    test_empty_molecules();
    test_molecule_count();

    std::cout << "\n──────────────────────────\n";
    std::cout << "PASSED: " << n_pass << "\n";
    std::cout << "FAILED: " << n_fail << "\n";
    if (n_fail > 0) {
        std::cerr << "[test_molecule_info] " << n_fail << " assertion(s) FAILED\n";
        return 1;
    }
    return 0;
}
