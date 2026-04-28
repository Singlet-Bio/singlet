// test_loom_writer.cpp
// Tests for G-LOOM: Loom v3 export (loom_writer.h)
// Run: ./test_loom_writer
// All verification uses the HDF5 C API to read back the written file.

#include <cassert>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <hdf5.h>
#include "singlet-pileup/loom_writer.h"

using namespace singlet;

// ── Counters ─────────────────────────────────────────────────────────────────
static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, msg) \
    do { \
        if (!(cond)) { \
            std::cerr << "FAIL [" << __LINE__ << "]: " << (msg) << "\n"; \
            ++n_fail; \
        } else { \
            ++n_pass; \
        } \
    } while (0)

// ── HDF5 read-back helpers ───────────────────────────────────────────────────

static bool group_exists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

/// Read a scalar VL-string attribute. Returns "<missing>" if not found.
static std::string read_str_attr(hid_t loc, const char* name) {
    if (H5Aexists(loc, name) <= 0) return "<missing>";
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return "<open_fail>";
    hid_t atype = H5Aget_type(attr);
    char* val = nullptr;
    H5Aread(attr, atype, &val);
    std::string result = (val ? val : "");
    H5free_memory(val);
    H5Tclose(atype);
    H5Aclose(attr);
    return result;
}

/// Read a single VL-string element at index `idx` from a 1-D string dataset.
static std::string read_str_elem(hid_t loc, const char* name, hsize_t idx) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return "<ds_open_fail>";
    hid_t ftype  = H5Dget_type(ds);
    hid_t fspace = H5Dget_space(ds);

    hsize_t start = idx, count = 1;
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET,
                        &start, nullptr, &count, nullptr);
    hsize_t one = 1;
    hid_t mspace = H5Screate_simple(1, &one, nullptr);

    char* buf = nullptr;
    H5Dread(ds, ftype, mspace, fspace, H5P_DEFAULT, &buf);
    std::string result = (buf ? buf : "");
    H5Treclaim(ftype, mspace, H5P_DEFAULT, &buf);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Tclose(ftype);
    H5Dclose(ds);
    return result;
}

/// Read a single int32 value from a 2-D dataset at [row, col].
static int32_t read_int32_2d(hid_t loc, const char* name,
                              hsize_t row, hsize_t col) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return -999999;

    hid_t fspace = H5Dget_space(ds);
    hsize_t start[2] = { row, col };
    hsize_t count[2] = { 1, 1 };
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET,
                        start, nullptr, count, nullptr);
    hid_t mspace = H5Screate(H5S_SCALAR);

    int32_t val = -999999;
    H5Dread(ds, H5T_NATIVE_INT32, mspace, fspace, H5P_DEFAULT, &val);

    H5Sclose(mspace);
    H5Sclose(fspace);
    H5Dclose(ds);
    return val;
}

/// Read 2-D dataset dimensions. Returns {0,0} on failure.
static std::pair<hsize_t, hsize_t> read_2d_dims(hid_t loc, const char* name) {
    hid_t ds = H5Dopen2(loc, name, H5P_DEFAULT);
    if (ds < 0) return {0, 0};
    hid_t sp = H5Dget_space(ds);
    hsize_t dims[2] = {0, 0};
    H5Sget_simple_extent_dims(sp, dims, nullptr);
    H5Sclose(sp);
    H5Dclose(ds);
    return {dims[0], dims[1]};
}

// ── Test data ────────────────────────────────────────────────────────────────
//
// 5 genes × 3 cells
//   Cell 0: gene0=10, gene2=5
//   Cell 1: gene1=3,  gene3=7,  gene4=1
//   Cell 2: gene0=2
//
// Spliced layer:
//   Cell 0: gene0=8, gene2=4
//   Cell 1: gene1=2
//   Cell 2: (empty)

static const int32_t  kIndptr[]  = { 0, 2, 5, 6 };
static const int32_t  kIndices[] = { 0, 2, 1, 3, 4, 0 };
static const uint32_t kData[]    = { 10, 5, 3, 7, 1, 2 };

static const int32_t  kSplIp[]  = { 0, 2, 3, 3 };
static const int32_t  kSplIdx[] = { 0, 2, 1 };
static const uint32_t kSplD[]   = { 8, 4, 2 };

static const std::vector<std::string> kGeneNames =
    { "GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E" };
static const std::vector<std::string> kGeneIds =
    { "ENSG0001", "ENSG0002", "ENSG0003", "ENSG0004", "ENSG0005" };
static const std::vector<std::string> kBarcodes =
    { "AAACCCAAG-1", "AAACCCAAT-1", "AAACCCAAU-1" };

// ── Tests ────────────────────────────────────────────────────────────────────

static void test_basic_write_and_verify() {
    const std::string path = "/tmp/test_loom_basic.loom";

    // Build config with spliced layer
    LoomWriteConfig cfg;
    cfg.filepath      = path;
    cfg.indptr        = kIndptr;
    cfg.indices       = kIndices;
    cfg.data          = kData;
    cfg.nnz           = 6;
    cfg.n_genes       = 5;
    cfg.n_cells       = 3;
    cfg.gene_names    = &kGeneNames;
    cfg.gene_ids      = &kGeneIds;
    cfg.cell_barcodes = &kBarcodes;

    LoomWriteConfig::Layer spl;
    spl.name    = "spliced";
    spl.indptr  = kSplIp;
    spl.indices = kSplIdx;
    spl.data    = kSplD;
    spl.nnz     = 3;
    cfg.layers.push_back(spl);

    // ── assertion 1: write succeeds
    CHECK(write_loom(cfg), "write_loom basic returned false");

    // ── Open file
    hid_t f = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    CHECK(f >= 0, "cannot open written loom file");
    if (f < 0) { std::remove(path.c_str()); return; }

    // ── assertion 2,3: LOOM_SPEC_VERSION, CreationDate attributes
    CHECK(read_str_attr(f, "LOOM_SPEC_VERSION") == "3.0.0",
          "LOOM_SPEC_VERSION != '3.0.0'");
    CHECK(!read_str_attr(f, "CreationDate").empty(),
          "CreationDate attribute missing or empty");

    // ── assertion 4,5: /matrix dimensions [5, 3]
    auto [mr, mc] = read_2d_dims(f, "matrix");
    CHECK(mr == 5, "/matrix n_genes (rows) != 5");
    CHECK(mc == 3, "/matrix n_cells (cols) != 3");

    // ── assertions 6–11: /matrix values
    CHECK(read_int32_2d(f, "matrix", 0, 0) == 10, "/matrix[gene0,cell0] != 10");
    CHECK(read_int32_2d(f, "matrix", 2, 0) ==  5, "/matrix[gene2,cell0] != 5");
    CHECK(read_int32_2d(f, "matrix", 1, 1) ==  3, "/matrix[gene1,cell1] != 3");
    CHECK(read_int32_2d(f, "matrix", 3, 1) ==  7, "/matrix[gene3,cell1] != 7");
    CHECK(read_int32_2d(f, "matrix", 4, 1) ==  1, "/matrix[gene4,cell1] != 1");
    CHECK(read_int32_2d(f, "matrix", 0, 2) ==  2, "/matrix[gene0,cell2] != 2");

    // ── assertion 12: zero-fill for absent entries
    CHECK(read_int32_2d(f, "matrix", 1, 2) ==  0,
          "/matrix[gene1,cell2] should be 0 (zero-filled)");

    // ── assertion 13,14: /row_attrs group + Gene strings
    CHECK(group_exists(f, "row_attrs"), "/row_attrs group missing");
    {
        hid_t grp = H5Gopen2(f, "row_attrs", H5P_DEFAULT);
        if (grp >= 0) {
            // assertion 15,16
            CHECK(read_str_elem(grp, "Gene", 0) == "GENE_A",
                  "/row_attrs/Gene[0] != 'GENE_A'");
            CHECK(read_str_elem(grp, "Gene", 4) == "GENE_E",
                  "/row_attrs/Gene[4] != 'GENE_E'");
            // assertion 17
            CHECK(read_str_elem(grp, "Accession", 1) == "ENSG0002",
                  "/row_attrs/Accession[1] != 'ENSG0002'");
            // assertion 18
            CHECK(read_str_elem(grp, "Accession", 4) == "ENSG0005",
                  "/row_attrs/Accession[4] != 'ENSG0005'");
            H5Gclose(grp);
        }
    }

    // ── assertion 19,20: /col_attrs group + CellID strings
    CHECK(group_exists(f, "col_attrs"), "/col_attrs group missing");
    {
        hid_t grp = H5Gopen2(f, "col_attrs", H5P_DEFAULT);
        if (grp >= 0) {
            // assertions 21,22
            CHECK(read_str_elem(grp, "CellID", 0) == "AAACCCAAG-1",
                  "/col_attrs/CellID[0] mismatch");
            CHECK(read_str_elem(grp, "CellID", 2) == "AAACCCAAU-1",
                  "/col_attrs/CellID[2] mismatch");
            H5Gclose(grp);
        }
    }

    // ── assertions 23,24: required empty groups
    CHECK(group_exists(f, "row_graphs"), "/row_graphs group missing");
    CHECK(group_exists(f, "col_graphs"), "/col_graphs group missing");

    // ── assertions 25-30: /layers/spliced
    CHECK(group_exists(f, "layers"), "/layers group missing");
    {
        hid_t grp = H5Gopen2(f, "layers", H5P_DEFAULT);
        if (grp >= 0) {
            CHECK(group_exists(grp, "spliced"), "/layers/spliced missing");
            auto [sr, sc] = read_2d_dims(grp, "spliced");
            // assertions 26,27
            CHECK(sr == 5, "/layers/spliced rows != 5");
            CHECK(sc == 3, "/layers/spliced cols != 3");
            // assertions 28,29,30
            CHECK(read_int32_2d(grp, "spliced", 0, 0) == 8,
                  "/layers/spliced[gene0,cell0] != 8");
            CHECK(read_int32_2d(grp, "spliced", 2, 0) == 4,
                  "/layers/spliced[gene2,cell0] != 4");
            CHECK(read_int32_2d(grp, "spliced", 1, 1) == 2,
                  "/layers/spliced[gene1,cell1] != 2");
            H5Gclose(grp);
        }
    }

    H5Fclose(f);
    std::remove(path.c_str());
}

static void test_null_pointer_returns_false() {
    LoomWriteConfig cfg;
    cfg.filepath = "/tmp/test_loom_null.loom";
    cfg.indptr   = nullptr;  // null — must fail
    cfg.indices  = nullptr;
    cfg.data     = nullptr;
    cfg.n_genes  = 5;
    cfg.n_cells  = 3;
    // assertion 31
    CHECK(!write_loom(cfg), "write_loom should return false for null data pointers");
}

static void test_zero_cells_returns_false() {
    static const int32_t  ip[]  = { 0 };
    static const int32_t  idx[] = { 0 };
    static const uint32_t dat[] = { 1 };
    LoomWriteConfig cfg;
    cfg.filepath = "/tmp/test_loom_zero.loom";
    cfg.indptr   = ip;
    cfg.indices  = idx;
    cfg.data     = dat;
    cfg.n_genes  = 5;
    cfg.n_cells  = 0;  // zero cells — must fail
    // assertion 32
    CHECK(!write_loom(cfg), "write_loom should return false for n_cells=0");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    test_basic_write_and_verify();
    test_null_pointer_returns_false();
    test_zero_cells_returns_false();

    std::cout << "loom_writer: " << n_pass << " pass, " << n_fail << " fail\n";
    return (n_fail > 0) ? 1 : 0;
}
