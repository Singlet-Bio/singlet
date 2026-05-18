// SPDX-License-Identifier: MIT
// test_h5ad_writer.cpp
// Tests for G-H5AD: AnnData .h5ad export (h5ad_writer.h)
// Run: ./test_h5ad_writer
// All tests use the HDF5 C API for both write (h5ad_writer) and read-back verification.

#include <cassert>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>
#include <map>

#include <hdf5.h>
#include "singlet/pileup/h5ad_writer.h"

using namespace singlet;

// ── Read-back helpers ────────────────────────────────────────────────────────

[[maybe_unused]] static bool group_exists(hid_t loc, const char* name) {
    return H5Lexists(loc, name, H5P_DEFAULT) > 0;
}

[[maybe_unused]] static std::string read_str_attr(hid_t loc, const char* name) {
    hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
    if (attr < 0) return "<missing>";
    hid_t atype = H5Aget_type(attr);
    char* val = nullptr;
    H5Aread(attr, atype, &val);
    std::string result = val ? val : "";
    H5free_memory(val);
    H5Tclose(atype);
    H5Aclose(attr);
    return result;
}

// Read a fixed 1-D int64 dataset into a vector
static std::vector<int64_t> read_int64_dataset(hid_t loc, const char* path) {
    hid_t dset = H5Dopen2(loc, path, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<int64_t> buf(n);
    if (n > 0)
        H5Dread(dset, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    return buf;
}

// Read a fixed 1-D int32 dataset into a vector
static std::vector<int32_t> read_int32_dataset(hid_t loc, const char* path) {
    hid_t dset = H5Dopen2(loc, path, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<int32_t> buf(n);
    if (n > 0)
        H5Dread(dset, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    return buf;
}

// Read a fixed 1-D float32 dataset into a vector
static std::vector<float> read_float32_dataset(hid_t loc, const char* path) {
    hid_t dset = H5Dopen2(loc, path, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<float> buf(n);
    if (n > 0)
        H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, buf.data());
    H5Sclose(space);
    H5Dclose(dset);
    return buf;
}

// Read a VL string dataset into a vector of strings
static std::vector<std::string> read_str_dataset(hid_t loc, const char* path) {
    hid_t dset = H5Dopen2(loc, path, H5P_DEFAULT);
    if (dset < 0) return {};
    hid_t ftype = H5Dget_type(dset);
    hid_t space = H5Dget_space(dset);
    hsize_t n = 0;
    H5Sget_simple_extent_dims(space, &n, nullptr);
    std::vector<char*> raw(n, nullptr);
    std::vector<std::string> result;
    if (n > 0) {
        H5Dread(dset, ftype, H5S_ALL, H5S_ALL, H5P_DEFAULT, raw.data());
        result.reserve(n);
        for (hsize_t i = 0; i < n; ++i) result.push_back(raw[i] ? raw[i] : "");
        H5Treclaim(ftype, space, H5P_DEFAULT, raw.data());
    }
    H5Sclose(space);
    H5Tclose(ftype);
    H5Dclose(dset);
    return result;
}

// Read shape attribute (int64[2]) from a group
static std::pair<int64_t,int64_t> read_shape_attr(hid_t loc) {
    hid_t attr  = H5Aopen(loc, "shape", H5P_DEFAULT);
    if (attr < 0) return {-1, -1};
    int64_t shape[2] = {-1, -1};
    H5Aread(attr, H5T_NATIVE_LLONG, shape);
    H5Aclose(attr);
    return {shape[0], shape[1]};
}

// ── Small 3-gene × 2-cell CSC matrix used across multiple tests ──────────────
//
//   Cell 0: gene 0 → 5,  gene 2 → 3
//   Cell 1: gene 1 → 7
//
//   CSC format  (genes as rows, cells as columns):
//     indptr  = [0, 2, 3]          (cell 0 has 2 nnz, cell 1 has 1 nnz)
//     indices = [0, 2, 1]          (gene indices for each nnz)
//     data    = [5, 3, 7]          (uint32 values)

static const uint32_t kSmallData[]    = {5, 3, 7};
static const int32_t  kSmallIndptr[]  = {0, 2, 3};
static const int32_t  kSmallIndices[] = {0, 2, 1};

static H5adWriteConfig make_small_cfg(const std::string& path) {
    static std::vector<std::string> genes   = {"GeneA", "GeneB", "GeneC"};
    static std::vector<std::string> gene_ids = {"ENSG01", "ENSG02", "ENSG03"};
    static std::vector<std::string> cells   = {"AAACCC", "TTTGGG"};

    H5adWriteConfig cfg;
    cfg.filepath      = path;
    cfg.indptr        = kSmallIndptr;
    cfg.indices       = kSmallIndices;
    cfg.data          = kSmallData;
    cfg.nnz           = 3;
    cfg.n_genes       = 3;
    cfg.n_cells       = 2;
    cfg.data_is_float = false;
    cfg.gene_names    = &genes;
    cfg.gene_ids      = &gene_ids;
    cfg.cell_barcodes = &cells;
    cfg.metadata["singlet_version"] = "0.3.0";
    cfg.metadata["organism"]         = "Homo sapiens";
    return cfg;
}

// ── Test 1: Basic write — returns true, file exists ──────────────────────────

static void test_basic_write() {
    const std::string path = "/tmp/test_h5ad_basic.h5ad";
    auto cfg = make_small_cfg(path);
    bool ok = write_h5ad(cfg);
    (void)ok;
    assert(ok && "write_h5ad returned false");                     // assertion 1

    FILE* f = fopen(path.c_str(), "rb");
    assert(f && "output .h5ad file does not exist");               // assertion 2
    fclose(f);
    std::remove(path.c_str());
    std::cout << "  test_basic_write: PASS\n";
}

// ── Test 2: HDF5 structural groups all present ───────────────────────────────

static void test_hdf5_structure() {
    const std::string path = "/tmp/test_h5ad_struct.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0 && "cannot open written .h5ad");               // assertion 3

    assert(group_exists(fid, "X")      && "X group missing");      // assertion 4
    assert(group_exists(fid, "obs")    && "obs group missing");     // assertion 5
    assert(group_exists(fid, "var")    && "var group missing");     // assertion 6
    assert(group_exists(fid, "obsm")   && "obsm group missing");    // assertion 7
    assert(group_exists(fid, "varm")   && "varm group missing");    // assertion 8
    assert(group_exists(fid, "obsp")   && "obsp group missing");    // assertion 9
    assert(group_exists(fid, "varp")   && "varp group missing");    // assertion 10
    assert(group_exists(fid, "uns")    && "uns group missing");     // assertion 11
    assert(group_exists(fid, "layers") && "layers group missing");  // assertion 12

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_hdf5_structure: PASS\n";
}

// ── Test 3: X matrix data values read back correctly ─────────────────────────

static void test_x_matrix_data() {
    const std::string path = "/tmp/test_h5ad_xdata.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    // indptr: int64 [0, 2, 3]
    auto indptr = read_int64_dataset(fid, "/X/indptr");
    assert(indptr.size() == 3 && "indptr length wrong");           // assertion 13
    assert(indptr[0] == 0 && indptr[1] == 2 && indptr[2] == 3 &&
           "indptr values wrong");                                  // assertion 14

    // indices: int32 [0, 2, 1]
    auto indices = read_int32_dataset(fid, "/X/indices");
    assert(indices.size() == 3 && "indices length wrong");         // assertion 15
    assert(indices[0] == 0 && indices[1] == 2 && indices[2] == 1 &&
           "indices values wrong");                                 // assertion 16

    // data: float32 [5.0, 3.0, 7.0]
    auto data = read_float32_dataset(fid, "/X/data");
    assert(data.size() == 3 && "data length wrong");               // assertion 17
    assert(data[0] == 5.0f && data[1] == 3.0f && data[2] == 7.0f &&
           "data values wrong");                                    // assertion 18

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_x_matrix_data: PASS\n";
}

// ── Test 4: Encoding attributes on root and X ────────────────────────────────

static void test_encoding_attributes() {
    const std::string path = "/tmp/test_h5ad_enc.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    // Root
    assert(read_str_attr(fid, "encoding-type")    == "anndata" &&
           "root encoding-type wrong");                            // assertion 19
    assert(read_str_attr(fid, "encoding-version") == "0.1.0"   &&
           "root encoding-version wrong");                         // assertion 20

    // X group
    hid_t x = H5Gopen2(fid, "X", H5P_DEFAULT);
    assert(x >= 0);
    assert(read_str_attr(x, "encoding-type")    == "csr_matrix" &&
           "X encoding-type wrong");                               // assertion 21
    assert(read_str_attr(x, "encoding-version") == "0.1.0"      &&
           "X encoding-version wrong");                            // assertion 22
    H5Gclose(x);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_encoding_attributes: PASS\n";
}

// ── Test 5: Shape attribute on X ─────────────────────────────────────────────

static void test_shape_attribute() {
    const std::string path = "/tmp/test_h5ad_shape.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    hid_t x = H5Gopen2(fid, "X", H5P_DEFAULT);
    assert(x >= 0);
    [[maybe_unused]] auto [n_cells, n_genes] = read_shape_attr(x);
    assert(n_cells == 2 && "shape n_cells wrong");                 // assertion 23
    assert(n_genes == 3 && "shape n_genes wrong");                 // assertion 24
    H5Gclose(x);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_shape_attribute: PASS\n";
}

// ── Test 6: obs/_index and var/_index string datasets ────────────────────────

static void test_string_datasets() {
    const std::string path = "/tmp/test_h5ad_str.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    auto barcodes = read_str_dataset(fid, "/obs/_index");
    assert(barcodes.size() == 2 && "obs/_index length wrong");     // assertion 25
    assert(barcodes[0] == "AAACCC" && barcodes[1] == "TTTGGG" &&
           "obs/_index values wrong");                             // assertion 26

    auto gene_names = read_str_dataset(fid, "/var/_index");
    assert(gene_names.size() == 3 && "var/_index length wrong");   // assertion 27
    assert(gene_names[0] == "GeneA" && gene_names[1] == "GeneB" &&
           gene_names[2] == "GeneC" && "var/_index values wrong"); // assertion 28

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_string_datasets: PASS\n";
}

// ── Test 7: var/gene_ids dataset present and correct when provided ────────────

static void test_gene_ids_present() {
    const std::string path = "/tmp/test_h5ad_geneids.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    assert(group_exists(fid, "/var/gene_ids") && "var/gene_ids missing");  // assertion 29
    auto ids = read_str_dataset(fid, "/var/gene_ids");
    assert(ids.size() == 3 && "var/gene_ids length wrong");               // assertion 30
    assert(ids[0] == "ENSG01" && ids[2] == "ENSG03" &&
           "var/gene_ids values wrong");                                   // assertion 31

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_gene_ids_present: PASS\n";
}

// ── Test 8: layers — write a layer and verify CSR structure ──────────────────

static void test_layers() {
    const std::string path = "/tmp/test_h5ad_layers.h5ad";

    static const uint32_t ldata[]    = {1, 2};
    static const int32_t  lindptr[]  = {0, 1, 2};  // one nnz per cell
    static const int32_t  lindices[] = {0, 1};

    auto cfg = make_small_cfg(path);
    H5adWriteConfig::Layer spliced;
    spliced.name           = "spliced";
    spliced.indptr         = lindptr;
    spliced.indices        = lindices;
    spliced.data           = ldata;
    spliced.nnz            = 2;
    spliced.data_is_float  = false;
    cfg.layers.push_back(spliced);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    assert(group_exists(fid, "/layers/spliced") &&
           "layers/spliced group missing");                         // assertion 32

    hid_t lg = H5Gopen2(fid, "/layers/spliced", H5P_DEFAULT);
    assert(lg >= 0);
    assert(read_str_attr(lg, "encoding-type") == "csr_matrix" &&
           "layer encoding-type wrong");                            // assertion 33

    auto layer_data = read_float32_dataset(fid, "/layers/spliced/data");
    assert(layer_data.size() == 2 && "layer data length wrong");   // assertion 34
    assert(layer_data[0] == 1.0f && layer_data[1] == 2.0f &&
           "layer data values wrong");                             // assertion 35

    H5Gclose(lg);
    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_layers: PASS\n";
}

// ── Test 9: uns/singlet/ metadata written as string attributes ──────────────

static void test_metadata() {
    const std::string path = "/tmp/test_h5ad_meta.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    assert(group_exists(fid, "/uns/singlet") && "uns/singlet missing"); // assertion 36

    hid_t sg = H5Gopen2(fid, "/uns/singlet", H5P_DEFAULT);
    assert(sg >= 0);
    assert(read_str_attr(sg, "singlet_version") == "0.3.0" &&
           "singlet_version attr wrong");                          // assertion 37
    assert(read_str_attr(sg, "organism") == "Homo sapiens" &&
           "organism attr wrong");                                  // assertion 38
    H5Gclose(sg);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_metadata: PASS\n";
}

// ── Test 10: Empty matrix (0 cells) — must not crash ─────────────────────────

static void test_empty_matrix() {
    const std::string path = "/tmp/test_h5ad_empty.h5ad";

    static const int32_t empty_indptr[]  = {0};  // length 1 (n_cells+1 = 0+1)
    static std::vector<std::string> genes = {"GeneX", "GeneY"};

    H5adWriteConfig cfg;
    cfg.filepath      = path;
    cfg.indptr        = empty_indptr;
    cfg.indices       = nullptr;
    cfg.data          = nullptr;
    cfg.nnz           = 0;
    cfg.n_genes       = 2;
    cfg.n_cells       = 0;
    cfg.data_is_float = false;
    cfg.gene_names    = &genes;

    bool ok = write_h5ad(cfg);
    (void)ok;
    assert(ok && "empty matrix write returned false");             // assertion 39

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0 && "empty matrix file cannot be opened");      // assertion 40

    hid_t x = H5Gopen2(fid, "X", H5P_DEFAULT);
    assert(x >= 0 && "empty matrix X group missing");              // assertion 41
    [[maybe_unused]] auto [n_cells, n_genes] = read_shape_attr(x);
    assert(n_cells == 0 && "empty: shape n_cells should be 0");    // assertion 42
    assert(n_genes == 2 && "empty: shape n_genes should be 2");    // assertion 43
    H5Gclose(x);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_empty_matrix: PASS\n";
}

// ── Test 11: Large matrix (1000 genes × 500 cells) write + read-back ─────────

static void test_large_matrix() {
    const std::string path = "/tmp/test_h5ad_large.h5ad";
    const uint32_t N_GENES = 1000;
    const uint32_t N_CELLS = 500;
    const uint32_t NNZ_PER_CELL = 50;  // sparse

    // Build CSC: each cell has 50 evenly-spaced gene hits
    std::vector<int32_t>  indptr(N_CELLS + 1);
    std::vector<int32_t>  indices;
    std::vector<uint32_t> data;
    indices.reserve(N_CELLS * NNZ_PER_CELL);
    data.reserve(N_CELLS * NNZ_PER_CELL);

    for (uint32_t c = 0; c <= N_CELLS; ++c) indptr[c] = c * NNZ_PER_CELL;
    for (uint32_t c = 0; c < N_CELLS; ++c) {
        for (uint32_t k = 0; k < NNZ_PER_CELL; ++k) {
            indices.push_back(static_cast<int32_t>((c * 7 + k * 19) % N_GENES));
            data.push_back(c + k + 1);
        }
    }

    std::vector<std::string> gene_names(N_GENES);
    std::vector<std::string> cell_names(N_CELLS);
    for (uint32_t i = 0; i < N_GENES; ++i) gene_names[i] = "G" + std::to_string(i);
    for (uint32_t i = 0; i < N_CELLS; ++i) cell_names[i] = "C" + std::to_string(i);

    H5adWriteConfig cfg;
    cfg.filepath      = path;
    cfg.indptr        = indptr.data();
    cfg.indices       = indices.data();
    cfg.data          = data.data();
    cfg.nnz           = static_cast<uint64_t>(N_CELLS) * NNZ_PER_CELL;
    cfg.n_genes       = N_GENES;
    cfg.n_cells       = N_CELLS;
    cfg.data_is_float = false;
    cfg.gene_names    = &gene_names;
    cfg.cell_barcodes = &cell_names;

    bool ok = write_h5ad(cfg);
    (void)ok;
    assert(ok && "large matrix write failed");                     // assertion 44

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0 && "large matrix file cannot be opened");      // assertion 45

    hid_t x = H5Gopen2(fid, "X", H5P_DEFAULT);
    assert(x >= 0);
    [[maybe_unused]] auto [nc, ng] = read_shape_attr(x);
    assert(nc == N_CELLS && ng == N_GENES && "large matrix shape wrong"); // assertion 46

    auto data_back = read_float32_dataset(fid, "/X/data");
    assert(data_back.size() == cfg.nnz && "large matrix nnz mismatch");  // assertion 47

    // Spot check: first value should be 0+0+1 = 1.0f
    assert(data_back[0] == 1.0f && "large matrix spot-check data[0]");   // assertion 48

    H5Gclose(x);
    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_large_matrix: PASS\n";
}

// ── Test 12: Float data input → float32 in HDF5 ──────────────────────────────

static void test_float_data() {
    const std::string path = "/tmp/test_h5ad_float.h5ad";

    static const float   fdata[]    = {1.5f, 2.5f, 3.5f};
    static const int32_t findptr[]  = {0, 2, 3};
    static const int32_t findices[] = {0, 2, 1};
    static std::vector<std::string> genes = {"GA", "GB", "GC"};
    static std::vector<std::string> cells = {"C0", "C1"};

    H5adWriteConfig cfg;
    cfg.filepath      = path;
    cfg.indptr        = findptr;
    cfg.indices       = findices;
    cfg.data          = fdata;
    cfg.nnz           = 3;
    cfg.n_genes       = 3;
    cfg.n_cells       = 2;
    cfg.data_is_float = true;
    cfg.gene_names    = &genes;
    cfg.cell_barcodes = &cells;

    bool ok = write_h5ad(cfg);
    (void)ok;
    assert(ok && "float data write failed");                       // assertion 49

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    auto data_back = read_float32_dataset(fid, "/X/data");
    assert(data_back.size() == 3 && "float data size wrong");      // assertion 50
    assert(data_back[0] == 1.5f && data_back[1] == 2.5f &&
           data_back[2] == 3.5f && "float data values wrong");     // assertion 51

    // Verify the HDF5 type stored is float (H5T_FLOAT class)
    hid_t dset = H5Dopen2(fid, "/X/data", H5P_DEFAULT);
    assert(dset >= 0);
    hid_t dtype = H5Dget_type(dset);
    assert(H5Tget_class(dtype) == H5T_FLOAT && "data stored type not float"); // assertion 52
    assert(H5Tget_size(dtype) == 4 && "data stored type not float32");        // assertion 53
    H5Tclose(dtype);
    H5Dclose(dset);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_float_data: PASS\n";
}

// ── Test 13: No gene_ids — var/gene_ids dataset must NOT exist ───────────────

static void test_no_gene_ids() {
    const std::string path = "/tmp/test_h5ad_nogeneids.h5ad";
    auto cfg = make_small_cfg(path);
    cfg.gene_ids = nullptr;  // remove gene_ids
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    int exists = H5Lexists(fid, "/var/gene_ids", H5P_DEFAULT);
    (void)exists;
    assert(exists == 0 && "var/gene_ids should not exist when gene_ids=nullptr"); // assertion 54

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_no_gene_ids: PASS\n";
}

// ── Test 14: obs and var DataFrame attrs ─────────────────────────────────────

static void test_dataframe_attributes() {
    const std::string path = "/tmp/test_h5ad_df.h5ad";
    auto cfg = make_small_cfg(path);
    write_h5ad(cfg);

    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    hid_t fid = H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    assert(fid >= 0);

    hid_t obs = H5Gopen2(fid, "obs", H5P_DEFAULT);
    assert(obs >= 0);
    assert(read_str_attr(obs, "encoding-type")    == "dataframe" &&
           "obs encoding-type wrong");                              // assertion 55
    assert(read_str_attr(obs, "encoding-version") == "0.2.0"     &&
           "obs encoding-version wrong");                           // assertion 56
    assert(read_str_attr(obs, "_index") == "_index" &&
           "obs _index attr wrong");                               // assertion 57
    H5Gclose(obs);

    hid_t var = H5Gopen2(fid, "var", H5P_DEFAULT);
    assert(var >= 0);
    assert(read_str_attr(var, "encoding-type") == "dataframe" &&
           "var encoding-type wrong");                              // assertion 58
    H5Gclose(var);

    H5Fclose(fid);
    std::remove(path.c_str());
    std::cout << "  test_dataframe_attributes: PASS\n";
}

// ── main ─────────────────────────────────────────────────────────────────────

int main() {
    // Silence HDF5 default error printing globally for all tests
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    std::cout << "[test_h5ad_writer] Running 14 tests...\n";
    test_basic_write();           // assertions 1-2
    test_hdf5_structure();        // assertions 3-12
    test_x_matrix_data();         // assertions 13-18
    test_encoding_attributes();   // assertions 19-22
    test_shape_attribute();       // assertions 23-24
    test_string_datasets();       // assertions 25-28
    test_gene_ids_present();      // assertions 29-31
    test_layers();                // assertions 32-35
    test_metadata();              // assertions 36-38
    test_empty_matrix();          // assertions 39-43
    test_large_matrix();          // assertions 44-48
    test_float_data();            // assertions 49-53
    test_no_gene_ids();           // assertion  54
    test_dataframe_attributes();  // assertions 55-58

    std::cout << "[test_h5ad_writer] All 58 assertions passed.\n";
    return 0;
}
