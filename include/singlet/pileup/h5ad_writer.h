// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: h5ad_writer.h
// Export sparse count matrices as Scanpy-compatible AnnData .h5ad files.
// Uses HDF5 C API directly — header-only, no HighFive, no H5Cpp.
//
// LAYOUT  (AnnData 0.1.0 / dataframe 0.2.0 encoding):
//   /  [attrs: encoding-type="anndata", encoding-version="0.1.0"]
//   ├── X/         CSR cells × genes  (indptr int64, indices int32, data float32)
//   ├── obs/       DataFrame  — cell barcodes (_index)
//   ├── var/       DataFrame  — gene names (_index), optional gene_ids
//   ├── layers/    Named extra matrices (same CSR layout as X)
//   ├── obsm/      (empty)
//   ├── varm/      (empty)
//   ├── obsp/      (empty)
//   ├── varp/      (empty)
//   └── uns/singlet/  Key-value string metadata
//
// CSC→CSR EQUIVALENCE:
//   Our internal GeneCSC has columns=cells, rows=genes.
//   A CSC (genes×cells) IS the CSR of (cells×genes) — same arrays, transposed interpretation.
//   indptr[cell_j..cell_j+1] → non-zeros for cell j (gene indices in indices[]).

#include <hdf5.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "h5_util.h"

namespace singlet {

// ── Public API ──────────────────────────────────────────────────────────────

struct H5adWriteConfig {
    std::string filepath;

    // Matrix data in GeneCSC layout (genes × cells):
    //   columns = cells  →  indptr length = n_cells + 1
    //   rows    = genes  →  indices are gene indices
    // Written to X as CSR cells × genes (same arrays, reinterpreted).
    const int32_t* indptr;      // CSC column pointers, length n_cells + 1
    const int32_t* indices;     // CSC row (gene) indices, length nnz
    const void*    data;        // Non-zero values, length nnz
    uint64_t       nnz;
    uint32_t       n_genes;
    uint32_t       n_cells;
    bool           data_is_float = false;  // true → float32 input; false → uint32_t input

    const std::vector<std::string>* gene_names   = nullptr;  // var/_index
    const std::vector<std::string>* gene_ids      = nullptr;  // var/gene_ids (Ensembl)
    const std::vector<std::string>* cell_barcodes = nullptr;  // obs/_index

    struct Layer {
        std::string    name;          // e.g. "spliced", "unspliced"
        const int32_t* indptr;
        const int32_t* indices;
        const void*    data;
        uint64_t       nnz;
        bool           data_is_float = false;
    };
    std::vector<Layer> layers;

    std::map<std::string, std::string> metadata;  // written to uns/singlet/
};

// write_h5ad() — main export entry point.
// Returns true on success; on failure logs to stderr and removes partial file.
inline bool write_h5ad(const H5adWriteConfig& cfg);

// ── Implementation helpers ───────────────────────────────────────────────────

namespace h5ad_detail {

static_assert(sizeof(long long)  == 8, "long long must be 8 bytes for indptr int64");
static_assert(sizeof(int)        == 4, "int must be 4 bytes for indices int32");
static_assert(sizeof(float)      == 4, "float must be 4 bytes for data float32");

// Shared HDF5 boilerplate — canonical definitions live in h5_util.h.
using ::singlet::pileup::h5::make_vlen_str_type;
using ::singlet::pileup::h5::write_str_attr;
using ::singlet::pileup::h5::write_numeric_dataset;
// h5ad spells its VL-string dataset helper write_vlen_str_dataset.
using ::singlet::pileup::h5::write_vlen_str_dataset;

// Write a VL-string array attribute on loc (empty array is valid: n==0).
inline bool write_str_array_attr(hid_t loc, const char* name,
                                  const std::vector<std::string>& values) {
    hid_t stype = make_vlen_str_type();
    hsize_t n   = static_cast<hsize_t>(values.size());
    hid_t space = H5Screate_simple(1, &n, nullptr);
    hid_t attr  = H5Acreate2(loc, name, stype, space, H5P_DEFAULT, H5P_DEFAULT);
    bool ok     = (attr >= 0);
    if (ok) {
        if (n > 0) {
            std::vector<const char*> ptrs;
            ptrs.reserve(n);
            for (const auto& s : values) ptrs.push_back(s.c_str());
            ok = (H5Awrite(attr, stype, ptrs.data()) >= 0);
        }
        // n==0: attribute exists but holds no data; no write call needed
        H5Aclose(attr);
    }
    H5Sclose(space);
    H5Tclose(stype);
    return ok;
}

// Write a CSR matrix group already opened as grp.
// Our CSC (genes×cells) arrays map directly: indptr[n_cells+1], indices[nnz], data[nnz].
inline bool write_csr_group(hid_t grp,
                             const int32_t* indptr,
                             const int32_t* indices,
                             const void*    data,
                             uint64_t       nnz,
                             uint32_t       n_cells,
                             uint32_t       n_genes,
                             bool           data_is_float) {
    bool ok = true;

    // indptr: int32 source → int64 file (AnnData spec requires int64)
    {
        std::vector<int64_t> ptr64(n_cells + 1, 0);
        if (indptr) {
            for (uint32_t i = 0; i <= n_cells; ++i)
                ptr64[i] = static_cast<int64_t>(indptr[i]);
        }
        ok = write_numeric_dataset(grp, "indptr",
                                   static_cast<hsize_t>(n_cells + 1),
                                   H5T_NATIVE_LLONG, H5T_NATIVE_LLONG,
                                   ptr64.data()) && ok;
    }

    // indices: int32
    ok = write_numeric_dataset(grp, "indices",
                               static_cast<hsize_t>(nnz),
                               H5T_NATIVE_INT, H5T_NATIVE_INT,
                               (nnz > 0 ? static_cast<const void*>(indices) : nullptr)) && ok;

    // data: float32 — convert uint32→float if needed
    if (data_is_float) {
        ok = write_numeric_dataset(grp, "data",
                                   static_cast<hsize_t>(nnz),
                                   H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,
                                   (nnz > 0 ? data : nullptr)) && ok;
    } else {
        std::vector<float> fdata;
        const void* src_ptr = nullptr;
        if (nnz > 0 && data != nullptr) {
            fdata.resize(nnz);
            const uint32_t* src = static_cast<const uint32_t*>(data);
            for (uint64_t i = 0; i < nnz; ++i) fdata[i] = static_cast<float>(src[i]);
            src_ptr = fdata.data();
        }
        ok = write_numeric_dataset(grp, "data",
                                   static_cast<hsize_t>(nnz),
                                   H5T_NATIVE_FLOAT, H5T_NATIVE_FLOAT,
                                   src_ptr) && ok;
    }

    // Encoding attributes
    ok = write_str_attr(grp, "encoding-type",    "csr_matrix") && ok;
    ok = write_str_attr(grp, "encoding-version", "0.1.0")      && ok;

    // Shape: int64[2] = {n_cells, n_genes}
    {
        int64_t  shape[2]  = { static_cast<int64_t>(n_cells),
                                static_cast<int64_t>(n_genes) };
        hsize_t  sdims[1]  = { 2 };
        hid_t    sspace    = H5Screate_simple(1, sdims, nullptr);
        hid_t    sattr     = H5Acreate2(grp, "shape", H5T_NATIVE_LLONG,
                                         sspace, H5P_DEFAULT, H5P_DEFAULT);
        if (sattr >= 0) {
            ok = (H5Awrite(sattr, H5T_NATIVE_LLONG, shape) >= 0) && ok;
            H5Aclose(sattr);
        } else { ok = false; }
        H5Sclose(sspace);
    }

    return ok;
}

// Write obs or var DataFrame group under parent.
// index_strings: barcodes (obs) or gene names (var); nullptr → generate "0","1",...
// extra_col / extra_col_name: optional second dataset (e.g. gene_ids)
// column_order: value of the column-order array attribute
inline bool write_dataframe_group(hid_t parent,
                                   const char* group_name,
                                   const std::vector<std::string>* index_strings,
                                   const std::vector<std::string>* extra_col,
                                   const char* extra_col_name,
                                   uint32_t n_default,
                                   const std::vector<std::string>& column_order) {
    hid_t grp = H5Gcreate2(parent, group_name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (grp < 0) return false;
    bool ok = true;

    // _index dataset
    if (index_strings && !index_strings->empty()) {
        ok = write_vlen_str_dataset(grp, "_index", *index_strings) && ok;
    } else {
        std::vector<std::string> defaults;
        defaults.reserve(n_default);
        for (uint32_t i = 0; i < n_default; ++i) defaults.push_back(std::to_string(i));
        ok = write_vlen_str_dataset(grp, "_index", defaults) && ok;
    }

    // Optional extra column (gene_ids etc.)
    if (extra_col && extra_col_name && !extra_col->empty()) {
        ok = write_vlen_str_dataset(grp, extra_col_name, *extra_col) && ok;
    }

    // DataFrame attributes
    ok = write_str_attr(grp, "_index",           "_index")   && ok;
    ok = write_str_attr(grp, "encoding-type",    "dataframe") && ok;
    ok = write_str_attr(grp, "encoding-version", "0.2.0")    && ok;
    ok = write_str_array_attr(grp, "column-order", column_order) && ok;

    H5Gclose(grp);
    return ok;
}

}  // namespace h5ad_detail

// ── write_h5ad() implementation ──────────────────────────────────────────────

inline bool write_h5ad(const H5adWriteConfig& cfg) {
    // Silence HDF5 default error stack printing; we log our own messages.
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

    hid_t fid = H5Fcreate(cfg.filepath.c_str(), H5F_ACC_TRUNC,
                           H5P_DEFAULT, H5P_DEFAULT);
    if (fid < 0) {
        std::cerr << "[h5ad_writer] Cannot create: " << cfg.filepath << "\n";
        return false;
    }

    bool ok = true;

    // Root AnnData encoding attributes
    ok = h5ad_detail::write_str_attr(fid, "encoding-type",    "anndata") && ok;
    ok = h5ad_detail::write_str_attr(fid, "encoding-version", "0.1.0")   && ok;

    // X — primary count matrix (CSR cells × genes)
    {
        hid_t x_grp = H5Gcreate2(fid, "X", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (x_grp < 0) { ok = false; }
        else {
            ok = h5ad_detail::write_csr_group(x_grp,
                                               cfg.indptr, cfg.indices, cfg.data,
                                               cfg.nnz, cfg.n_cells, cfg.n_genes,
                                               cfg.data_is_float) && ok;
            H5Gclose(x_grp);
        }
    }

    // obs — cell barcode DataFrame
    {
        std::vector<std::string> obs_col_order;  // empty — no extra columns
        ok = h5ad_detail::write_dataframe_group(fid, "obs",
                                                 cfg.cell_barcodes,
                                                 nullptr, nullptr,
                                                 cfg.n_cells,
                                                 obs_col_order) && ok;
    }

    // var — gene name DataFrame
    {
        std::vector<std::string> var_col_order;
        if (cfg.gene_ids && !cfg.gene_ids->empty()) var_col_order.push_back("gene_ids");
        ok = h5ad_detail::write_dataframe_group(fid, "var",
                                                 cfg.gene_names,
                                                 cfg.gene_ids, "gene_ids",
                                                 cfg.n_genes,
                                                 var_col_order) && ok;
    }

    // Required empty groups (Scanpy checks for their existence)
    const char* empty_grp_names[] = {"obsm", "varm", "obsp", "varp"};
    for (const char* name : empty_grp_names) {
        hid_t grp = H5Gcreate2(fid, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp >= 0) H5Gclose(grp);
        else ok = false;
    }

    // layers/ — named extra matrices
    {
        hid_t l_root = H5Gcreate2(fid, "layers", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (l_root < 0) { ok = false; }
        else {
            for (const auto& layer : cfg.layers) {
                hid_t l_grp = H5Gcreate2(l_root, layer.name.c_str(),
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
                if (l_grp < 0) { ok = false; continue; }
                ok = h5ad_detail::write_csr_group(l_grp,
                                                   layer.indptr, layer.indices, layer.data,
                                                   layer.nnz, cfg.n_cells, cfg.n_genes,
                                                   layer.data_is_float) && ok;
                H5Gclose(l_grp);
            }
            H5Gclose(l_root);
        }
    }

    // uns/singlet/ — pipeline key-value metadata
    {
        hid_t uns_grp = H5Gcreate2(fid, "uns", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (uns_grp >= 0) {
            hid_t sg_grp = H5Gcreate2(uns_grp, "singlet",
                                       H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            if (sg_grp >= 0) {
                for (const auto& kv : cfg.metadata) {
                    h5ad_detail::write_str_attr(sg_grp,
                                                kv.first.c_str(),
                                                kv.second.c_str());
                }
                H5Gclose(sg_grp);
            }
            H5Gclose(uns_grp);
        }
    }

    H5Fclose(fid);

    if (!ok) {
        std::cerr << "[h5ad_writer] Write failed, removing partial file: "
                  << cfg.filepath << "\n";
        std::remove(cfg.filepath.c_str());
    }
    return ok;
}

}  // namespace singlet
