// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: loom_writer.h
// Loom v3 format export (loompy.org specification).
// Uses HDF5 C API directly — header-only, no HighFive or H5Cpp dependency.
//
// LAYOUT (Loom v3.0.0):
//   / [attrs: LOOM_SPEC_VERSION="3.0.0", CreationDate=ISO-8601]
//   ├── matrix          [n_genes × n_cells] int32, chunked+shuffle+gzip=4
//   ├── row_attrs/
//   │   ├── Gene        [n_genes] VL string  (gene names / symbols)
//   │   └── Accession   [n_genes] VL string  (Ensembl gene IDs)
//   ├── col_attrs/
//   │   └── CellID      [n_cells] VL string  (cell barcodes)
//   ├── col_graphs/     (empty group; required by spec)
//   ├── row_graphs/     (empty group; required by spec)
//   └── layers/         (optional; named matrices, same [n_genes × n_cells] shape)
//       ├── spliced     [n_genes × n_cells] int32
//       └── unspliced   [n_genes × n_cells] int32
//
// Input matrix is in GeneCSC layout (genes × cells):
//   indptr [n_cells+1], indices [nnz] = gene row indices, data [nnz] = uint32_t counts.
// Written to /matrix as a row-major [n_genes, n_cells] HDF5 dataset.
// Column-by-column hyperslab writes keep peak RAM at O(n_genes × chunk_cols).
//
// Compatible downstream tools: loompy, velocyto, scVelo, linnarssonlab/loomR.

#include <hdf5.h>
#include <cstdint>
#include <cstring>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "h5_util.h"

namespace singlet {

// ── Public API ───────────────────────────────────────────────────────────────

struct LoomWriteConfig {
    std::string filepath;

    // Main matrix in GeneCSC layout (genes × cells):
    //   indptr  : length n_cells + 1
    //   indices : gene row indices, length nnz
    //   data    : uint32_t count values, length nnz
    const int32_t*  indptr  = nullptr;
    const int32_t*  indices = nullptr;
    const uint32_t* data    = nullptr;
    uint64_t        nnz     = 0;
    uint32_t        n_genes = 0;
    uint32_t        n_cells = 0;

    const std::vector<std::string>* gene_names    = nullptr;  // /row_attrs/Gene
    const std::vector<std::string>* gene_ids       = nullptr;  // /row_attrs/Accession
    const std::vector<std::string>* cell_barcodes  = nullptr;  // /col_attrs/CellID

    /// Optional additional matrices (same [n_genes × n_cells] shape as /matrix).
    struct Layer {
        std::string     name;     // e.g. "spliced", "unspliced"
        const int32_t*  indptr  = nullptr;
        const int32_t*  indices = nullptr;
        const uint32_t* data    = nullptr;
        uint64_t        nnz     = 0;
    };
    std::vector<Layer> layers;
};

/// Write a .loom file. Returns true on success; on failure logs to stderr and
/// removes the partial output file.
inline bool write_loom(const LoomWriteConfig& cfg);

// ── Implementation ───────────────────────────────────────────────────────────

namespace loom_detail {

// Shared HDF5 boilerplate — canonical definitions live in h5_util.h.
using ::singlet::pileup::h5::make_vlen_str_type;
using ::singlet::pileup::h5::write_str_attr;
// loom historically spells its VL-string dataset helper write_str_dataset.
inline bool write_str_dataset(hid_t loc, const char* name,
                              const std::vector<std::string>& values) {
    return ::singlet::pileup::h5::write_vlen_str_dataset(loc, name, values);
}

/// Write a dense [n_genes × n_cells] int32 HDF5 dataset named `name` under
/// `loc` from a GeneCSC sparse matrix. Uses column-by-column hyperslab writes
/// (batch_size = chunk_cols) to bound peak memory at ~8 MB for typical genomes.
inline bool write_dense_matrix(hid_t loc, const char* name,
                                const int32_t* indptr, const int32_t* indices,
                                const uint32_t* data,
                                uint32_t n_genes, uint32_t n_cells) {
    if (n_genes == 0 || n_cells == 0) return false;

    // Chunk: 64×64 (loompy default — good for both row and column access)
    hsize_t chunk_rows = static_cast<hsize_t>(std::min(n_genes, 64u));
    hsize_t chunk_cols = static_cast<hsize_t>(std::min(n_cells,  64u));
    uint32_t batch     = static_cast<uint32_t>(chunk_cols);

    // Create [n_genes, n_cells] dataset with chunking, SHUFFLE + DEFLATE-4
    hsize_t dims[2] = { static_cast<hsize_t>(n_genes),
                        static_cast<hsize_t>(n_cells) };
    hid_t fspace = H5Screate_simple(2, dims, nullptr);
    hid_t dcpl   = H5Pcreate(H5P_DATASET_CREATE);
    hsize_t cdims[2] = { chunk_rows, chunk_cols };
    H5Pset_chunk(dcpl, 2, cdims);
    H5Pset_shuffle(dcpl);
    H5Pset_deflate(dcpl, 4);
    static const int32_t kFill = 0;
    H5Pset_fill_value(dcpl, H5T_NATIVE_INT32, &kFill);

    hid_t ds = H5Dcreate2(loc, name, H5T_NATIVE_INT32, fspace,
                           H5P_DEFAULT, dcpl, H5P_DEFAULT);
    H5Pclose(dcpl);
    H5Sclose(fspace);
    if (ds < 0) return false;

    // Dense buffer: [n_genes × batch] in row-major order
    // Buffer element [g, j] = buf[g * batch + j]  (j = cell index within batch)
    std::vector<int32_t> buf(static_cast<size_t>(n_genes) * batch, 0);

    bool ok = true;
    for (uint32_t col_start = 0; col_start < n_cells && ok; col_start += batch) {
        uint32_t col_end   = std::min(col_start + batch, n_cells);
        uint32_t this_batch = col_end - col_start;

        // Zero buffer (only the active [n_genes × this_batch] portion)
        std::fill(buf.begin(),
                  buf.begin() + static_cast<ptrdiff_t>(n_genes) * this_batch,
                  0);

        // Expand sparse columns into dense [n_genes × this_batch] buffer
        for (uint32_t j = col_start; j < col_end; ++j) {
            uint32_t local_j = j - col_start;
            for (int32_t k = indptr[j]; k < indptr[j + 1]; ++k) {
                uint32_t g = static_cast<uint32_t>(indices[k]);
                // Row-major in a [n_genes × this_batch] virtual block:
                //   element[g][local_j] = buf[g * this_batch + local_j]
                buf[static_cast<size_t>(g) * this_batch + local_j] =
                    static_cast<int32_t>(data[k]);
            }
        }

        // Write [n_genes, this_batch] hyperslab at [0, col_start] in file
        hsize_t mem_dims[2] = { static_cast<hsize_t>(n_genes),
                                 static_cast<hsize_t>(this_batch) };
        hid_t mspace = H5Screate_simple(2, mem_dims, nullptr);
        hid_t fsel   = H5Dget_space(ds);
        hsize_t hstart[2] = { 0, static_cast<hsize_t>(col_start) };
        hsize_t hcount[2] = { static_cast<hsize_t>(n_genes),
                               static_cast<hsize_t>(this_batch) };
        H5Sselect_hyperslab(fsel, H5S_SELECT_SET,
                            hstart, nullptr, hcount, nullptr);
        ok = (H5Dwrite(ds, H5T_NATIVE_INT32, mspace, fsel,
                       H5P_DEFAULT, buf.data()) >= 0);
        H5Sclose(mspace);
        H5Sclose(fsel);
    }

    H5Dclose(ds);
    return ok;
}

}  // namespace loom_detail

// ── write_loom ───────────────────────────────────────────────────────────────

inline bool write_loom(const LoomWriteConfig& cfg) {
    if (!cfg.indptr || !cfg.indices || !cfg.data) return false;
    if (cfg.n_genes == 0 || cfg.n_cells == 0)    return false;

    hid_t file = H5Fcreate(cfg.filepath.c_str(), H5F_ACC_TRUNC,
                            H5P_DEFAULT, H5P_DEFAULT);
    if (file < 0) {
        std::cerr << "[loom] ERROR: cannot create \"" << cfg.filepath << "\"\n";
        return false;
    }

    // Helper: close+remove on error
    auto fail = [&]() -> bool {
        H5Fclose(file);
        std::remove(cfg.filepath.c_str());
        return false;
    };

    // ── Root attributes ──────────────────────────────────────────────────
    loom_detail::write_str_attr(file, "LOOM_SPEC_VERSION", "3.0.0");
    {
        time_t t = time(nullptr);
        char buf[32];
        std::strftime(buf, sizeof(buf), "%Y-%m-%dT%H:%M:%S", std::gmtime(&t));
        loom_detail::write_str_attr(file, "CreationDate", buf);
    }

    // ── Required empty groups ────────────────────────────────────────────
    {
        hid_t g = H5Gcreate2(file, "row_graphs",
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g >= 0) H5Gclose(g);
    }
    {
        hid_t g = H5Gcreate2(file, "col_graphs",
                              H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (g >= 0) H5Gclose(g);
    }

    // ── /row_attrs/ ─────────────────────────────────────────────────────
    {
        hid_t grp = H5Gcreate2(file, "row_attrs",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0) return fail();

        // Gene names (symbols)
        {
            std::vector<std::string> names;
            if (cfg.gene_names && cfg.gene_names->size() == cfg.n_genes) {
                names = *cfg.gene_names;
            } else {
                names.assign(cfg.n_genes, "");
            }
            loom_detail::write_str_dataset(grp, "Gene", names);
        }

        // Ensembl accession IDs
        {
            std::vector<std::string> ids;
            if (cfg.gene_ids && cfg.gene_ids->size() == cfg.n_genes) {
                ids = *cfg.gene_ids;
            } else {
                ids.assign(cfg.n_genes, "");
            }
            loom_detail::write_str_dataset(grp, "Accession", ids);
        }

        H5Gclose(grp);
    }

    // ── /col_attrs/ ─────────────────────────────────────────────────────
    {
        hid_t grp = H5Gcreate2(file, "col_attrs",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0) return fail();

        {
            std::vector<std::string> ids;
            if (cfg.cell_barcodes && cfg.cell_barcodes->size() == cfg.n_cells) {
                ids = *cfg.cell_barcodes;
            } else {
                ids.assign(cfg.n_cells, "");
            }
            loom_detail::write_str_dataset(grp, "CellID", ids);
        }

        H5Gclose(grp);
    }

    // ── /matrix ─────────────────────────────────────────────────────────
    if (!loom_detail::write_dense_matrix(file, "matrix",
                                         cfg.indptr, cfg.indices, cfg.data,
                                         cfg.n_genes, cfg.n_cells)) {
        std::cerr << "[loom] ERROR: failed to write /matrix\n";
        return fail();
    }

    // ── /layers/ ────────────────────────────────────────────────────────
    if (!cfg.layers.empty()) {
        hid_t grp = H5Gcreate2(file, "layers",
                                H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        if (grp < 0) return fail();

        for (const auto& lyr : cfg.layers) {
            if (!lyr.indptr || !lyr.indices || !lyr.data) continue;
            if (!loom_detail::write_dense_matrix(grp, lyr.name.c_str(),
                                                 lyr.indptr, lyr.indices, lyr.data,
                                                 cfg.n_genes, cfg.n_cells)) {
                std::cerr << "[loom] WARNING: failed to write /layers/"
                          << lyr.name << "\n";
                // Continue — partial layers are still useful
            }
        }

        H5Gclose(grp);
    }

    H5Fclose(file);
    return true;
}

}  // namespace singlet
