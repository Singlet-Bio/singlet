#pragma once
// singlet-pileup: mtx_writer.h
// Write sparse matrices in Matrix Market format for downstream conversion to .1pz

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>
#include <zlib.h>

#include "sparse_accumulator.h"

namespace singlet {

// Write a CSC matrix as Matrix Market (.mtx.gz) format — gzip-compressed for Scanpy/Seurat
template <typename T>
bool write_mtx(const std::string& path,
               uint32_t n_rows, uint32_t n_cols,
               const std::vector<int32_t>& indptr,
               const std::vector<int32_t>& indices,
               const std::vector<T>& data) {
    gzFile gz = gzopen(path.c_str(), "wb");
    if (!gz) {
        std::cerr << "[mtx_writer] Cannot open: " << path << "\n";
        return false;
    }
    gzbuffer(gz, 256 * 1024);

    // Header
    constexpr bool is_int = std::is_integral_v<T>;
    gzprintf(gz, "%%%%MatrixMarket matrix coordinate %s general\n",
             is_int ? "integer" : "real");
    gzprintf(gz, "%u %u %zu\n", n_rows, n_cols, data.size());

    // Entries (1-indexed, CSC: iterate columns)
    for (uint32_t j = 0; j < n_cols; ++j) {
        for (int32_t k = indptr[j]; k < indptr[j + 1]; ++k) {
            if constexpr (is_int) {
                gzprintf(gz, "%d %u %lld\n", indices[k] + 1, j + 1,
                         static_cast<long long>(data[k]));
            } else {
                gzprintf(gz, "%d %u %g\n", indices[k] + 1, j + 1,
                         static_cast<double>(data[k]));
            }
        }
    }

    gzclose(gz);
    return true;
}

// Write row names (features) to gzipped TSV
bool write_names(const std::string& path, const std::vector<std::string>& names) {
    gzFile gz = gzopen(path.c_str(), "wb");
    if (!gz) return false;
    for (const auto& name : names) {
        gzprintf(gz, "%s\n", name.c_str());
    }
    gzclose(gz);
    return true;
}

// Write CellRanger-compatible features.tsv.gz (3 columns: gene_id\tgene_name\tfeature_type)
inline bool write_features_10x(const std::string& path,
                               const std::vector<std::string>& gene_ids,
                               const std::vector<std::string>& gene_names,
                               const std::string& feature_type = "Gene Expression") {
    gzFile gz = gzopen(path.c_str(), "wb");
    if (!gz) {
        std::cerr << "[mtx_writer] Cannot open: " << path << "\n";
        return false;
    }
    const size_t n = gene_ids.size();
    for (size_t i = 0; i < n; ++i) {
        const char* name = (i < gene_names.size()) ? gene_names[i].c_str() : gene_ids[i].c_str();
        gzprintf(gz, "%s\t%s\t%s\n", gene_ids[i].c_str(), name, feature_type.c_str());
    }
    gzclose(gz);
    return true;
}

// Write CellRanger-compatible barcodes.tsv.gz (barcode-{gem_well} format)
inline bool write_barcodes_10x(const std::string& path,
                               const std::vector<std::string>& barcodes,
                               int gem_well = 1) {
    gzFile gz = gzopen(path.c_str(), "wb");
    if (!gz) {
        std::cerr << "[mtx_writer] Cannot open: " << path << "\n";
        return false;
    }
    for (const auto& bc : barcodes) {
        gzprintf(gz, "%s-%d\n", bc.c_str(), gem_well);
    }
    gzclose(gz);
    return true;
}

// Export a SparseAccumulator to MTX + sidecar files
template <typename T>
bool export_accumulator(const std::string& out_dir,
                        const std::string& prefix,
                        const SparseAccumulator<T>& acc,
                        const std::vector<std::string>& feature_names,
                        const std::vector<std::string>& barcode_names) {
    auto csc = acc.to_csc();

    std::string mtx_path = out_dir + "/" + prefix + ".mtx.gz";
    std::string feat_path = out_dir + "/" + prefix + "_features.tsv.gz";
    std::string bc_path = out_dir + "/" + prefix + "_barcodes.tsv.gz";

    if (!write_mtx(mtx_path, csc.nrows, csc.ncols,
                   csc.indptr, csc.indices, csc.data)) {
        return false;
    }

    write_names(feat_path, feature_names);
    write_names(bc_path, barcode_names);

    std::cerr << "[mtx_writer] Wrote " << prefix << ": "
              << csc.nrows << " features × "
              << csc.ncols << " barcodes, "
              << csc.data.size() << " nnz → " << mtx_path << "\n";
    return true;
}

}  // namespace singlet
