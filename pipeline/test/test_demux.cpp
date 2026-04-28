// Test harness: read AD/DP from MTX format, run donor_demux, write assignments
// Usage: test_demux <ad.mtx> <dp.mtx> <barcodes.tsv> <n_donors> <output.tsv>

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include "singlet-pileup/donor_demux.h"

// Minimal COO MTX reader
struct COOEntry { int32_t row, col; uint8_t val; };

static bool read_mtx(const std::string& path, int32_t& nrows, int32_t& ncols,
                     std::vector<int32_t>& indptr, std::vector<int32_t>& indices,
                     std::vector<uint8_t>& data) {
    std::ifstream f(path);
    if (!f) return false;
    std::string line;
    // Skip header
    while (std::getline(f, line) && line[0] == '%') {}
    // Dimensions
    std::istringstream dim(line);
    int32_t nnz;
    dim >> nrows >> ncols >> nnz;

    std::vector<COOEntry> entries;
    entries.reserve(nnz);
    for (int i = 0; i < nnz; ++i) {
        int r, c, v;
        f >> r >> c >> v;
        entries.push_back({(int32_t)(r-1), (int32_t)(c-1), (uint8_t)std::min(v, 255)});
    }

    // COO to CSC
    indptr.assign(ncols + 1, 0);
    for (auto& e : entries) indptr[e.col + 1]++;
    for (int j = 0; j < ncols; ++j) indptr[j+1] += indptr[j];
    indices.resize(nnz);
    data.resize(nnz);
    std::vector<int32_t> pos(indptr.begin(), indptr.end());
    for (auto& e : entries) {
        int32_t p = pos[e.col]++;
        indices[p] = e.row;
        data[p] = e.val;
    }
    return true;
}

int main(int argc, char* argv[]) {
    if (argc < 6) {
        std::cerr << "Usage: test_demux <ad.mtx> <dp.mtx> <barcodes.tsv> <n_donors> <output.tsv>\n";
        return 1;
    }

    std::string ad_path = argv[1], dp_path = argv[2], bc_path = argv[3];
    int n_donors = std::atoi(argv[4]);
    std::string out_path = argv[5];

    // Read barcodes
    std::vector<std::string> barcodes;
    {
        std::ifstream f(bc_path);
        std::string line;
        while (std::getline(f, line)) if (!line.empty()) barcodes.push_back(line);
    }

    // Read matrices
    int32_t ad_rows, ad_cols, dp_rows, dp_cols;
    std::vector<int32_t> ad_indptr, ad_indices, dp_indptr, dp_indices;
    std::vector<uint8_t> ad_data, dp_data;

    if (!read_mtx(ad_path, ad_rows, ad_cols, ad_indptr, ad_indices, ad_data)) {
        std::cerr << "Failed to read AD: " << ad_path << "\n";
        return 1;
    }
    if (!read_mtx(dp_path, dp_rows, dp_cols, dp_indptr, dp_indices, dp_data)) {
        std::cerr << "Failed to read DP: " << dp_path << "\n";
        return 1;
    }

    std::cerr << "AD: " << ad_rows << "x" << ad_cols << " nnz=" << ad_data.size() << "\n";
    std::cerr << "DP: " << dp_rows << "x" << dp_cols << " nnz=" << dp_data.size() << "\n";
    std::cerr << "Barcodes: " << barcodes.size() << "\n";

    // Run demux
    singlet::DonorDemuxConfig cfg;
    cfg.n_donors = n_donors;
    cfg.threads = 8;
    cfg.seed = 42;

    auto assignments = singlet::run_demux(
        (uint32_t)ad_rows, (uint32_t)ad_cols,
        ad_indptr.data(), ad_indices.data(), ad_data.data(), ad_data.size(),
        dp_indptr.data(), dp_indices.data(), dp_data.data(), dp_data.size(),
        cfg);

    // Write assignments
    singlet::write_donor_assignments(out_path, assignments, barcodes);

    return 0;
}
