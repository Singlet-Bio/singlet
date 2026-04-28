// N5: Standalone EmptyDrops cell calling tool
// Reads singlify MTX output, runs call_cells_emptydrops, writes cell_calls.tsv
//
// Build (no CMake needed):
//   g++ -std=c++17 -O2 -I../include -o /tmp/cell_caller src/cell_caller.cpp
// Run:
//   /tmp/cell_caller /dev/shm/n5_mtx/exon_counts.mtx \
//     /dev/shm/n5_mtx/exon_counts_barcodes.tsv \
//     /dev/shm/n5_mtx/cell_calls.tsv

#include "singlet-pileup/cell_calling.h"
#include "singlet-pileup/sparse_accumulator.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace singlet;

// ── MTX reader → SparseAccumulator<uint16_t> ──────────────────────────────────
// Market Exchange format: header lines starting with %, then "rows cols nnz",
// then 1-based (row col val) entries (row = gene, col = barcode).
static SparseAccumulator<uint16_t>::CSCMatrix read_mtx(
    const std::string& mtx_path,
    const std::vector<std::string>& barcodes,
    uint32_t& n_genes_out)
{
    std::ifstream f(mtx_path);
    if (!f.is_open()) throw std::runtime_error("Cannot open: " + mtx_path);

    std::string line;

    // Skip comment/banner lines
    while (std::getline(f, line))
        if (!line.empty() && line[0] != '%') break;

    // Dimension line
    uint32_t n_genes, n_barcodes;
    uint64_t nnz;
    {
        std::istringstream ss(line);
        ss >> n_genes >> n_barcodes >> nnz;
    }
    if (n_barcodes != static_cast<uint32_t>(barcodes.size()))
        throw std::runtime_error("barcode count mismatch");
    n_genes_out = n_genes;

    SparseAccumulator<uint16_t> acc;
    acc.set_barcodes(barcodes);
    acc.set_n_features(n_genes);

    uint32_t row, col;
    uint64_t val;
    while (f >> row >> col >> val) {
        // 1-based → 0-based; MTX is row=gene, col=barcode
        uint16_t v16 = (val > 65535u) ? 65535u : static_cast<uint16_t>(val);
        acc.increment(row - 1, col - 1, v16);
    }

    return acc.to_csc();
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        std::cerr << "Usage: cell_caller <exon_counts.mtx> <barcodes.tsv> <out_cell_calls.tsv>\n";
        std::cerr << "Options (appended after positional):\n";
        std::cerr << "  --fdr    F    FDR threshold (default 0.01)\n";
        std::cerr << "  --lower  N    Min UMI  (default 100)\n";
        std::cerr << "  --bins   N    n_ambient_bins (default 20000)\n";
        return 1;
    }

    std::string mtx_path      = argv[1];
    std::string barcodes_path = argv[2];
    std::string out_path      = argv[3];
    double   fdr_threshold    = 0.01;
    uint64_t lower            = 100;
    int      n_bins           = 20000;

    for (int i = 4; i < argc - 1; ++i) {
        std::string a = argv[i];
        if      (a == "--fdr")   fdr_threshold = std::atof(argv[++i]);
        else if (a == "--lower") lower         = std::stoull(argv[++i]);
        else if (a == "--bins")  n_bins        = std::atoi(argv[++i]);
    }

    // ── Read barcodes ─────────────────────────────────────────────────────────
    std::vector<std::string> barcodes;
    {
        std::ifstream f(barcodes_path);
        if (!f.is_open()) { std::cerr << "Cannot open barcodes: " << barcodes_path << "\n"; return 1; }
        std::string bc;
        while (std::getline(f, bc)) if (!bc.empty()) barcodes.push_back(bc);
    }
    std::cerr << "[cell_caller] Barcodes: " << barcodes.size() << "\n";

    // ── Load MTX ──────────────────────────────────────────────────────────────
    auto t0 = std::chrono::high_resolution_clock::now();
    uint32_t n_genes = 0;
    auto csc = read_mtx(mtx_path, barcodes, n_genes);
    std::cerr << "[cell_caller] Matrix: " << n_genes << " genes × " << barcodes.size()
              << " barcodes, nnz=" << csc.indices.size() << "\n";

    // ── Compute per-barcode totals from CSC ───────────────────────────────────
    const uint32_t n_bc = static_cast<uint32_t>(barcodes.size());
    std::vector<uint64_t> counts_per_bc(n_bc, 0);
    for (uint32_t b = 0; b < n_bc; ++b) {
        int32_t start = csc.indptr[b];
        int32_t end   = csc.indptr[b + 1];
        for (int32_t k = start; k < end; ++k)
            counts_per_bc[b] += csc.data[k];
    }

    // ── Cell calling ──────────────────────────────────────────────────────────
    auto result = call_cells_emptydrops(counts_per_bc, csc, fdr_threshold, lower, n_bins);
    auto t1     = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cerr << "[cell_caller] n_ambient=" << result.n_ambient
              << " ambient_total=" << static_cast<uint64_t>(result.ambient_total)
              << " tested=" << result.tested_indices.size()
              << " cells_called=" << result.cell_indices.size()
              << " elapsed=" << elapsed << "s\n";

    // ── Write TSV ─────────────────────────────────────────────────────────────
    write_cell_calls(out_path, result, barcodes, counts_per_bc, fdr_threshold);
    std::cerr << "[cell_caller] Wrote " << out_path << "\n";

    return 0;
}
