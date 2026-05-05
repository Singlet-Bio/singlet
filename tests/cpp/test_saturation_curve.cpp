// test_saturation_curve.cpp — Unit tests for G-SATCURVE: saturation downsampling curves
//
// Tests the analytical Lander-Waterman + Poisson gene approximation.
// Uses a synthetic CSC (100 genes × 50 cells) with known counts.

#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

// We need GeneModel, SparseAccumulator, and CellReadStats.
// Avoid pulling in GTF parsing; build minimal mocks instead.

// ── Minimal SparseAccumulator mock (just the CSCMatrix typedef) ──
#include "singlet/pileup/sparse_accumulator.h"

// ── Minimal read_stats (CellReadStats struct) ──
#include "singlet/pileup/read_stats.h"

// ── GeneModel — we need exon_to_gene(); use a thin wrapper around the real class ──
#include "singlet/pileup/gene_model.h"

// ── Feature under test ──
#include "singlet/pileup/saturation_curve.h"

using namespace singlet;

// ── Helpers ──
static bool approx_eq(double a, double b, double tol = 0.01) {
    return std::abs(a - b) <= tol * std::max(1.0, std::abs(b));
}

// Build a minimal CSCMatrix with n_genes genes × n_cells cells.
// Each cell gets uniform count `count_per_gene` for all n_active_genes genes.
static SparseAccumulator<uint16_t>::CSCMatrix
make_uniform_csc(uint32_t n_genes, uint32_t n_cells,
                 uint32_t n_active_genes, uint16_t count_per_gene) {
    SparseAccumulator<uint16_t>::CSCMatrix csc;
    csc.nrows = n_genes;
    csc.ncols = n_cells;

    uint32_t nnz_per_cell = n_active_genes;
    csc.indptr.resize(n_cells + 1);
    csc.indptr[0] = 0;
    for (uint32_t c = 0; c < n_cells; ++c)
        csc.indptr[c + 1] = csc.indptr[c] + static_cast<int32_t>(nnz_per_cell);

    const int32_t total_nnz = csc.indptr[n_cells];
    csc.indices.resize(static_cast<size_t>(total_nnz));
    csc.data.resize(static_cast<size_t>(total_nnz), count_per_gene);

    // Each cell: genes 0 … n_active_genes-1
    for (uint32_t c = 0; c < n_cells; ++c)
        for (uint32_t g = 0; g < n_active_genes; ++g)
            csc.indices[csc.indptr[c] + static_cast<int32_t>(g)] =
                static_cast<int32_t>(g);

    return csc;
}

// Build a GeneModel where exon_to_gene maps exon i → gene i (1-to-1)
// for n_features total exon intervals. Writes a temp BED to /tmp.
// Populates an already-constructed GeneModel by reference (avoids copy/move).
static void init_identity_gene_model(GeneModel& gm, uint32_t n_features) {
    const std::string tmp = "/tmp/test_satcurve_genes.bed";
    {
        std::ofstream f(tmp);
        // BED format: chrom start end gene_name
        for (uint32_t i = 0; i < n_features; ++i) {
            f << "chr1\t" << (i * 1000) << "\t" << (i * 1000 + 100)
              << "\tGENE_" << i << "\n";
        }
    }
    bool ok = gm.load_bed(tmp);
    assert(ok && "load_bed must succeed");
    // Under BED load: exon_to_gene_[i] == i (1-to-1)
    assert(gm.exon_to_gene(0) == 0 && "exon 0 -> gene 0");
    assert(gm.exon_to_gene(n_features - 1) == n_features - 1 && "exon n-1 -> gene n-1");
}

// ── Test 1: monotonicity ──
static void test_monotonicity() {
    const uint32_t N_GENES = 100;
    const uint32_t N_CELLS = 50;
    const uint32_t N_ACTIVE = 80;  // each cell has 80 genes
    const uint16_t COUNT = 5;      // 5 UMIs per gene per cell

    auto csc = make_uniform_csc(N_GENES, N_CELLS, N_ACTIVE, COUNT);
    GeneModel gm;
    init_identity_gene_model(gm, N_GENES);

    // Build CellReadStats: total_reads = 2x unique for dup_rate = 50%
    std::vector<CellReadStats> stats(N_CELLS);
    for (uint32_t c = 0; c < N_CELLS; ++c) {
        stats[c].unique_umis = N_ACTIVE * COUNT;      // 400
        stats[c].total_reads = N_ACTIVE * COUNT * 2;  // 800 (50% dup)
        stats[c].dup_reads = stats[c].total_reads - stats[c].unique_umis;
        stats[c].dup_rate = static_cast<float>(stats[c].dup_reads) / static_cast<float>(stats[c].total_reads);
    }

    auto curve = singlet::compute_saturation_curve(stats, csc, gm);

    assert(!curve.empty() && "curve must not be empty");
    assert(curve.size() == 6 && "default fractions should give 6 points");

    // Check monotonicity
    for (size_t i = 1; i < curve.size(); ++i) {
        assert(curve[i].median_umis >= curve[i - 1].median_umis - 1e-6 && "UMIs must be non-decreasing");
        assert(curve[i].median_genes >= curve[i - 1].median_genes - 1e-6 && "genes must be non-decreasing");
        assert(curve[i].sampled_reads >= curve[i - 1].sampled_reads && "sampled_reads must be non-decreasing");
        assert(curve[i].fraction > curve[i - 1].fraction && "fractions must be strictly increasing");
    }
    std::cerr << "PASS test_monotonicity\n";
}

// ── Test 2: fraction=1.0 matches actual UMIs ──
static void test_full_depth_umis() {
    const uint32_t N_GENES = 50;
    const uint32_t N_CELLS = 20;
    const uint32_t N_ACTIVE = 40;
    const uint16_t COUNT = 3;

    auto csc = make_uniform_csc(N_GENES, N_CELLS, N_ACTIVE, COUNT);
    GeneModel gm;
    init_identity_gene_model(gm, N_GENES);

    // No duplication: total_reads == unique_umis → rpu=1, sat=0
    std::vector<CellReadStats> stats(N_CELLS);
    const uint32_t expected_umis = N_ACTIVE * COUNT;  // 120
    for (uint32_t c = 0; c < N_CELLS; ++c) {
        stats[c].unique_umis = expected_umis;
        stats[c].total_reads = expected_umis;  // no duplicates
    }

    auto curve = singlet::compute_saturation_curve(stats, csc, gm);
    assert(!curve.empty());

    // At f=1.0, formula: U * (1 - exp(-N*1/U)) = U*(1-exp(-1)) when N==U
    // = 120 * (1 - 1/e) ≈ 120 * 0.6321 ≈ 75.85
    // But with N==U and rpu=1: umis_at_1 = U*(1-exp(-1)) ≈ 75.9
    // (not 120 because rpu=1 means saturation is already happening)
    const auto& last = curve.back();
    assert(std::abs(last.fraction - 1.0) < 1e-9 && "last point must be f=1.0");
    // Median should match across all equal cells
    assert(std::abs(last.median_umis - last.mean_umis) < 1e-6 && "all cells identical → median == mean");

    // Verify formula: U*(1-exp(-N/U)) = 120*(1-exp(-1)) ≈ 75.85
    double expected_formula = expected_umis * (1.0 - std::exp(-1.0));
    assert(approx_eq(last.median_umis, expected_formula, 0.001) && "UMI formula check at f=1.0");

    std::cerr << "PASS test_full_depth_umis  median_umis@1.0=" << last.median_umis
              << " (expected≈" << expected_formula << ")\n";
}

// ── Test 3: empty inputs return empty ──
static void test_empty_inputs() {
    SparseAccumulator<uint16_t>::CSCMatrix empty_csc;
    empty_csc.ncols = 0;
    empty_csc.nrows = 0;
    GeneModel gm;
    std::vector<CellReadStats> empty_stats;

    auto curve1 = singlet::compute_saturation_curve(empty_stats, empty_csc, gm);
    assert(curve1.empty() && "empty stats → empty curve");

    auto curve2 = singlet::compute_saturation_curve(empty_stats, empty_csc, gm, {0.5, 1.0});
    assert(curve2.empty() && "empty csc → empty curve");

    std::cerr << "PASS test_empty_inputs\n";
}

// ── Test 4: genes at f=1.0 ≤ n_active_genes ──
static void test_gene_count_upper_bound() {
    const uint32_t N_GENES = 60;
    const uint32_t N_CELLS = 10;
    const uint32_t N_ACTIVE = 50;
    const uint16_t COUNT = 10;  // high count → high detection probability

    auto csc = make_uniform_csc(N_GENES, N_CELLS, N_ACTIVE, COUNT);
    GeneModel gm;
    init_identity_gene_model(gm, N_GENES);

    std::vector<CellReadStats> stats(N_CELLS);
    // High duplication: rpu = 5 → strong detection probability at f=1.0
    for (uint32_t c = 0; c < N_CELLS; ++c) {
        stats[c].unique_umis = N_ACTIVE * COUNT;      // 500
        stats[c].total_reads = N_ACTIVE * COUNT * 5;  // 2500
        stats[c].dup_reads = stats[c].total_reads - stats[c].unique_umis;
        stats[c].dup_rate = static_cast<float>(stats[c].dup_reads) / static_cast<float>(stats[c].total_reads);
    }

    auto curve = singlet::compute_saturation_curve(stats, csc, gm);
    assert(!curve.empty());

    const auto& last = curve.back();
    // genes_at_f=1 = N_ACTIVE * (1 - exp(-COUNT * rpu)) = 50*(1-exp(-50)) ≈ 50
    assert(last.median_genes <= static_cast<double>(N_ACTIVE) + 0.01 && "genes/cell cannot exceed active gene count");
    assert(last.median_genes >= static_cast<double>(N_ACTIVE) * 0.9 && "at high depth + high rpu, nearly all genes should be detected");

    std::cerr << "PASS test_gene_count_upper_bound  median_genes@1.0="
              << last.median_genes << " (n_active=" << N_ACTIVE << ")\n";
}

// ── Test 5: write TSV round-trip ──
static void test_write_tsv() {
    std::vector<singlet::SaturationPoint> curve;
    for (int i = 0; i < 3; ++i) {
        singlet::SaturationPoint pt;
        pt.fraction = (i + 1) * 0.33;
        pt.sampled_reads = static_cast<uint64_t>((i + 1) * 1000);
        pt.median_umis = (i + 1) * 100.0;
        pt.median_genes = (i + 1) * 50.0;
        pt.mean_umis = pt.median_umis;
        pt.mean_genes = pt.median_genes;
        curve.push_back(pt);
    }

    const std::string path = "/tmp/test_saturation_curve.tsv";
    bool ok = singlet::write_saturation_curve_tsv(path, curve);
    assert(ok && "write_saturation_curve_tsv should succeed");

    // Verify file has header + 3 data rows
    std::ifstream f(path);
    assert(f && "TSV file should be readable");
    int line_count = 0;
    std::string line;
    while (std::getline(f, line)) ++line_count;
    assert(line_count == 4 && "header + 3 data rows expected");

    std::cerr << "PASS test_write_tsv\n";
}

int main() {
    std::cerr << "=== test_saturation_curve ===\n";
    test_monotonicity();
    test_full_depth_umis();
    test_empty_inputs();
    test_gene_count_upper_bound();
    test_write_tsv();
    std::cerr << "ALL PASS\n";
    return 0;
}
