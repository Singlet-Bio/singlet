// N22: Unit tests for full-whitelist ambient profiling in EmptyDrops cell calling.
//
// Verifies that when wl_ambient_gene_raw is provided:
//   1. The ambient profile is built from WL data (not auto-discovered barcodes)
//   2. n_ambient is computed from wl_umi_counts (barcodes with 1–50 UMI)
//   3. Cell calling results improve: low-UMI auto-discovered barcodes no longer
//      form a degenerate ambient pool leading to knee fallback
//   4. Original (non-WL) path still works when wl_ambient_gene_raw == nullptr
//
// Run via ctest or standalone:
//   g++ -std=c++17 -O2 -I../include -o /tmp/test_wl_ambient
//       test/test_wl_ambient.cpp && /tmp/test_wl_ambient

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/sparse_accumulator.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <vector>

using namespace singlet;

// Helper: build a CSC matrix from (gene, barcode, count) triplets.
static SparseAccumulator<uint16_t>::CSCMatrix make_csc(
    uint32_t n_genes, uint32_t n_barcodes,
    const std::vector<std::tuple<uint32_t,uint32_t,uint16_t>>& entries)
{
    SparseAccumulator<uint16_t> acc;
    std::vector<std::string> dummy_bc(n_barcodes, "X");
    acc.set_barcodes(dummy_bc);
    acc.set_n_features(n_genes);
    for (auto& [g, b, v] : entries)
        acc.increment(g, b, v);
    return acc.to_csc();
}

int main() {
    int passed = 0, failed = 0;
    auto PASS = [&](const char* name) {
        ++passed;
        std::cerr << "PASS: " << name << "\n";
    };
    auto FAIL = [&](const char* name, const char* msg) {
        ++failed;
        std::cerr << "FAIL: " << name << " — " << msg << "\n";
    };

    // ── Test 1: WL ambient path builds profile from wl_ambient_gene_raw ─────
    // Setup:
    //   20 genes.  10 cells (bc 0–9) with ~2000 UMI concentrated on genes 0–4.
    //   0 auto-discovered ambient barcodes (all have UMI > lower=100).
    //   → Without WL: trigger knee/WL-ambient fallback.
    //   → With WL profile: concentrate on genes 5–9 (different from cells).
    //   Expect: cells are called (deviance from WL-ambient profile is large).
    {
        const uint32_t N_GENES  = 20;
        const uint32_t N_CELLS  = 10;
        const uint32_t N_BC     = N_CELLS;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: UMIs concentrated on genes 0–4
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 5; ++g) {
                uint16_t v = static_cast<uint16_t>(400 + b * 10);
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);

        // WL ambient profile: concentrated on genes 5–9 (distinct from cells)
        // Simulates 100 WL barcodes with 20 reads each on genes 5–9
        std::vector<uint64_t> wl_genes(N_GENES, 0ULL);
        for (uint32_t g = 5; g < 10; ++g)
            wl_genes[g] = 2000ULL;  // 100 barcodes × 20 reads on genes 5–9

        // WL per-barcode UMI counts: 100 barcodes each with 20 UMI (≤ wl_ambient_ceil=50)
        std::vector<uint32_t> wl_umi(100, 20u);

        // call with WL ambient
        auto result = call_cells_emptydrops(
            counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000,
            &wl_umi, &wl_genes, /*wl_ambient_ceil=*/50);

        // With a WL ambient profile on genes 5–9 and cells on genes 0–4,
        // the deviance should be strong → all 10 barcodes should be called.
        if (result.cell_indices.size() == N_CELLS)
            PASS("wl_ambient: all 10 cells called with WL profile");
        else
            FAIL("wl_ambient: wrong cell count",
                 ("n_cells=" + std::to_string(result.cell_indices.size())).c_str());

        // n_ambient should be set from WL counts (100 barcodes with UMI=20 ≤ 50)
        if (result.n_ambient == 100)
            PASS("wl_ambient: n_ambient=100 from wl_umi_counts");
        else
            FAIL("wl_ambient: n_ambient mismatch",
                 ("n_ambient=" + std::to_string(result.n_ambient)).c_str());
    }

    // ── Test 2: nullptr WL parameters → original path still works ────────────
    // Same setup but with ambient barcodes in the CSC (count ≤ 100).
    {
        const uint32_t N_GENES   = 20;
        const uint32_t N_CELLS   = 5;
        const uint32_t N_AMBIENT = 100;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: concentrated on genes 0–3
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 4; ++g) {
                uint16_t v = static_cast<uint16_t>(600);
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }
        // Ambient (bc 5–104): flat profile on all 20 genes, 20 UMI each
        for (uint32_t b = N_CELLS; b < N_BC; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                uint16_t v = 1;
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);

        // Call with no WL data — original auto-discovered ambient path
        auto result = call_cells_emptydrops(
            counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000,
            /*wl_umi_counts=*/nullptr,
            /*wl_ambient_gene_raw=*/nullptr);

        if (result.cell_indices.size() == N_CELLS)
            PASS("no_wl: all 5 cells called via original path");
        else
            FAIL("no_wl: wrong cell count",
                 ("n_cells=" + std::to_string(result.cell_indices.size())).c_str());

        // n_ambient should be ~100 from auto-discovered ambient
        if (result.n_ambient >= 90 && result.n_ambient <= 110)
            PASS("no_wl: n_ambient ≈ 100 from auto-discovered");
        else
            FAIL("no_wl: n_ambient mismatch",
                 ("n_ambient=" + std::to_string(result.n_ambient)).c_str());
    }

    // ── Test 3: WL profile with zero data falls through to original path ─────
    {
        const uint32_t N_GENES   = 10;
        const uint32_t N_AMBIENT = 50;
        const uint32_t N_CELLS   = 3;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 3; ++g) {
                entries.emplace_back(g, b, static_cast<uint16_t>(700));
                counts[b] += 700;
            }
        }
        for (uint32_t b = N_CELLS; b < N_BC; ++b) {
            for (uint32_t g = 0; g < N_GENES; ++g) {
                entries.emplace_back(g, b, 1u);
                counts[b] += 1;
            }
        }
        auto csc = make_csc(N_GENES, N_BC, entries);

        // All-zero WL gene array → should fall through to original ambient path
        std::vector<uint64_t> wl_genes_zero(N_GENES, 0ULL);
        std::vector<uint32_t> wl_umi_zero(10, 5u);

        auto result = call_cells_emptydrops(
            counts, csc, 0.01, 100, 50000, 500, 2000,
            &wl_umi_zero, &wl_genes_zero, 50);

        if (result.cell_indices.size() == N_CELLS)
            PASS("wl_zero: correct fallthrough to original path");
        else
            FAIL("wl_zero: unexpected cell count",
                 ("n_cells=" + std::to_string(result.cell_indices.size())).c_str());
    }

    // ── Test 4: n_ambient estimated from read total when wl_umi_counts=nullptr ─
    {
        const uint32_t N_GENES = 5;
        const uint32_t N_CELLS = 2;
        const uint32_t N_BC    = N_CELLS;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            entries.emplace_back(0u, b, 800u);
            counts[b] = 800;
        }
        auto csc = make_csc(N_GENES, N_BC, entries);

        // WL genes: 500 reads on gene 4 (total=500), but no per-barcode counts
        std::vector<uint64_t> wl_genes(N_GENES, 0ULL);
        wl_genes[4] = 500ULL;

        auto result = call_cells_emptydrops(
            counts, csc, 0.01, 100, 50000, 500, 2000,
            /*wl_umi_counts=*/nullptr, &wl_genes, 50);

        // n_ambient estimated as 500/25 = 20 (rough estimate)
        bool n_amb_ok = (result.n_ambient >= 10 && result.n_ambient <= 40);
        if (n_amb_ok)
            PASS("wl_no_umi_counts: n_ambient estimated from reads");
        else
            FAIL("wl_no_umi_counts: n_ambient out of range",
                 ("n_ambient=" + std::to_string(result.n_ambient)).c_str());
    }

    std::cerr << "\n" << passed << "/" << (passed + failed) << " tests passed.\n";
    return (failed == 0) ? 0 : 1;
}
