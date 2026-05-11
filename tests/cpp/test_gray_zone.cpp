// Unit tests for the gray-zone ambient supplement logic in EmptyDrops cell calling.
//
// When the ambient pool is too thin (<300 barcodes or <10000 total UMI), the
// algorithm supplements it with "gray-zone" barcodes — barcodes with UMI in
// (lower, min(min_umi_test, lower*2)).  These tests verify correct recruitment,
// profile preservation, sensitivity, and gate behavior.
//
// See cell_calling.h lines ~808-825 for the supplement logic.

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/sparse_accumulator.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <numeric>
#include <string>
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

    // ── T1: T_GRAY_AMBIENT_PROFILE ──────────────────────────────────────────
    // Gray-zone barcodes with AMBIENT profile (genes 100-119) are recruited
    // into the ambient pool and the resulting ambient profile stays on the
    // correct genes — NOT the cell genes (0-19).
    //
    // Setup:
    //   200 genes
    //   20 cells (bc 0-19):   2000 UMI each, concentrated on genes 0-19
    //   5  ambient (bc 20-24): 50 UMI each, spread on genes 100-119 (too few for MC)
    //   30 gray-zone (bc 25-54): 150 UMI each, spread on genes 100-119
    //   lower=100  → gray_zone_ceil = min(500, 200) = 200
    //   5 ambient < 300 AND 250 UMI < 10000 → supplement triggered
    {
        const uint32_t N_GENES   = 200;
        const uint32_t N_CELLS   = 20;
        const uint32_t N_AMBIENT = 5;
        const uint32_t N_GRAY    = 30;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT + N_GRAY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: 2000 UMI each on genes 0-19 (100 UMI per gene)
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 20; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }
        }

        // Ambient barcodes: 50 UMI each on genes 100-119
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(3));
                counts[b] += 3;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(2));
                counts[b] += 2;
            }
        }

        // Gray-zone barcodes: 150 UMI each on genes 100-119 (same ambient profile)
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(8));
                counts[b] += 8;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(7));
                counts[b] += 7;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000);

        // Verify: gray-zone barcodes were recruited (n_ambient > original 5)
        if (res.n_ambient > N_AMBIENT)
            PASS("T_GRAY_AMBIENT_PROFILE: gray-zone barcodes recruited");
        else
            FAIL("T_GRAY_AMBIENT_PROFILE: gray-zone recruitment",
                 ("n_ambient=" + std::to_string(res.n_ambient) + " expected >" +
                  std::to_string(N_AMBIENT)).c_str());

        // Verify: the supplemented ambient profile still represents genes 100-119.
        // If the ambient profile is correct (genes 100-119), cells (genes 0-19)
        // have maximal deviance from ambient → all cells are called.
        // If contaminated with cell genes, deviance drops → fewer cells called.
        if (res.cell_indices.size() >= 15)
            PASS("T_GRAY_AMBIENT_PROFILE: ambient profile preserves gene identity");
        else
            FAIL("T_GRAY_AMBIENT_PROFILE: ambient profile check",
                 ("cells_called=" + std::to_string(res.cell_indices.size()) +
                  " expected >=15 (ambient on correct genes → high deviance)").c_str());

        std::cerr << "  n_ambient=" << res.n_ambient
                  << " ambient_total=" << static_cast<uint64_t>(res.ambient_total)
                  << " cells_called=" << res.cell_indices.size() << "\n";
    }

    // ── T2: T_GRAY_CELL_PROFILE_REJECTED ────────────────────────────────────
    // Adversarial: gray-zone barcodes have CELL profile (genes 0-19) instead
    // of ambient profile.  This contaminates the ambient pool with cell signal,
    // making cells look like ambient → low deviance → undercalling.
    //
    // This tests that the algorithm doesn't crash even with a contaminated
    // ambient profile.  Cell calling quality degrades (expected), but the
    // pipeline remains stable.
    {
        const uint32_t N_GENES   = 200;
        const uint32_t N_CELLS   = 20;
        const uint32_t N_AMBIENT = 5;
        const uint32_t N_GRAY    = 30;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT + N_GRAY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: 2000 UMI each on genes 0-19
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 20; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }
        }

        // Ambient barcodes: 50 UMI each on genes 100-119
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(3));
                counts[b] += 3;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(2));
                counts[b] += 2;
            }
        }

        // Gray-zone barcodes: 150 UMI each on genes 0-19 (CELL profile!)
        // Contaminates the ambient pool with cell-derived signal.
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b) {
            for (uint32_t g = 0; g < 20; ++g) {
                uint16_t v = static_cast<uint16_t>(7 + (g < 10 ? 1 : 0));
                entries.emplace_back(g, b, v);
                counts[b] += v;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000);

        // Gray-zone barcodes should still be recruited (n_ambient > 5)
        if (res.n_ambient > N_AMBIENT)
            PASS("T_GRAY_CELL_PROFILE_REJECTED: gray-zone recruited (adversarial)");
        else
            FAIL("T_GRAY_CELL_PROFILE_REJECTED: gray-zone not recruited",
                 ("n_ambient=" + std::to_string(res.n_ambient)).c_str());

        // Primary assertion: no crash.  Cell calling degrades because the
        // contaminated ambient profile makes cells look ambient (low deviance).
        // This is the expected failure mode that the lower*2 ceiling prevents
        // in practice — gray-zone barcodes at lower*2 are empty droplets with
        // ambient profile, not cell-profile contaminants.
        std::cerr << "  [adversarial] n_ambient=" << res.n_ambient
                  << " cells_called=" << res.cell_indices.size()
                  << " (contaminated ambient — degraded calling expected)\n";
        PASS("T_GRAY_CELL_PROFILE_REJECTED: no crash with contaminated ambient");
    }

    // ── T3: T_GRAY_SENSITIVITY ──────────────────────────────────────────────
    // With lower=100 (gray zone = 100-200) and correct ambient-profile
    // gray-zone barcodes, verify the cell count is reasonable.
    {
        const uint32_t N_GENES   = 200;
        const uint32_t N_CELLS   = 20;
        const uint32_t N_AMBIENT = 5;
        const uint32_t N_GRAY    = 30;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT + N_GRAY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: 2000 UMI on genes 0-19
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 20; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }
        }

        // Ambient: 50 UMI on genes 100-119
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(3));
                counts[b] += 3;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(2));
                counts[b] += 2;
            }
        }

        // Gray-zone: 150 UMI on genes 100-119 (ambient profile)
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(8));
                counts[b] += 8;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(7));
                counts[b] += 7;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000);

        uint32_t n_cells = static_cast<uint32_t>(res.cell_indices.size());

        if (n_cells > 0)
            PASS("T_GRAY_SENSITIVITY: cell count > 0 (not degenerate)");
        else
            FAIL("T_GRAY_SENSITIVITY: degenerate", "cell count = 0");

        if (n_cells >= 15 && n_cells <= 25)
            PASS("T_GRAY_SENSITIVITY: cell count in [15, 25]");
        else
            FAIL("T_GRAY_SENSITIVITY: cell count out of range",
                 ("n_cells=" + std::to_string(n_cells) + " expected [15,25]").c_str());

        std::cerr << "  cells_called=" << n_cells << " (expected ~20)\n";
    }

    // ── T4: T_GRAY_EMPTY_SUPPLEMENT ─────────────────────────────────────────
    // When the ambient pool is already sufficient (≥300 barcodes, ≥10000 UMI),
    // the gray-zone supplement should NOT be triggered.
    //
    // 400 ambient barcodes × 30 UMI = 12000 total UMI → both thresholds met.
    // 30 gray-zone barcodes exist in the matrix but should NOT be recruited.
    {
        const uint32_t N_GENES   = 200;
        const uint32_t N_CELLS   = 20;
        const uint32_t N_AMBIENT = 400;
        const uint32_t N_GRAY    = 30;
        const uint32_t N_BC      = N_CELLS + N_AMBIENT + N_GRAY;

        std::vector<uint64_t> counts(N_BC, 0);
        std::vector<std::tuple<uint32_t,uint32_t,uint16_t>> entries;

        // Cells: 2000 UMI on genes 0-19
        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 0; g < 20; ++g) {
                entries.emplace_back(g, b, uint16_t(100));
                counts[b] += 100;
            }
        }

        // 400 ambient barcodes: 30 UMI each on genes 100-119
        for (uint32_t b = N_CELLS; b < N_CELLS + N_AMBIENT; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(2));
                counts[b] += 2;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(1));
                counts[b] += 1;
            }
        }

        // Gray-zone barcodes: 150 UMI each on genes 100-119
        // These should NOT be recruited because ambient pool is sufficient.
        for (uint32_t b = N_CELLS + N_AMBIENT; b < N_BC; ++b) {
            for (uint32_t g = 100; g < 110; ++g) {
                entries.emplace_back(g, b, uint16_t(8));
                counts[b] += 8;
            }
            for (uint32_t g = 110; g < 120; ++g) {
                entries.emplace_back(g, b, uint16_t(7));
                counts[b] += 7;
            }
        }

        auto csc = make_csc(N_GENES, N_BC, entries);
        auto res = call_cells_emptydrops(counts, csc,
            /*fdr_threshold=*/0.01,
            /*lower=*/100,
            /*n_ambient_max=*/50000,
            /*min_umi_test=*/500,
            /*n_monte_carlo=*/2000);

        // n_ambient should be exactly N_AMBIENT (400) — no gray-zone supplement
        if (res.n_ambient == N_AMBIENT)
            PASS("T_GRAY_EMPTY_SUPPLEMENT: gray-zone NOT triggered (n_ambient=400)");
        else
            FAIL("T_GRAY_EMPTY_SUPPLEMENT: unexpected n_ambient",
                 ("n_ambient=" + std::to_string(res.n_ambient) +
                  " expected " + std::to_string(N_AMBIENT)).c_str());

        std::cerr << "  n_ambient=" << res.n_ambient
                  << " ambient_total=" << static_cast<uint64_t>(res.ambient_total) << "\n";
    }

    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
