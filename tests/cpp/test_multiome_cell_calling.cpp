// SPDX-License-Identifier: MIT
// Test: multiome EmptyDrops threshold adjustment
// Verifies that multiome-appropriate thresholds (lower=30, min_umi_test=100)
// are necessary for calling cells with low UMI counts typical of multiome GEX.

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/multiome_router.h"
#include "singlet/pileup/sparse_accumulator.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace singlet;

// Helper: build a CSC matrix from (gene, barcode, count) triplets.
static SparseAccumulator<uint16_t>::CSCMatrix make_csc(
    uint32_t n_genes,
    uint32_t n_barcodes,
    const std::vector<std::tuple<uint32_t, uint32_t, uint16_t>>& entries)
{
    SparseAccumulator<uint16_t> acc;
    std::vector<std::string> dummy_bc(n_barcodes, "X");
    acc.set_barcodes(dummy_bc);
    acc.set_n_features(n_genes);
    for (auto& [g, b, v] : entries)
        acc.increment(g, b, v);
    return acc.to_csc();
}

// Build the shared multiome-like matrix:
//   200 genes, 30 cells (300 UMI each on genes 0-19), 200 empties (20 UMI each on genes 50-69)
struct MultiomeMatrix {
    std::vector<uint64_t> counts;
    SparseAccumulator<uint16_t>::CSCMatrix csc;

    static constexpr uint32_t N_GENES   = 200;
    static constexpr uint32_t N_CELLS   = 30;
    static constexpr uint32_t N_EMPTIES = 200;
    static constexpr uint32_t N_BC      = N_CELLS + N_EMPTIES;
};

static MultiomeMatrix build_multiome_matrix() {
    MultiomeMatrix m;
    m.counts.resize(MultiomeMatrix::N_BC, 0);
    std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> entries;

    // 30 cells: 300 UMI each, concentrated on genes 0-19
    for (uint32_t b = 0; b < MultiomeMatrix::N_CELLS; ++b) {
        for (uint32_t g = 0; g < 20; ++g) {
            uint16_t v = 15;  // 20 genes × 15 = 300 UMI per cell
            entries.emplace_back(g, b, v);
            m.counts[b] += v;
        }
    }

    // 200 empties: 20 UMI each, spread across genes 50-69
    for (uint32_t b = MultiomeMatrix::N_CELLS; b < MultiomeMatrix::N_BC; ++b) {
        for (uint32_t g = 50; g < 70; ++g) {
            uint16_t v = 1;  // 20 genes × 1 = 20 UMI per empty
            entries.emplace_back(g, b, v);
            m.counts[b] += v;
        }
    }

    m.csc = make_csc(MultiomeMatrix::N_GENES, MultiomeMatrix::N_BC, entries);
    return m;
}

// ── T_MULTIOME_LOW_UMI_CALLING ──────────────────────────────────────────────
// Multiome thresholds (lower=30, min_umi_test=100) should call low-UMI cells.
static bool test_multiome_low_umi_calling() {
    auto m = build_multiome_matrix();

    // Multiome thresholds: lower=30, min_umi_test=100
    auto res = call_cells_emptydrops(
        m.counts, m.csc,
        /*fdr_threshold=*/0.01,
        /*lower=*/30,
        /*n_ambient_max=*/200000,
        /*min_umi_test=*/100,
        /*n_monte_carlo=*/10000);

    // Count how many of the 30 true cells were called
    uint32_t cells_called = 0;
    uint32_t empties_called = 0;
    for (uint32_t idx : res.cell_indices) {
        if (idx < MultiomeMatrix::N_CELLS)
            ++cells_called;
        else
            ++empties_called;
    }

    double recall = static_cast<double>(cells_called) / MultiomeMatrix::N_CELLS;
    std::cerr << "  [multiome_low_umi] cells_called=" << cells_called
              << "/" << MultiomeMatrix::N_CELLS
              << " recall=" << recall
              << " empties_called=" << empties_called
              << " n_ambient=" << res.n_ambient << "\n";

    bool ok = true;
    if (cells_called < 25) {
        std::cerr << "  FAIL: recall " << recall << " < 0.83 (expected ≥25/30)\n";
        ok = false;
    }
    if (empties_called > 0) {
        std::cerr << "  FAIL: " << empties_called << " empty barcodes called\n";
        ok = false;
    }
    return ok;
}

// ── T_MULTIOME_FAILS_AT_STANDARD_THRESHOLDS ─────────────────────────────────
// Standard thresholds (lower=100, min_umi_test=500) should fail to call these cells
// because cells have only 300 UMI, below min_umi_test=500.
static bool test_multiome_fails_at_standard_thresholds() {
    auto m = build_multiome_matrix();

    // Standard thresholds: lower=100, min_umi_test=500
    auto res = call_cells_emptydrops(
        m.counts, m.csc,
        /*fdr_threshold=*/0.01,
        /*lower=*/100,
        /*n_ambient_max=*/200000,
        /*min_umi_test=*/500,
        /*n_monte_carlo=*/10000);

    uint32_t cells_called = 0;
    for (uint32_t idx : res.cell_indices) {
        if (idx < MultiomeMatrix::N_CELLS)
            ++cells_called;
    }

    std::cerr << "  [standard_thresholds] cells_called=" << cells_called
              << " (expected 0, cells have 300 UMI < min_umi_test=500)\n";

    if (cells_called != 0) {
        std::cerr << "  FAIL: expected 0 cells called at standard thresholds\n";
        return false;
    }
    return true;
}

// ── T_MULTIOME_PROTOCOL_DETECTION ────────────────────────────────────────────
// Verify is_multiome_protocol() identifies multiome vs standard protocols.
static bool test_multiome_protocol_detection() {
    bool ok = true;

    auto check = [&](const std::string& proto, bool expected, const char* label) {
        bool result = is_multiome_protocol(proto);
        if (result != expected) {
            std::cerr << "  FAIL: is_multiome_protocol(\"" << proto
                      << "\") = " << result << ", expected " << expected
                      << " [" << label << "]\n";
            ok = false;
        }
    };

    check("10x-multiome-gex", true,  "multiome GEX");
    check("10x-arc-gex",      true,  "arc GEX");
    check("10x-arc-atac",     true,  "arc ATAC");
    check("10x-3p-v3",        false, "3' v3");
    check("10x-5p-v2",        false, "5' v2");
    check("",                  false, "empty string");

    return ok;
}

// ─────────────────────────────────────────────────────────────────────────────

int main() {
    int passed = 0, failed = 0;

    auto run = [&](const char* name, bool (*fn)()) {
        std::cerr << "Running: " << name << "\n";
        if (fn()) {
            ++passed;
            std::cerr << "PASS: " << name << "\n";
        } else {
            ++failed;
            std::cerr << "FAIL: " << name << "\n";
        }
    };

    run("T_MULTIOME_LOW_UMI_CALLING",              test_multiome_low_umi_calling);
    run("T_MULTIOME_FAILS_AT_STANDARD_THRESHOLDS",  test_multiome_fails_at_standard_thresholds);
    run("T_MULTIOME_PROTOCOL_DETECTION",             test_multiome_protocol_detection);

    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    if (failed > 0) {
        std::cerr << "SOME TESTS FAILED\n";
        return 1;
    }
    std::puts("All multiome cell calling tests passed.");
    return 0;
}
