// SPDX-License-Identifier: MIT
// Realistic ambient RNA correction tests with PBMC-like synthetic matrix.
//
// Tests:
//   T_CORRECTION_ACCURACY_10PCT  — correction recovers cell-specific signal at ρ=0.10
//   T_CORRECTION_ACCURACY_20PCT  — correction recovers cell-specific signal at ρ=0.20
//   T_NO_OVERCORRECTION          — no negative counts; background genes unchanged
//   T_IDENTITY_NO_CONTAMINATION  — clean matrix unchanged after correction
//   T_AMBIENT_PROFILE_SHAPE      — estimated ambient profile matches known composition

#include "singlet/pileup/ambient_correction.h"
#include "singlet/pileup/sparse_accumulator.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <vector>

using namespace singlet;

// ── Synthetic matrix parameters ─────────────────────────────────────────────

static constexpr uint32_t N_GENES    = 200;
static constexpr uint32_t N_CELLS    = 100;   // barcodes 0..99
static constexpr uint32_t N_EMPTY    = 500;   // barcodes 100..599
static constexpr uint32_t N_BARCODES = N_CELLS + N_EMPTY;

static constexpr uint16_t HK_COUNT   = 100;   // housekeeping genes 0-9 per cell
static constexpr uint16_t CS_COUNT   = 25;    // cell-specific genes 10-49 per cell
static constexpr uint16_t EMPTY_HK   = 3;     // housekeeping genes 0-9 per empty barcode
static constexpr uint64_t LOWER_UMI  = 50;    // empty threshold (empties have ~30 UMI)

// Clean cell total: 10×100 + 40×25 = 2000
static constexpr double CLEAN_TOTAL = 10.0 * HK_COUNT + 40.0 * CS_COUNT;

// True ambient profile: genes 0-9 share equal mass, rest zero.
static double true_ambient_frac(uint32_t g) {
    return (g < 10) ? 0.1 : 0.0;
}

// ── Helper: build CSC from (gene, barcode, count) triplets ──────────────────

static CSCu16 make_csc(
    uint32_t n_genes, uint32_t n_barcodes,
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

// ── Build combined (cells + empties) matrix with given contamination ρ ──────

struct MatrixBundle {
    CSCu16                 counts;
    std::vector<uint64_t>  bc_totals;
    std::vector<uint32_t>  cell_indices;
};

static MatrixBundle build_matrix(double rho) {
    std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> entries;
    entries.reserve(N_CELLS * 50 + N_EMPTY * 10);
    std::vector<uint64_t> bc_totals(N_BARCODES, 0);

    // Cell barcodes 0..N_CELLS-1
    for (uint32_t b = 0; b < N_CELLS; ++b) {
        uint64_t total = 0;
        for (uint32_t g = 0; g < N_GENES; ++g) {
            double clean = 0.0;
            if (g < 10)       clean = HK_COUNT;
            else if (g < 50)  clean = CS_COUNT;

            double val = (1.0 - rho) * clean
                       + rho * CLEAN_TOTAL * true_ambient_frac(g);
            auto ival = static_cast<uint16_t>(std::round(val));
            if (ival > 0) {
                entries.emplace_back(g, b, ival);
                total += ival;
            }
        }
        bc_totals[b] = total;
    }

    // Empty barcodes N_CELLS..N_BARCODES-1
    for (uint32_t b = N_CELLS; b < N_BARCODES; ++b) {
        uint64_t total = 0;
        for (uint32_t g = 0; g < 10; ++g) {
            entries.emplace_back(g, b, EMPTY_HK);
            total += EMPTY_HK;
        }
        bc_totals[b] = total;
    }

    std::vector<uint32_t> cell_idx(N_CELLS);
    std::iota(cell_idx.begin(), cell_idx.end(), 0);

    return {make_csc(N_GENES, N_BARCODES, entries), bc_totals, cell_idx};
}

// ── Retrieve a single count from a CSC matrix ──────────────────────────────

static uint16_t get_count(const CSCu16& m, uint32_t gene, uint32_t barcode) {
    const int32_t st = m.indptr[barcode];
    const int32_t en = m.indptr[barcode + 1];
    for (int32_t k = st; k < en; ++k) {
        if (static_cast<uint32_t>(m.indices[k]) == gene)
            return m.data[k];
    }
    return 0;
}

// ── Test harness ────────────────────────────────────────────────────────────

static int g_passed = 0, g_failed = 0;

static void PASS(const char* name) {
    ++g_passed;
    std::cerr << "PASS: " << name << "\n";
}
static void FAIL(const char* name, const char* msg) {
    ++g_failed;
    std::cerr << "FAIL: " << name << " — " << msg << "\n";
}

// ── main ────────────────────────────────────────────────────────────────────

int main() {
    // Build clean and contaminated matrices
    auto clean    = build_matrix(0.00);
    auto cont10   = build_matrix(0.10);
    auto cont20   = build_matrix(0.20);

    // Estimate ambient profile from empty barcodes (shared across all matrices)
    AmbientProfile ambient = estimate_ambient(
        clean.counts, clean.bc_totals, LOWER_UMI);

    // ── T_AMBIENT_PROFILE_SHAPE ─────────────────────────────────────────────
    {
        double mass_0_9   = 0.0;
        double mass_50_199 = 0.0;
        for (uint32_t g = 0; g < N_GENES; ++g) {
            if (g < 10)       mass_0_9    += ambient.gene_fractions[g];
            else if (g >= 50) mass_50_199 += ambient.gene_fractions[g];
        }

        bool ok = true;
        char buf[256];
        if (mass_0_9 < 0.90) {
            snprintf(buf, sizeof(buf),
                     "genes 0-9 mass=%.4f, expected >=0.90", mass_0_9);
            FAIL("T_AMBIENT_PROFILE_SHAPE", buf);
            ok = false;
        }
        if (mass_50_199 > 0.02) {
            snprintf(buf, sizeof(buf),
                     "genes 50-199 mass=%.4f, expected <=0.02", mass_50_199);
            FAIL("T_AMBIENT_PROFILE_SHAPE", buf);
            ok = false;
        }
        if (ok) PASS("T_AMBIENT_PROFILE_SHAPE");
    }

    // Helper lambda: correct a matrix using a known ρ for every cell.
    auto run_correction = [&](MatrixBundle& mat, double rho) -> CSCu16 {
        std::vector<CellContamination> contam(mat.cell_indices.size());
        for (auto& c : contam) c.rho = rho;
        return correct_counts(
            mat.counts, ambient, contam, mat.cell_indices, mat.bc_totals);
    };

    // ── T_CORRECTION_ACCURACY_10PCT ─────────────────────────────────────────
    {
        CSCu16 corrected = run_correction(cont10, 0.10);

        double total_rel_err = 0.0;
        uint32_t n_checks    = 0;
        bool within_20pct    = true;

        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 10; g < 50; ++g) {
                double corr = get_count(corrected, g, b);
                double rel  = std::abs(corr - CS_COUNT) / CS_COUNT;
                total_rel_err += rel;
                ++n_checks;
                if (rel > 0.20) within_20pct = false;
            }
        }

        double avg = total_rel_err / n_checks;
        char buf[256];
        if (!within_20pct) {
            FAIL("T_CORRECTION_ACCURACY_10PCT",
                 "some cell-specific genes >20% relative error");
        } else if (avg > 0.15) {
            snprintf(buf, sizeof(buf),
                     "avg relative error=%.4f, expected <=0.15", avg);
            FAIL("T_CORRECTION_ACCURACY_10PCT", buf);
        } else {
            snprintf(buf, sizeof(buf), "(avg_rel_err=%.4f)", avg);
            std::cerr << "  ";
            PASS("T_CORRECTION_ACCURACY_10PCT");
            std::cerr << "       " << buf << "\n";
        }
    }

    // ── T_CORRECTION_ACCURACY_20PCT ─────────────────────────────────────────
    {
        CSCu16 corrected = run_correction(cont20, 0.20);

        double total_rel_err = 0.0;
        uint32_t n_checks    = 0;

        for (uint32_t b = 0; b < N_CELLS; ++b) {
            for (uint32_t g = 10; g < 50; ++g) {
                double corr = get_count(corrected, g, b);
                double rel  = std::abs(corr - CS_COUNT) / CS_COUNT;
                total_rel_err += rel;
                ++n_checks;
            }
        }

        double avg = total_rel_err / n_checks;
        char buf[256];
        if (avg > 0.25) {
            snprintf(buf, sizeof(buf),
                     "avg relative error=%.4f, expected <=0.25", avg);
            FAIL("T_CORRECTION_ACCURACY_20PCT", buf);
        } else {
            snprintf(buf, sizeof(buf), "(avg_rel_err=%.4f)", avg);
            std::cerr << "  ";
            PASS("T_CORRECTION_ACCURACY_20PCT");
            std::cerr << "       " << buf << "\n";
        }
    }

    // ── T_NO_OVERCORRECTION ─────────────────────────────────────────────────
    {
        CSCu16 corrected10 = run_correction(cont10, 0.10);
        CSCu16 corrected20 = run_correction(cont20, 0.20);

        bool ok = true;
        char buf[256];

        // All stored values must be strictly positive (zeros are dropped)
        for (const auto* m : {&corrected10, &corrected20}) {
            for (size_t i = 0; i < m->data.size(); ++i) {
                if (m->data[i] == 0) {
                    FAIL("T_NO_OVERCORRECTION",
                         "zero stored in sparse corrected matrix");
                    ok = false;
                    break;
                }
            }
            if (!ok) break;
        }

        // Background genes (50-199) should be unchanged (already ~0)
        if (ok) {
            for (uint32_t b = 0; b < N_CELLS && ok; ++b) {
                for (uint32_t g = 50; g < N_GENES && ok; ++g) {
                    uint16_t o10 = get_count(cont10.counts, g, b);
                    uint16_t c10 = get_count(corrected10, g, b);
                    uint16_t o20 = get_count(cont20.counts, g, b);
                    uint16_t c20 = get_count(corrected20, g, b);
                    if (c10 != o10 || c20 != o20) {
                        snprintf(buf, sizeof(buf),
                                 "bg gene %u bc %u changed: %u->%u / %u->%u",
                                 g, b, o10, c10, o20, c20);
                        FAIL("T_NO_OVERCORRECTION", buf);
                        ok = false;
                    }
                }
            }
        }

        if (ok) PASS("T_NO_OVERCORRECTION");
    }

    // ── T_IDENTITY_NO_CONTAMINATION ─────────────────────────────────────────
    {
        // Correct clean matrix with ρ=0 → should be identical to input
        std::vector<CellContamination> zero_contam(clean.cell_indices.size());
        for (auto& c : zero_contam) c.rho = 0.0;
        CSCu16 corrected = correct_counts(
            clean.counts, ambient, zero_contam,
            clean.cell_indices, clean.bc_totals);

        bool ok = true;
        char buf[256];
        for (uint32_t b = 0; b < N_CELLS && ok; ++b) {
            for (uint32_t g = 0; g < N_GENES && ok; ++g) {
                int diff = static_cast<int>(get_count(corrected, g, b))
                         - static_cast<int>(get_count(clean.counts, g, b));
                if (std::abs(diff) > 1) {
                    snprintf(buf, sizeof(buf),
                             "gene=%u bc=%u diff=%d", g, b, diff);
                    FAIL("T_IDENTITY_NO_CONTAMINATION", buf);
                    ok = false;
                }
            }
        }
        if (ok) PASS("T_IDENTITY_NO_CONTAMINATION");
    }

    // ── Summary ─────────────────────────────────────────────────────────────
    std::cerr << "\n" << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed ? 1 : 0;
}
