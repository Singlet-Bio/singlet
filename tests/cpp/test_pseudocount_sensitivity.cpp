// SPDX-License-Identifier: MIT
// Pseudocount sensitivity analysis for EmptyDrops cell calling.
// Ablates pseudo in {0.05, 0.1, 0.2, 0.3, 0.5, 1.0} against a synthetic
// matrix with overlapping cell/ambient gene programs, measuring recall,
// precision, and FPR.  Asserts robustness across the [0.1, 0.5] band and
// identifies the optimal pseudo value.

#include "singlet/pileup/cell_calling.h"
#include "singlet/pileup/sparse_accumulator.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

using namespace singlet;

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

struct SweepResult {
    double pseudo;
    double recall;
    double precision;
    double fpr;
    uint32_t tp, fp, fn, tn;
};

// Run EmptyDrops MC pipeline at a given pseudo value by replicating the core
// machinery (ambient profile + pseudo → deviance → MC p-values → BH FDR).
static SweepResult run_at_pseudo(
    double pseudo,
    uint32_t n_genes,
    uint32_t n_cells,
    uint32_t n_empty,
    const std::vector<uint64_t>& counts,
    const SparseAccumulator<uint16_t>::CSCMatrix& csc,
    uint64_t lower,
    uint64_t min_umi_test,
    double fdr_threshold,
    int n_mc)
{
    const uint32_t n_bc = n_cells + n_empty;

    std::vector<double> ambient(n_genes, 0.0);
    for (uint32_t b = 0; b < n_bc; ++b) {
        if (counts[b] > lower) continue;
        int32_t start = csc.indptr[b];
        int32_t end   = csc.indptr[b + 1];
        for (int32_t k = start; k < end; ++k)
            ambient[csc.indices[k]] += static_cast<double>(csc.data[k]);
    }

    double psum = 0.0;
    for (uint32_t g = 0; g < n_genes; ++g) {
        ambient[g] += pseudo;
        psum += ambient[g];
    }
    for (uint32_t g = 0; g < n_genes; ++g)
        ambient[g] /= psum;

    std::vector<double> log_ambient(n_genes);
    for (uint32_t g = 0; g < n_genes; ++g)
        log_ambient[g] = std::log(ambient[g]);

    std::vector<uint32_t> tested;
    std::vector<double> deviances;
    for (uint32_t b = 0; b < n_bc; ++b) {
        uint64_t N = counts[b];
        if (N <= lower || N < min_umi_test) continue;
        tested.push_back(b);
        double dev = detail::compute_deviance(
            csc.indices.data(), csc.data.data(),
            csc.indptr[b], csc.indptr[b + 1],
            log_ambient, N);
        deviances.push_back(dev);
    }

    std::vector<double> pvals = detail::mc_emptydrops_pvalues(
        tested, deviances, counts, ambient, log_ambient, n_mc);

    std::vector<double> fdr;
    detail::bh_fdr(pvals, fdr);

    uint32_t tp = 0, fp = 0, fn = 0, tn = 0;
    std::vector<bool> called(n_bc, false);
    for (size_t i = 0; i < tested.size(); ++i) {
        if (fdr[i] < fdr_threshold)
            called[tested[i]] = true;
    }
    for (uint32_t b = 0; b < n_cells; ++b) {
        if (called[b]) ++tp; else ++fn;
    }
    for (uint32_t b = n_cells; b < n_bc; ++b) {
        if (called[b]) ++fp; else ++tn;
    }

    double recall    = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
    double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 1.0;
    double fpr_val   = (fp + tn > 0) ? static_cast<double>(fp) / (fp + tn) : 0.0;

    return {pseudo, recall, precision, fpr_val, tp, fp, fn, tn};
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

    // ── 1. Build realistic synthetic matrix ─────────────────────────────────
    //
    //   5000 genes — large K makes pseudo mass non-negligible relative to
    //   ambient_tot (~15000), which is the mechanism that makes the null
    //   deviance distribution sensitive to pseudo.
    //
    //   Gene layout:
    //     0-49   : cell-type-specific (only expressed by cells)
    //     50-69  : ambient pool (empties + cells both express these)
    //     70-4999: dark / unused
    //
    //   30 "easy" cells (bc 0-29):  ~1500 UMI
    //     genes 0-49 at ~25 UMI (1250) + genes 50-59 at ~25 UMI (250)
    //     These have massive deviance from 50 unseen genes → always called.
    //
    //   20 "hard" cells (bc 30-49): ~600 UMI
    //     genes 0-39 at ~2 UMI (80) + genes 50-69 at ~26 UMI (520)
    //     Ambient gene expression matches the profile well (low deviance).
    //     Only the 40 cell-specific genes at 2 UMI provide signal.
    //     At high pseudo (1.0), the MC null samples ~150 unseen genes →
    //     null deviance exceeds the cell's → hard cells are missed.
    //
    //   500 empties (bc 50-549): genes 50-69 at ~1.5 UMI each = ~30 UMI

    const uint32_t N_GENES = 5000;
    const uint32_t N_EASY  = 30;
    const uint32_t N_HARD  = 20;
    const uint32_t N_CELLS = N_EASY + N_HARD;
    const uint32_t N_EMPTY = 500;
    const uint32_t N_BC    = N_CELLS + N_EMPTY;
    const uint64_t LOWER   = 100;
    const uint64_t MIN_UMI_TEST = 500;
    const double   FDR_THR = 0.01;
    const int      N_MC    = 10000;

    std::vector<uint64_t> counts(N_BC, 0);
    std::vector<std::tuple<uint32_t, uint32_t, uint16_t>> entries;
    std::mt19937 gen(42);

    // Easy cells (bc 0-29): strong cell-specific program
    for (uint32_t b = 0; b < N_EASY; ++b) {
        for (uint32_t g = 0; g < 50; ++g) {
            uint16_t v = static_cast<uint16_t>(22 + gen() % 7); // 22-28
            entries.emplace_back(g, b, v);
            counts[b] += v;
        }
        for (uint32_t g = 50; g < 60; ++g) {
            uint16_t v = static_cast<uint16_t>(22 + gen() % 7);
            entries.emplace_back(g, b, v);
            counts[b] += v;
        }
    }

    // Hard cells (bc 30-49): weak cell-specific, mostly ambient expression
    for (uint32_t b = N_EASY; b < N_CELLS; ++b) {
        // 50 cell-specific genes at ~2 UMI each (100 total)
        for (uint32_t g = 0; g < 50; ++g) {
            uint16_t v = static_cast<uint16_t>(1 + gen() % 3); // 1-3
            entries.emplace_back(g, b, v);
            counts[b] += v;
        }
        // 20 ambient genes at ~25 UMI each (500 total, matches ambient profile)
        for (uint32_t g = 50; g < 70; ++g) {
            uint16_t v = static_cast<uint16_t>(23 + gen() % 5); // 23-27
            entries.emplace_back(g, b, v);
            counts[b] += v;
        }
    }

    // Empty barcodes (bc 50-549): ambient genes 50-69
    for (uint32_t b = N_CELLS; b < N_BC; ++b) {
        for (uint32_t g = 50; g < 70; ++g) {
            uint16_t v = static_cast<uint16_t>(1 + gen() % 2); // 1-2
            entries.emplace_back(g, b, v);
            counts[b] += v;
        }
    }

    auto csc = make_csc(N_GENES, N_BC, entries);

    // Print UMI summary
    uint64_t easy_sum = 0, hard_sum = 0, empty_sum = 0;
    for (uint32_t b = 0; b < N_EASY; ++b) easy_sum += counts[b];
    for (uint32_t b = N_EASY; b < N_CELLS; ++b) hard_sum += counts[b];
    for (uint32_t b = N_CELLS; b < N_BC; ++b) empty_sum += counts[b];

    std::cerr << "\n═══════════════════════════════════════════════════════\n";
    std::cerr << "  Pseudocount sensitivity ablation (EmptyDrops MC)\n";
    std::cerr << "  " << N_GENES << " genes, " << N_CELLS << " cells ("
              << N_EASY << " easy + " << N_HARD << " hard), "
              << N_EMPTY << " empties\n";
    std::cerr << "  Easy avg UMI = " << easy_sum / N_EASY
              << ", Hard avg UMI = " << hard_sum / N_HARD
              << ", Empty avg UMI = " << empty_sum / N_EMPTY << "\n";
    std::cerr << "  lower=" << LOWER << " min_umi_test=" << MIN_UMI_TEST
              << " ambient_tot=" << empty_sum << "\n";
    std::cerr << "═══════════════════════════════════════════════════════\n\n";

    // ── 2. Ablation sweep ───────────────────────────────────────────────────
    std::vector<double> pseudo_values = {0.05, 0.1, 0.2, 0.3, 0.5, 1.0};
    std::vector<SweepResult> results;

    for (double pv : pseudo_values) {
        auto r = run_at_pseudo(pv, N_GENES, N_CELLS, N_EMPTY,
                               counts, csc, LOWER, MIN_UMI_TEST,
                               FDR_THR, N_MC);
        results.push_back(r);
    }

    // Print results table
    std::cerr << std::fixed << std::setprecision(4);
    std::cerr << "┌──────────┬────────┬───────────┬────────┬────┬────┬────┬─────┐\n";
    std::cerr << "│  pseudo  │ recall │ precision │  FPR   │ TP │ FP │ FN │  TN │\n";
    std::cerr << "├──────────┼────────┼───────────┼────────┼────┼────┼────┼─────┤\n";
    for (auto& r : results) {
        std::cerr << "│  " << std::setw(6) << r.pseudo
                  << "  │ " << std::setw(6) << r.recall
                  << " │   " << std::setw(6) << r.precision
                  << "  │ " << std::setw(6) << r.fpr
                  << " │ " << std::setw(2) << r.tp
                  << " │ " << std::setw(2) << r.fp
                  << " │ " << std::setw(2) << r.fn
                  << " │ " << std::setw(3) << r.tn
                  << " │\n";
    }
    std::cerr << "└──────────┴────────┴───────────┴────────┴────┴────┴────┴─────┘\n\n";

    // ── 3. Robustness band test ─────────────────────────────────────────────
    std::vector<double> robust_band = {0.1, 0.2, 0.3, 0.5};
    for (double pv : robust_band) {
        const SweepResult* sr = nullptr;
        for (auto& r : results)
            if (std::abs(r.pseudo - pv) < 1e-9) { sr = &r; break; }
        assert(sr != nullptr);

        std::string name_recall = "robustness: pseudo=" + std::to_string(pv) + " recall >= 0.80";
        std::string name_fpr    = "robustness: pseudo=" + std::to_string(pv) + " FPR <= 0.05";

        if (sr->recall >= 0.80) PASS(name_recall.c_str());
        else FAIL(name_recall.c_str(),
                  ("recall=" + std::to_string(sr->recall)).c_str());

        if (sr->fpr <= 0.05) PASS(name_fpr.c_str());
        else FAIL(name_fpr.c_str(),
                  ("FPR=" + std::to_string(sr->fpr)).c_str());
    }

    // ── 4. Optimal-value test ───────────────────────────────────────────────
    //   Minimize (1 - recall) + FPR. Ties broken by preferring the largest
    //   pseudo (most regularization at equal performance).
    double best_score = 1e9;
    double best_pseudo = -1;
    for (auto& r : results) {
        double score = (1.0 - r.recall) + r.fpr;
        if (score < best_score ||
            (std::abs(score - best_score) < 1e-12 && r.pseudo > best_pseudo)) {
            best_score = score;
            best_pseudo = r.pseudo;
        }
    }
    std::cerr << "Optimal pseudo = " << best_pseudo
              << " (score = " << best_score << ")\n\n";

    if (best_pseudo >= 0.1 && best_pseudo <= 0.5) {
        PASS("optimal pseudo in [0.1, 0.5]");
    } else {
        FAIL("optimal pseudo in [0.1, 0.5]",
             ("best_pseudo=" + std::to_string(best_pseudo)).c_str());
    }

    std::cerr << "\n" << passed << " passed, " << failed << " failed\n";
    return failed > 0 ? 1 : 0;
}
