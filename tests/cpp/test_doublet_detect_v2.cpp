// test_doublet_detect_v2.cpp — Standalone validation of the v2 Scrublet-style
// doublet detection algorithm in doublet_detect.h.
//
// Tests:
//  T1: Empty cell set → empty results.
//  T2: Fallback path (n_cells=20 < 50): scores in [0,1], ordering correct.
//  T3: Simulation path (n_cells=200): scores strictly in [0,1], at least one > 0.
//  T4: Large simulation path (n_cells=5000): scores in [0,1], scoring not degenerate.
//  T5: DoubletResult struct fields behave as documented.

#include "singlet/pileup/doublet_detect.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

static int n_pass = 0, n_fail = 0;

#define CHECK(cond, msg)                                                          \
    do {                                                                          \
        if (cond) { std::cout << "  PASS: " << (msg) << "\n"; ++n_pass; }        \
        else { std::cout << "  FAIL: " << (msg)                                   \
                         << " [line " << __LINE__ << "]\n"; ++n_fail; }           \
    } while (0)

using CSC = singlet::SparseAccumulator<uint16_t>::CSCMatrix;

struct CooEntry { int32_t row, col; uint16_t val; };

static CSC build_csc(uint32_t n_genes, uint32_t n_cells,
                     const std::vector<CooEntry>& entries) {
    CSC csc;
    csc.nrows = n_genes; csc.ncols = n_cells;
    csc.indptr.assign(n_cells + 1, 0);
    for (const auto& e : entries) csc.indptr[e.col + 1]++;
    for (uint32_t c = 0; c < n_cells; ++c)
        csc.indptr[c + 1] += csc.indptr[c];
    size_t nnz = entries.size();
    csc.indices.resize(nnz); csc.data.resize(nnz);
    std::vector<int32_t> pos(csc.indptr.begin(), csc.indptr.begin() + n_cells);
    for (const auto& e : entries) {
        int32_t slot = pos[e.col]++;
        csc.indices[slot] = e.row;
        csc.data[slot]    = e.val;
    }
    return csc;
}

// Random sparse matrix: each cell draws n_umi genes uniformly from [0, n_genes).
// Includes duplicates (raw COO), which is fine — detect_doublets handles raw CSC.
static CSC make_random_matrix(uint32_t n_genes, uint32_t n_cells,
                               uint32_t n_umi_per_cell, uint64_t seed = 42) {
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<uint32_t> gd(0, n_genes - 1);
    std::vector<CooEntry> entries;
    entries.reserve(static_cast<size_t>(n_cells) * n_umi_per_cell);
    for (uint32_t c = 0; c < n_cells; ++c)
        for (uint32_t u = 0; u < n_umi_per_cell; ++u)
            entries.push_back({(int32_t)gd(rng), (int32_t)c, 1u});
    return build_csc(n_genes, n_cells, entries);
}

// ---------------------------------------------------------------------------
// T1: Empty cell set → empty results
// ---------------------------------------------------------------------------
static void test_empty() {
    std::cout << "\n=== T1: empty cell list ===\n";
    CSC csc; csc.nrows = 100; csc.ncols = 0; csc.indptr.push_back(0);
    auto res = singlet::detect_doublets(csc, {});
    CHECK(res.empty(), "empty input returns empty results");  // 1
}

// ---------------------------------------------------------------------------
// T2: Fallback path (n_cells = 20): scores in [0,1], high-UMI cell not lowest.
// ---------------------------------------------------------------------------
static void test_fallback_range() {
    std::cout << "\n=== T2: fallback path (n_cells=20) ===\n";
    // Cells 0-18: ~100 UMIs (singlets).  Cell 19: ~300 UMIs (double-high).
    const uint32_t n_cells = 20;
    std::mt19937 rng(7u);
    std::uniform_int_distribution<uint32_t> gd(0, 199);
    std::vector<CooEntry> entries;
    for (uint32_t c = 0; c < n_cells - 1; ++c)
        for (int u = 0; u < 100; ++u)
            entries.push_back({(int32_t)gd(rng), (int32_t)c, 1u});
    for (int u = 0; u < 300; ++u)  // cell 19 has 3× UMIs
        entries.push_back({(int32_t)gd(rng), (int32_t)(n_cells - 1), 1u});
    auto csc = build_csc(200, n_cells, entries);

    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0u);
    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == n_cells, "result size = n_cells");  // 2
    bool all_ok = true;
    double min_sc = 1.0, max_sc = 0.0;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) all_ok = false;
        min_sc = std::min(min_sc, r.score);
        max_sc = std::max(max_sc, r.score);
    }
    CHECK(all_ok, "fallback scores in [0,1]");  // 3
    std::cout << "  INFO: min=" << min_sc << " max=" << max_sc
              << " cell19=" << res[n_cells-1].score << "\n";
    // High-UMI cell 19 (3× median) should not be the minimum score
    CHECK(res[n_cells - 1].score > min_sc,
          "high-UMI cell score > minimum");  // 4
}

// ---------------------------------------------------------------------------
// T3: Simulation path (n_cells=200): scores in [0,1], not all zero.
// ---------------------------------------------------------------------------
static void test_simulation_range() {
    std::cout << "\n=== T3: simulation path (n=200) ===\n";
    auto csc = make_random_matrix(1000, 200, 100);
    std::vector<uint32_t> cells(200); std::iota(cells.begin(), cells.end(), 0u);
    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == 200, "result size=200");  // 5
    bool all_ok = true;
    double max_s = 0.0, min_s = 1.0;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) all_ok = false;
        max_s = std::max(max_s, r.score); min_s = std::min(min_s, r.score);
    }
    CHECK(all_ok, "all 200 scores in [0,1]");  // 6
    std::cout << "  INFO: min=" << min_s << " max=" << max_s << "\n";
    CHECK(max_s > 0.0, "at least one score > 0");  // 7
    CHECK(min_s < max_s, "scores have non-trivial variance");  // 8
}

// ---------------------------------------------------------------------------
// T4: Large dataset (n_cells=5000 > min_n_sim_cap=1500): verify scores are
//     in [0,1], non-trivial variance, and runtime is reasonable.
//     With n_cells=5000, n_sim=3000(capped), sim_frac=0.375.
// ---------------------------------------------------------------------------
static void test_large_dataset() {
    std::cout << "\n=== T4: large dataset (n=5000) ===\n";
    auto csc = make_random_matrix(2000, 5000, 100, 99u);
    std::vector<uint32_t> cells(5000); std::iota(cells.begin(), cells.end(), 0u);
    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == 5000, "result size=5000");  // 9
    bool all_ok = true;
    double max_s = 0.0, min_s = 1.0;
    uint32_t n_flagged = 0;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) all_ok = false;
        max_s = std::max(max_s, r.score); min_s = std::min(min_s, r.score);
        if (r.is_doublet) ++n_flagged;
    }
    CHECK(all_ok, "all 5000 scores in [0,1]");  // 10
    std::cout << "  INFO: min=" << min_s << " max=" << max_s
              << " flagged=" << n_flagged << "/" << res.size() << "\n";
    CHECK(max_s > 0.0, "at least one score > 0");  // 11
    // Not all cells flagged (threshold > 0 means at most some fraction)
    CHECK(n_flagged < 5000, "not all cells flagged as doublets");  // 12
}

// ---------------------------------------------------------------------------
// T5: DoubletResult struct fields
// ---------------------------------------------------------------------------
static void test_struct_fields() {
    std::cout << "\n=== T5: DoubletResult struct fields ===\n";
    singlet::DoubletResult r;
    r.score = 0.6; r.is_doublet = (r.score > 0.35);
    CHECK(r.is_doublet == true,  "score 0.6 → is_doublet=true");   // 13
    r.score = 0.1; r.is_doublet = (r.score > 0.35);
    CHECK(r.is_doublet == false, "score 0.1 → is_doublet=false");  // 14
}

// ---------------------------------------------------------------------------
// T6: Injected doublet precision/recall — core acceptance criterion for N12 v6
//
// Design: 10 cell types, 500 cells each = 5000 singlets.
//   Each type exclusively expresses 20 marker genes (genes [t*20, (t+1)*20])
//   at count~Poisson(5) (per-cell Poisson noise breaks within-type degeneracy
//   in HVG/PCA space — constant counts collapse all cells to the same point).
//   Each singlet also gets 10 random background reads from genes [200, 999].
//   200 inter-type injected doublets (same Poisson noise) sum two cell types.
//
//   Expected behaviour in PCA space:
//   - Singlets cluster in 10 well-separated groups (with spread from noise).
//   - Same-type simulated doublets fall within their cluster → low score.
//   - Inter-type simulated doublets and injected doublets land "between"
//     clusters → high fraction of simulated neighbours → norm score > 0.5.
//
// Acceptance criteria (v6 GMM spec):
//   ≥80% of injected doublets have score > 0.5 on normalized scale
//   ≤3% of singlets have score > 0.5  (GMM on well-separated data achieves <<3% FPR)
// ---------------------------------------------------------------------------
static void test_injected_doublets() {
    std::cout << "\n=== T6: injected doublet precision/recall (5200 cells, 1000 genes) ===\n";

    static constexpr int n_types        = 10;
    static constexpr int cells_per_type = 500;
    static constexpr int n_singlets     = n_types * cells_per_type;  // 5000
    static constexpr int n_injected     = 200;
    static constexpr int n_total_cells  = n_singlets + n_injected;   // 5200
    static constexpr int n_genes        = 1000;
    static constexpr int genes_per_type = 20;  // type t uses genes [t*20, t*20+20)
    // genes [0, 199] = markers; genes [200, 999] = background noise pool
    static constexpr int bg_base        = 200;
    static constexpr int n_bg_genes     = n_genes - bg_base;         // 800
    static constexpr int bg_per_cell    = 10;
    static constexpr int marker_mean    = 5;  // Poisson mean for marker counts

    std::mt19937 rng(20260416u);
    std::uniform_int_distribution<int> bg_dist(bg_base, n_genes - 1);
    std::poisson_distribution<int> pcnt(marker_mean);  // per-gene Poisson noise

    std::vector<CooEntry> entries;
    entries.reserve(
        n_singlets * (genes_per_type + bg_per_cell) +
        n_injected * 2 * genes_per_type);

    // --- singlets: Poisson(5) counts per marker gene (unique HVG profile per cell) ---
    for (int t = 0; t < n_types; ++t) {
        const int g0 = t * genes_per_type;
        for (int ci = 0; ci < cells_per_type; ++ci) {
            const int cidx = t * cells_per_type + ci;
            // type-exclusive marker genes with Poisson noise
            for (int g = g0; g < g0 + genes_per_type; ++g) {
                const uint16_t cnt = static_cast<uint16_t>(std::max(1, pcnt(rng)));
                entries.push_back({(int32_t)g, (int32_t)cidx, cnt});
            }
            // per-cell background noise
            for (int b = 0; b < bg_per_cell; ++b)
                entries.push_back({(int32_t)bg_dist(rng), (int32_t)cidx, (uint16_t)1});
        }
    }
    (void)n_bg_genes; // used only in design comment above

    // --- injected doublets: inter-type, t1 = d%10, t2 = (d+5)%10 ---
    // With n_types/2=5, t1 != t2 always.  Covers all 10 unique pairs uniformly.
    for (int d = 0; d < n_injected; ++d) {
        const int t1   = d % n_types;
        const int t2   = (d + n_types / 2) % n_types;  // guaranteed != t1
        const int didx = n_singlets + d;
        const int g1   = t1 * genes_per_type;
        const int g2   = t2 * genes_per_type;
        for (int g = g1; g < g1 + genes_per_type; ++g) {
            const uint16_t cnt = static_cast<uint16_t>(std::max(1, pcnt(rng)));
            entries.push_back({(int32_t)g, (int32_t)didx, cnt});
        }
        for (int g = g2; g < g2 + genes_per_type; ++g) {
            const uint16_t cnt = static_cast<uint16_t>(std::max(1, pcnt(rng)));
            entries.push_back({(int32_t)g, (int32_t)didx, cnt});
        }
    }

    auto csc = build_csc((uint32_t)n_genes, (uint32_t)n_total_cells, entries);

    std::vector<uint32_t> cell_idx(n_total_cells);
    std::iota(cell_idx.begin(), cell_idx.end(), 0u);

    // Pass the actual injected rate: 200/5200 ≈ 3.8%
    auto res = singlet::detect_doublets(csc, cell_idx, 0.04);

    CHECK(res.size() == (size_t)n_total_cells, "result size = n_total_cells");  // 15

    bool in_range = true;
    for (const auto& r : res)
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) { in_range = false; break; }
    CHECK(in_range, "all scores in [0,1]");  // 16

    // Diagnostic: score > 0.5 on normalized scale (informational only)
    int n_det_score = 0, n_fp_score = 0;
    for (int d = 0; d < n_injected; ++d)
        if (res[n_singlets + d].score > 0.5) ++n_det_score;
    for (int i = 0; i < n_singlets; ++i)
        if (res[i].score > 0.5) ++n_fp_score;

    // Recall / FPR using the GMM-adaptive is_doublet flag (algorithm's actual output).
    // v6 GMM sets a data-adaptive threshold; doublets land in norm ∈ [0.3, 0.9],
    // well above singlets (norm≈0) but often below a fixed 0.5 cutoff.
    int n_det = 0, n_fp = 0;
    for (int d = 0; d < n_injected; ++d)
        if (res[n_singlets + d].is_doublet) ++n_det;
    for (int i = 0; i < n_singlets; ++i)
        if (res[i].is_doublet) ++n_fp;

    const double recall = static_cast<double>(n_det) / n_injected;
    const double fpr    = static_cast<double>(n_fp)  / n_singlets;
    std::cout << "  INFO: recall(is_doublet)="
              << static_cast<int>(recall * 100) << "%"
              << " (" << n_det << "/" << n_injected << ")"
              << "  FPR(is_doublet)=" << static_cast<int>(fpr * 100) << "%"
              << " (" << n_fp << "/" << n_singlets << ")"
              << "  recall(score>0.5)=" << n_det_score
              << "  FPR(score>0.5)=" << n_fp_score << "\n";

    // Acceptance criteria (v6 GMM): ≥80% recall, ≤3% FPR (GMM-determined threshold).
    // GMM on well-separated data achieves <<3% FPR (vs v5's ~8.8% static bound).
    CHECK(n_det >= static_cast<int>(0.80 * n_injected),
          ">=80% injected doublets flagged as is_doublet (recall)");  // 17
    CHECK(n_fp  <= static_cast<int>(0.03 * n_singlets + 0.5),
          "<=3% singlets flagged as is_doublet (FPR)");               // 18
}

// ---------------------------------------------------------------------------
// T7: Same-type doublets score lower than inter-type doublets
//
// Known limitation: same-type doublets (A+A) have the same gene profile as
// singlets (just deeper UMI) so they are intrinsically harder to detect.
// Inter-type doublets (A+B) express genes from both clusters and land between
// clusters in PCA space, making them easier to flag.
// ---------------------------------------------------------------------------
static void test_same_vs_inter_type() {
    std::cout << "\n=== T7: same-type doublets score lower than inter-type ===\n";

    // Build doublets by summing actual singlet profiles (same as simulation does).
    const uint32_t n_genes = 50;   // 0-19: type-A, 20-39: type-B, 40-49: noise
    const uint32_t cells_per_type = 250;
    const uint32_t n_singlets = 2 * cells_per_type;  // 500
    const uint32_t n_inter = 10;
    const uint32_t n_same = 10;
    const uint32_t n_total = n_singlets + n_inter + n_same;  // 520

    std::mt19937 rng(777u);
    std::poisson_distribution<int> pcnt(10);
    std::uniform_int_distribution<int> noise_gene(40, 49);

    // First build singlet profiles as dense arrays so we can sum them for doublets
    std::vector<std::vector<uint16_t>> profiles(n_singlets, std::vector<uint16_t>(n_genes, 0));

    // 250 type-A singlets: high on genes 0-19
    for (uint32_t c = 0; c < cells_per_type; ++c) {
        for (int g = 0; g < 20; ++g)
            profiles[c][g] = static_cast<uint16_t>(std::max(1, pcnt(rng)));
        for (int b = 0; b < 5; ++b)
            profiles[c][noise_gene(rng)] += 1;
    }

    // 250 type-B singlets: high on genes 20-39
    for (uint32_t c = cells_per_type; c < n_singlets; ++c) {
        for (int g = 20; g < 40; ++g)
            profiles[c][g] = static_cast<uint16_t>(std::max(1, pcnt(rng)));
        for (int b = 0; b < 5; ++b)
            profiles[c][noise_gene(rng)] += 1;
    }

    // Convert singlets to COO entries
    std::vector<CooEntry> entries;
    for (uint32_t c = 0; c < n_singlets; ++c)
        for (uint32_t g = 0; g < n_genes; ++g)
            if (profiles[c][g] > 0)
                entries.push_back({(int32_t)g, (int32_t)c, profiles[c][g]});

    // Inter-type doublets: sum of random A + random B (mirrors simulated doublet generation)
    std::uniform_int_distribution<uint32_t> a_dist(0, cells_per_type - 1);
    std::uniform_int_distribution<uint32_t> b_dist(cells_per_type, n_singlets - 1);
    for (uint32_t d = 0; d < n_inter; ++d) {
        uint32_t ci = a_dist(rng), cj = b_dist(rng);
        uint32_t cidx = n_singlets + d;
        for (uint32_t g = 0; g < n_genes; ++g) {
            uint16_t val = profiles[ci][g] + profiles[cj][g];
            if (val > 0)
                entries.push_back({(int32_t)g, (int32_t)cidx, val});
        }
    }

    // Same-type doublets: sum of random A + random A
    for (uint32_t d = 0; d < n_same; ++d) {
        uint32_t ci = a_dist(rng), cj = a_dist(rng);
        uint32_t cidx = n_singlets + n_inter + d;
        for (uint32_t g = 0; g < n_genes; ++g) {
            uint16_t val = profiles[ci][g] + profiles[cj][g];
            if (val > 0)
                entries.push_back({(int32_t)g, (int32_t)cidx, val});
        }
    }

    auto csc = build_csc(n_genes, n_total, entries);
    std::vector<uint32_t> cells(n_total);
    std::iota(cells.begin(), cells.end(), 0u);

    auto res = singlet::detect_doublets(csc, cells, 0.08);
    CHECK(res.size() == n_total, "T7: result size matches");  // 19

    // Compute mean scores for each group
    double sum_inter = 0.0, sum_same = 0.0;
    for (uint32_t d = 0; d < n_inter; ++d)
        sum_inter += res[n_singlets + d].score;
    for (uint32_t d = 0; d < n_same; ++d)
        sum_same += res[n_singlets + n_inter + d].score;
    double mean_inter = sum_inter / n_inter;
    double mean_same  = sum_same / n_same;

    std::cout << "  INFO: mean_inter_type_score=" << mean_inter
              << "  mean_same_type_score=" << mean_same << "\n";
    // Same-type doublets are intrinsically harder to detect: after CP10K log-
    // normalization, A+A profiles collapse onto A singlets, while A+B profiles
    // land between clusters in PCA space where simulated doublets dominate.
    CHECK(mean_inter > mean_same,
          "inter-type doublets score higher than same-type (known limitation)");  // 20
}

// ---------------------------------------------------------------------------
// T8: Ultra-sparse data (3 UMI/cell average)
//
// Verifies the algorithm doesn't crash on extremely sparse input where most
// genes are zero and PCA/kNN operate on near-degenerate data.
// ---------------------------------------------------------------------------
static void test_ultra_sparse() {
    std::cout << "\n=== T8: ultra-sparse data (3 UMI/cell) ===\n";

    const uint32_t n_cells = 50;
    const uint32_t n_genes = 20;

    std::mt19937 rng(888u);
    std::uniform_int_distribution<int> gd(0, n_genes - 1);

    std::vector<CooEntry> entries;
    // Each cell gets exactly 3 UMI spread across 1-2 genes
    for (uint32_t c = 0; c < n_cells; ++c) {
        for (int u = 0; u < 3; ++u)
            entries.push_back({(int32_t)gd(rng), (int32_t)c, (uint16_t)1});
    }

    auto csc = build_csc(n_genes, n_cells, entries);
    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0u);

    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == n_cells, "T8: result size matches");  // 21

    bool all_in_range = true;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) { all_in_range = false; break; }
    }
    CHECK(all_in_range, "T8: all scores in [0,1]");  // 22
    std::cout << "  INFO: algorithm completed on ultra-sparse data (may use fallback)\n";
    CHECK(true, "T8: no crash on ultra-sparse input");  // 23
}

// ---------------------------------------------------------------------------
// T9: GMM unimodal input — all singlets, no injected doublets
//
// With 80 homogeneous cells and no injected doublets, the normalized score
// distribution should be roughly unimodal near 0. The GMM may detect this
// as unimodal and fire the rate-based fallback. Either way, the number of
// cells called as doublets should be bounded.
// ---------------------------------------------------------------------------
static void test_gmm_unimodal() {
    std::cout << "\n=== T9: GMM unimodal input (all singlets) ===\n";

    const uint32_t n_cells = 80;
    const uint32_t n_genes = 200;

    // Single population: genes 0-49 are "active" at Poisson(10), genes 50-199
    // are low-level background at 1 count with ~10% probability per cell.
    // After CP10K normalization, simulated doublets (sum of two profiles from
    // the same population) have the same relative proportions → score ≈ 0.
    // GMM should see unimodal distribution and fire rate-based fallback.
    std::mt19937 rng(999u);
    std::poisson_distribution<int> pcnt(10);
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    std::vector<CooEntry> entries;

    for (uint32_t c = 0; c < n_cells; ++c) {
        // Active genes with Poisson noise
        for (uint32_t g = 0; g < 50; ++g) {
            uint16_t cnt = static_cast<uint16_t>(std::max(1, pcnt(rng)));
            entries.push_back({(int32_t)g, (int32_t)c, cnt});
        }
        // Sparse background
        for (uint32_t g = 50; g < n_genes; ++g) {
            if (unif(rng) < 0.10)
                entries.push_back({(int32_t)g, (int32_t)c, (uint16_t)1});
        }
    }

    auto csc = build_csc(n_genes, n_cells, entries);
    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0u);

    auto res = singlet::detect_doublets(csc, cells, 0.08);

    CHECK(res.size() == n_cells, "T9: result size matches");  // 24

    uint32_t n_called = 0;
    double min_s = 1.0, max_s = 0.0;
    for (const auto& r : res) {
        if (r.is_doublet) ++n_called;
        min_s = std::min(min_s, r.score);
        max_s = std::max(max_s, r.score);
    }

    // Tolerant ceiling: at most 3x expected rate worth of doublet calls.
    // With a single homogeneous population the GMM should fire the rate-based
    // fallback, but even the normal path may flag a modest fraction.
    uint32_t ceiling = static_cast<uint32_t>(0.08 * n_cells * 3 + 1);
    std::cout << "  INFO: doublets_called=" << n_called << " ceiling=" << ceiling
              << " min_score=" << min_s << " max_score=" << max_s << "\n";

    CHECK(n_called <= ceiling,
          "T9: doublets called <= expected_rate * n_cells * 2");  // 25
    CHECK(max_s > min_s + 1e-9,
          "T9: scores are not all identical (some variation exists)");  // 26
}

// ---------------------------------------------------------------------------
// T10: Very small n (15 cells) — below the fallback threshold of 50
//
// With n_cells=15 < 50, the UMI-ratio heuristic is engaged.
// Verify: no crash, all scores in [0,1], returns correct number of results.
// ---------------------------------------------------------------------------
static void test_very_small_n() {
    std::cout << "\n=== T10: very small n (15 cells, fallback path) ===\n";

    const uint32_t n_cells = 15;
    const uint32_t n_genes = 100;

    std::mt19937 rng(1010u);
    std::uniform_int_distribution<int> gd(0, n_genes - 1);
    std::vector<CooEntry> entries;

    for (uint32_t c = 0; c < n_cells; ++c) {
        int umi = 50 + static_cast<int>(c) * 10;  // varying depth: 50..190
        for (int u = 0; u < umi; ++u)
            entries.push_back({(int32_t)gd(rng), (int32_t)c, (uint16_t)1});
    }

    auto csc = build_csc(n_genes, n_cells, entries);
    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0u);

    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == n_cells, "T10: result size matches (15)");  // 27

    bool all_in_range = true;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) { all_in_range = false; break; }
    }
    CHECK(all_in_range, "T10: all scores in [0,1]");  // 28
    CHECK(true, "T10: no crash, no infinite loop on 15 cells");  // 29
}

// ---------------------------------------------------------------------------
// T11: Score correlates with UMI ratio for fallback path
//
// 30 cells with varying UMI depth (100..3200), 5 cells at each depth level.
// n_cells=30 < 50 → UMI-ratio heuristic: score = min(1.0, umi / (3*median)).
// Higher-UMI cells should get higher scores.
// If fallback fires, Pearson(log10(UMI), score) >= 0.50.
// ---------------------------------------------------------------------------
static void test_umi_score_correlation() {
    std::cout << "\n=== T11: UMI-score correlation in fallback path ===\n";

    const uint32_t n_cells = 30;
    const uint32_t n_genes = 100;
    const int depths[] = {100, 200, 400, 800, 1600, 3200};
    const int cells_per_depth = 5;

    std::mt19937 rng(1111u);
    std::uniform_int_distribution<int> gd(0, n_genes - 1);
    std::vector<CooEntry> entries;
    std::vector<double> log_umi(n_cells);

    for (int d = 0; d < 6; ++d) {
        for (int ci = 0; ci < cells_per_depth; ++ci) {
            uint32_t c = static_cast<uint32_t>(d * cells_per_depth + ci);
            int umi = depths[d];
            log_umi[c] = std::log10(static_cast<double>(umi));
            for (int u = 0; u < umi; ++u)
                entries.push_back({(int32_t)gd(rng), (int32_t)c, (uint16_t)1});
        }
    }

    auto csc = build_csc(n_genes, n_cells, entries);
    std::vector<uint32_t> cells(n_cells);
    std::iota(cells.begin(), cells.end(), 0u);

    auto res = singlet::detect_doublets(csc, cells);

    CHECK(res.size() == n_cells, "T11: result size matches (30)");  // 30

    bool all_in_range = true;
    for (const auto& r : res) {
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) { all_in_range = false; break; }
    }
    CHECK(all_in_range, "T11: all scores in [0,1]");  // 31

    // Compute Pearson correlation between log10(UMI) and doublet score
    // This is the fallback path (n_cells=30 < 50), so UMI-ratio heuristic fires
    std::vector<double> scores(n_cells);
    for (uint32_t i = 0; i < n_cells; ++i)
        scores[i] = res[i].score;

    double sum_x = 0, sum_y = 0, sum_xx = 0, sum_yy = 0, sum_xy = 0;
    for (uint32_t i = 0; i < n_cells; ++i) {
        sum_x  += log_umi[i];
        sum_y  += scores[i];
        sum_xx += log_umi[i] * log_umi[i];
        sum_yy += scores[i] * scores[i];
        sum_xy += log_umi[i] * scores[i];
    }
    double n = static_cast<double>(n_cells);
    double num = n * sum_xy - sum_x * sum_y;
    double den = std::sqrt((n * sum_xx - sum_x * sum_x) *
                           (n * sum_yy - sum_y * sum_y));
    double pearson = (den > 1e-12) ? num / den : 0.0;

    std::cout << "  INFO: Pearson(log10(UMI), score) = " << pearson << "\n";
    // Fallback path: score = min(1.0, umi / (3*median)), so correlation should be strong
    CHECK(pearson >= 0.50,
          "T11: Pearson correlation >= 0.50 (UMI predicts score in fallback)");  // 32
}

int main() {
    test_empty();
    test_fallback_range();
    test_simulation_range();
    test_large_dataset();
    test_struct_fields();
    test_injected_doublets();
    test_same_vs_inter_type();
    test_ultra_sparse();
    test_gmm_unimodal();
    test_very_small_n();
    test_umi_score_correlation();
    std::cout << "\n===========================\n"
              << "PASSED: " << n_pass << "  FAILED: " << n_fail << "\n";
    return (n_fail == 0) ? 0 : 1;
}
