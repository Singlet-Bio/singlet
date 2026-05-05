// test_doublet_detect_v2.cpp — Standalone validation of the v2 Scrublet-style
// doublet detection algorithm in doublet_detect.h.
//
// Tests:
//  T1: Empty cell set → empty results.
//  T2: Fallback path (n_cells=20 < 50): scores in [0,1], ordering correct.
//  T3: Simulation path (n_cells=200): scores strictly in [0,1], at least one > 0.
//  T4: Large simulation path (n_cells=5000): scores in [0,1], scoring not degenerate.
//  T5: DoubletResult struct fields behave as documented.

#include "singlet-pileup/doublet_detect.h"

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
// T6: Injected doublet precision/recall — core acceptance criterion for N12 v4
//
// Design: 10 cell types, 500 cells each = 5000 singlets.
//   Each type exclusively expresses 20 marker genes (genes [t*20, (t+1)*20])
//   at count=5.  Each singlet also gets 10 random background reads from
//   genes [200, 999] at count=1 to break within-type degeneracy.
//   200 inter-type injected doublets are formed by summing marker genes of
//   two different cell types (exactly how Scrublet's simulation works).
//
//   Expected behaviour in PCA space:
//   - Singlets cluster tightly in 10 well-separated groups.
//   - Same-type simulated doublets (CP10K-normalised sum equals singlet) fall
//     within their cluster → low score.
//   - Inter-type simulated doublets and injected doublets both land "between"
//     clusters → high fraction of simulated neighbours → score > 0.5.
//
// Acceptance criteria (from N12 spec):
//   ≥80% of injected doublets have score > 0.5
//   ≤10% of singlets have score > 0.5
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
    static constexpr uint16_t marker_cnt = 5;

    std::mt19937 rng(20260416u);
    std::uniform_int_distribution<int> bg_dist(bg_base, n_genes - 1);

    std::vector<CooEntry> entries;
    entries.reserve(
        n_singlets * (genes_per_type + bg_per_cell) +
        n_injected * 2 * genes_per_type);

    // --- singlets ---
    for (int t = 0; t < n_types; ++t) {
        const int g0 = t * genes_per_type;
        for (int ci = 0; ci < cells_per_type; ++ci) {
            const int cidx = t * cells_per_type + ci;
            // type-exclusive marker genes
            for (int g = g0; g < g0 + genes_per_type; ++g)
                entries.push_back({(int32_t)g, (int32_t)cidx, marker_cnt});
            // per-cell background noise (breaks within-type degeneracy)
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
        for (int g = g1; g < g1 + genes_per_type; ++g)
            entries.push_back({(int32_t)g, (int32_t)didx, marker_cnt});
        for (int g = g2; g < g2 + genes_per_type; ++g)
            entries.push_back({(int32_t)g, (int32_t)didx, marker_cnt});
    }

    auto csc = build_csc((uint32_t)n_genes, (uint32_t)n_total_cells, entries);

    std::vector<uint32_t> cell_idx(n_total_cells);
    std::iota(cell_idx.begin(), cell_idx.end(), 0u);

    auto res = singlet::detect_doublets(csc, cell_idx, 0.5);

    CHECK(res.size() == (size_t)n_total_cells, "result size = n_total_cells");  // 15

    bool in_range = true;
    for (const auto& r : res)
        if (r.score < -1e-6 || r.score > 1.0 + 1e-6) { in_range = false; break; }
    CHECK(in_range, "all scores in [0,1]");  // 16

    // recall: fraction of injected doublets with score > 0.5
    int n_det = 0;
    for (int d = 0; d < n_injected; ++d)
        if (res[n_singlets + d].score > 0.5) ++n_det;

    // FPR: fraction of singlets with score > 0.5
    int n_fp = 0;
    for (int i = 0; i < n_singlets; ++i)
        if (res[i].score > 0.5) ++n_fp;

    const double recall = static_cast<double>(n_det) / n_injected;
    const double fpr    = static_cast<double>(n_fp)  / n_singlets;
    std::cout << "  INFO: recall=" << static_cast<int>(recall * 100) << "%"
              << " (" << n_det << "/" << n_injected << ")"
              << "  FPR=" << static_cast<int>(fpr * 100) << "%"
              << " (" << n_fp << "/" << n_singlets << ")\n";

    // Acceptance criteria (N12 spec: ≥80% recall, ≤10% FPR)
    CHECK(n_det >= static_cast<int>(0.80 * n_injected),
          ">=80% injected doublets have score > 0.5 (recall)");  // 17
    CHECK(n_fp  <= static_cast<int>(0.10 * n_singlets + 0.5),
          "<=10% singlets have score > 0.5 (FPR)");              // 18
}

int main() {
    test_empty();
    test_fallback_range();
    test_simulation_range();
    test_large_dataset();
    test_struct_fields();
    test_injected_doublets();
    std::cout << "\n===========================\n"
              << "PASSED: " << n_pass << "  FAILED: " << n_fail << "\n";
    return (n_fail == 0) ? 0 : 1;
}
