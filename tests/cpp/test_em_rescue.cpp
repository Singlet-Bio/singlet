// test/test_em_rescue.cpp
// Unit tests for G-EM: equivalence-class EM multi-mapper rescue.
//
// Tests:
//   1. Basic 3-gene, 2-EC EM — known expected proportional output
//   2. Convergence within 50 iterations
//   3. Single-cell counts sum correctly
//   4. Reads with unknown second gene (UINT32_MAX) are skipped
//   5. UMI deduplication within same (bc, ec)
//   6. Empty ambig_reads produces empty output
//   7. EM with uniform prior converges to 50/50

#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <vector>

#include "singlet/pileup/em_rescue.h"

using namespace singlet;
using namespace singlet::em_rescue;

static int n_pass = 0;
static int n_fail = 0;

#define CHECK(cond, name)                                                        \
    do {                                                                         \
        if (cond) {                                                              \
            std::cout << "  PASS: " << (name) << "\n";                           \
            ++n_pass;                                                            \
        } else {                                                                 \
            std::cout << "  FAIL: " << (name) << " [line " << __LINE__ << "]\n"; \
            ++n_fail;                                                            \
        }                                                                        \
    } while (0)

#define CHECK_NEAR(a, b, eps, name) CHECK(std::abs((double)(a) - (double)(b)) <= (eps), name)

// Helper: build a minimal GeneCSC from a dense matrix [n_genes x n_cells]
struct SimpleGeneCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint32_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

static SimpleGeneCSC make_gene_csc(const std::vector<std::vector<uint32_t>>& mat) {
    // mat[gene][cell]
    uint32_t n_genes = static_cast<uint32_t>(mat.size());
    uint32_t n_cells = n_genes > 0 ? static_cast<uint32_t>(mat[0].size()) : 0;
    SimpleGeneCSC csc;
    csc.nrows = n_genes;
    csc.ncols = n_cells;
    csc.indptr.assign(n_cells + 1, 0);
    // Build CSC: iterate column-first
    for (uint32_t c = 0; c < n_cells; ++c) {
        for (uint32_t g = 0; g < n_genes; ++g) {
            if (mat[g][c] > 0) {
                csc.indices.push_back(static_cast<int32_t>(g));
                csc.data.push_back(mat[g][c]);
            }
        }
        csc.indptr[c + 1] = static_cast<int32_t>(csc.indices.size());
    }
    return csc;
}

static AmbigRead make_ar(int32_t bc, const char* umi, uint32_t ga, uint32_t gb) {
    AmbigRead ar;
    ar.bc_idx = bc;
    ar.gene_a = ga;
    ar.gene_b = gb;
    ar.umi_len = static_cast<uint8_t>(std::strlen(umi));
    std::memcpy(ar.umi, umi, ar.umi_len);
    return ar;
}

// ── Test 1: Basic proportional rescue ─────────────────────────────────────
// 3 genes A=0, B=1, C=2, 1 cell
// Unique counts: A=100, B=50, C=10
// EC1: {A,B} with 20 reads; EC2: {B,C} with 10 reads
// Expected: A gets ~13.3 from EC1, B gets ~6.7+~8.3, C gets ~1.7
static void test_basic_em() {
    std::cout << "Test 1: basic proportional EM rescue\n";

    // 1 cell
    // gene_csc: gene 0=A=100, gene 1=B=50, gene 2=C=10
    auto csc = make_gene_csc({{100}, {50}, {10}});  // [gene][cell]

    std::vector<AmbigRead> ambig;
    // EC1: {A(0), B(1)}, 20 reads
    for (int i = 0; i < 20; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "ec1u%03d", i);
        ambig.push_back(make_ar(0, umi, 0, 1));
    }
    // EC2: {B(1), C(2)}, 10 reads
    for (int i = 0; i < 10; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "ec2u%03d", i);
        ambig.push_back(make_ar(0, umi, 1, 2));
    }

    EMRescueStats stats;
    auto em_csc = run_em_rescue(ambig, csc, /*max_iter=*/100, /*tol=*/0.001, &stats);

    CHECK(stats.n_ambig_reads == 30, "30 ambig reads submitted");
    CHECK(stats.n_valid_reads == 30, "30 valid (both genes known)");
    CHECK(stats.n_deduped_umis == 30, "30 unique UMIs (all distinct)");
    CHECK(stats.n_equiv_classes == 2, "2 equivalence classes");
    CHECK(stats.n_iterations <= 50, "converged within 50 iterations");
    CHECK(em_csc.nrows == 3, "3 genes in output");
    CHECK(em_csc.ncols == 1, "1 cell in output");

    // Extract rescued counts per gene for cell 0
    std::vector<uint32_t> rescued(3, 0);
    for (int32_t k = em_csc.indptr[0]; k < em_csc.indptr[1]; ++k)
        rescued[em_csc.indices[k]] = em_csc.data[k];

    uint32_t total_rescued = rescued[0] + rescued[1] + rescued[2];
    CHECK(total_rescued >= 25 && total_rescued <= 30,
          "total rescued ~25-30 (rounding from fractions)");

    // A gets ~13 from EC1 (100/150*20=13.3), B gets ~7+8=15.3, C gets ~1.7
    CHECK(rescued[0] >= 10 && rescued[0] <= 17, "gene A rescued 10-17");
    CHECK(rescued[1] >= 10 && rescued[1] <= 20, "gene B rescued 10-20");
    CHECK(rescued[2] >= 0 && rescued[2] <= 5, "gene C rescued 0-5");
}

// ── Test 2: Convergence ───────────────────────────────────────────────────
static void test_convergence() {
    std::cout << "Test 2: EM convergence within max_iter\n";

    auto csc = make_gene_csc({{200, 50}, {100, 30}, {10, 5}});
    std::vector<AmbigRead> ambig;
    for (int bc = 0; bc < 2; ++bc) {
        for (int i = 0; i < 50; ++i) {
            char umi[10];
            std::snprintf(umi, sizeof(umi), "b%du%03d", bc, i);
            ambig.push_back(make_ar(bc, umi, 0, 1));
        }
        for (int i = 0; i < 20; ++i) {
            char umi[10];
            std::snprintf(umi, sizeof(umi), "b%dv%03d", bc, i);
            ambig.push_back(make_ar(bc, umi, 1, 2));
        }
    }

    EMRescueStats stats;
    run_em_rescue(ambig, csc, /*max_iter=*/50, /*tol=*/0.01, &stats);

    CHECK(stats.n_iterations <= 50, "converged within 50 iterations");
    CHECK(stats.n_iterations >= 2, "ran at least 2 iterations");
}

// ── Test 3: Per-cell counts sum ───────────────────────────────────────────
static void test_per_cell_sum() {
    std::cout << "Test 3: per-cell rescued counts sum check\n";

    // 2 cells, 2 genes
    auto csc = make_gene_csc({{100, 200}, {50, 40}});

    std::vector<AmbigRead> ambig;
    // Cell 0: EC {0,1} with 10 reads
    for (int i = 0; i < 10; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "c0u%02d", i);
        ambig.push_back(make_ar(0, umi, 0, 1));
    }
    // Cell 1: EC {0,1} with 15 reads
    for (int i = 0; i < 15; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "c1u%02d", i);
        ambig.push_back(make_ar(1, umi, 0, 1));
    }

    EMRescueStats stats;
    auto em_csc = run_em_rescue(ambig, csc, 50, 0.01, &stats);

    // Each cell: sum of rescued counts over genes should equal that cell's
    // UMI-deduped EC count (allowing for rounding: total <= n_reads, >= n_reads - n_genes)
    auto cell_sum = [&](uint32_t c) -> uint32_t {
        uint32_t s = 0;
        for (int32_t k = em_csc.indptr[c]; k < em_csc.indptr[c + 1]; ++k)
            s += em_csc.data[k];
        return s;
    };

    uint32_t s0 = cell_sum(0);
    uint32_t s1 = cell_sum(1);
    CHECK(s0 >= 8 && s0 <= 10, "cell 0 total rescued in [8,10]");
    CHECK(s1 >= 13 && s1 <= 15, "cell 1 total rescued in [13,15]");
}

// ── Test 4: Reads with unknown gene_b are skipped ─────────────────────────
static void test_skip_unknown() {
    std::cout << "Test 4: reads with UINT32_MAX gene_b are skipped\n";

    auto csc = make_gene_csc({{100}, {50}});

    std::vector<AmbigRead> ambig;
    // 5 valid reads
    for (int i = 0; i < 5; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "v%04d", i);
        ambig.push_back(make_ar(0, umi, 0, 1));
    }
    // 3 invalid (gene_b = UINT32_MAX)
    for (int i = 0; i < 3; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "x%04d", i);
        ambig.push_back(make_ar(0, umi, 0, UINT32_MAX));
    }

    EMRescueStats stats;
    run_em_rescue(ambig, csc, 50, 0.01, &stats);

    CHECK(stats.n_valid_reads == 5, "only 5 valid reads (3 skipped)");
    CHECK(stats.n_deduped_umis == 5, "5 unique UMIs counted");
}

// ── Test 5: UMI deduplication within same equivalence class ───────────────
static void test_umi_dedup() {
    std::cout << "Test 5: UMI dedup within same (bc, ec)\n";

    auto csc = make_gene_csc({{100}, {50}});

    std::vector<AmbigRead> ambig;
    // 10 reads with 5 distinct UMIs → should deduplicate to 5
    for (int i = 0; i < 10; ++i) {
        char umi[6];
        std::snprintf(umi, sizeof(umi), "u%04d", i % 5);
        ambig.push_back(make_ar(0, umi, 0, 1));
    }

    EMRescueStats stats;
    auto em_csc = run_em_rescue(ambig, csc, 50, 0.01, &stats);

    CHECK(stats.n_deduped_umis == 5, "5 unique UMIs after dedup");
    uint32_t total = 0;
    for (int32_t k = em_csc.indptr[0]; k < em_csc.indptr[1]; ++k)
        total += em_csc.data[k];
    CHECK(total <= 5, "total rescued counts <= 5 (UMI-bounded)");
}

// ── Test 6: Empty input ───────────────────────────────────────────────────
static void test_empty_input() {
    std::cout << "Test 6: empty ambig_reads produces empty output\n";

    auto csc = make_gene_csc({{100, 50}, {30, 20}});
    std::vector<AmbigRead> ambig;

    EMRescueStats stats;
    auto em_csc = run_em_rescue(ambig, csc, 50, 0.01, &stats);

    CHECK(stats.n_ambig_reads == 0, "0 ambig reads");
    CHECK(stats.n_rescued_ints == 0, "0 rescued counts");
    CHECK(em_csc.nrows == 2, "correct nrows");
    CHECK(em_csc.ncols == 2, "correct ncols");
    bool all_zero = true;
    for (int32_t c = 0; c < 2; ++c)
        if (em_csc.indptr[c + 1] > em_csc.indptr[c]) all_zero = false;
    CHECK(all_zero, "no nonzero entries in output");
}

// ── Test 7: Uniform prior → 50/50 split ──────────────────────────────────
static void test_uniform_prior() {
    std::cout << "Test 7: uniform prior yields ~50/50 split\n";

    // Both genes have equal counts
    auto csc = make_gene_csc({{100}, {100}});

    std::vector<AmbigRead> ambig;
    for (int i = 0; i < 20; ++i) {
        char umi[8];
        std::snprintf(umi, sizeof(umi), "u%04d", i);
        ambig.push_back(make_ar(0, umi, 0, 1));
    }

    EMRescueStats stats;
    auto em_csc = run_em_rescue(ambig, csc, 50, 0.01, &stats);

    std::vector<uint32_t> rescued(2, 0);
    for (int32_t k = em_csc.indptr[0]; k < em_csc.indptr[1]; ++k)
        rescued[em_csc.indices[k]] = em_csc.data[k];

    // With equal prior, both genes should get ~10 each (within 2)
    CHECK(rescued[0] >= 8 && rescued[0] <= 12, "gene 0 gets ~10 with uniform prior");
    CHECK(rescued[1] >= 8 && rescued[1] <= 12, "gene 1 gets ~10 with uniform prior");
    CHECK(rescued[0] + rescued[1] >= 18 && rescued[0] + rescued[1] <= 20,
          "total ~20 with 20 input reads");
}

// ── Test 8: AmbigRead struct is POD-compatible ────────────────────────────
static void test_ambig_struct() {
    std::cout << "Test 8: AmbigRead struct layout\n";

    AmbigRead ar;
    CHECK(ar.bc_idx == -1, "default bc_idx = -1");
    CHECK(ar.gene_a == UINT32_MAX, "default gene_a = UINT32_MAX");
    CHECK(ar.gene_b == UINT32_MAX, "default gene_b = UINT32_MAX");
    CHECK(ar.umi_len == 0, "default umi_len = 0");
}

// ── main ──────────────────────────────────────────────────────────────────

int main() {
    std::cout << "=== test_em_rescue (G-EM) ===\n";

    test_basic_em();
    test_convergence();
    test_per_cell_sum();
    test_skip_unknown();
    test_umi_dedup();
    test_empty_input();
    test_uniform_prior();
    test_ambig_struct();

    std::cout << "\n=== Results: " << n_pass << " passed, " << n_fail << " failed ===\n";
    return (n_fail == 0) ? 0 : 1;
}
