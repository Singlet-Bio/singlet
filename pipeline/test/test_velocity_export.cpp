// test/test_velocity_export.cpp
// Unit tests for G-VELOCITY: spliced/unspliced/ambiguous per-gene matrices.
//
// Tests:
//   1. collapse_intron_to_gene produces correct gene-level sums
//   2. Spliced + unspliced = total gene counts (when no ambiguous reads)
//   3. make_empty_gene_csc produces correctly shaped all-zero matrix
//   4. Single-intron, single-cell edge case
//   5. Multiple introns per gene are summed correctly

#include <cstdint>
#include <iostream>
#include <vector>

// Only velocity.h — no htslib dependency
#include "singlet-pileup/velocity.h"

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

// ── Minimal mock types ─────────────────────────────────────────────────────

struct MockCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint16_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

static MockCSC make_csc(uint32_t nrows, uint32_t ncols,
                        const std::vector<std::tuple<uint32_t, uint32_t, uint16_t>>& entries) {
    MockCSC csc;
    csc.nrows = nrows;
    csc.ncols = ncols;
    csc.indptr.assign(ncols + 1, 0);

    for (auto& [r, c, v] : entries) csc.indptr[c + 1]++;
    for (uint32_t j = 0; j < ncols; ++j) csc.indptr[j + 1] += csc.indptr[j];

    csc.indices.resize(entries.size());
    csc.data.resize(entries.size());
    std::vector<int32_t> pos(csc.indptr.begin(), csc.indptr.end());
    for (auto& [r, c, v] : entries) {
        int32_t k = pos[c]++;
        csc.indices[k] = static_cast<int32_t>(r);
        csc.data[k] = v;
    }
    return csc;
}

// Minimal GeneModel duck-type: only needs n_genes(), intron_to_gene()
struct MockGeneModel {
    std::vector<uint32_t> intron2gene;
    uint32_t n_genes_ = 0;

    uint32_t n_genes() const { return n_genes_; }
    uint32_t intron_to_gene(uint32_t i) const {
        return i < intron2gene.size() ? intron2gene[i] : UINT32_MAX;
    }
};

// Sum all values for gene g across all cells in a GeneCSC
static uint64_t gene_total(const singlet::GeneCSC<uint32_t>& g_csc, uint32_t gene_idx) {
    uint64_t s = 0;
    for (uint32_t j = 0; j < g_csc.ncols; ++j)
        for (int32_t k = g_csc.indptr[j]; k < g_csc.indptr[j + 1]; ++k)
            if (static_cast<uint32_t>(g_csc.indices[k]) == gene_idx)
                s += g_csc.data[k];
    return s;
}

// ── Test 1: collapse_intron_to_gene basic ─────────────────────────────────
static void test_collapse_intron_basic() {
    std::cout << "Test 1: collapse_intron_to_gene basic gene sums\n";
    // 3 introns → 2 genes, 2 cells
    // intron 0 → gene 0, intron 1 → gene 0, intron 2 → gene 1
    MockGeneModel gm;
    gm.n_genes_ = 2;
    gm.intron2gene = {0, 0, 1};

    auto csc = make_csc(3, 2, {{0, 0, 5}, {2, 0, 3}, {1, 1, 7}, {2, 1, 2}});

    auto out = singlet::collapse_intron_to_gene(csc, gm);

    CHECK(out.nrows == 2, "nrows=2");
    CHECK(out.ncols == 2, "ncols=2");
    // gene 0: cell0=5, cell1=7  → total=12
    // gene 1: cell0=3, cell1=2  → total=5
    CHECK(gene_total(out, 0) == 12, "gene0 total=12 (5+7)");
    CHECK(gene_total(out, 1) == 5, "gene1 total=5  (3+2)");
}

// ── Test 2: spliced + unspliced = total ───────────────────────────────────
static void test_spliced_plus_unspliced_equals_total() {
    std::cout << "Test 2: spliced + unspliced = total gene counts\n";
    // 2 intron intervals → 2 genes, 2 cells (proxy for both exon and intron collapse)
    MockGeneModel gm;
    gm.n_genes_ = 2;
    gm.intron2gene = {0, 1};

    // "spliced" proxy: row 0 (gene 0) and row 1 (gene 1)
    auto exon_csc = make_csc(2, 2, {{0, 0, 10}, {1, 0, 4}, {0, 1, 6}, {1, 1, 8}});
    // "unspliced" proxy: same mapping
    auto intron_csc = make_csc(2, 2, {{0, 0, 3}, {1, 0, 2}, {0, 1, 1}, {1, 1, 5}});

    auto spliced = singlet::collapse_intron_to_gene(exon_csc, gm);
    auto unspliced = singlet::collapse_intron_to_gene(intron_csc, gm);

    // gene0: spliced=10+6=16, unspliced=3+1=4, total=20
    // gene1: spliced=4+8=12,  unspliced=2+5=7, total=19
    CHECK(gene_total(spliced, 0) == 16, "spliced gene0=16");
    CHECK(gene_total(unspliced, 0) == 4, "unspliced gene0=4");
    CHECK(gene_total(spliced, 0) + gene_total(unspliced, 0) == 20,
          "gene0: spliced+unspliced=20");
    CHECK(gene_total(spliced, 1) + gene_total(unspliced, 1) == 19,
          "gene1: spliced+unspliced=19");
}

// ── Test 3: make_empty_gene_csc dimensions ───────────────────────────────
static void test_empty_csc_dimensions() {
    std::cout << "Test 3: make_empty_gene_csc dimensions\n";
    auto empty = singlet::make_empty_gene_csc<uint32_t>(500, 200);
    CHECK(empty.nrows == 500, "nrows=500");
    CHECK(empty.ncols == 200, "ncols=200");
    CHECK(empty.data.empty(), "data is empty");
    CHECK((uint32_t)empty.indptr.size() == 201, "indptr size=ncols+1");
    uint32_t total = 0;
    for (uint32_t j = 0; j < 200; ++j)
        total += static_cast<uint32_t>(empty.indptr[j + 1] - empty.indptr[j]);
    CHECK(total == 0, "all indptr diffs=0");
}

// ── Test 4: single-intron, single-cell ──────────────────────────────────
static void test_single_intron_single_cell() {
    std::cout << "Test 4: single intron, single cell\n";
    MockGeneModel gm;
    gm.n_genes_ = 1;
    gm.intron2gene = {0};

    auto csc = make_csc(1, 1, {{0, 0, 42}});
    auto out = singlet::collapse_intron_to_gene(csc, gm);

    CHECK(out.nrows == 1, "nrows=1");
    CHECK(out.ncols == 1, "ncols=1");
    CHECK(gene_total(out, 0) == 42, "gene0 total=42");
}

// ── Test 5: multiple introns per gene summed ─────────────────────────────
static void test_multi_intron_per_gene() {
    std::cout << "Test 5: multiple introns per gene are summed\n";
    MockGeneModel gm;
    gm.n_genes_ = 1;
    gm.intron2gene = {0, 0, 0, 0, 0};

    auto csc = make_csc(5, 1, {{0, 0, 1}, {1, 0, 2}, {2, 0, 3}, {3, 0, 4}, {4, 0, 5}});
    auto out = singlet::collapse_intron_to_gene(csc, gm);

    CHECK(gene_total(out, 0) == 15, "gene0 total=15 (1+2+3+4+5)");
}

// ── main ─────────────────────────────────────────────────────────────────
int main() {
    test_collapse_intron_basic();
    test_spliced_plus_unspliced_equals_total();
    test_empty_csc_dimensions();
    test_single_intron_single_cell();
    test_multi_intron_per_gene();

    std::cout << "\n"
              << n_pass << "/" << (n_pass + n_fail) << " tests passed\n";
    return n_fail > 0 ? 1 : 0;
}
