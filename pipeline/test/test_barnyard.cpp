// test/test_barnyard.cpp
// Unit tests for G-BARNYARD: per-cell species classification from dual-genome.
//
// Tests:
//   1. classify_gene_species — Ensembl ID prefix detection (ENSG, ENSMUSG)
//   2. classify_gene_species — CellRanger three-underscore prefix (GRCh38___, GRCm39___)
//   3. classify_gene_species — simple underscore prefix (hg38_, mm10_)
//   4. is_barnyard_experiment — true for mixed, false for single-species
//   5. classify_barnyard — Cell 0: 90% human → HUMAN
//   6. classify_barnyard — Cell 1: 95% mouse → MOUSE
//   7. classify_barnyard — Cell 2: 50/50 → DOUBLET
//   8. classify_barnyard — Cell 3: 85% human → AMBIGUOUS
//   9. classify_barnyard — correct summary counts
//  10. classify_barnyard — cell with 0 UMIs → UNKNOWN
//  11. write_barnyard_tsv — columns, header, and data correct
//  12. human_fraction computed correctly

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "singlet-pileup/barnyard.h"

using namespace singlet;

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

// Minimal CSC (column-sparse) for tests
struct SimpleCSC {
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint32_t> data;
    uint32_t nrows;
    uint32_t ncols;
};

// Build a CSC from a dense [n_genes x n_cells] matrix
static SimpleCSC make_csc(const std::vector<std::vector<uint32_t>>& mat) {
    uint32_t n_genes = static_cast<uint32_t>(mat.size());
    uint32_t n_cells = n_genes > 0 ? static_cast<uint32_t>(mat[0].size()) : 0;
    SimpleCSC csc;
    csc.nrows = n_genes;
    csc.ncols = n_cells;
    csc.indptr.assign(n_cells + 1, 0);
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

// ── Test 1: Ensembl ID prefix ──────────────────────────────────────────
static void test_ensembl_prefix() {
    std::cout << "Test 1: Ensembl gene ID prefix detection\n";
    std::vector<std::string> ids = {"ENSG00000111640", "ENSMUSG00000000001", "ENSDARG00000000001", "LOC123"};
    std::vector<std::string> names = {"GAPDH", "Gapdh", "gapdh", "LOC123"};
    auto sp = classify_gene_species(ids, names);
    CHECK(sp[0] == static_cast<uint8_t>(SpeciesCode::HUMAN), "ENSG → HUMAN");
    CHECK(sp[1] == static_cast<uint8_t>(SpeciesCode::MOUSE), "ENSMUSG → MOUSE");
    CHECK(sp[2] == static_cast<uint8_t>(SpeciesCode::ZEBRAFISH), "ENSDARG → ZEBRAFISH");
    CHECK(sp[3] == static_cast<uint8_t>(SpeciesCode::UNKNOWN), "LOC123 → UNKNOWN");
}

// ── Test 2: CellRanger three-underscore prefix ─────────────────────────
static void test_cellranger_prefix() {
    std::cout << "Test 2: CellRanger three-underscore gene name prefix\n";
    std::vector<std::string> ids = {"unknown0", "unknown1", "unknown2", "unknown3"};
    std::vector<std::string> names = {"GRCh38___GAPDH", "GRCm39___Gapdh", "hg38___TP53", "mm10___Trp53"};
    auto sp = classify_gene_species(ids, names);
    CHECK(sp[0] == static_cast<uint8_t>(SpeciesCode::HUMAN), "GRCh38___* → HUMAN");
    CHECK(sp[1] == static_cast<uint8_t>(SpeciesCode::MOUSE), "GRCm39___* → MOUSE");
    CHECK(sp[2] == static_cast<uint8_t>(SpeciesCode::HUMAN), "hg38___* → HUMAN");
    CHECK(sp[3] == static_cast<uint8_t>(SpeciesCode::MOUSE), "mm10___* → MOUSE");
}

// ── Test 3: Simple underscore prefix ──────────────────────────────────
static void test_simple_prefix() {
    std::cout << "Test 3: Simple underscore gene name prefix\n";
    std::vector<std::string> ids = {"unknownA", "unknownB", "unknownC", "unknownD"};
    std::vector<std::string> names = {"hg38_GAPDH", "mm10_Gapdh", "GRCh38_TP53", "GRCm39_Trp53"};
    auto sp = classify_gene_species(ids, names);
    CHECK(sp[0] == static_cast<uint8_t>(SpeciesCode::HUMAN), "hg38_* → HUMAN");
    CHECK(sp[1] == static_cast<uint8_t>(SpeciesCode::MOUSE), "mm10_* → MOUSE");
    CHECK(sp[2] == static_cast<uint8_t>(SpeciesCode::HUMAN), "GRCh38_* → HUMAN");
    CHECK(sp[3] == static_cast<uint8_t>(SpeciesCode::MOUSE), "GRCm39_* → MOUSE");
}

// ── Test 4: is_barnyard_experiment ────────────────────────────────────
static void test_is_barnyard() {
    std::cout << "Test 4: is_barnyard_experiment detection\n";
    // Mixed: human + mouse
    std::vector<std::string> ids_mix = {"ENSG00000111640", "ENSMUSG00000000001"};
    std::vector<std::string> names_mix = {"GAPDH", "Gapdh"};
    CHECK(is_barnyard_experiment(ids_mix, names_mix), "human+mouse → barnyard");

    // Single species — human only
    std::vector<std::string> ids_h = {"ENSG00000111640", "ENSG00000139618"};
    std::vector<std::string> names_h = {"GAPDH", "BRCA2"};
    CHECK(!is_barnyard_experiment(ids_h, names_h), "human-only → not barnyard");

    // Single species — mouse only
    std::vector<std::string> ids_m = {"ENSMUSG00000000001", "ENSMUSG00000000003"};
    std::vector<std::string> names_m = {"Gnai3", "Pbsn"};
    CHECK(!is_barnyard_experiment(ids_m, names_m), "mouse-only → not barnyard");

    // All unknown
    std::vector<std::string> ids_u = {"LOC1", "LOC2"};
    std::vector<std::string> names_u = {"gene1", "gene2"};
    CHECK(!is_barnyard_experiment(ids_u, names_u), "all-unknown → not barnyard");
}

// ── Tests 5-10: classify_barnyard ────────────────────────────────────--
//
// 6 genes: genes 0-2 are human (ENSG*), genes 3-5 are mouse (ENSMUSG*)
// 5 cells:
//   Cell 0: 90 human UMIs, 10 mouse UMIs → 90% human → HUMAN
//   Cell 1: 5 human UMIs,  95 mouse UMIs → 95% mouse → MOUSE
//   Cell 2: 50 human UMIs, 50 mouse UMIs → 50% each  → DOUBLET
//   Cell 3: 85 human UMIs, 15 mouse UMIs → 85% human → AMBIGUOUS
//   Cell 4: 0 human UMIs,  0 mouse UMIs  → no UMIs   → UNKNOWN
//
// Distribute per-species counts across 3 genes each for realism.
static void test_classify_barnyard_cells() {
    std::cout << "Tests 5-10: classify_barnyard per-cell calls\n";

    // gene_species: 0,1,2=human; 3,4,5=mouse
    std::vector<uint8_t> gs = {
        static_cast<uint8_t>(SpeciesCode::HUMAN),
        static_cast<uint8_t>(SpeciesCode::HUMAN),
        static_cast<uint8_t>(SpeciesCode::HUMAN),
        static_cast<uint8_t>(SpeciesCode::MOUSE),
        static_cast<uint8_t>(SpeciesCode::MOUSE),
        static_cast<uint8_t>(SpeciesCode::MOUSE),
    };

    // mat[gene][cell]: 6 genes × 5 cells
    // Distribute human across genes 0-2, mouse across genes 3-5
    std::vector<std::vector<uint32_t>> mat = {
        // gene 0 (human)
        {30, 2, 17, 30, 0},
        // gene 1 (human)
        {30, 1, 17, 30, 0},
        // gene 2 (human)
        {30, 2, 16, 25, 0},
        // gene 3 (mouse)
        {4, 32, 17, 5, 0},
        // gene 4 (mouse)
        {3, 32, 17, 5, 0},
        // gene 5 (mouse)
        {3, 31, 16, 5, 0},
    };
    // Cell counts:  H=90,M=10 | H=5,M=95 | H=50,M=50 | H=85,M=15 | H=0,M=0
    // doublet_min_fraction=0.20: cell3 mouse=15%<20% → AMBIGUOUS (not DOUBLET)

    SimpleCSC csc = make_csc(mat);
    BarnyardResult res = classify_barnyard(csc, gs, 0.9f, 0.2f);

    // Test 5: Cell 0 → HUMAN (90% human)
    CHECK(res.calls[0] == SpeciesCall::HUMAN, "Test 5: cell0 90% human → HUMAN");
    CHECK(res.human_umis[0] == 90, "Test 5: cell0 human_umis=90");

    // Test 6: Cell 1 → MOUSE (95% mouse)
    CHECK(res.calls[1] == SpeciesCall::MOUSE, "Test 6: cell1 95% mouse → MOUSE");
    CHECK(res.mouse_umis[1] == 95, "Test 6: cell1 mouse_umis=95");

    // Test 7: Cell 2 → DOUBLET (50/50)
    CHECK(res.calls[2] == SpeciesCall::DOUBLET, "Test 7: cell2 50/50 → DOUBLET");

    // Test 8: Cell 3 → AMBIGUOUS (85% human, below 90% threshold)
    CHECK(res.calls[3] == SpeciesCall::AMBIGUOUS, "Test 8: cell3 85% human → AMBIGUOUS");

    // Test 9: Summary counts
    CHECK(res.n_human == 1, "Test 9: n_human=1");
    CHECK(res.n_mouse == 1, "Test 9: n_mouse=1");
    CHECK(res.n_doublet == 1, "Test 9: n_doublet=1");
    CHECK(res.n_ambiguous == 1, "Test 9: n_ambiguous=1");

    // Test 10: Cell 4 → UNKNOWN (zero UMIs)
    CHECK(res.calls[4] == SpeciesCall::UNKNOWN, "Test 10: cell4 empty → UNKNOWN");
}

// ── Test 11: human_fraction ────────────────────────────────────────────
static void test_human_fraction() {
    std::cout << "Test 11: human_fraction accuracy\n";
    std::vector<uint8_t> gs = {
        static_cast<uint8_t>(SpeciesCode::HUMAN),
        static_cast<uint8_t>(SpeciesCode::MOUSE),
    };
    // 1 human gene, 1 mouse gene, 1 cell: 3 human + 7 mouse = 10 total, fraction=0.3
    std::vector<std::vector<uint32_t>> mat = {{3}, {7}};
    SimpleCSC csc = make_csc(mat);
    BarnyardResult res = classify_barnyard(csc, gs, 0.9f);
    CHECK_NEAR(res.human_fraction[0], 0.3f, 1e-5f, "human_fraction = 0.3000");
}

// ── Test 12: write_barnyard_tsv ────────────────────────────────────────
static void test_write_tsv() {
    std::cout << "Test 12: write_barnyard_tsv output format\n";

    BarnyardResult res;
    res.calls = {SpeciesCall::HUMAN, SpeciesCall::MOUSE, SpeciesCall::DOUBLET};
    res.human_umis = {90, 5, 50};
    res.mouse_umis = {10, 95, 50};
    res.human_fraction = {0.9f, 0.05f, 0.5f};
    std::vector<std::string> bcs = {"AAACCTGAGAAACCAT", "AAACCTGAGAAACCGC", "AAACCTGAGAAACCTG"};

    const std::string path = "/tmp/test_barnyard_out.tsv";
    bool ok = write_barnyard_tsv(path, res, bcs);
    CHECK(ok, "write_barnyard_tsv returns true");

    std::ifstream in(path);
    std::string header;
    std::getline(in, header);
    CHECK(header == "barcode\tspecies_call\thuman_umis\tmouse_umis\thuman_fraction",
          "TSV header correct");

    std::string line1;
    std::getline(in, line1);
    // Should start with barcode and contain "human"
    CHECK(line1.find("AAACCTGAGAAACCAT") != std::string::npos, "TSV row 1 has barcode");
    CHECK(line1.find("human") != std::string::npos, "TSV row 1 species=human");
    CHECK(line1.find("90") != std::string::npos, "TSV row 1 human_umis=90");

    std::string line2;
    std::getline(in, line2);
    CHECK(line2.find("mouse") != std::string::npos, "TSV row 2 species=mouse");

    std::string line3;
    std::getline(in, line3);
    CHECK(line3.find("doublet") != std::string::npos, "TSV row 3 species=doublet");
}

// ── main ──────────────────────────────────────────────────────────────
int main() {
    std::cout << "=== G-BARNYARD unit tests ===\n\n";
    test_ensembl_prefix();
    test_cellranger_prefix();
    test_simple_prefix();
    test_is_barnyard();
    test_classify_barnyard_cells();
    test_human_fraction();
    test_write_tsv();

    std::cout << "\n=== Results: " << n_pass << " passed, " << n_fail << " failed ===\n";
    return (n_fail == 0) ? 0 : 1;
}
