// SPDX-License-Identifier: MIT
// test_barnyard_route.cpp
// Unit tests for G-BARNYARD-ROUTE: per-cell species routing.
//
// Test matrix: 6 genes × 4 cells
//   Genes 0-2: human (ENSG00000001/2/3)
//   Genes 3-5: mouse (ENSMUSG00000001/2/3)
//
//   Cell 0: human singlet  — genes 0,1,2 → counts 10,8,6  ; mouse = 0   → HUMAN
//   Cell 1: mouse singlet  — genes 3,4,5 → counts 15,12,9 ; human = 0   → MOUSE
//   Cell 2: doublet        — genes 0,3   → counts 5,5      ; 50/50       → DOUBLET
//   Cell 3: human singlet  — genes 0,2   → counts 20,18   ; mouse = 0   → HUMAN
//
// Expected after route_barnyard():
//   human matrix: 3 human genes × 2 human cells (cells 0 and 3)
//   mouse matrix: 3 mouse genes × 1 mouse cell  (cell 1)
//   doublet cell (cell 2) excluded from both.

#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "singlet/pileup/barnyard.h"
#include "singlet/pileup/barnyard_route.h"

// ── Minimal CSC struct ────────────────────────────────────────────────────
struct TestCSC {
    uint32_t nrows;  // genes
    uint32_t ncols;  // cells
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<uint32_t> data;
};

// ── Helpers ───────────────────────────────────────────────────────────────

// Sum all counts in a SpeciesCSC column.
[[maybe_unused]] static uint32_t col_sum(const singlet::SpeciesCSC& csc, uint32_t col) {
    uint32_t s = 0;
    for (int32_t i = csc.indptr[col]; i < csc.indptr[col + 1]; ++i)
        s += csc.data[i];
    return s;
}

// Find count for a specific (row, col) in a SpeciesCSC.
[[maybe_unused]] static uint32_t get_val(const singlet::SpeciesCSC& csc, int32_t row, uint32_t col) {
    for (int32_t i = csc.indptr[col]; i < csc.indptr[col + 1]; ++i)
        if (csc.indices[i] == row) return csc.data[i];
    return 0;
}

int main() {
    // ── Build the test CSC (column-sparse, sorted row indices within col) ──
    // 6 genes, 4 cells
    //   Cell 0: g0=10, g1=8, g2=6
    //   Cell 1: g3=15, g4=12, g5=9
    //   Cell 2: g0=5,  g3=5
    //   Cell 3: g0=20, g2=18

    TestCSC csc;
    csc.nrows = 6;
    csc.ncols = 4;
    csc.indptr = {0, 3, 6, 8, 10};
    csc.indices = {0, 1, 2,  // cell 0
                   3, 4, 5,  // cell 1
                   0, 3,     // cell 2
                   0, 2};    // cell 3
    csc.data = {10, 8, 6,    // cell 0
                15, 12, 9,   // cell 1
                5, 5,        // cell 2
                20, 18};     // cell 3

    // ── Gene species ──────────────────────────────────────────────────────
    std::vector<std::string> gene_ids = {
        "ENSG00000001", "ENSG00000002", "ENSG00000003",
        "ENSMUSG00000001", "ENSMUSG00000002", "ENSMUSG00000003"};
    std::vector<std::string> gene_names = {
        "GAPDH", "ACTB", "B2M",
        "Gapdh", "Actb", "B2m"};

    auto gene_species = singlet::classify_gene_species(gene_ids, gene_names);
    // Verify classification
    assert(gene_species[0] == static_cast<uint8_t>(singlet::SpeciesCode::HUMAN));
    assert(gene_species[1] == static_cast<uint8_t>(singlet::SpeciesCode::HUMAN));
    assert(gene_species[2] == static_cast<uint8_t>(singlet::SpeciesCode::HUMAN));
    assert(gene_species[3] == static_cast<uint8_t>(singlet::SpeciesCode::MOUSE));
    assert(gene_species[4] == static_cast<uint8_t>(singlet::SpeciesCode::MOUSE));
    assert(gene_species[5] == static_cast<uint8_t>(singlet::SpeciesCode::MOUSE));

    // ── Barnyard classification ───────────────────────────────────────────
    auto barnyard = singlet::classify_barnyard(csc, gene_species);

    assert(barnyard.calls[0] == singlet::SpeciesCall::HUMAN);
    assert(barnyard.calls[1] == singlet::SpeciesCall::MOUSE);
    assert(barnyard.calls[2] == singlet::SpeciesCall::DOUBLET);
    assert(barnyard.calls[3] == singlet::SpeciesCall::HUMAN);

    // ── Barcodes ──────────────────────────────────────────────────────────
    std::vector<std::string> barcodes = {
        "AAACCCAAGAAACACT", "AACCGAAAGCGTCTAT",
        "ATCGATCGATCGATCG", "TTGGTTGGTTGGTTGG"};

    // ── Route ─────────────────────────────────────────────────────────────
    auto routed = singlet::route_barnyard(
        csc, gene_species, barnyard.calls, gene_names, gene_ids, barcodes);

    const auto& H = routed.human;
    const auto& M = routed.mouse;

    // ── Dimension checks ──────────────────────────────────────────────────
    // Human: 3 human genes × 2 human cells (cells 0 and 3)
    assert(H.nrows == 3 && "human nrows must be 3");
    assert(H.ncols == 2 && "human ncols must be 2");

    // Mouse: 3 mouse genes × 1 mouse cell (cell 1)
    assert(M.nrows == 3 && "mouse nrows must be 3");
    assert(M.ncols == 1 && "mouse ncols must be 1");

    // ── Gene ID/name remapping ────────────────────────────────────────────
    assert(H.gene_ids[0] == "ENSG00000001");
    assert(H.gene_ids[1] == "ENSG00000002");
    assert(H.gene_ids[2] == "ENSG00000003");
    assert(H.gene_names[0] == "GAPDH");

    assert(M.gene_ids[0] == "ENSMUSG00000001");
    assert(M.gene_ids[1] == "ENSMUSG00000002");
    assert(M.gene_ids[2] == "ENSMUSG00000003");
    assert(M.gene_names[0] == "Gapdh");

    // ── Barcode filtering ─────────────────────────────────────────────────
    // Human cells: cell 0 and cell 3
    assert(H.barcodes.size() == 2);
    assert(H.barcodes[0] == barcodes[0]);
    assert(H.barcodes[1] == barcodes[3]);

    // Mouse cells: only cell 1
    assert(M.barcodes.size() == 1);
    assert(M.barcodes[0] == barcodes[1]);

    // ── Species name ──────────────────────────────────────────────────────
    assert(H.species_name == "human");
    assert(M.species_name == "mouse");

    // ── Count correctness ─────────────────────────────────────────────────
    // Human column 0 = original cell 0: g0=10, g1=8, g2=6 → sum=24
    assert(col_sum(H, 0) == 24);
    assert(get_val(H, 0, 0) == 10);  // GAPDH gene (new row 0 = human gene 0)
    assert(get_val(H, 1, 0) == 8);   // ACTB  gene (new row 1)
    assert(get_val(H, 2, 0) == 6);   // B2M   gene (new row 2)

    // Human column 1 = original cell 3: g0=20, g2=18 → sum=38
    assert(col_sum(H, 1) == 38);
    assert(get_val(H, 0, 1) == 20);
    assert(get_val(H, 1, 1) == 0);  // ACTB absent in cell 3
    assert(get_val(H, 2, 1) == 18);

    // Mouse column 0 = original cell 1: g3=15, g4=12, g5=9 → sum=36
    assert(col_sum(M, 0) == 36);
    assert(get_val(M, 0, 0) == 15);  // Gapdh (new row 0 = mouse gene 0)
    assert(get_val(M, 1, 0) == 12);
    assert(get_val(M, 2, 0) == 9);

    // ── Doublet exclusion ─────────────────────────────────────────────────
    // Cell 2 is DOUBLET — its barcode must NOT appear in either species.
    for (const auto& bc : H.barcodes) {
        (void)bc;
        assert(bc != barcodes[2] && "doublet cell must not appear in human matrix");
    }
    for (const auto& bc : M.barcodes) {
        (void)bc;
        assert(bc != barcodes[2] && "doublet cell must not appear in mouse matrix");
    }

    // ── indptr consistency ────────────────────────────────────────────────
    assert(static_cast<int32_t>(H.indptr.size()) == static_cast<int32_t>(H.ncols) + 1);
    assert(static_cast<int32_t>(M.indptr.size()) == static_cast<int32_t>(M.ncols) + 1);
    assert(H.indptr[0] == 0);
    assert(M.indptr[0] == 0);
    assert(static_cast<uint32_t>(H.indptr[H.ncols]) == H.indices.size());
    assert(static_cast<uint32_t>(M.indptr[M.ncols]) == M.indices.size());

    std::cout << "All barnyard_route tests passed.\n";
    std::cout << "  Human: " << H.nrows << " genes x " << H.ncols << " cells\n";
    std::cout << "  Mouse: " << M.nrows << " genes x " << M.ncols << " cells\n";
    return 0;
}
