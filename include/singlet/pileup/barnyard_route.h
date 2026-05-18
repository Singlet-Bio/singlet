// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: barnyard_route.h
// G-BARNYARD-ROUTE: Per-cell species routing for barnyard mixture experiments.
//
// After classify_barnyard() assigns each cell to HUMAN/MOUSE/DOUBLET/AMBIGUOUS,
// this module splits the unified gene × cell CSC into per-species submatrices:
//   - human matrix:  human genes (ENSG*)  × human singlet cells
//   - mouse matrix:  mouse genes (ENSMUSG*) × mouse singlet cells
// Doublet and ambiguous cells are excluded from both.

#include <cstdint>
#include <string>
#include <vector>

#include "barnyard.h"

namespace singlet {

// ── Per-species CSC submatrix ────────────────────────────────────────────

struct SpeciesCSC {
    // CSC sparse matrix: nrows (genes) × ncols (cells)
    std::vector<int32_t> indptr;   // size ncols+1
    std::vector<int32_t> indices;  // gene row indices (0-based within species subset)
    std::vector<uint32_t> data;    // UMI counts
    uint32_t nrows = 0;            // number of genes in this species
    uint32_t ncols = 0;            // number of singlet cells for this species

    std::vector<std::string> gene_names;  // remapped to species subset
    std::vector<std::string> gene_ids;    // remapped to species subset
    std::vector<std::string> barcodes;    // filtered to singlet cells
    std::string species_name;             // "human" or "mouse"
};

// ── Routing result ───────────────────────────────────────────────────────

struct BarnyardRouteResult {
    SpeciesCSC human;
    SpeciesCSC mouse;
    // Doublet/ambiguous/unknown cells are excluded from both matrices.
};

// ── Core routing function ────────────────────────────────────────────────

/// Route a unified barnyard gene × cell CSC into per-species submatrices.
///
/// @tparam CSC  Any struct with:
///              - uint32_t nrows, ncols
///              - std::vector<int32_t> indptr   [ncols+1]
///              - std::vector<int32_t> indices  (gene row indices)
///              - std::vector<uint32_t> data    (counts)
///
/// @param gene_csc     Full gene × cell CSC from pileup (nrows=genes, ncols=cells).
/// @param gene_species Per-gene species assignment from classify_gene_species().
///                     Values: 0=UNKNOWN, 1=HUMAN, 2=MOUSE (SpeciesCode).
/// @param cell_calls   Per-cell species calls from classify_barnyard().
/// @param gene_names   Full gene name list (length = gene_csc.nrows).
/// @param gene_ids     Full gene ID list (length = gene_csc.nrows).
/// @param barcodes     Full barcode list (length = gene_csc.ncols).
/// @return             BarnyardRouteResult with human and mouse SpeciesCSC.
template <typename CSC>
inline BarnyardRouteResult route_barnyard(
    const CSC& gene_csc,
    const std::vector<uint8_t>& gene_species,
    const std::vector<SpeciesCall>& cell_calls,
    const std::vector<std::string>& gene_names,
    const std::vector<std::string>& gene_ids,
    const std::vector<std::string>& barcodes) {
    const uint32_t n_genes = gene_csc.nrows;
    const uint32_t n_cells = gene_csc.ncols;

    BarnyardRouteResult result;

    // ── Build gene index maps for each species ───────────────────────────
    // gene_map_sp[old_gene_idx] = new_gene_idx_in_species (-1 = not in species)
    std::vector<int32_t> human_gene_map(n_genes, -1);
    std::vector<int32_t> mouse_gene_map(n_genes, -1);

    uint32_t n_human_genes = 0;
    uint32_t n_mouse_genes = 0;

    for (uint32_t g = 0; g < n_genes && g < gene_species.size(); ++g) {
        if (gene_species[g] == static_cast<uint8_t>(SpeciesCode::HUMAN)) {
            result.human.gene_names.push_back(g < gene_names.size() ? gene_names[g] : "");
            result.human.gene_ids.push_back(g < gene_ids.size() ? gene_ids[g] : "");
            human_gene_map[g] = static_cast<int32_t>(n_human_genes++);
        } else if (gene_species[g] == static_cast<uint8_t>(SpeciesCode::MOUSE)) {
            result.mouse.gene_names.push_back(g < gene_names.size() ? gene_names[g] : "");
            result.mouse.gene_ids.push_back(g < gene_ids.size() ? gene_ids[g] : "");
            mouse_gene_map[g] = static_cast<int32_t>(n_mouse_genes++);
        }
    }

    result.human.nrows = n_human_genes;
    result.mouse.nrows = n_mouse_genes;
    result.human.species_name = "human";
    result.mouse.species_name = "mouse";

    // ── Build cell index maps ────────────────────────────────────────────
    // Only HUMAN singlets → human matrix; MOUSE singlets → mouse matrix.
    uint32_t n_human_cells = 0;
    uint32_t n_mouse_cells = 0;

    std::vector<int32_t> human_cell_map(n_cells, -1);
    std::vector<int32_t> mouse_cell_map(n_cells, -1);

    for (uint32_t c = 0; c < n_cells && c < cell_calls.size(); ++c) {
        if (cell_calls[c] == SpeciesCall::HUMAN) {
            result.human.barcodes.push_back(c < barcodes.size() ? barcodes[c] : "");
            human_cell_map[c] = static_cast<int32_t>(n_human_cells++);
        } else if (cell_calls[c] == SpeciesCall::MOUSE) {
            result.mouse.barcodes.push_back(c < barcodes.size() ? barcodes[c] : "");
            mouse_cell_map[c] = static_cast<int32_t>(n_mouse_cells++);
        }
    }

    result.human.ncols = n_human_cells;
    result.mouse.ncols = n_mouse_cells;

    // ── Allocate indptr arrays ───────────────────────────────────────────
    result.human.indptr.assign(n_human_cells + 1, 0);
    result.mouse.indptr.assign(n_mouse_cells + 1, 0);

    // ── Pass 1: count nnz per output column ─────────────────────────────
    for (uint32_t c = 0; c < n_cells; ++c) {
        int32_t hcol = (c < cell_calls.size() && cell_calls[c] == SpeciesCall::HUMAN)
                           ? human_cell_map[c]
                           : -1;
        int32_t mcol = (c < cell_calls.size() && cell_calls[c] == SpeciesCall::MOUSE)
                           ? mouse_cell_map[c]
                           : -1;

        if (hcol < 0 && mcol < 0) continue;

        int32_t col_start = gene_csc.indptr[c];
        int32_t col_end = gene_csc.indptr[c + 1];

        for (int32_t i = col_start; i < col_end; ++i) {
            uint32_t g = static_cast<uint32_t>(gene_csc.indices[i]);
            if (g >= n_genes) continue;
            if (hcol >= 0 && human_gene_map[g] >= 0)
                ++result.human.indptr[hcol + 1];
            if (mcol >= 0 && mouse_gene_map[g] >= 0)
                ++result.mouse.indptr[mcol + 1];
        }
    }

    // ── Prefix-sum indptr ────────────────────────────────────────────────
    for (uint32_t c = 0; c < n_human_cells; ++c)
        result.human.indptr[c + 1] += result.human.indptr[c];
    for (uint32_t c = 0; c < n_mouse_cells; ++c)
        result.mouse.indptr[c + 1] += result.mouse.indptr[c];

    uint32_t human_nnz = static_cast<uint32_t>(result.human.indptr[n_human_cells]);
    uint32_t mouse_nnz = static_cast<uint32_t>(result.mouse.indptr[n_mouse_cells]);

    result.human.indices.resize(human_nnz);
    result.human.data.resize(human_nnz);
    result.mouse.indices.resize(mouse_nnz);
    result.mouse.data.resize(mouse_nnz);

    // ── Pass 2: fill entries (use fill-position cursor per column) ───────
    std::vector<int32_t> h_cursor(n_human_cells, 0);
    std::vector<int32_t> m_cursor(n_mouse_cells, 0);
    for (uint32_t c = 0; c < n_human_cells; ++c)
        h_cursor[c] = result.human.indptr[c];
    for (uint32_t c = 0; c < n_mouse_cells; ++c)
        m_cursor[c] = result.mouse.indptr[c];

    for (uint32_t c = 0; c < n_cells; ++c) {
        if (c >= cell_calls.size()) continue;
        int32_t hcol = (cell_calls[c] == SpeciesCall::HUMAN) ? human_cell_map[c] : -1;
        int32_t mcol = (cell_calls[c] == SpeciesCall::MOUSE) ? mouse_cell_map[c] : -1;
        if (hcol < 0 && mcol < 0) continue;

        int32_t col_start = gene_csc.indptr[c];
        int32_t col_end = gene_csc.indptr[c + 1];

        for (int32_t i = col_start; i < col_end; ++i) {
            uint32_t g = static_cast<uint32_t>(gene_csc.indices[i]);
            uint32_t v = gene_csc.data[i];
            if (g >= n_genes) continue;

            if (hcol >= 0) {
                int32_t new_g = human_gene_map[g];
                if (new_g >= 0) {
                    int32_t pos = h_cursor[hcol]++;
                    result.human.indices[pos] = new_g;
                    result.human.data[pos] = v;
                }
            }
            if (mcol >= 0) {
                int32_t new_g = mouse_gene_map[g];
                if (new_g >= 0) {
                    int32_t pos = m_cursor[mcol]++;
                    result.mouse.indices[pos] = new_g;
                    result.mouse.data[pos] = v;
                }
            }
        }
    }

    return result;
}

}  // namespace singlet
