// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: barnyard.h
// G-BARNYARD: Per-cell species classification from dual-genome alignment.
//
// For mixed-species (barnyard) experiments, classifies each cell as
// human, mouse, doublet, or ambiguous based on gene-level UMI counts.
// Enables doublet rate estimation and cross-contamination QC.
//
// Classification rules (default thresholds):
//   HUMAN:     human_fraction >= singlet_threshold    (default 0.90)
//   MOUSE:     mouse_fraction  >= singlet_threshold   (default 0.90)
//   DOUBLET:   both species have fraction >= doublet_min_fraction  (default 0.20)
//   AMBIGUOUS: below singlet threshold, with one species < doublet_min_fraction
//              (e.g. 85/15 split — likely contaminated singlet, not true doublet)
//
// Gene species detection (priority order):
//   1. Gene ID prefix:   ENSG* → human, ENSMUSG* → mouse, ENSDARG* → zebrafish
//   2. Gene name prefix: GRCh38_*, hg38_* → human; GRCm39_*, mm10_* → mouse
//   3. Three-underscore CellRanger joint-ref style: GRCh38___* / GRCm39___*

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace singlet {

// ── Species enum ─────────────────────────────────────────────────────────

// Species codes stored in the gene_species vector.
// uint8_t allows extension to >2 species in future.
enum class SpeciesCode : uint8_t {
    UNKNOWN = 0,
    HUMAN = 1,
    MOUSE = 2,
    ZEBRAFISH = 3,
    // Future: add species 4, 5, …
};

// Per-cell classification result.
enum class SpeciesCall { HUMAN,
                         MOUSE,
                         DOUBLET,
                         AMBIGUOUS,
                         UNKNOWN };

// ── Result struct ────────────────────────────────────────────────────────

struct BarnyardResult {
    std::vector<SpeciesCall> calls;     // per-cell species assignment
    std::vector<uint32_t> human_umis;   // per-cell human UMI count
    std::vector<uint32_t> mouse_umis;   // per-cell mouse UMI count
    std::vector<float> human_fraction;  // human / (human + mouse)

    uint32_t n_human = 0;
    uint32_t n_mouse = 0;
    uint32_t n_doublet = 0;
    uint32_t n_ambiguous = 0;
};

// ── Gene species classification ──────────────────────────────────────────

// Infer species from a single gene ID or name.
// Returns SpeciesCode (UNKNOWN if unrecognised).
inline SpeciesCode infer_gene_species_one(const std::string& gene_id,
                                          const std::string& gene_name) {
    // Priority 1 — Ensembl gene ID prefix
    if (gene_id.size() >= 4) {
        if (gene_id.substr(0, 4) == "ENSG") return SpeciesCode::HUMAN;
        if (gene_id.size() >= 7 &&
            gene_id.substr(0, 7) == "ENSMUSG") return SpeciesCode::MOUSE;
        if (gene_id.size() >= 7 &&
            gene_id.substr(0, 7) == "ENSDARG") return SpeciesCode::ZEBRAFISH;
    }

    // Priority 2 — CellRanger three-underscore joint-ref prefix on gene name
    // e.g. "GRCh38___GAPDH", "GRCm39___Gapdh"
    auto has_prefix3 = [](const std::string& s, const std::string& pfx) {
        if (s.size() < pfx.size() + 3) return false;
        if (s.substr(0, pfx.size()) != pfx) return false;
        return s.substr(pfx.size(), 3) == "___";
    };
    if (has_prefix3(gene_name, "GRCh38") || has_prefix3(gene_name, "hg38"))
        return SpeciesCode::HUMAN;
    if (has_prefix3(gene_name, "GRCm39") || has_prefix3(gene_name, "mm10"))
        return SpeciesCode::MOUSE;

    // Priority 3 — Simple underscore prefix on gene name
    // e.g. "hg38_GAPDH", "mm10_Gapdh", "GRCh38_GAPDH", "GRCm39_Gapdh"
    auto has_prefix = [](const std::string& s, const std::string& pfx) {
        if (s.size() <= pfx.size() + 1) return false;
        return s.substr(0, pfx.size() + 1) == pfx + "_";
    };
    if (has_prefix(gene_name, "GRCh38") || has_prefix(gene_name, "hg38"))
        return SpeciesCode::HUMAN;
    if (has_prefix(gene_name, "GRCm39") || has_prefix(gene_name, "mm10"))
        return SpeciesCode::MOUSE;

    return SpeciesCode::UNKNOWN;
}

/// Classify every gene in the model by species.
/// Returns a vector of SpeciesCode, one per gene.
inline std::vector<uint8_t> classify_gene_species(
    const std::vector<std::string>& gene_ids,
    const std::vector<std::string>& gene_names) {
    const size_t n = gene_ids.size();
    std::vector<uint8_t> result(n, static_cast<uint8_t>(SpeciesCode::UNKNOWN));
    for (size_t i = 0; i < n; ++i) {
        const std::string& nm = (i < gene_names.size()) ? gene_names[i] : "";
        result[i] = static_cast<uint8_t>(infer_gene_species_one(gene_ids[i], nm));
    }
    return result;
}

/// Returns true when the gene_species vector contains at least one human gene
/// AND at least one mouse gene — i.e. this looks like a barnyard experiment.
inline bool is_barnyard_experiment(const std::vector<uint8_t>& gene_species) {
    bool has_human = false, has_mouse = false;
    for (uint8_t sp : gene_species) {
        if (sp == static_cast<uint8_t>(SpeciesCode::HUMAN)) has_human = true;
        if (sp == static_cast<uint8_t>(SpeciesCode::MOUSE)) has_mouse = true;
        if (has_human && has_mouse) return true;
    }
    return false;
}

/// Convenience overload that classifies genes and checks for barnyard.
inline bool is_barnyard_experiment(const std::vector<std::string>& gene_ids,
                                   const std::vector<std::string>& gene_names) {
    return is_barnyard_experiment(classify_gene_species(gene_ids, gene_names));
}

// ── Per-cell classification ──────────────────────────────────────────────

/// Run barnyard classification on a CSC gene-count matrix.
///
/// @tparam CSC  Any struct with:
///              - uint32_t ncols  (number of cells)
///              - uint32_t nrows  (number of genes)
///              - std::vector<int32_t> indptr   [ncols+1]
///              - std::vector<int32_t> indices  (gene indices, row-major within col)
///              - std::vector<uint32_t> data    (counts)
///
/// @param gene_species       Output of classify_gene_species(), one uint8_t per gene.
/// @param singlet_threshold  Fraction threshold to call singlet (default 0.90).
///                           A cell is HUMAN if human_fraction >= singlet_threshold.
/// @param doublet_min_fraction  Minimum fraction EACH species must have to call DOUBLET
///                              (default 0.20). Cells below singlet_threshold but with
///                              one species < doublet_min_fraction are AMBIGUOUS rather
///                              than DOUBLET (e.g. 85/15 split is likely a contaminated
///                              singlet, not a true doublet).
template <typename CSC>
inline BarnyardResult classify_barnyard(const CSC& gene_csc,
                                        const std::vector<uint8_t>& gene_species,
                                        float singlet_threshold = 0.9f,
                                        float doublet_min_fraction = 0.2f) {
    const uint32_t n_cells = gene_csc.ncols;
    BarnyardResult res;
    res.calls.resize(n_cells, SpeciesCall::UNKNOWN);
    res.human_umis.assign(n_cells, 0u);
    res.mouse_umis.assign(n_cells, 0u);
    res.human_fraction.assign(n_cells, 0.0f);

    const float doublet_min = doublet_min_fraction;

    for (uint32_t c = 0; c < n_cells; ++c) {
        int32_t col_start = gene_csc.indptr[c];
        int32_t col_end = gene_csc.indptr[c + 1];

        uint32_t h_cnt = 0, m_cnt = 0;
        for (int32_t i = col_start; i < col_end; ++i) {
            uint32_t g = static_cast<uint32_t>(gene_csc.indices[i]);
            uint32_t v = gene_csc.data[i];
            if (g < gene_species.size()) {
                uint8_t sp = gene_species[g];
                if (sp == static_cast<uint8_t>(SpeciesCode::HUMAN))
                    h_cnt += v;
                else if (sp == static_cast<uint8_t>(SpeciesCode::MOUSE))
                    m_cnt += v;
            }
        }

        res.human_umis[c] = h_cnt;
        res.mouse_umis[c] = m_cnt;

        uint32_t total = h_cnt + m_cnt;
        if (total == 0) {
            res.calls[c] = SpeciesCall::UNKNOWN;
            res.human_fraction[c] = 0.0f;
            continue;
        }

        float hf = static_cast<float>(h_cnt) / static_cast<float>(total);
        float mf = 1.0f - hf;
        res.human_fraction[c] = hf;

        if (hf >= singlet_threshold) {
            res.calls[c] = SpeciesCall::HUMAN;
            ++res.n_human;
        } else if (mf >= singlet_threshold) {
            res.calls[c] = SpeciesCall::MOUSE;
            ++res.n_mouse;
        } else if (hf >= doublet_min && mf >= doublet_min) {
            res.calls[c] = SpeciesCall::DOUBLET;
            ++res.n_doublet;
        } else {
            res.calls[c] = SpeciesCall::AMBIGUOUS;
            ++res.n_ambiguous;
        }
    }
    return res;
}

// ── TSV output ───────────────────────────────────────────────────────────

/// Write barnyard results to a TSV file.
/// Columns: barcode  species_call  human_umis  mouse_umis  human_fraction
inline bool write_barnyard_tsv(const std::string& path,
                               const BarnyardResult& result,
                               const std::vector<std::string>& barcodes) {
    std::ofstream ofs(path);
    if (!ofs.is_open()) return false;

    ofs << "barcode\tspecies_call\thuman_umis\tmouse_umis\thuman_fraction\n";

    auto call_str = [](SpeciesCall c) -> const char* {
        switch (c) {
            case SpeciesCall::HUMAN:
                return "human";
            case SpeciesCall::MOUSE:
                return "mouse";
            case SpeciesCall::DOUBLET:
                return "doublet";
            case SpeciesCall::AMBIGUOUS:
                return "ambiguous";
            default:
                return "unknown";
        }
    };

    const size_t n = result.calls.size();
    for (size_t i = 0; i < n; ++i) {
        const std::string& bc = (i < barcodes.size()) ? barcodes[i] : "";
        ofs << bc << '\t'
            << call_str(result.calls[i]) << '\t'
            << result.human_umis[i] << '\t'
            << result.mouse_umis[i] << '\t';
        // Write human_fraction with 4 decimal places
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4f", result.human_fraction[i]);
        ofs << buf << '\n';
    }
    return ofs.good();
}

}  // namespace singlet
