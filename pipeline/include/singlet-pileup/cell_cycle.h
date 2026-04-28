#pragma once
// singlet-pileup: cell_cycle.h
// N19: Cell cycle phase scoring (Tirosh et al. 2016 / Seurat approach).
//
// Assigns each cell a cell cycle phase (G1, S, G2M) based on expression of
// canonical marker genes, using the simplified score:
//   Score_S   = mean(log1p(counts of S-phase genes with ≥1 read))
//   Score_G2M = mean(log1p(counts of G2M-phase genes with ≥1 read))
// Mean is taken only over genes present in the count matrix; missing genes
// contribute 0 to the sum but do NOT reduce the denominator (conservative).
//
// Phase assignment:
//   S    if Score_S > Score_G2M AND Score_S > threshold
//   G2M  if Score_G2M > Score_S AND Score_G2M > threshold
//   G1   otherwise
//
// Usage:
//   auto result = singlet::score_cell_cycle(exon_csc, engine.gene_model(), barcodes);
//   singlet::write_cell_cycle_tsv(out_prefix + "/cell_cycle_scores.tsv", result, barcodes);

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "gene_model.h"

namespace singlet {

// ─── Tirosh et al. 2016 gene lists ───────────────────────────────────────────

/// S-phase marker genes (Tirosh 2016).
static const std::unordered_set<std::string> kSPhaseGenes = {
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG",
    "GINS2", "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS",
    "RFC2", "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2",
    "UBR7", "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6",
    "EXO1", "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN",
    "POLA1", "CHAF1B", "BRIP1", "E2F8"
};

/// G2M-phase marker genes (Tirosh 2016).
static const std::unordered_set<std::string> kG2MPhaseGenes = {
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A",
    "NDC80", "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3",
    "FAM64A", "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1",
    "KIF11", "ANP32E", "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3",
    "HN1", "CDC20", "TTK", "CDC25C", "KIF2C", "RANGAP1", "NCAPD2",
    "DLGAP5", "CDCA2", "CDCA8", "ECT2", "KIF23", "HMMR", "AURKA",
    "PSRC1", "ANLN", "LBR", "CKAP5", "CENPE", "CTCF", "NEK2", "G2E3",
    "GAS2L3", "CBX5", "CENPA"
};

// ─── Result types ─────────────────────────────────────────────────────────────

/// Per-cell cell cycle score and phase assignment.
struct CellCycleEntry {
    double      s_score   = 0.0;    ///< Mean log1p S-phase gene expression
    double      g2m_score = 0.0;    ///< Mean log1p G2M-phase gene expression
    std::string phase     = "G1";   ///< "G1", "S", or "G2M"
};

/// Result of cell cycle scoring for a whole sample.
struct CellCycleResult {
    std::vector<CellCycleEntry> cells;          ///< One entry per barcode column
    uint32_t n_s_genes_detected   = 0;          ///< S-phase genes found in matrix
    uint32_t n_g2m_genes_detected = 0;          ///< G2M-phase genes found in matrix
    uint32_t n_g1                 = 0;
    uint32_t n_s                  = 0;
    uint32_t n_g2m                = 0;
};

// ─── Core algorithm ──────────────────────────────────────────────────────────

/// Score cell cycle phase for every cell in an exon count CSC matrix.
///
/// @tparam CSCMat   CSC matrix type with indptr, indices, data, nrows, ncols
/// @param  exon_csc Exon count matrix (features × barcodes) in CSC format
/// @param  model    GeneModel loaded from GTF (must have gene hierarchy)
/// @param  threshold Minimum mean-log1p score to call S or G2M (default 0.1)
/// @return          CellCycleResult with per-cell scores and phase calls
template <typename CSCMat>
CellCycleResult score_cell_cycle(const CSCMat& exon_csc,
                                 const GeneModel& model,
                                 double threshold = 0.1) {
    CellCycleResult result;

    if (!model.has_gene_hierarchy() || exon_csc.nrows == 0 || exon_csc.ncols == 0) {
        result.cells.resize(exon_csc.ncols);
        return result;
    }

    // ── Build exon-row → phase lookup ──────────────────────────────────────
    // 0 = other, 1 = S, 2 = G2M
    std::vector<uint8_t> row_phase(exon_csc.nrows, 0);

    std::unordered_set<std::string> s_genes_seen, g2m_genes_seen;

    const auto& genes = model.genes();
    for (const auto& gene : genes) {
        bool is_s   = kSPhaseGenes.count(gene.gene_name)   > 0;
        bool is_g2m = kG2MPhaseGenes.count(gene.gene_name) > 0;
        if (!is_s && !is_g2m) continue;

        if (is_s)   s_genes_seen.insert(gene.gene_name);
        if (is_g2m) g2m_genes_seen.insert(gene.gene_name);

        uint8_t phase_tag = is_s ? 1 : 2;
        for (uint32_t ei : gene.exon_indices) {
            if (ei < exon_csc.nrows) {
                // If an exon row is claimed by both (shouldn't happen), S wins
                if (row_phase[ei] == 0) row_phase[ei] = phase_tag;
            }
        }
    }

    result.n_s_genes_detected   = static_cast<uint32_t>(s_genes_seen.size());
    result.n_g2m_genes_detected = static_cast<uint32_t>(g2m_genes_seen.size());

    const uint32_t n_s_total   = static_cast<uint32_t>(kSPhaseGenes.size());
    const uint32_t n_g2m_total = static_cast<uint32_t>(kG2MPhaseGenes.size());

    // ── Per-cell scoring ───────────────────────────────────────────────────
    // Denominator = total marker genes of each type (not just detected),
    // consistent with Seurat's approach (missing = 0 contribution).
    const uint32_t ncols = exon_csc.ncols;
    result.cells.resize(ncols);

    for (uint32_t j = 0; j < ncols; ++j) {
        double s_sum = 0.0, g2m_sum = 0.0;

        for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k) {
            uint32_t row = static_cast<uint32_t>(exon_csc.indices[k]);
            if (row >= exon_csc.nrows) continue;
            uint8_t ph = row_phase[row];
            if (ph == 0) continue;
            double lv = std::log1p(static_cast<double>(exon_csc.data[k]));
            if (ph == 1) s_sum   += lv;
            else         g2m_sum += lv;
        }

        double s_score   = (n_s_total   > 0) ? s_sum   / n_s_total   : 0.0;
        double g2m_score = (n_g2m_total > 0) ? g2m_sum / n_g2m_total : 0.0;

        std::string phase = "G1";
        if (s_score > g2m_score && s_score > threshold) {
            phase = "S";
        } else if (g2m_score > s_score && g2m_score > threshold) {
            phase = "G2M";
        }

        result.cells[j] = {s_score, g2m_score, phase};

        if (phase == "S")        ++result.n_s;
        else if (phase == "G2M") ++result.n_g2m;
        else                     ++result.n_g1;
    }

    return result;
}

// ─── Output ──────────────────────────────────────────────────────────────────

/// Write cell cycle scores to a TSV file.
///
/// Columns: barcode, phase, s_score, g2m_score
///
/// @param path     Output file path (e.g. "outdir/cell_cycle_scores.tsv")
/// @param result   CellCycleResult from score_cell_cycle()
/// @param barcodes Barcode strings, one per column
inline void write_cell_cycle_tsv(const std::string& path,
                                  const CellCycleResult& result,
                                  const std::vector<std::string>& barcodes) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[cell_cycle] Cannot write: " << path << "\n";
        return;
    }

    f << "barcode\tphase\ts_score\tg2m_score\n";

    const uint32_t n = static_cast<uint32_t>(
        std::min(result.cells.size(), barcodes.size()));

    char buf[64];
    for (uint32_t i = 0; i < n; ++i) {
        const auto& e = result.cells[i];
        f << barcodes[i] << '\t' << e.phase << '\t';
        snprintf(buf, sizeof(buf), "%.6f", e.s_score);   f << buf << '\t';
        snprintf(buf, sizeof(buf), "%.6f", e.g2m_score); f << buf << '\n';
    }
}

}  // namespace singlet
