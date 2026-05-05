#pragma once
// singlet-pileup: cell_qc_metrics.h
// Per-cell QC metrics computed from accumulated CSC count matrices.
// Called after COO→CSC conversion; no BAM re-processing required.
//
// Metrics:
//   total_umis      — total UMI count (exon + intron)
//   total_genes     — genes with ≥1 UMI
//   mt_pct          — % UMIs from mitochondrial genes (chrM / chrMT / MT)
//   ribo_pct        — % UMIs from ribosomal protein genes (RPL* / RPS*)
//   intronic_pct    — % UMIs from introns vs total (exon + intron)

#include <cctype>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "gene_model.h"
#include "sparse_accumulator.h"

namespace singlet {

struct CellQCMetrics {
    std::vector<uint32_t> total_umis;   ///< sum of exon + intron UMIs per cell
    std::vector<uint32_t> total_genes;  ///< genes detected (≥1 UMI) per cell
    std::vector<float>    mt_pct;       ///< % UMIs from MT genes (0–100)
    std::vector<float>    ribo_pct;     ///< % UMIs from ribosomal genes (0–100)
    std::vector<float>    intronic_pct; ///< % UMIs intronic out of total (0–100)
};

// ── Classification helpers ──────────────────────────────────────────────────

/// True if gene is mitochondrial: chrom is chrM / MT / chrMT,
/// or gene_name begins with "MT-" / "mt-".
inline bool is_mt_gene(const GeneRecord& g) {
    const auto& c = g.chrom;
    if (c == "chrM" || c == "MT" || c == "chrMT") return true;
    const auto& n = g.gene_name;
    if (n.size() >= 3) {
        // Check "MT-" prefix (human) or "mt-" (mouse) case-insensitively
        if ((n[0] == 'M' || n[0] == 'm') &&
            (n[1] == 'T' || n[1] == 't') &&
             n[2] == '-') return true;
    }
    return false;
}

/// True if gene is a ribosomal protein gene: name begins with RPL or RPS
/// (case-insensitive first 3 characters; handles human RPL* / RPS* and
/// mouse Rpl* / Rps*).
inline bool is_ribo_gene(const GeneRecord& g) {
    const auto& n = g.gene_name;
    if (n.size() < 3) return false;
    const char c0 = static_cast<char>(std::toupper(static_cast<unsigned char>(n[0])));
    const char c1 = static_cast<char>(std::toupper(static_cast<unsigned char>(n[1])));
    const char c2 = static_cast<char>(std::toupper(static_cast<unsigned char>(n[2])));
    return c0 == 'R' && c1 == 'P' && (c2 == 'L' || c2 == 'S');
}

// ── Core computation ────────────────────────────────────────────────────────

/// Compute per-cell QC metrics from exon and intron CSC matrices.
///
/// @param exon_csc    Exon count matrix (nrows=n_exons,  ncols=n_cells) — uint16_t CSC
/// @param intron_csc  Intron count matrix (nrows=n_introns, ncols=n_cells) — may be empty
///                    (if intron counting was disabled, indptr will be empty)
/// @param gene_model  GeneModel built from GTF; must have gene hierarchy
///                    (has_gene_hierarchy() == true)
/// @return            Per-cell CellQCMetrics
inline CellQCMetrics compute_cell_qc(
    const SparseAccumulator<uint16_t>::CSCMatrix& exon_csc,
    const SparseAccumulator<uint16_t>::CSCMatrix& intron_csc,
    const GeneModel& gene_model)
{
    const uint32_t n_cells   = exon_csc.ncols;
    const uint32_t n_exons   = exon_csc.nrows;
    const uint32_t n_introns = intron_csc.nrows;
    const uint32_t n_genes   = gene_model.n_genes();

    const bool has_introns = !intron_csc.indptr.empty()
                              && intron_csc.ncols == n_cells
                              && n_introns > 0;

    // ── Build per-feature MT and ribo flags (O(n_genes)) ──────────────────
    std::vector<bool> exon_is_mt(n_exons, false),   exon_is_ribo(n_exons, false);
    std::vector<bool> intron_is_mt(n_introns, false), intron_is_ribo(n_introns, false);

    const auto& genes = gene_model.genes();
    for (const auto& g : genes) {
        const bool mt   = is_mt_gene(g);
        const bool ribo = is_ribo_gene(g);
        if (!mt && !ribo) continue;

        for (uint32_t ei : g.exon_indices) {
            if (ei < n_exons) {
                if (mt)   exon_is_mt[ei]   = true;
                if (ribo) exon_is_ribo[ei] = true;
            }
        }
        if (has_introns) {
            for (uint32_t ii : g.intron_indices) {
                if (ii < n_introns) {
                    if (mt)   intron_is_mt[ii]   = true;
                    if (ribo) intron_is_ribo[ii] = true;
                }
            }
        }
    }

    // Count MT and ribo genes for diagnostic log
    uint32_t n_mt_genes = 0, n_ribo_genes = 0;
    for (const auto& g : genes) {
        if (is_mt_gene(g))   ++n_mt_genes;
        if (is_ribo_gene(g)) ++n_ribo_genes;
    }
    std::cerr << "[cell_qc] Gene flags: " << n_mt_genes << " MT genes, "
              << n_ribo_genes << " ribosomal genes out of " << n_genes << " total\n";

    // ── Per-cell accumulation ─────────────────────────────────────────────
    CellQCMetrics qc;
    qc.total_umis.resize(n_cells, 0);
    qc.total_genes.resize(n_cells, 0);
    qc.mt_pct.resize(n_cells, 0.f);
    qc.ribo_pct.resize(n_cells, 0.f);
    qc.intronic_pct.resize(n_cells, 0.f);

    // Generation-counter trick for O(nnz) gene detection without per-cell reset.
    // gene_gen[gi] == current cell generation → gene gi was seen this cell.
    // Initialized to 0; first cell uses generation 1.
    std::vector<uint32_t> gene_gen(n_genes, 0);
    std::vector<uint64_t> gene_sum(n_genes, 0);  // scratch; valid when gene_gen matches

    for (uint32_t c = 0; c < n_cells; ++c) {
        const uint32_t gen = c + 1;  // unique non-zero generation id for this cell

        uint64_t exon_total = 0, intron_total = 0;
        uint64_t mt_total   = 0, ribo_total   = 0;

        // touched_genes: indices into gene_sum that need to be checked
        // Use a small local vector; cleared each iteration.
        std::vector<uint32_t> touched;

        // ── Sum exon counts ──
        for (int32_t k = exon_csc.indptr[c]; k < exon_csc.indptr[c + 1]; ++k) {
            const uint32_t row = static_cast<uint32_t>(exon_csc.indices[k]);
            const uint64_t cnt = static_cast<uint64_t>(exon_csc.data[k]);
            exon_total += cnt;
            if (row < n_exons) {
                if (exon_is_mt[row])   mt_total   += cnt;
                if (exon_is_ribo[row]) ribo_total += cnt;
                // Gene-level tracking
                const uint32_t gi = gene_model.exon_to_gene(row);
                if (gi < n_genes) {
                    if (gene_gen[gi] != gen) {
                        gene_gen[gi] = gen;
                        gene_sum[gi] = 0;
                        touched.push_back(gi);
                    }
                    gene_sum[gi] += cnt;
                }
            }
        }

        // ── Sum intron counts ──
        if (has_introns) {
            for (int32_t k = intron_csc.indptr[c]; k < intron_csc.indptr[c + 1]; ++k) {
                const uint32_t row = static_cast<uint32_t>(intron_csc.indices[k]);
                const uint64_t cnt = static_cast<uint64_t>(intron_csc.data[k]);
                intron_total += cnt;
                if (row < n_introns) {
                    if (intron_is_mt[row])   mt_total   += cnt;
                    if (intron_is_ribo[row]) ribo_total += cnt;
                    const uint32_t gi = gene_model.intron_to_gene(row);
                    if (gi < n_genes) {
                        if (gene_gen[gi] != gen) {
                            gene_gen[gi] = gen;
                            gene_sum[gi] = 0;
                            touched.push_back(gi);
                        }
                        gene_sum[gi] += cnt;
                    }
                }
            }
        }

        // ── Count detected genes ──
        uint32_t genes_detected = 0;
        for (uint32_t gi : touched) {
            if (gene_sum[gi] > 0) ++genes_detected;
        }

        const uint64_t total = exon_total + intron_total;
        qc.total_umis[c]  = static_cast<uint32_t>(total > UINT32_MAX ? UINT32_MAX : total);
        qc.total_genes[c] = genes_detected;
        if (total > 0) {
            const double inv = 100.0 / static_cast<double>(total);
            qc.mt_pct[c]       = static_cast<float>(static_cast<double>(mt_total)      * inv);
            qc.ribo_pct[c]     = static_cast<float>(static_cast<double>(ribo_total)    * inv);
            qc.intronic_pct[c] = static_cast<float>(static_cast<double>(intron_total)  * inv);
        }
    }

    return qc;
}

// ── TSV writer ────────────────────────────────────────────────────────────

/// Write per-cell QC metrics to a tab-separated file.
/// Columns: barcode, total_umis, total_genes, mt_pct, ribo_pct, intronic_pct
inline void write_cell_qc_tsv(const std::string& path,
                               const CellQCMetrics& qc,
                               const std::vector<std::string>& barcodes)
{
    std::ofstream f(path);
    if (!f) {
        std::cerr << "[cell_qc] ERROR: cannot write: " << path << "\n";
        return;
    }
    f << std::fixed << std::setprecision(2);
    f << "barcode\ttotal_umis\ttotal_genes\tmt_pct\tribo_pct\tintronic_pct\n";

    const uint32_t n = static_cast<uint32_t>(barcodes.size());
    for (uint32_t i = 0; i < n; ++i) {
        f << barcodes[i]        << '\t'
          << qc.total_umis[i]  << '\t'
          << qc.total_genes[i] << '\t'
          << qc.mt_pct[i]      << '\t'
          << qc.ribo_pct[i]    << '\t'
          << qc.intronic_pct[i] << '\n';
    }
    std::cerr << "[cell_qc] Wrote: " << path << " (" << n << " cells)\n";
}

} // namespace singlet
