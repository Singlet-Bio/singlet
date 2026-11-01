// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: export.h
// Shared export logic for writing pileup results to disk.
// Used by both the singlet-pileup CLI and the unified singlet pipeline.
//
// Exports: COO→CSC conversion, mt heteroplasmy, donor demux, matrix I/O.
// All operations are parallelized where safe.

#include <algorithm>
#include <chrono>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include "pileup_engine.h"
#include "cell_qc_metrics.h"
#include "mtx_writer.h"
#include "pz_writer.h"
#include "donor_demux.h"
#include "mt_heteroplasmy.h"
#include "mt_event_caller.h"
#include "provenance.h"
#include "saturation.h"
#include "saturation_curve.h"
#include "cell_calling.h"
#include "../nvtx.h"
#include "called_v2.h"
#include "sex_calling.h"
#include "cell_cycle.h"
#include "ancestry.h"
#include "splice_psi.h"
#include "ase.h"
#include "ambient_correction.h"
#include "doublet_detect.h"
#include "read_stats.h"
#include "velocity.h"
#include "metrics_summary.h"
#include "barnyard.h"
#include "tpm_fpkm.h"
#include "rrna_detect.h"
#include "h5ad_writer.h"
#include "loom_writer.h"
#include "summary_json.h"
#include "multiome_router.h"
#include "atomic_io.h"
#include "validate_output.h"
#include "mt_reference.h"
#include "sj_features.h"
#include "star_cell_read_stats.h"

namespace singlet {

/// Gene-level sparse matrix in CSC format.
/// Canonical layout lives in sparse_accumulator.h as
/// SparseAccumulator<OutT>::CSCMatrix; GeneCSC is just an alias of it.
#ifndef SINGLET_GENE_CSC_DEFINED
#define SINGLET_GENE_CSC_DEFINED
template <typename OutT>
using GeneCSC = typename SparseAccumulator<OutT>::CSCMatrix;
#endif  // SINGLET_GENE_CSC_DEFINED

/// Collapse exon-interval-level CSC to gene-level CSC by summing counts.
/// Works for both integer (uint16_t → uint32_t) and floating-point (double) CSC data.
/// @param exon_csc  Exon-interval sparse matrix (n_exon_intervals × n_cells)
/// @param gm        GeneModel providing exon_to_gene() mapping and n_genes()
/// @return          Gene-level sparse matrix (n_genes × n_cells)
template <typename CSCType>
auto collapse_exon_to_gene(const CSCType& exon_csc, const GeneModel& gm) {
    using InT  = typename std::remove_reference_t<decltype(exon_csc.data)>::value_type;
    using OutT = std::conditional_t<std::is_integral_v<InT>, uint32_t, double>;

    const uint32_t n_genes = gm.n_genes();
    const uint32_t n_cells = exon_csc.ncols;

    GeneCSC<OutT> out;
    out.nrows = n_genes;
    out.ncols = n_cells;
    out.indptr.resize(n_cells + 1, 0);

    std::vector<OutT>    acc(n_genes, OutT(0));
    std::vector<uint32_t> active;
    active.reserve(1024);

    for (uint32_t j = 0; j < n_cells; ++j) {
        for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k) {
            uint32_t exon_idx = static_cast<uint32_t>(exon_csc.indices[k]);
            uint32_t gene_idx = gm.exon_to_gene(exon_idx);
            if (gene_idx < n_genes) {
                if (acc[gene_idx] == OutT(0)) active.push_back(gene_idx);
                acc[gene_idx] += static_cast<OutT>(exon_csc.data[k]);
            }
        }
        std::sort(active.begin(), active.end());
        for (uint32_t g : active) {
            out.indices.push_back(static_cast<int32_t>(g));
            out.data.push_back(acc[g]);
            acc[g] = OutT(0);
        }
        active.clear();
        out.indptr[j + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

/// Collapse exon CSC + intron CSC to a single gene-level count matrix.
/// Sums contributions from both exon_to_gene() and intron_to_gene() mappings.
/// When intron_csc is empty (ncols==0 or data empty), behaves identically to
/// collapse_exon_to_gene().
template <typename CSCType>
auto collapse_to_gene_counts(const CSCType& exon_csc,
                              const CSCType& intron_csc,
                              const GeneModel& gm) {
    using InT  = typename std::remove_reference_t<decltype(exon_csc.data)>::value_type;
    using OutT = std::conditional_t<std::is_integral_v<InT>, uint32_t, double>;

    const uint32_t n_genes = gm.n_genes();
    const uint32_t n_cells = exon_csc.ncols;

    GeneCSC<OutT> out;
    out.nrows = n_genes;
    out.ncols = n_cells;
    out.indptr.resize(n_cells + 1, 0);

    std::vector<OutT>     acc(n_genes, OutT(0));
    std::vector<uint32_t> active;
    active.reserve(1024);

    const bool has_introns = (intron_csc.ncols > 0 && !intron_csc.data.empty());

    for (uint32_t j = 0; j < n_cells; ++j) {
        // Accumulate exon contributions
        for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k) {
            uint32_t gene_idx = gm.exon_to_gene(
                static_cast<uint32_t>(exon_csc.indices[k]));
            if (gene_idx < n_genes) {
                if (acc[gene_idx] == OutT(0)) active.push_back(gene_idx);
                acc[gene_idx] += static_cast<OutT>(exon_csc.data[k]);
            }
        }
        // Accumulate intron contributions
        if (has_introns) {
            for (int32_t k = intron_csc.indptr[j]; k < intron_csc.indptr[j + 1]; ++k) {
                uint32_t gene_idx = gm.intron_to_gene(
                    static_cast<uint32_t>(intron_csc.indices[k]));
                if (gene_idx < n_genes) {
                    if (acc[gene_idx] == OutT(0)) active.push_back(gene_idx);
                    acc[gene_idx] += static_cast<OutT>(intron_csc.data[k]);
                }
            }
        }
        std::sort(active.begin(), active.end());
        for (uint32_t g : active) {
            out.indices.push_back(static_cast<int32_t>(g));
            out.data.push_back(acc[g]);
            acc[g] = OutT(0);
        }
        active.clear();
        out.indptr[j + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

/// Gene-level a - b (b may be null) as uint32 counts; a and b are genes × cells.
/// Used for the spliced layer: exon_counts holds spliced + ambiguous molecules,
/// so spliced = collapse_exon_to_gene(exon) - ambiguous.
template <typename ACSC, typename BCSC>
GeneCSC<uint32_t> gene_csc_minus(const ACSC& a, const BCSC* b) {
    GeneCSC<uint32_t> out;
    out.nrows = a.nrows;
    out.ncols = a.ncols;
    out.indptr.assign(static_cast<size_t>(a.ncols) + 1, 0);
    std::vector<int64_t> acc(a.nrows, 0);
    std::vector<uint32_t> active;
    const bool has_b = b && b->ncols == a.ncols && !b->data.empty();
    for (uint32_t j = 0; j < a.ncols; ++j) {
        for (int32_t k = a.indptr[j]; k < a.indptr[j + 1]; ++k) {
            uint32_t g = static_cast<uint32_t>(a.indices[k]);
            if (acc[g] == 0) active.push_back(g);
            acc[g] += static_cast<int64_t>(a.data[k]);
        }
        if (has_b)
            for (int32_t k = b->indptr[j]; k < b->indptr[j + 1]; ++k) {
                uint32_t g = static_cast<uint32_t>(b->indices[k]);
                if (acc[g] == 0) active.push_back(g);
                acc[g] -= static_cast<int64_t>(b->data[k]);
            }
        std::sort(active.begin(), active.end());
        for (uint32_t g : active) {
            if (acc[g] > 0) {
                out.indices.push_back(static_cast<int32_t>(g));
                out.data.push_back(static_cast<uint32_t>(acc[g]));
            }
            acc[g] = 0;
        }
        active.clear();
        out.indptr[j + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

/// Columns `cols` (ascending) of a CSC matrix, in that order.
template <typename CSC>
CSC csc_select_columns(const CSC& a, const std::vector<uint32_t>& cols) {
    CSC out;
    out.nrows = a.nrows;
    out.ncols = static_cast<uint32_t>(cols.size());
    out.indptr.assign(cols.size() + 1, 0);
    for (size_t j = 0; j < cols.size(); ++j) {
        const uint32_t c = cols[j];
        if (c < a.ncols)
            for (int32_t k = a.indptr[c]; k < a.indptr[c + 1]; ++k) {
                out.indices.push_back(a.indices[k]);
                out.data.push_back(a.data[k]);
            }
        out.indptr[j + 1] = static_cast<int32_t>(out.indices.size());
    }
    return out;
}

/// Configuration for export_results().
struct ExportConfig {
    std::string out_prefix;            ///< Output directory path
    std::string output_format = "1pz"; ///< "1pz", "mtx", "h5ad", or "all" (1pz+h5ad)
    bool pipeline_mode = false;        ///< Enable demux + mt heteroplasmy
    int n_donors = -1;                 ///< Number of donors (-1 = auto)
    int threads = 1;                   ///< Threads for demux
    std::map<std::string, std::string> user_meta; ///< GEO/pipeline metadata embedded in every .1pz
    ProvenanceConfig provenance;       ///< N8: provenance manifest (written if input_file non-empty)
    // N5: EmptyDrops cell calling
    bool run_cell_calling = false;     ///< Run EmptyDrops cell calling after pileup
    /// "called_v2" (default): CellRanger2.2 knee on the pileup's exon counts, then the called_v2 EmptyDrops call of
    /// docs/called-v2.md on exon + intron gene counts. "star_solo": STARsolo's filtered barcodes, else the legacy
    /// EmptyDrops chain (the behaviour before 2026-10). "emptydrops": the legacy chain without STARsolo.
    std::string cell_caller = "called_v2";
    double fdr_threshold = 0.01;       ///< BH-FDR cutoff for cell calls (STARsolo EmptyDrops_CR default)
    uint64_t lower_umi = 100;          ///< Lower UMI bound for ambient pool estimation
    uint64_t min_umi_test = 500;       ///< Min UMI to be tested by EmptyDrops (STARsolo umiMin=500)
    // N11: Ambient RNA correction
    bool run_ambient_correction = false; ///< Write ambient_corrected_counts (always on in pipeline_mode)
    // G-PERMITLIST: permit-list cell selection modes
    int forced_cells = 0;              ///< Force exactly N cells (top-N by UMI); 0=off, skips EmptyDrops
    int expect_cells = 0;              ///< Hint for cell calling (stored; EmptyDrops integration pending)
    // G-METRICS: pileup stats for metrics_summary.csv
    const PileupStats* pileup_stats = nullptr; ///< Non-owning pointer; valid for duration of export_results()
    bool write_raw_matrix = false;             ///< Write raw (unfiltered) barcode matrix alongside filtered
    // §3.6 required fields passed from the pipeline driver (src/pipeline/singlet.cpp)
    int         protocol_id   = 0;
    std::string protocol_name;
    std::string species;
    std::string reference_build;
    double      peak_rss_gb   = 0.0;
    std::string memory_tier   = "unknown";
    bool        nonhost_was_screened    = false;
    int         nonhost_species_above_em = 0;
    // §3.2 per-stage read statistics
    std::vector<singlet::PipelineStageStats> stage_stats;
    // Genome directory for MT reference extraction
    std::string genome_dir;
};

/// Result statistics from export_results().
struct ExportStats {
    double csc_time_s = 0;
    double export_time_s = 0;
    int n_csc_threads = 0;
    int n_write_threads = 0;
    uint64_t n_called_cells = 0;  // called cells (0 when cell calling ran and found none)
    int exit_code = 0;            // 0=success, 4=zero_cells
    std::vector<std::string> failed_writes;  // .1pz outputs that could not be written
    std::string cell_caller;      // method that produced the cell set
    std::string cell_call_json;   // knee + called_v2 parameters (JSON object), empty for other methods
    uint64_t n_knee_called = 0;   // knee ("called") set size when called_v2 ran
};

/// Export all pileup results to disk.
///
/// Handles the full output pipeline:
///   1. Parallel COO → CSC conversion
///   2. mt heteroplasmy computation (pipeline mode)
///   3. Donor demultiplexing (pipeline mode)
///   4. Parallel matrix writes (.1pz or .mtx)
///   5. Donor assignment output (pipeline mode)
///
/// @param engine      PileupEngine with completed run() data
/// @param pileup_cfg  Original PileupConfig (for feature flags)
/// @param export_cfg  Export-specific configuration
/// @return            Timing statistics
inline ExportStats export_results(const PileupEngine& engine,
                                  const PileupConfig& pileup_cfg,
                                  const ExportConfig& export_cfg) {
    ExportStats result;
    auto export_t0 = std::chrono::high_resolution_clock::now();
    singlet::NvtxChain nvexp;   // NVTX: one range per phase below (SINGLET_NVTX=1)

    const auto& barcodes = engine.barcodes();
    const bool use_1pz  = (export_cfg.output_format == "1pz" || export_cfg.output_format == "all");
    const bool use_h5ad = (export_cfg.output_format == "h5ad" || export_cfg.output_format == "all");
    const bool use_loom = (export_cfg.output_format == "loom");
    const auto& out_prefix = export_cfg.out_prefix;

    // ── Phase 1: COO → CSC (parallel) ──
    nvexp.next("export: Phase 1: COO  CSC (parallel)");
    auto csc_t0 = std::chrono::high_resolution_clock::now();

    SparseAccumulator<uint16_t>::CSCMatrix snp_ad_csc, snp_dp_csc;
    // Two exon / intron splits of the same molecules (DirectionalUmiStore::emit_molecule):
    //  - layer_exon / layer_intron (+ amb_csc): the alevin-fry USA labels (unspliced = any intron read). Used ONLY for the
    //    published exon_counts / intron_counts / ambiguous_counts matrices and the h5ad / loom spliced / unspliced / ambiguous
    //    layers.
    //  - exon_csc / intron_csc: the pre-USA split, legacy exon = USA exon + mixed_exon, legacy intron = USA intron - mixed_intron
    //    (mixed: unspliced molecules that also have an exon-only read, which the old labels counted as ambiguous in the exon
    //    matrix). Every method validated on that split reads it: cell QC, sex, cell cycle, read stats / saturation, the cell
    //    caller (knee, EmptyDrops, called_v2), ambient (incl. the intron-rich guard), doublets, the gene-expression matrices
    //    (exon + intron: the same either way) and the summary metrics.
    SparseAccumulator<uint16_t>::CSCMatrix exon_csc, intron_csc, sj_csc, mt_csc, guide_csc, amb_csc;
    SparseAccumulator<uint16_t>::CSCMatrix mixed_exon_csc, mixed_intron_csc, layer_exon_store, layer_intron_store;
    const SparseAccumulator<uint16_t>::CSCMatrix* layer_exon = &exon_csc;     // USA labels: exon_counts and the spliced layer
    const SparseAccumulator<uint16_t>::CSCMatrix* layer_intron = &intron_csc; // USA labels: intron_counts and the unspliced layer
    SparseAccumulator<uint32_t>::CSCMatrix exon_reads_csc, intron_reads_csc;   // reads per molecule, same layout as the UMI matrices

    std::vector<std::thread> csc_threads;
    bool need_snp_csc = !pileup_cfg.snp_path.empty() ||
                         engine.aim_snp_start() != UINT32_MAX;
    if (need_snp_csc) {
        csc_threads.emplace_back([&]() { snp_ad_csc = engine.snp_ad().to_csc(); });
        csc_threads.emplace_back([&]() { snp_dp_csc = engine.snp_dp().to_csc(); });
    }
    if (pileup_cfg.count_mt && engine.mt_alleles().nnz_raw() > 0)
        csc_threads.emplace_back([&]() { mt_csc = engine.mt_alleles().to_csc(); });
    if (!pileup_cfg.exon_gtf_path.empty()) {
        csc_threads.emplace_back([&]() { exon_csc = engine.exons().to_csc(); });
        if (engine.reads_matrix_on()) {
            csc_threads.emplace_back([&]() { exon_reads_csc = engine.exon_reads().to_csc(); });
            if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0)
                csc_threads.emplace_back([&]() { intron_reads_csc = engine.intron_reads().to_csc(); });
        }
        if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0) {
            csc_threads.emplace_back([&]() { intron_csc = engine.introns().to_csc(); });
            csc_threads.emplace_back([&]() { amb_csc = engine.ambiguous().to_csc(); });
            if (engine.mixed_exon().nnz_raw() > 0) {
                csc_threads.emplace_back([&]() { mixed_exon_csc = engine.mixed_exon().to_csc(); });
                csc_threads.emplace_back([&]() { mixed_intron_csc = engine.mixed_intron().to_csc(); });
            }
        }
        if (pileup_cfg.count_sj && !engine.sj_names().empty())
            csc_threads.emplace_back([&]() { sj_csc = engine.splice_junctions().to_csc(); });
    }
    // N18: CRISPR guide counts CSC conversion
    if (!pileup_cfg.guide_ref_path.empty() && engine.guide_counts().nnz_raw() > 0)
        csc_threads.emplace_back([&]() { guide_csc = engine.guide_counts().to_csc(); });
    // N17: VDJ gene usage CSC conversion
    SparseAccumulator<uint16_t>::CSCMatrix vdj_csc;
    if (engine.has_vdj() && engine.vdj_counts().nnz_raw() > 0)
        csc_threads.emplace_back([&]() { vdj_csc = engine.vdj_counts().to_csc(); });

    for (auto& t : csc_threads) t.join();
    // The pre-USA split (see the declarations above): keep the USA matrices for the published layers, and rebind exon_csc /
    // intron_csc to the legacy ones for every other consumer. No mixed molecules: the two splits are the same matrices.
    // At the uint16 cap (65535) legacy exon stays exact and legacy intron undercounts only a saturated entry, never wraps
    // (csc_add_signed, sparse_accumulator.h; tests/cpp/test_csc_add_signed.cpp).
    if (mixed_exon_csc.ncols > 0 && mixed_exon_csc.ncols == exon_csc.ncols && intron_csc.ncols == exon_csc.ncols) {
        layer_exon_store = std::move(exon_csc); layer_intron_store = std::move(intron_csc);
        layer_exon = &layer_exon_store; layer_intron = &layer_intron_store;
        std::thread tl([&]() { intron_csc = csc_add_signed(layer_intron_store, mixed_intron_csc, -1); });
        exon_csc = csc_add_signed(layer_exon_store, mixed_exon_csc, +1);
        tl.join();
        mixed_exon_csc = {}; mixed_intron_csc = {};
    }
    // Ambiguous molecules per gene: always genes × barcodes, possibly all zero
    if (amb_csc.ncols != exon_csc.ncols && exon_csc.ncols > 0) {
        amb_csc = {};
        amb_csc.nrows = engine.gene_model().n_genes();
        amb_csc.ncols = exon_csc.ncols;
        amb_csc.indptr.assign(static_cast<size_t>(exon_csc.ncols) + 1, 0);
    }

    auto csc_t1 = std::chrono::high_resolution_clock::now();
    result.csc_time_s = std::chrono::duration<double>(csc_t1 - csc_t0).count();
    result.n_csc_threads = static_cast<int>(csc_threads.size());
    std::cerr << "[export] CSC conversion: " << result.csc_time_s
              << "s (" << result.n_csc_threads << " threads)\n";

    // ── Phase 1.5: Cell QC metrics (compute only; write deferred to Phase 1.8) ──
    nvexp.next("export: Phase 1.5: Cell QC metrics (compute only write deferred to Phase 1.8)");
    // Computed from already-converted CSC matrices — no BAM re-processing.
    // Only runs when a GTF gene model with hierarchy is available.
    CellQCMetrics cell_qc;
    bool have_cell_qc = false;
    if (!pileup_cfg.exon_gtf_path.empty()
        && engine.gene_model().has_gene_hierarchy()
        && exon_csc.ncols > 0) {
        cell_qc = singlet::compute_cell_qc(exon_csc, intron_csc,
                                           engine.gene_model());
        have_cell_qc = true;
    }

    // ── Phase 1.55: Sex calling (N14) ──
    nvexp.next("export: Phase 1.55: Sex calling (N14)");
    // Runs when a GTF gene model is present and exon data is available.
    if (!pileup_cfg.exon_gtf_path.empty()
        && engine.gene_model().has_gene_hierarchy()
        && exon_csc.ncols > 0) {
        auto t_sex0 = std::chrono::high_resolution_clock::now();
        auto sex_result = singlet::call_sex(exon_csc, engine.gene_model());
        singlet::write_sex_call_json(out_prefix + "/sex_call.json", sex_result);
        auto t_sex1 = std::chrono::high_resolution_clock::now();
        double sex_s = std::chrono::duration<double>(t_sex1 - t_sex0).count();
        std::cerr << "[sex_calling] sex=" << sex_result.sex
                  << " confidence=" << sex_result.confidence
                  << " xist_cpm=" << sex_result.xist_cpm
                  << " y_cpm=" << sex_result.y_cpm
                  << " n_y_genes=" << sex_result.n_y_genes_detected
                  << " time=" << sex_s << "s\n";
    }

    // ── Phase 1.57: Cell cycle scoring (N19) ──
    nvexp.next("export: Phase 1.57: Cell cycle scoring (N19)");
    // Runs when a GTF gene model with hierarchy is present and exon data available.
    CellCycleResult cc_result_cycle;
    if (!pileup_cfg.exon_gtf_path.empty()
        && engine.gene_model().has_gene_hierarchy()
        && exon_csc.ncols > 0) {
        auto t_cc0 = std::chrono::high_resolution_clock::now();
        cc_result_cycle = singlet::score_cell_cycle(exon_csc, engine.gene_model());
        singlet::write_cell_cycle_tsv(
            out_prefix + "/cell_cycle_scores.tsv", cc_result_cycle, barcodes);
        auto t_cc1 = std::chrono::high_resolution_clock::now();
        double cc_s = std::chrono::duration<double>(t_cc1 - t_cc0).count();
        uint32_t n_total = cc_result_cycle.n_g1 + cc_result_cycle.n_s + cc_result_cycle.n_g2m;
        std::cerr << "[cell_cycle] G1=" << cc_result_cycle.n_g1
                  << " S=" << cc_result_cycle.n_s
                  << " G2M=" << cc_result_cycle.n_g2m
                  << " (" << (n_total > 0 ? 100.0 * cc_result_cycle.n_g1 / n_total : 0.0) << "%G1)"
                  << " s_genes=" << cc_result_cycle.n_s_genes_detected
                  << " g2m_genes=" << cc_result_cycle.n_g2m_genes_detected
                  << " time=" << cc_s << "s\n";
    }

    // ── Phase 1.58: Per-cell read statistics (N20) ──
    nvexp.next("export: Phase 1.58: Per-cell read statistics (N20)");
    // Derived from per_cell_reads_ (pre-dedup barcoded reads) + CSC column sums.
    // Zero overhead: all data already in memory, no BAM re-processing.
    std::vector<CellReadStats> cell_read_stats;
    if (!engine.per_cell_reads().empty() && exon_csc.ncols > 0) {
        auto t_rs0 = std::chrono::high_resolution_clock::now();
        const int32_t* intron_ip = pileup_cfg.count_introns && intron_csc.ncols > 0
                                   ? intron_csc.indptr.data() : nullptr;
        const uint16_t* intron_d = pileup_cfg.count_introns && intron_csc.data.size() > 0
                                   ? intron_csc.data.data() : nullptr;
        cell_read_stats = singlet::compute_read_stats(
            engine.per_cell_reads(),
            exon_csc.indptr.data(), exon_csc.data.data(),
            intron_ip, intron_d,
            exon_csc.ncols);
        // Per-stage read_stats.tsv (§3.2 schema): write from pipeline stage data
        if (!export_cfg.stage_stats.empty()) {
            singlet::write_stage_read_stats_tsv(out_prefix + "/read_stats.tsv", export_cfg.stage_stats);
        }
        double med_dr = singlet::median_dup_rate(cell_read_stats);
        auto t_rs1 = std::chrono::high_resolution_clock::now();
        double rs_s = std::chrono::duration<double>(t_rs1 - t_rs0).count();
        std::cerr << "[read_stats] n_cells=" << cell_read_stats.size()
                  << " median_dup_rate=" << med_dr
                  << " time=" << rs_s << "s\n";
    }

    // ── Phase 1.56: Ancestry classification (N13) ──
    nvexp.next("export: Phase 1.56: Ancestry classification (N13)");
    // Uses the embedded AIM panel injected into the SNP pileup at load time.
    // Runs regardless of whether --snps was provided by the user.
    if (engine.aim_snp_start() != UINT32_MAX && snp_dp_csc.ncols > 0) {
        auto t_anc0 = std::chrono::high_resolution_clock::now();
        const auto& aims = singlet::get_aim_panel();
        auto anc_result = singlet::classify_ancestry(
            snp_ad_csc, snp_dp_csc,
            engine.aim_snp_start(), aims);
        singlet::write_ancestry_json(out_prefix + "/ancestry_call.json", anc_result, aims);
        auto t_anc1 = std::chrono::high_resolution_clock::now();
        double anc_s = std::chrono::duration<double>(t_anc1 - t_anc0).count();
        if (anc_result.low_data) {
            std::cerr << "[ancestry] insufficient_data n_informative="
                      << anc_result.n_informative
                      << " (need " << singlet::kAncestryMinInformative << ") time="
                      << anc_s << "s\n";
        } else {
            std::cerr << "[ancestry] " << anc_result.ancestry
                      << " confidence=" << anc_result.confidence
                      << " n_informative=" << anc_result.n_informative
                      << " n_covered=" << anc_result.n_covered
                      << " time=" << anc_s << "s\n";
        }
    }

    // ── Phase 1.6: Sequencing saturation (N7, directional dedup path only) ──
    nvexp.next("export: Phase 1.6: Sequencing saturation (N7, directional dedup path only)");
    // Uses the DirectionalUmiStore groups built during run() — no extra BAM pass.
    std::vector<CellSaturation> sat_cells;
    std::vector<std::pair<uint32_t, std::pair<uint64_t, uint64_t>>> gpu_sat;
    if (pileup_cfg.umi_dedup_directional && engine.gpu_saturation(gpu_sat)) {   // W11 SINGLET_GPU_PILEUP=on: same inputs, from the GPU
        for (const auto& [bc, ru] : gpu_sat) {
            CellSaturation cs;
            cs.barcode_idx = bc; cs.total_reads = ru.first; cs.unique_umis = ru.second;
            if (cs.total_reads > 0 && cs.unique_umis > 0) {
                cs.saturation = 1.0 - static_cast<double>(cs.unique_umis) / static_cast<double>(cs.total_reads);
                cs.reads_per_umi = static_cast<double>(cs.total_reads) / static_cast<double>(cs.unique_umis);
            }
            sat_cells.push_back(cs);
        }
        if (!sat_cells.empty()) {
            double med_sat = singlet::median_saturation(sat_cells);
            singlet::write_saturation_tsv(out_prefix + "/saturation_metrics.tsv", sat_cells, barcodes);
            std::cerr << "[saturation] median_saturation=" << med_sat << " n_cells=" << sat_cells.size() << " (GPU molecule list)\n";
        }
    } else if (pileup_cfg.umi_dedup_directional && !engine.dir_exon_store().molecules_empty()) {
        sat_cells = singlet::compute_saturation(
            engine.dir_exon_store(), pileup_cfg.umi_len);
        if (!sat_cells.empty()) {
            double med_sat = singlet::median_saturation(sat_cells);
            singlet::write_saturation_tsv(
                out_prefix + "/saturation_metrics.tsv", sat_cells, barcodes);
            std::cerr << "[saturation] median_saturation=" << med_sat
                      << " n_cells=" << sat_cells.size() << "\n";
        }
    }

    // ── Phase 1.65: Sequencing saturation curve (G-SATCURVE) ──
    nvexp.next("export: Phase 1.65: Sequencing saturation curve (G-SATCURVE)");
    // Analytical downsampling at 6 depth fractions using CellReadStats + exon CSC.
    // Zero pileup overhead — uses data already computed in Phase 1.58.
    if (!engine.per_cell_reads().empty() && exon_csc.ncols > 0
        && engine.gene_model().has_gene_hierarchy()) {
        auto t_sc0 = std::chrono::high_resolution_clock::now();
        const int32_t* intron_ip = pileup_cfg.count_introns && intron_csc.ncols > 0
                                   ? intron_csc.indptr.data() : nullptr;
        const uint16_t* intron_d = pileup_cfg.count_introns && intron_csc.data.size() > 0
                                   ? intron_csc.data.data() : nullptr;
        auto rs_sc = singlet::compute_read_stats(
            engine.per_cell_reads(),
            exon_csc.indptr.data(), exon_csc.data.data(),
            intron_ip, intron_d,
            exon_csc.ncols);
        auto sat_curve = singlet::compute_saturation_curve(
            rs_sc, exon_csc, engine.gene_model());
        if (!sat_curve.empty()) {
            singlet::write_saturation_curve_tsv(
                out_prefix + "/saturation_curve.tsv", sat_curve);
            const auto& last = sat_curve.back();
            auto t_sc1 = std::chrono::high_resolution_clock::now();
            double sc_s = std::chrono::duration<double>(t_sc1 - t_sc0).count();
            std::cerr << "[saturation_curve] median_umis@1.0=" << last.median_umis
                      << " median_genes@1.0=" << last.median_genes
                      << " time=" << sc_s << "s\n";
        }
    }

    // ── Phase 1.7: Cell calling (N5, EmptyDrops-style) ──
    nvexp.next("export: Phase 1.7: Cell calling (N5, EmptyDrops-style)");
    // Declare outside so Phase 1.75 (ambient correction) and N8 (provenance) can access them.
    std::vector<uint64_t> bc_totals;
    CellCallResult cc_result;
    CSCu16 exon_corrected_csc;
    std::string cell_caller_method;  ///< which cell caller ran (written to provenance.json)

    if (export_cfg.run_cell_calling && !pileup_cfg.exon_gtf_path.empty() && exon_csc.ncols > 0) {
        // Sum each column (barcode) of exon CSC to get per-barcode UMI totals.
        // These totals are used for barcode ranking (lower/test UMI thresholds).
        // exon_csc is the pre-USA (legacy) exon matrix, so the knee and the umi_exon column of cell_calls.tsv stay on the exonic
        // total they were validated on: a label change does not move the cell set.
        bc_totals.assign(exon_csc.ncols, 0);
        for (uint32_t j = 0; j < exon_csc.ncols; ++j)
            for (int32_t k = exon_csc.indptr[j]; k < exon_csc.indptr[j + 1]; ++k)
                bc_totals[j] += static_cast<uint64_t>(exon_csc.data[k]);

        // N22-GENE: EmptyDrops operates at GENE level (38k features), not exon level
        // (310k features).  Passing exon_csc directly causes systematic overcalling:
        //   - 310k features vs ~38k in ambient pool → 272k "dark" features with only
        //     pseudocount probability → every tested barcode expressing any dark exon
        //     has astronomical deviance → p≈0 → FDR≈0 → called as cell.
        // Fix: collapse exon_csc → gene_csc (n_genes × n_barcodes) before calling
        // EmptyDrops.  The wl_ambient_gene_counts() vector is already gene-level and
        // now matches the feature space exactly (n_genes entries on both sides).
        auto gene_csc_for_ed = collapse_exon_to_gene(exon_csc, engine.gene_model());

        // ── called_v2 (default): knee on pileup exon counts, then EmptyDrops on exon + intron genes ──────────
        // The knee ("called") is STARsolo's CellRanger2.2 3000 0.99 10 rule, computed here on the pileup's own
        // counts after strand resolution, so 5' libraries no longer get a knee from wrong-strand STARsolo counts.
        bool used_called_v2 = false;
        if (export_cfg.cell_caller == "called_v2" && export_cfg.forced_cells <= 0 &&
            engine.exon_names().size() != exon_csc.nrows)
            std::cerr << "[cell_calling] WARNING: exon feature names do not match the exon matrix; called_v2 "
                         "skipped, using the legacy caller\n";
        else if (export_cfg.cell_caller == "called_v2" && export_cfg.forced_cells <= 0) {
            auto t_v2 = std::chrono::high_resolution_clock::now();
            std::vector<int64_t> exon_tot(bc_totals.begin(), bc_totals.end());
            int64_t knee_umi_min = 0;
            std::vector<uint8_t> knee_called = calledv2::cellranger22(exon_tot, 3000, 0.99, 10, &knee_umi_min);
            const bool with_introns = intron_csc.ncols == exon_csc.ncols && intron_csc.ncols > 0;
            calledv2::GeneCounts gx = calledv2::gene_counts(exon_csc, engine.exon_names(),
                                                              with_introns ? &intron_csc : nullptr,
                                                              with_introns ? &engine.intron_names() : nullptr);
            // is_cell is EmptyDrops_CR on exon-only counts (STARsolo's Gene counts), the reference tool's own input
            calledv2::GeneCounts gx_exon = calledv2::gene_counts(
                exon_csc, engine.exon_names(), static_cast<const decltype(exon_csc)*>(nullptr),
                static_cast<const std::vector<std::string>*>(nullptr));
            std::string proto = export_cfg.protocol_name;
            if (proto.empty() || proto == "unknown") {
                auto it = export_cfg.user_meta.find("protocol");
                if (it != export_cfg.user_meta.end()) proto = it->second;
            }
            calledv2::Result cv = calledv2::call(gx, knee_called, proto, calledv2::Params(),
                                                 std::max(1, export_cfg.threads), &gx_exon);
            cv.seconds = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_v2).count();
            cc_result = CellCallResult{};
            for (uint32_t j = 0; j < cv.is_cell.size(); ++j)
                if (cv.is_cell[j]) cc_result.cell_indices.push_back(j);
            cc_result.n_ambient = static_cast<uint32_t>(cv.n_ambient);
            cc_result.ambient_total = static_cast<double>(cv.ambient_umi);
            cell_caller_method = "called_v2";
            result.n_knee_called = static_cast<uint64_t>(cv.n_called);
            {
                std::ostringstream js;
                js << "{\n    \"knee\": {\"method\": \"CellRanger2.2\", \"counts\": \"pileup exon\", "
                   << "\"n_expected\": 3000, \"max_percentile\": 0.99, \"max_min_ratio\": 10, \"umi_min\": "
                   << knee_umi_min << ", \"n_called\": " << cv.n_called << "},\n    \"called_v2\": "
                   << calledv2::params_json(cv, "    ") << "\n  }";
                result.cell_call_json = js.str();
            }
            calledv2::write_cell_calls_tsv(out_prefix + "/cell_calls.tsv", barcodes, bc_totals, cv, knee_called);
            std::cerr << "[cell_calling] called_v2: knee umi_min=" << knee_umi_min << " called=" << cv.n_called
                      << " lower=" << cv.lower << " retain=" << cv.retain_total << " alpha=" << cv.alpha
                      << " tested=" << cv.n_tested << " is_cell=" << cv.n_is_cell << " lowrna=" << cv.n_lowrna
                      << " called_v2(2.1)=" << cv.n_called_v2 << " (+" << cv.n_added
                      << ")" << (cv.error.empty() ? std::string() : " note=" + cv.error) << " time=" << cv.seconds
                      << "s\n";
            used_called_v2 = true;
        }

        // ── STAR Solo.out preferred cell caller ───────────────────────────────────
        // When STAR ran in CB_UMI_Simple mode (10x-style) it writes
        // Solo.out/Gene/filtered/barcodes.tsv — a knee-called, well-calibrated cell
        // set.  Use it directly and skip EmptyDrops entirely.
        // Fall back to EmptyDrops when the file is absent (CB_samTagOut protocols:
        // Drop-seq / sci-RNA / etc. do not produce Solo.out).
        bool used_star_solo = false;
        if (!used_called_v2 && export_cfg.cell_caller != "emptydrops") {
            const std::string star_bc_path =
                out_prefix + "/star_Solo.out/Gene/filtered/barcodes.tsv";
            std::ifstream star_bc_f(star_bc_path);
            if (star_bc_f.is_open()) {
                // Build O(1) barcode-string → column-index lookup from the engine's
                // already-loaded barcode list.
                std::unordered_map<std::string, uint32_t> bc_to_idx;
                bc_to_idx.reserve(barcodes.size());
                for (size_t i = 0; i < barcodes.size(); ++i)
                    bc_to_idx.emplace(barcodes[i], static_cast<uint32_t>(i));

                std::vector<uint32_t> star_cells;
                std::string line;
                uint32_t n_missing = 0;
                while (std::getline(star_bc_f, line)) {
                    if (line.empty()) continue;
                    if (line.back() == '\r') line.pop_back();  // strip Windows CR
                    auto it = bc_to_idx.find(line);
                    if (it != bc_to_idx.end()) {
                        star_cells.push_back(it->second);
                    } else {
                        ++n_missing;
                        if (n_missing <= 5)
                            std::cerr << "[cell_calling] STAR Solo barcode not in bc_index: "
                                      << line << "\n";
                    }
                }
                if (!star_cells.empty()) {
                    cc_result.cell_indices   = std::move(star_cells);
                    // Populate sentinel values so write_cell_calls() produces valid TSV.
                    cc_result.tested_indices = cc_result.cell_indices;
                    cc_result.deviance.assign(cc_result.cell_indices.size(), 0.0);
                    cc_result.fdr.assign(cc_result.cell_indices.size(),      0.0);
                    cell_caller_method = "star_solo";
                    used_star_solo     = true;
                    std::cerr << "[cell_calling] Using STAR Solo.out filtered barcodes: "
                              << cc_result.cell_indices.size()
                              << " cells from " << star_bc_path;
                    if (n_missing > 0)
                        std::cerr << " (" << n_missing << " not in bc_index, skipped)";
                    std::cerr << "\n";
                } else {
                    std::cerr << "[cell_calling] STAR Solo.out found but 0 barcodes matched"
                                 " bc_index — falling back to EmptyDrops\n";
                }
            } else {
                std::cerr << "[cell_calling] No STAR Solo.out filtered barcodes at "
                          << star_bc_path << " — using EmptyDrops\n";
            }
        }

        auto t_cc0 = std::chrono::high_resolution_clock::now();
        if (!used_star_solo && !used_called_v2) {
        if (export_cfg.forced_cells > 0) {
            // G-PERMITLIST: forced-cells — top-N barcodes by UMI count, skip EmptyDrops
            std::vector<uint32_t> order(exon_csc.ncols);
            std::iota(order.begin(), order.end(), 0u);
            std::sort(order.begin(), order.end(),
                      [&](uint32_t a, uint32_t b) { return bc_totals[a] > bc_totals[b]; });
            uint32_t n = std::min(static_cast<uint32_t>(export_cfg.forced_cells),
                                  static_cast<uint32_t>(order.size()));
            cc_result.cell_indices.assign(order.begin(), order.begin() + n);
            std::cerr << "[cell_calling] forced_cells=" << n
                      << " (top-" << n << " by UMI count)\n";
            cell_caller_method = "forced_cells";
        } else {
            // Multiome GEX libraries have ~3-10× fewer RNA UMI per cell (sequencing
            // depth split between RNA and ATAC).  Lower the ambient/test thresholds
            // to avoid the 0-cell bug where all cells fall below min_umi_test.
            uint64_t effective_lower = export_cfg.lower_umi;
            uint64_t effective_min_umi_test = export_cfg.min_umi_test;
            if (is_multiome_protocol(export_cfg.protocol_name)) {
                effective_lower = std::min(effective_lower, uint64_t(30));
                effective_min_umi_test = std::min(effective_min_umi_test, uint64_t(100));
                std::cerr << "[cell_calling] Multiome detected (" << export_cfg.protocol_name
                          << ") — adjusted thresholds: lower=" << effective_lower
                          << " min_umi_test=" << effective_min_umi_test << "\n";
            }
            cc_result = singlet::call_cells_emptydrops(
                bc_totals, gene_csc_for_ed, export_cfg.fdr_threshold, effective_lower,
                50000, effective_min_umi_test, 10000,
                // N22: full-whitelist ambient override (nullptr when whitelist not loaded)
                engine.wl_umi_counts().empty()             ? nullptr : &engine.wl_umi_counts(),
                engine.wl_ambient_gene_counts().empty()    ? nullptr : &engine.wl_ambient_gene_counts(),
                engine.wl_ambient_ceil());
            std::cerr << "[cell_calling] n_ambient=" << cc_result.n_ambient
                      << " n_tested=" << cc_result.tested_indices.size()
                      << " n_cells=" << cc_result.cell_indices.size() << "\n";

            // ── EMPTYDROPS-OVERCALL-FALLBACK: CellRanger2 threshold when ambient is unusable ─
            //
            // The pipeline's auto-discovery keeps only barcodes with ≥ discovery_threshold reads.
            // After UMI dedup, tested barcodes (UMI ≥ min_umi_test=500) are almost entirely
            // genuine cells — their gene expression genuinely deviates from ambient RNA.
            // A high call rate (>95%) is EXPECTED and CORRECT behavior in this setting.
            //
            // The CR2 fallback (pct99_umi/10) is designed for full-whitelist EmptyDrops where
            // 100% call rate indicates true miscalibration of the ambient profile.  For
            // The pipeline's pre-filtered barcode set, applying CR2 is overly aggressive: it
            // uses threshold ~1078 UMI and cuts ~30% of valid cells that STARsolo calls.
            //
            // CR2 fallback ONLY triggers when the ambient pool is genuinely unusable:
            //   - n_ambient < 100 AND total ambient UMI < 1000
            // This protects against the rare case of truly degenerate ambient profiling.
            // Otherwise, the EmptyDrops result stands.
            if (!cc_result.tested_indices.empty() &&
                cc_result.tested_indices.size() >= 200 &&
                cc_result.n_ambient < 100 &&
                cc_result.ambient_total < 1000.0) {
                double call_rate = static_cast<double>(cc_result.cell_indices.size()) /
                                   static_cast<double>(cc_result.tested_indices.size());
                if (call_rate > 0.95) {
                    std::cerr << "[cell_calling] EmptyDrops call_rate="
                              << static_cast<int>(call_rate * 100 + 0.5)
                              << "% AND n_ambient=" << cc_result.n_ambient
                              << " ambient_total=" << static_cast<uint64_t>(cc_result.ambient_total)
                              << " — ambient genuinely unusable; using CR2 fallback\n";

                    // Sort tested barcodes by UMI descending
                    std::vector<uint32_t> sorted_test = cc_result.tested_indices;
                    std::sort(sorted_test.begin(), sorted_test.end(),
                              [&](uint32_t a, uint32_t b) {
                                  return bc_totals[a] > bc_totals[b];
                              });

                    // 99th-percentile barcode (top 1%) UMI
                    const size_t pct99_idx = std::max(size_t(1), sorted_test.size() / 100) - 1;
                    const uint64_t pct99_umi = bc_totals[sorted_test[pct99_idx]];
                    // CellRanger2: cells = barcodes with UMI ≥ 10% of 99th-pct UMI
                    const uint64_t cell_threshold = std::max(uint64_t(1), pct99_umi / 10);

                    cc_result.cell_indices.clear();
                    for (uint32_t bc : sorted_test)
                        if (bc_totals[bc] >= cell_threshold)
                            cc_result.cell_indices.push_back(bc);
                    cell_caller_method = "cr2_fallback";
                    std::cerr << "[cell_calling] CR2 fallback: pct99_umi=" << pct99_umi
                              << " threshold=" << cell_threshold
                              << " n_cells=" << cc_result.cell_indices.size() << "\n";
                }
            }

            // ── EMPTYDROPS-ZERO-FALLBACK: top-N by UMI when EmptyDrops calls 0 cells ──
            // EmptyDrops can return 0 cells on shallow libraries (no clear knee) or deep
            // libraries where ambient saturation defeats the LRT.  When this happens and
            // enough barcodes have non-trivial UMI signal, fall back to top-N by total
            // UMI rather than reporting zero cells.
            if (cc_result.cell_indices.empty()) {
                // Count barcodes with >= 100 UMI
                uint32_t n_above_100 = 0;
                for (uint64_t c : bc_totals)
                    if (c >= 100) ++n_above_100;

                if (n_above_100 >= 10) {
                    // Determine N: min(expected_cells_hint, 500, n_above_100)
                    uint32_t fallback_n = std::min(
                        static_cast<uint32_t>(500u), n_above_100);
                    if (export_cfg.expect_cells > 0)
                        fallback_n = std::min(
                            fallback_n,
                            static_cast<uint32_t>(export_cfg.expect_cells));

                    std::cerr << "[cell_calling] EmptyDrops returned 0 cells — "
                              << "falling back to top-" << fallback_n
                              << " by UMI (n_barcodes_above_100=" << n_above_100 << ")\n";

                    // Rank barcodes descending by total UMI
                    std::vector<uint32_t> order(exon_csc.ncols);
                    std::iota(order.begin(), order.end(), 0u);
                    std::sort(order.begin(), order.end(),
                              [&](uint32_t a, uint32_t b) {
                                  return bc_totals[a] > bc_totals[b];
                              });

                    cc_result.cell_indices.assign(
                        order.begin(),
                        order.begin() + static_cast<ptrdiff_t>(fallback_n));
                    // Populate tested_indices, deviance, fdr with sentinel values so
                    // write_cell_calls() can iterate without UB (matches knee fallback
                    // sentinel convention: deviance=0, fdr=0 for called cells).
                    cc_result.tested_indices = cc_result.cell_indices;
                    cc_result.deviance.assign(fallback_n, 0.0);
                    cc_result.fdr.assign(fallback_n, 0.0);

                    cell_caller_method = "top_n_fallback";
                    std::cerr << "[cell_calling] top_n_fallback: called "
                              << cc_result.cell_indices.size() << " cells\n";
                } else {
                    std::cerr << "[cell_calling] EmptyDrops returned 0 cells and only "
                              << n_above_100 << " barcodes >= 100 UMI — sample is empty\n";
                    cell_caller_method = "emptydrops";
                }
            } else {
                // Only assign if not already set by a fallback (e.g. cr2_fallback overrides
                // must survive this block).  Determine whether EmptyDrops itself or the knee
                // fallback ran (knee fallback sets n_ambient=0 and logs its own message).
                if (cell_caller_method.empty())
                    cell_caller_method = (cc_result.n_ambient == 0) ? "knee_fallback" : "emptydrops";
            }
        }
        } // end if (!used_star_solo)
        auto t_cc1 = std::chrono::high_resolution_clock::now();
        double cc_s = std::chrono::duration<double>(t_cc1 - t_cc0).count();
        if (!used_called_v2)
            singlet::write_cell_calls(out_prefix + "/cell_calls.tsv",
                                      cc_result, barcodes, bc_totals,
                                      export_cfg.fdr_threshold,
                                      cell_caller_method);
        result.cell_caller = cell_caller_method;
        std::cerr << "[cell_calling] n_cells=" << cc_result.cell_indices.size()
                  << " cell_caller=" << cell_caller_method
                  << " time=" << cc_s << "s\n";

        // G-METRICS: write metrics_summary.csv whenever cell calling ran
        if (export_cfg.pileup_stats != nullptr
            && engine.gene_model().has_gene_hierarchy()) {
            singlet::write_metrics_summary_csv(
                out_prefix + "/metrics_summary.csv",
                *export_cfg.pileup_stats,
                cc_result.cell_indices,
                bc_totals,
                exon_csc,
                engine.gene_model());
        }
    }

    // -- Phase 1.75: Ambient RNA correction (N11, gene-level DecontX-style cluster EM) --
    // Runs whenever cell calling produced a non-empty cell set. The estimator works on GENE counts (exon, plus intron
    // counts when present): ambient profile from the empty droplets (1..lower_umi UMIs), quick deterministic clusters,
    // per-cluster native profile, per-cell Beta-prior EM for rho (docs: include/singlet/pileup/ambient_correction.h).
    // Always writes ambient_profile.tsv (gene level), ambient_contamination.tsv/.json (called cells).
    // The integer-corrected gene x barcode matrix (ambient_corrected_counts, called cells only, raw counts untouched)
    // is produced in pipeline mode or when run_ambient_correction is set; it is written by the "N11" block below.
    GeneCSC<uint32_t> ambient_corrected_csc;
    std::vector<std::string> ambient_feature_names;  ///< gene_id_gene_name, same strings as ambiguous_counts
    // kept for the final ambient_contamination.json, which is rewritten once the doublet calls exist (headline excludes them)
    singlet::AmbientCorrectionResult amb_saved;
    std::vector<uint32_t> amb_saved_cells;
    singlet::AmbientDecontConfig amb_saved_cfg;
    bool amb_saved_have = false;
    if (!cc_result.cell_indices.empty() && exon_csc.ncols > 0 && !bc_totals.empty()
        && engine.gene_model().n_genes() > 0) {
        auto t_amb0 = std::chrono::high_resolution_clock::now();
        const bool amb_has_int = pileup_cfg.count_introns && intron_csc.data.size() > 0
                                 && intron_csc.ncols == exon_csc.ncols;
        // order-free-rules R10: the pool is defined by each barcode's own counts (5..lower_umi UMIs; every barcode
        // list the engine builds holds all barcodes with >= 5 reads), cap ties and the cells' order in the fit follow
        // the barcode sequence, never the column order of the list.
        std::vector<uint32_t> amb_seq_rank(barcodes.size());
        {
            std::vector<uint32_t> by_seq(barcodes.size());
            for (uint32_t b = 0; b < by_seq.size(); ++b) by_seq[b] = b;
            std::sort(by_seq.begin(), by_seq.end(),
                      [&](uint32_t a, uint32_t b2) { return barcodes[a] != barcodes[b2] ? barcodes[a] < barcodes[b2] : a < b2; });
            for (uint32_t r = 0; r < by_seq.size(); ++r) amb_seq_rank[by_seq[r]] = r;
        }
        std::vector<uint32_t> amb_cells = cc_result.cell_indices;
        if (amb_seq_rank.size() == exon_csc.ncols)
            std::sort(amb_cells.begin(), amb_cells.end(),
                      [&](uint32_t a, uint32_t b2) { return amb_seq_rank[a] < amb_seq_rank[b2]; });
        singlet::AmbientDecontConfig amb_cfg;
        amb_cfg.lower_umi = export_cfg.lower_umi;
        amb_cfg.pool_min_umi = 5;
        if (amb_seq_rank.size() == exon_csc.ncols) amb_cfg.pool_tie_rank = &amb_seq_rank;
        // R10b: the fit's random numbers come from each cell's barcode (packed 2 bits per base, first base highest),
        // 10 k-means restarts
        std::vector<uint64_t> amb_keys(barcodes.size(), 0);
        for (size_t b = 0; b < barcodes.size(); ++b) {
            uint64_t k = 0;
            const auto& s = barcodes[b];
            for (size_t i = 0; i < s.size() && i < 32; ++i) {
                const char c = static_cast<char>(s[i] | 32);
                k = (k << 2) | (c == 'a' ? 0u : c == 'c' ? 1u : c == 'g' ? 2u : 3u);
            }
            amb_keys[b] = k;
        }
        if (amb_keys.size() == exon_csc.ncols) amb_cfg.barcode_keys = &amb_keys;
        amb_cfg.kmeans_restarts = 10;
        amb_cfg.ensemble = 32;   // R10c: smallest M with per-cell rho >= 0.99 stream vs sorted and under a 1% read drop
        amb_cfg.threads = export_cfg.threads > 1 ? static_cast<uint32_t>(export_cfg.threads) : 0u;
        // Exon and intron gene counts enter as separate features (the soup is exon-rich, cells are 20-30% intronic);
        // contamination is estimated from the exon features, the correction acts on both and is folded back to genes.
        auto amb_exon_gene = collapse_exon_to_gene(exon_csc, engine.gene_model());
        singlet::AmbientCorrectionResult amb;
        if (amb_has_int) {
            auto amb_intron_gene = [&]() {
                // intron-only gene collapse = collapse of (no exon counts, intron counts)
                std::decay_t<decltype(exon_csc)> zero_exon;
                zero_exon.nrows = exon_csc.nrows; zero_exon.ncols = exon_csc.ncols;
                zero_exon.indptr.assign(static_cast<size_t>(exon_csc.ncols) + 1, 0);
                return collapse_to_gene_counts(zero_exon, intron_csc, engine.gene_model());
            }();
            amb = singlet::correct_ambient_gene_level_split(amb_exon_gene, amb_intron_gene, amb_cells, amb_cfg);
        } else {
            amb = singlet::correct_ambient_gene_level(amb_exon_gene, amb_cells, amb_cfg);
        }
        amb_exon_gene = decltype(amb_exon_gene)();  // release the all-barcode gene matrix
        {
            const auto& ids = engine.gene_ids();
            const auto& gnames = engine.gene_names();
            ambient_feature_names.reserve(ids.size());
            for (size_t i = 0; i < ids.size(); ++i)
                ambient_feature_names.push_back(ids[i] + "_" + (i < gnames.size() ? gnames[i] : std::string()));
        }
        singlet::write_ambient_profile(
            out_prefix + "/ambient_profile.tsv", amb.profile, ambient_feature_names);
        singlet::write_contamination_tsv(
            out_prefix + "/ambient_contamination.tsv",
            amb.contam, amb_cells, barcodes);
        singlet::write_ambient_json(out_prefix + "/ambient_contamination.json", amb, amb_cfg);
        const bool want_corrected = export_cfg.pipeline_mode || export_cfg.run_ambient_correction;
        if (want_corrected && amb.ok && ambient_feature_names.size() == amb.corrected.nrows)
            ambient_corrected_csc = std::move(amb.corrected);
        {   // a copy without the (moved or unused) corrected matrix
            auto keep_corrected = std::move(amb.corrected);
            amb_saved = amb;
            amb.corrected = std::move(keep_corrected);
            amb_saved_cells = amb_cells;
            amb_saved_cfg = amb_cfg;
            amb_saved_have = true;
        }
        auto t_amb1 = std::chrono::high_resolution_clock::now();
        double amb_s = std::chrono::duration<double>(t_amb1 - t_amb0).count();
        std::cerr << "[ambient] method=" << (amb.ok ? amb.method : std::string("none_no_ambient_pool"))
                  << " n_ambient_barcodes=" << amb.profile.n_ambient_barcodes
                  << " total_ambient_umi=" << static_cast<uint64_t>(amb.profile.total_ambient_umi)
                  << " n_clusters=" << amb.n_clusters
                  << " mean_rho=" << amb.mean_rho
                  << " median_rho=" << amb.median_rho
                  << " depth_gamma=" << amb.depth_gamma << " prior_reads=" << amb.prior_reads << " clusters_capped=" << amb.n_clusters_capped
                  << " removed_umis=" << amb.umis_removed << "/" << amb.umis_observed
                  << " n_cells=" << amb.contam.size()
                  << " corrected=" << (ambient_corrected_csc.data.empty() ? "no" : "yes")
                  << " time=" << amb_s << "s\n";
    }

    // ── Phase 1.76: Doublet detection (N12) ──
    nvexp.next("export: Phase 1.76: Doublet detection (N12)");
    // Runs when we have called cells OR when --barcodes was provided (all barcodes
    // are treated as cells in that case, cc_result.cell_indices is empty).
    // Score: Scrublet-style simulation (8000 HVG genes of exon + intron counts, 30 PCs, simulated pair-sums, kNN) plus a
    // logistic classifier on the neighbour fraction and library size (see doublet_detect.h).  The call is capped at the
    // expected multiplet rate (0.8% per 1,000 cells, <= 25%) times the detectable fraction.
    //
    // FIX-ZEROCELL-HANG: when no cells were called AND the whitelist was used as-is
    // (--barcodes not explicitly provided), exon_csc.ncols can be the full whitelist
    // size (~3.6M barcodes).  Running detect_doublets on that is O(n^2) kNN → hang.
    // Guard: skip doublet detection when (a) cc_result is empty (no called cells) AND
    // (b) ncols > 50000 (whitelist-scale, not user-provided barcodes).
    // If --barcodes was explicitly given, ncols is the user list (typically ≤50K) → ok.
    static constexpr uint32_t DOUBLET_MAX_BARCODES = 50000;
    std::vector<DoubletResult> dbl_results;
    std::vector<uint32_t> dbl_cell_indices;
    {
        // Build cell index set: prefer cc_result (cell calling ran), else all barcodes
        if (!cc_result.cell_indices.empty()) {
            dbl_cell_indices = cc_result.cell_indices;
        } else if (exon_csc.ncols > 0 && !pileup_cfg.exon_gtf_path.empty()) {
            if (exon_csc.ncols > DOUBLET_MAX_BARCODES) {
                // Full whitelist with 0 called cells — skip to avoid O(n^2) kNN hang.
                std::cerr << "[doublet] SKIP: ncols=" << exon_csc.ncols
                          << " > " << DOUBLET_MAX_BARCODES
                          << " and no called cells — 0-cell sample, skipping doublet detection\n";
            } else {
                // --barcodes provided with a small explicit list: treat all as cells.
                dbl_cell_indices.resize(exon_csc.ncols);
                std::iota(dbl_cell_indices.begin(), dbl_cell_indices.end(), 0u);
            }
        }
        if (!dbl_cell_indices.empty() && exon_csc.ncols > 0) {
            auto t_dbl0 = std::chrono::high_resolution_clock::now();
            // v7 scorer: exon intervals are folded to genes inside detect_doublets (called cells only), the multiplet
            // rate is the 10x rule from the cell count (0.8% per 1,000 cells, <= 25%) and the call is capped by it.
            std::vector<uint32_t> dbl_exon2gene(exon_csc.nrows);
            for (uint32_t e = 0; e < exon_csc.nrows; ++e)
                dbl_exon2gene[e] = engine.gene_model().exon_to_gene(e);
            // Intron counts are a second source folded into the same genes (gene counts = exon + intron, as in the
            // STARsolo GeneFull matrix Scrublet is usually run on); +0.01 AUROC on the Kang pool.
            const bool dbl_with_introns = pileup_cfg.count_introns && intron_csc.ncols == exon_csc.ncols
                                          && !intron_csc.data.empty();
            std::vector<uint32_t> dbl_intron2gene;
            if (dbl_with_introns) {
                dbl_intron2gene.resize(intron_csc.nrows);
                for (uint32_t e = 0; e < intron_csc.nrows; ++e)
                    dbl_intron2gene[e] = engine.gene_model().intron_to_gene(e);
            }
            singlet::DoubletParams dbl_prm;
            dbl_prm.threads = export_cfg.threads > 1 ? export_cfg.threads : 0;  // 0 = all hardware threads
            singlet::DoubletSummary dbl_sum;
            dbl_results = singlet::detect_doublets(exon_csc, dbl_cell_indices, dbl_prm,
                                                   &dbl_exon2gene, engine.gene_model().n_genes(), &dbl_sum,
                                                   dbl_with_introns ? &intron_csc : nullptr,
                                                   dbl_with_introns ? &dbl_intron2gene : nullptr);
            singlet::write_doublet_tsv(
                out_prefix + "/doublet_scores.tsv",
                dbl_results, dbl_cell_indices, barcodes, exon_csc);
            auto t_dbl1 = std::chrono::high_resolution_clock::now();
            double dbl_s = std::chrono::duration<double>(t_dbl1 - t_dbl0).count();
            uint32_t n_doublets = 0;
            for (const auto& r : dbl_results) if (r.is_doublet) ++n_doublets;
            if (amb_saved_have && !dbl_results.empty()) {
                std::vector<char> dflag(exon_csc.ncols, 0);
                for (size_t di = 0; di < dbl_cell_indices.size() && di < dbl_results.size(); ++di)
                    if (dbl_cell_indices[di] < dflag.size()) dflag[dbl_cell_indices[di]] = dbl_results[di].is_doublet ? 1 : 0;
                std::vector<char> excl(amb_saved.contam.size(), 0);
                for (size_t i = 0; i < excl.size() && i < amb_saved_cells.size(); ++i)
                    if (amb_saved_cells[i] < dflag.size()) excl[i] = dflag[amb_saved_cells[i]];
                singlet::write_ambient_json(out_prefix + "/ambient_contamination.json", amb_saved, amb_saved_cfg, &excl);
            }
            std::cerr << "[doublet] n_cells=" << dbl_cell_indices.size()
                      << " n_doublets=" << n_doublets
                      << " doublet_rate=" << (dbl_results.empty() ? 0.0 :
                         100.0 * n_doublets / dbl_results.size()) << "%"
                      << " expected_rate=" << 100.0 * dbl_sum.expected_rate << "%"
                      << " detectable_fraction=" << dbl_sum.detectable_fraction
                      << " hvg=" << dbl_sum.n_hvg << " pcs=" << dbl_sum.n_pcs
                      << " time=" << dbl_s << "s\n";
        }  // if (!dbl_cell_indices.empty())
    }  // Phase 1.76 block

    // ── Phase 1.8: Write unified cell_qc_metrics.tsv ──
    nvexp.next("export: Phase 1.8: Write unified cellqcmetrics.tsv");
    // Merges data from Phase 1.5 (QC), 1.57 (cell cycle), 1.58 (read stats),
    // 1.6 (saturation), and 1.76 (doublet) into the schema-compliant TSV.
    // Aligner-level per-cell counts (lost once the BAM is deleted): STARsolo CellReads.stats (unmapped, antisense,
    // intergenic, mito, multi-mapped reads per barcode; star_cell_read_stats.h) and the engine's per-cell alignment
    // QC (nM mismatch rate over unique reads, NH>1 reads). Appended as columns after doublet_score.
    const std::string star_crs_path = out_prefix + "/star_Solo.out/Gene/CellReads.stats";
    StarCellReadStats star_crs = read_star_cell_read_stats(star_crs_path, barcodes);
    if (star_crs.found) {
        std::ofstream qj(out_prefix + "/read_qc_summary.json");
        if (qj)
            qj << "{\n  \"source\": \"STARsolo --soloCellReadStats Standard\",\n"
               << "  \"cb_not_in_passlist_reads\": " << star_crs.no_passlist.cb_match << ",\n"
               << "  \"cb_not_in_passlist_genome_unique\": " << star_crs.no_passlist.genome_u << ",\n"
               << "  \"cb_not_in_passlist_genome_multi\": " << star_crs.no_passlist.genome_m << ",\n"
               << "  \"barcodes_in_star_stats\": " << star_crs.n_rows << "\n}\n";
        std::error_code rm_ec;
        std::filesystem::remove(star_crs_path, rm_ec);  // parsed into cell_qc_metrics + read_qc_summary.json
        std::cerr << "[cell_qc] STAR CellReads.stats: " << star_crs.n_rows << " rows; no-passlist reads "
                  << star_crs.no_passlist.cb_match << "\n";
    }
    const auto& align_qc = engine.per_cell_align_qc();
    if (have_cell_qc) {
        const uint32_t n = static_cast<uint32_t>(barcodes.size());
        std::ofstream f(out_prefix + "/cell_qc_metrics.tsv");
        if (f) {
            f << std::fixed << std::setprecision(4);
            f << "barcode\tn_umi\tn_genes\tpct_mt\tpct_ribo\tintronic_pct"
                 "\tn_reads\tduplication_rate\tsaturation\tcell_cycle_phase\tdoublet_score"
                 "\tis_doublet\tmultimapped_reads\tmismatch_rate\tstar_cb_reads\tunmapped_reads"
                 "\tstar_multimapped_reads\texonic_reads\tintronic_reads\tantisense_reads\tmito_reads"
                 "\tintergenic_reads\n";
            std::unordered_map<uint32_t, int> dbl_call_map;
            for (size_t di = 0; di < dbl_results.size() && di < dbl_cell_indices.size(); ++di)
                dbl_call_map[dbl_cell_indices[di]] = dbl_results[di].is_doublet ? 1 : 0;

            // Build doublet score lookup (indexed by barcode column)
            std::unordered_map<uint32_t, double> dbl_score_map;
            for (size_t di = 0; di < dbl_results.size() && di < dbl_cell_indices.size(); ++di)
                dbl_score_map[dbl_cell_indices[di]] = dbl_results[di].score;

            for (uint32_t i = 0; i < n; ++i) {
                f << barcodes[i] << '\t'
                  << cell_qc.total_umis[i] << '\t'
                  << cell_qc.total_genes[i] << '\t'
                  << cell_qc.mt_pct[i] << '\t'
                  << cell_qc.ribo_pct[i] << '\t'
                  << cell_qc.intronic_pct[i] << '\t';
                // n_reads
                if (i < cell_read_stats.size())
                    f << cell_read_stats[i].total_reads;
                else
                    f << 0;
                f << '\t';
                // duplication_rate
                if (i < cell_read_stats.size())
                    f << cell_read_stats[i].dup_rate;
                else
                    f << "0.0000";
                f << '\t';
                // saturation
                if (i < sat_cells.size())
                    f << sat_cells[i].saturation;
                else
                    f << "NA";
                f << '\t';
                // cell_cycle_phase
                if (i < cc_result_cycle.cells.size())
                    f << cc_result_cycle.cells[i].phase;
                else
                    f << "NA";
                f << '\t';
                // doublet_score
                auto dit = dbl_score_map.find(i);
                if (dit != dbl_score_map.end())
                    f << dit->second;
                else
                    f << "NA";
                // is_doublet
                auto cit = dbl_call_map.find(i);
                f << '\t';
                if (cit != dbl_call_map.end()) f << cit->second; else f << "NA";
                // engine alignment QC
                if (i < align_qc.size()) {
                    const auto& q = align_qc[i];
                    f << '\t' << q.multimapped << '\t';
                    if (q.aligned_bases > 0)
                        f << std::setprecision(6) << static_cast<double>(q.mismatches) / static_cast<double>(q.aligned_bases)
                          << std::setprecision(4);
                    else
                        f << "NA";
                } else {
                    f << "\tNA\tNA";
                }
                // STARsolo read fates
                if (star_crs.found && i < star_crs.per_barcode.size() && star_crs.per_barcode[i].present) {
                    const auto& r = star_crs.per_barcode[i];
                    f << '\t' << r.cb_match << '\t' << r.unmapped() << '\t' << r.genome_m << '\t' << r.exonic << '\t'
                      << r.intronic << '\t' << r.antisense() << '\t' << r.mito << '\t' << r.intergenic();
                } else {
                    f << "\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA";
                }
                f << '\n';
            }
            std::cerr << "[cell_qc] Wrote: " << out_prefix << "/cell_qc_metrics.tsv (" << n << " cells)\n";
        }
    }

    // ── Phase 2: mt heteroplasmy (pipeline mode) ──
    nvexp.next("export: Phase 2: mt heteroplasmy (pipeline mode)");
    mt::MtHetResult mt_het;
    SparseAccumulator<uint16_t>::CSCMatrix mt_rev_csc;  // reverse-strand subset of mt_csc (forward = total - reverse)
    std::string mt_ref_seq;                              // chrM reference from the STAR genome (never the pool majority)
    if (pileup_cfg.count_mt && mt_csc.data.size() > 0) {
        if (engine.mt_alleles_rev().nnz_raw() > 0) mt_rev_csc = engine.mt_alleles_rev().to_csc();
        if (!export_cfg.genome_dir.empty()) {
            auto mt_ref = singlet::extract_mt_reference(export_cfg.genome_dir, export_cfg.reference_build);
            if (mt_ref.found) {
                mt_ref_seq = mt_ref.sequence;
                if (mt_ref.length != mt::MT_LEN)
                    std::cerr << "[mt] WARNING: genome " << mt_ref.contig_name << " is " << mt_ref.length
                              << " bp, pileup layout is " << mt::MT_LEN << " bp; positions beyond the shorter are not called\n";
            }
        }
    }
    if (export_cfg.pipeline_mode && mt_csc.data.size() > 0) {
        // Evidence comes from called cells only (ambient droplets carry cross-donor chrM reads).
        std::vector<uint8_t> mt_cell_mask;
        if (!cc_result.cell_indices.empty()) {
            mt_cell_mask.assign(mt_csc.ncols, 0);
            for (auto j : cc_result.cell_indices)
                if (static_cast<size_t>(j) < mt_cell_mask.size()) mt_cell_mask[static_cast<size_t>(j)] = 1;
        }
        mt::MtCounts mt_tot, mt_rev;
        mt_tot.indptr = mt_csc.indptr.data(); mt_tot.indices = mt_csc.indices.data();
        mt_tot.data = mt_csc.data.data();     mt_tot.n_cells = mt_csc.ncols;
        if (mt_rev_csc.data.size() > 0) {
            mt_rev.indptr = mt_rev_csc.indptr.data(); mt_rev.indices = mt_rev_csc.indices.data();
            mt_rev.data = mt_rev_csc.data.data();     mt_rev.n_cells = mt_rev_csc.ncols;
        }
        mt_het = mt::call_mt_variants(mt_tot, mt_rev.valid() ? &mt_rev : nullptr, mt_ref_seq,
                                      mt::MtCallerConfig(), mt_cell_mask.empty() ? nullptr : &mt_cell_mask);
        std::cerr << "[mt] reference=" << mt_het.ref_source << " pool-majority!=reference at "
                  << mt_het.n_ref_mismatch_pool << " positions; heteroplasmic sites=" << mt_het.n_variants
                  << " of " << mt_het.sites.size() << " evaluated\n";
        mt::write_mt_variants(out_prefix + "/mt_variants.tsv", mt_het);          // public: heteroplasmic sites only
        std::filesystem::create_directories(out_prefix + "/private");
        mt::write_mt_sites_private(out_prefix + "/private/mt_sites.tsv", mt_het); // private: all classes + reasons
    }

    // ── Phase 3: Demux + parallel writes ──
    nvexp.next("export: Phase 3: Demux + parallel writes");
    std::thread demux_thread;
    DemuxResult demux_result;
    if (export_cfg.pipeline_mode && !pileup_cfg.snp_path.empty() && snp_ad_csc.data.size() > 0) {
        demux_thread = std::thread([&]() {
            // B-G5-3: K=1 workaround — skip VB entirely; synthesize all-donor0 result.
            // vireoSNP (and our VB) crash or are meaningless at K=1; the correct answer
            // is trivially that every cell belongs to donor0.
            if (export_cfg.n_donors == 1) {
                demux_result = make_single_donor_result(
                    snp_ad_csc.nrows, snp_ad_csc.ncols,
                    snp_ad_csc.indptr.data(), snp_ad_csc.indices.data(), snp_ad_csc.data.data(),
                    snp_dp_csc.indptr.data(), snp_dp_csc.indices.data(), snp_dp_csc.data.data());
            } else {
                DonorDemuxConfig dcfg;
                dcfg.n_donors = export_cfg.n_donors;
                dcfg.threads = export_cfg.threads;
                dcfg.seed = 42;
                demux_result = run_demux(
                    snp_ad_csc.nrows, snp_ad_csc.ncols,
                    snp_ad_csc.indptr.data(), snp_ad_csc.indices.data(), snp_ad_csc.data.data(),
                    snp_ad_csc.data.size(),
                    snp_dp_csc.indptr.data(), snp_dp_csc.indices.data(), snp_dp_csc.data.data(),
                    snp_dp_csc.data.size(), dcfg,
                    cc_result.cell_indices); // filter VB to EmptyDrops-called cells only
            }
        });
    }

    // .1pz outputs whose write failed, named relative to out_prefix
    // ("exon_counts.1pz", "donor/snp_ad.1pz"). Appended to from several writer
    // threads, hence the mutex. Any entry marks the run fail_export_matrix in
    // summary.json (pack_gse then excludes the sample from its bundle), and no
    // 0x0 stub is written in its place further down: a silent stub is how
    // samples with cells but no data reached bundles.
    std::mutex failed_writes_mu;
    std::vector<std::string> failed_writes;
    auto record_failed_write = [&](const std::string& name) {
        std::lock_guard<std::mutex> lock(failed_writes_mu);
        failed_writes.push_back(name);
        std::cerr << "[export] ERROR: failed to write "
                  << out_prefix << "/" << name << "\n";
    };

    // Write helper — uses atomic write-to-tmp-then-rename for crash safety
    auto write_matrix = [&](const std::string& prefix, auto& csc,
                            const std::vector<std::string>& feature_names) {
        if (use_1pz) {
            if (!atomic_write_1pz(out_prefix + "/" + prefix + ".1pz",
                    csc.nrows, csc.ncols, csc.indptr, csc.indices, csc.data,
                    feature_names, barcodes, 3, 1024, 4, export_cfg.user_meta))
                record_failed_write(prefix + ".1pz");
        } else if (!use_h5ad) {
            write_mtx(out_prefix + "/" + prefix + ".mtx.gz",
                      csc.nrows, csc.ncols, csc.indptr, csc.indices, csc.data);
            write_names(out_prefix + "/" + prefix + "_features.tsv.gz", feature_names);
            write_names(out_prefix + "/" + prefix + "_barcodes.tsv.gz", barcodes);
        }
        // h5ad-only mode: sub-gene matrices not written (gene-level .h5ad written separately)
    };

    std::vector<std::thread> write_threads;

    // SNP matrices: always written when --snps is provided.
    // In pipeline mode the same data also feeds donor demux, but the raw per-cell
    // matrices (snp_ad.1pz / snp_dp.1pz) are independently required by downstream
    // tools (vireo, cellsnp-lite comparison, ASE, ancestry) and MUST be present.
    // Genotype data (per-cell SNP allele counts) is private: <out>/private/, never a public member
    // (docs/CELL-ARCHITECTURE.md: genotypes are never stored in public objects or served).
    if (!pileup_cfg.snp_path.empty()) {
        std::filesystem::create_directories(out_prefix + "/private");
        std::filesystem::create_directories(out_prefix + "/donor");
        write_threads.emplace_back([&]() { write_matrix("private/snp_ad", snp_ad_csc, engine.snp_names()); });
        write_threads.emplace_back([&]() { write_matrix("private/snp_dp", snp_dp_csc, engine.snp_names()); });
    }

    // N15: ASE — per-cell, per-SNP allele counts (both pipeline and non-pipeline).
    // Uses the already-computed snp_ad_csc / snp_dp_csc; no extra BAM pass.
    // Always written when --snps is provided (empty file if no het-SNP coverage).
    if (!pileup_cfg.snp_path.empty()) {
        write_threads.emplace_back([&]() {
            auto ase_entries = ase::compute_ase(snp_ad_csc, snp_dp_csc);
            ase::write_ase_tsv(out_prefix + "/private/ase_counts.tsv",
                               ase_entries, barcodes, engine.snp_names());
        });
    }

    // Gene model matrices
    if (!pileup_cfg.exon_gtf_path.empty()) {
        // exon_counts = the legacy split (exon_csc): every molecule with at least one exon-only read, as STARsolo Gene / Cell Ranger exonic
        // (a molecule that also has an intron-touching read is counted here from its exon-only read, and in intron_counts only when all its
        // reads touch an intron, so exon_counts + intron_counts count every molecule once = GeneFull). The alevin-fry USA labels are the
        // separate gene-level matrices spliced_counts / unspliced_counts / ambiguous_counts below.
        write_threads.emplace_back([&]() { write_matrix("exon_counts", exon_csc, engine.exon_names()); });
        // Reads behind each exon / intron UMI count (same rows, columns and codec, uint32): reads_per_umi = reads / UMIs
        if (engine.reads_matrix_on()) {
            write_threads.emplace_back([&]() { write_matrix("exon_reads", exon_reads_csc, engine.exon_names()); });
            if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0)
                write_threads.emplace_back([&]() { write_matrix("intron_reads", intron_reads_csc, engine.intron_names()); });
            write_threads.emplace_back([&]() {
                const std::string path = out_prefix + "/reads_per_umi_hist.tsv.gz";
                gzFile f = gzopen(path.c_str(), "wb6");
                if (!f) { record_failed_write("reads_per_umi_hist.tsv.gz"); return; }
                std::string line = "barcode";
                for (int k = 1; k < PileupEngine::kReadsHistBins; ++k) line += "\tn" + std::to_string(k);
                line += "\tn" + std::to_string(PileupEngine::kReadsHistBins) + "plus\n";
                gzwrite(f, line.data(), static_cast<unsigned>(line.size()));
                for (const auto& [bc, h] : engine.reads_per_umi_hist()) {
                    line = bc < barcodes.size() ? barcodes[bc] : std::to_string(bc);
                    for (uint32_t v : h) { line += '\t'; line += std::to_string(v); }
                    line += '\n';
                    gzwrite(f, line.data(), static_cast<unsigned>(line.size()));
                }
                gzclose(f);
            });
        }
        // Gene-level aggregation thread: TPM/FPKM, barnyard, h5ad/loom/mtx (no gene_counts.1pz)
        // Gene-level 1pz matrices are NOT written — they are redundant with exon_counts +
        // intron_counts and can be computed on the fly by downstream tools.
        if (engine.gene_model().n_genes() > 0) {
            write_threads.emplace_back([&]() {
                // X = exon + intron per gene: the total, the same from either split (legacy exon_csc / intron_csc here); the
                // spliced / unspliced / ambiguous layers below are the USA labels (layer_exon / layer_intron / amb_csc)
                const bool has_int = pileup_cfg.count_introns && intron_csc.data.size() > 0;
                auto gene_csc = has_int
                    ? collapse_to_gene_counts(exon_csc, intron_csc, engine.gene_model())
                    : collapse_exon_to_gene(exon_csc, engine.gene_model());
                const auto& gene_ids   = engine.gene_model().gene_ids();
                const auto& gene_names_vec = engine.gene_model().gene_names();

                // Non-1pz formats still write gene-level output for compatibility
                if (!use_1pz && !use_h5ad) {
                    // CellRanger-compatible output: filtered_feature_bc_matrix/
                    // filtered_feature_bc_matrix/ holds the called cells only (it held every barcode);
                    // raw_feature_bc_matrix/ (all barcodes) only with --raw-matrix. Without a cell call the
                    // filtered matrix keeps every barcode, as before.
                    std::string mtx_dir = out_prefix + "/filtered_feature_bc_matrix";
                    std::filesystem::create_directories(mtx_dir);
                    if (!cc_result.cell_indices.empty()) {
                        std::vector<uint32_t> cols(cc_result.cell_indices.begin(), cc_result.cell_indices.end());
                        std::sort(cols.begin(), cols.end());
                        auto fcsc = csc_select_columns(gene_csc, cols);
                        std::vector<std::string> fbc;
                        fbc.reserve(cols.size());
                        for (uint32_t c : cols) fbc.push_back(barcodes[c]);
                        write_mtx(mtx_dir + "/matrix.mtx.gz", fcsc.nrows, fcsc.ncols,
                                  fcsc.indptr, fcsc.indices, fcsc.data);
                        write_barcodes_10x(mtx_dir + "/barcodes.tsv.gz", fbc);
                    } else {
                        write_mtx(mtx_dir + "/matrix.mtx.gz",
                                  gene_csc.nrows, gene_csc.ncols,
                                  gene_csc.indptr, gene_csc.indices, gene_csc.data);
                        write_barcodes_10x(mtx_dir + "/barcodes.tsv.gz", barcodes);
                    }
                    write_features_10x(mtx_dir + "/features.tsv.gz", gene_ids, gene_names_vec);
                    if (export_cfg.write_raw_matrix) {
                        std::string raw_dir = out_prefix + "/raw_feature_bc_matrix";
                        std::filesystem::create_directories(raw_dir);
                        write_mtx(raw_dir + "/matrix.mtx.gz",
                                  gene_csc.nrows, gene_csc.ncols,
                                  gene_csc.indptr, gene_csc.indices, gene_csc.data);
                        write_features_10x(raw_dir + "/features.tsv.gz", gene_ids, gene_names_vec);
                        write_barcodes_10x(raw_dir + "/barcodes.tsv.gz", barcodes);
                    }
                }

                // G-H5AD: write gene-level counts as AnnData .h5ad
                if (use_h5ad) {
                    singlet::H5adWriteConfig h5cfg;
                    h5cfg.filepath      = out_prefix + "/filtered_feature_bc_matrix.h5ad";
                    h5cfg.indptr        = gene_csc.indptr.data();
                    h5cfg.indices       = gene_csc.indices.data();
                    h5cfg.data          = gene_csc.data.data();
                    h5cfg.nnz           = static_cast<uint64_t>(gene_csc.data.size());
                    h5cfg.n_genes       = gene_csc.nrows;
                    h5cfg.n_cells       = gene_csc.ncols;
                    h5cfg.gene_names    = &gene_names_vec;
                    h5cfg.gene_ids      = &gene_ids;
                    h5cfg.cell_barcodes = &barcodes;
                    h5cfg.metadata      = export_cfg.user_meta;
                    if (has_int) {
                        // Velocyto layers: each molecule in exactly one of them; X = their sum
                        auto spliced_csc   = gene_csc_minus(
                            collapse_exon_to_gene(*layer_exon, engine.gene_model()), &amb_csc);
                        auto unspliced_csc = collapse_intron_to_gene(*layer_intron, engine.gene_model());
                        auto ambiguous_csc = gene_csc_minus(amb_csc, static_cast<decltype(&amb_csc)>(nullptr));
                        h5cfg.layers.push_back({"spliced",
                            spliced_csc.indptr.data(), spliced_csc.indices.data(),
                            spliced_csc.data.data(),
                            static_cast<uint64_t>(spliced_csc.data.size())});
                        h5cfg.layers.push_back({"unspliced",
                            unspliced_csc.indptr.data(), unspliced_csc.indices.data(),
                            unspliced_csc.data.data(),
                            static_cast<uint64_t>(unspliced_csc.data.size())});
                        h5cfg.layers.push_back({"ambiguous",
                            ambiguous_csc.indptr.data(), ambiguous_csc.indices.data(),
                            ambiguous_csc.data.data(),
                            static_cast<uint64_t>(ambiguous_csc.data.size())});
                        bool ok = singlet::write_h5ad(h5cfg);
                        std::cerr << "[h5ad] " << (ok ? "wrote" : "FAILED") << " "
                                  << h5cfg.filepath << " n_genes=" << h5cfg.n_genes
                                  << " n_cells=" << h5cfg.n_cells
                                  << " nnz=" << h5cfg.nnz << " layers=3\n";
                    } else {
                        bool ok = singlet::write_h5ad(h5cfg);
                        std::cerr << "[h5ad] " << (ok ? "wrote" : "FAILED") << " "
                                  << h5cfg.filepath << " n_genes=" << h5cfg.n_genes
                                  << " n_cells=" << h5cfg.n_cells
                                  << " nnz=" << h5cfg.nnz << "\n";
                    }
                }

                // G-LOOM: write gene-level counts as Loom v3 (velocyto/scVelo compatible)
                if (use_loom) {
                    singlet::LoomWriteConfig lcfg;
                    lcfg.filepath      = out_prefix + "/filtered_feature_bc_matrix.loom";
                    lcfg.indptr        = gene_csc.indptr.data();
                    lcfg.indices       = gene_csc.indices.data();
                    lcfg.data          = gene_csc.data.data();
                    lcfg.nnz           = static_cast<uint64_t>(gene_csc.data.size());
                    lcfg.n_genes       = gene_csc.nrows;
                    lcfg.n_cells       = gene_csc.ncols;
                    lcfg.gene_names    = &gene_names_vec;
                    lcfg.gene_ids      = &gene_ids;
                    lcfg.cell_barcodes = &barcodes;
                    if (has_int) {
                        auto spl  = gene_csc_minus(
                            collapse_exon_to_gene(*layer_exon, engine.gene_model()), &amb_csc);
                        auto uspl = collapse_intron_to_gene(*layer_intron, engine.gene_model());
                        auto amb  = gene_csc_minus(amb_csc, static_cast<decltype(&amb_csc)>(nullptr));
                        LoomWriteConfig::Layer spl_layer, uspl_layer, amb_layer;
                        amb_layer.name    = "ambiguous";
                        amb_layer.indptr  = amb.indptr.data();
                        amb_layer.indices = amb.indices.data();
                        amb_layer.data    = amb.data.data();
                        amb_layer.nnz     = amb.data.size();
                        spl_layer.name    = "spliced";
                        spl_layer.indptr  = spl.indptr.data();
                        spl_layer.indices = spl.indices.data();
                        spl_layer.data    = spl.data.data();
                        spl_layer.nnz     = spl.data.size();
                        uspl_layer.name    = "unspliced";
                        uspl_layer.indptr  = uspl.indptr.data();
                        uspl_layer.indices = uspl.indices.data();
                        uspl_layer.data    = uspl.data.data();
                        uspl_layer.nnz     = uspl.data.size();
                        lcfg.layers.push_back(spl_layer);
                        lcfg.layers.push_back(uspl_layer);
                        lcfg.layers.push_back(amb_layer);
                        bool ok = singlet::write_loom(lcfg);
                        std::cerr << "[loom] " << (ok ? "wrote" : "FAILED") << " "
                                  << lcfg.filepath << " n_genes=" << lcfg.n_genes
                                  << " n_cells=" << lcfg.n_cells << " layers=3\n";
                    } else {
                        bool ok = singlet::write_loom(lcfg);
                        std::cerr << "[loom] " << (ok ? "wrote" : "FAILED") << " "
                                  << lcfg.filepath << " n_genes=" << lcfg.n_genes
                                  << " n_cells=" << lcfg.n_cells << "\n";
                    }
                }

                // G-BARNYARD: per-cell species classification (auto-detects barnyard experiments)
                {
                    auto gene_species = singlet::classify_gene_species(
                        gene_ids, engine.gene_model().gene_names());
                    if (singlet::is_barnyard_experiment(gene_species)) {
                        auto t_by0 = std::chrono::high_resolution_clock::now();
                        // Called cells only (the table covered every barcode, so empty droplets with a
                        // few UMIs of each species showed up as 69k "doublets" on hgmm1k).
                        std::vector<uint32_t> bcols(cc_result.cell_indices.begin(), cc_result.cell_indices.end());
                        if (bcols.empty()) { bcols.resize(gene_csc.ncols); std::iota(bcols.begin(), bcols.end(), 0u); }
                        std::sort(bcols.begin(), bcols.end());
                        auto bcsc = csc_select_columns(gene_csc, bcols);
                        std::vector<std::string> bbc;
                        bbc.reserve(bcols.size());
                        for (uint32_t c : bcols) bbc.push_back(barcodes[c]);
                        auto by = singlet::classify_barnyard(bcsc, gene_species);
                        singlet::write_barnyard_tsv(
                            out_prefix + "/barnyard_classification.tsv", by, bbc);
                        auto t_by1 = std::chrono::high_resolution_clock::now();
                        double by_s = std::chrono::duration<double>(t_by1 - t_by0).count();
                        std::cerr << "[barnyard] human=" << by.n_human
                                  << " mouse=" << by.n_mouse
                                  << " doublet=" << by.n_doublet
                                  << " ambiguous=" << by.n_ambiguous
                                  << " time=" << by_s << "s\n";
                    }
                }

                // G-TPM: aggregate TPM + FPKM across all cells (per-library normalization)
                {
                    auto t_tpm0 = std::chrono::high_resolution_clock::now();
                    const uint32_t ng = gene_csc.nrows;
                    std::vector<int32_t> gene_totals(ng, 0);
                    for (size_t _k = 0; _k < gene_csc.indices.size(); ++_k)
                        gene_totals[static_cast<size_t>(gene_csc.indices[_k])] +=
                            static_cast<int32_t>(gene_csc.data[_k]);
                    auto lengths = singlet::compute_gene_lengths(engine.gene_model());
                    auto tpm  = singlet::compute_tpm(gene_totals, lengths);
                    uint64_t total_m = export_cfg.pileup_stats
                                       ? export_cfg.pileup_stats->mapped_reads : 0;
                    auto fpkm = singlet::compute_fpkm(gene_totals, lengths, total_m);
                    singlet::write_gene_expression_tsv(
                        out_prefix + "/gene_expression.tsv", lengths, gene_totals, tpm, fpkm);
                    auto t_tpm1 = std::chrono::high_resolution_clock::now();
                    double tpm_s = std::chrono::duration<double>(t_tpm1 - t_tpm0).count();
                    double max_tpm = tpm.empty() ? 0.0 : *std::max_element(tpm.begin(), tpm.end());
                    std::cerr << "[tpm_fpkm] n_genes=" << ng
                              << " max_tpm=" << max_tpm
                              << " time=" << tpm_s << "s\n";
                }
            });
        }
        // N11: ambient-corrected gene-level matrix (genes x barcodes, called cells only; pipeline mode
        // or run_ambient_correction). Features gene_id_gene_name, as ambiguous_counts.
        if (ambient_corrected_csc.data.size() > 0)
            write_threads.emplace_back([&]() {
                write_matrix("ambient_corrected_counts", ambient_corrected_csc, ambient_feature_names);
            });
        if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0) {
            // intron_counts = the legacy split (intron_csc): molecules whose reads all touch an intron (GeneFull = exon_counts + intron_counts).
            write_threads.emplace_back([&]() { write_matrix("intron_counts", intron_csc, engine.intron_names()); });
            // The alevin-fry USA molecule labels, gene level, every molecule in exactly one of them:
            //   unspliced_counts  at least one intron-touching read
            //   spliced_counts    no intron-touching read, at least one junction read
            //   ambiguous_counts  exon-only reads, no junction read
            // (velocyto-style spliced = spliced_counts + ambiguous_counts; the h5ad / loom layers are the same three matrices.)
            write_threads.emplace_back([&]() {
                std::vector<std::string> names;
                const auto& ids = engine.gene_ids();
                const auto& gnames = engine.gene_names();
                names.reserve(ids.size());
                for (size_t i = 0; i < ids.size(); ++i) names.push_back(ids[i] + "_" + gnames[i]);
                auto spliced_g   = gene_csc_minus(collapse_exon_to_gene(*layer_exon, engine.gene_model()), &amb_csc);
                auto unspliced_g = collapse_intron_to_gene(*layer_intron, engine.gene_model());
                write_matrix("spliced_counts", spliced_g, names);
                write_matrix("unspliced_counts", unspliced_g, names);
                write_matrix("ambiguous_counts", amb_csc, names);
            });
        }
        if (pileup_cfg.count_sj && !engine.sj_names().empty())
            write_threads.emplace_back([&]() { write_matrix("sj_counts", sj_csc, engine.sj_names()); });
        // Poly(A) site usage (polya_sites.h): sites x barcodes molecules + per-site table with internal-priming flag.
        if (const char* pd = getenv("SINGLET_POLYA_DUMP"); pd && pileup_cfg.count_polya) {   // diagnostics: the raw (site, gene, cell, UMI hash) records, 24 bytes each
            if (FILE* f = fopen(pd, "wb")) {
                for (const auto& r : engine.polya_records()) { fwrite(&r.site, 8, 1, f); fwrite(&r.gene, 4, 1, f); fwrite(&r.bc, 4, 1, f); fwrite(&r.umi, 8, 1, f); }
                fclose(f);
            }
        }
        if (pileup_cfg.count_polya && !engine.polya_records().empty() && engine.gene_model().n_genes() > 0)
            write_threads.emplace_back([&]() {
                auto t_pa0 = std::chrono::high_resolution_clock::now();
                auto pa = build_polya_sites(engine.polya_records(), static_cast<uint32_t>(barcodes.size()),
                                            engine.tid_to_chrom(), engine.gene_ids(), engine.gene_names(),
                                            export_cfg.genome_dir);
                write_matrix("polya_counts", pa, pa.names);
                write_gz_text(out_prefix + "/polya_sites.tsv.gz", pa.table_tsv);
                std::cerr << "[polya] " << pa.n_records << " poly(A)-tailed reads -> " << pa.nrows << " sites ("
                          << pa.n_internal_priming << " flagged internal priming), nnz=" << pa.data.size()
                          << " time=" << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t_pa0).count()
                          << "s\n";
            });
        // sj_features.tsv.gz: STAR motif / annotation / read support and STARsolo-comparable flag per junction
        // (sj_features.h). All junctions stay in sj_counts; filtering is the reader's choice.
        if (pileup_cfg.count_sj && !engine.sj_names().empty() && !sj_csc.data.empty())
            write_threads.emplace_back([&]() {
                std::vector<uint8_t> is_cell;
                if (!cc_result.cell_indices.empty()) {
                    is_cell.assign(sj_csc.ncols, 0);
                    for (uint32_t c : cc_result.cell_indices) if (c < sj_csc.ncols) is_cell[c] = 1;
                }
                uint64_t n_pass = write_sj_features(out_prefix + "/sj_features.tsv.gz", engine.sj_names(), sj_csc,
                                                    is_cell, out_prefix + "/star_SJ.out.tab");
                std::cerr << "[sj_features] " << engine.sj_names().size() << " junctions, " << n_pass
                          << " in STAR SJ.out.tab (star_sj_pass)\n";
            });

        // G-PSI: per-cell splice junction PSI
        if (pileup_cfg.count_sj && !engine.sj_names().empty() && !sj_csc.data.empty()) {
            write_threads.emplace_back([&]() {
                auto psi = compute_psi(sj_csc, engine.sj_names());
                if (!psi.data.empty()) {
                    if (use_1pz) {
                        if (!atomic_write_1pz(out_prefix + "/splice_psi.1pz",
                                      psi.nrows, psi.ncols,
                                      psi.indptr, psi.indices, psi.data,
                                      engine.sj_names(), barcodes,
                                      3, 1024, 4, export_cfg.user_meta))
                            record_failed_write("splice_psi.1pz");
                    } else {
                        write_mtx(out_prefix + "/splice_psi.mtx.gz",
                                  psi.nrows, psi.ncols,
                                  psi.indptr, psi.indices, psi.data);
                        write_names(out_prefix + "/splice_psi_features.tsv.gz", engine.sj_names());
                        write_names(out_prefix + "/splice_psi_barcodes.tsv.gz", barcodes);
                    }
                    write_splice_events(out_prefix + "/splice_events.tsv",
                                        psi.events, engine.sj_names());
                    std::cerr << "[splice_psi] " << psi.events.size() << " events, "
                              << psi.data.size() << " PSI entries\n";
                }
            });
        }
    }

    // N18: CRISPR guide counts output
    if (!pileup_cfg.guide_ref_path.empty() && guide_csc.data.size() > 0) {
        auto gnames = engine.guide_names();
        write_threads.emplace_back([&, gnames]() { write_matrix("guide_counts", guide_csc, gnames); });
    }

    // N17: VDJ gene usage output
    if (engine.has_vdj() && vdj_csc.data.size() > 0) {
        const auto& vnames = engine.vdj_names();
        write_threads.emplace_back([&, vnames]() { write_matrix("vdj_gene_usage", vdj_csc, vnames); });
        std::cerr << "[vdj_counting] n_vdj_genes=" << vdj_csc.nrows
                  << " n_hits=" << vdj_csc.data.size() << "\n";
    } else if (engine.has_vdj()) {
        std::cerr << "[vdj_counting] No VDJ reads detected (0 hits)\n";
    }

    // mt outputs
    auto mt_feat_names = mt::mt_feature_names();
    auto het_names = (export_cfg.pipeline_mode && mt_het.n_variants > 0)
                         ? mt::het_feature_names(mt_het)
                         : std::vector<std::string>{};

    // Raw per-cell chrM allele counts are genotype-like (they identify donors): PRIVATE, both modes.
    // Layout: features = chrM:<1-based pos>:<A|C|G|T> (pos*4 + base) x barcodes; mt_allele_counts = all reads,
    // mt_allele_counts_rev = the reverse-strand subset (forward-strand count = total - reverse).
    // Columns of barcodes that are not called cells are written empty (every barcode stays a column, so the barcode
    // list matches the other members); without a cell call every column is kept.
    if (pileup_cfg.count_mt && mt_csc.data.size() > 0) {
        std::filesystem::create_directories(out_prefix + "/private");
        auto cells_only = [&](const SparseAccumulator<uint16_t>::CSCMatrix& a) {
            if (cc_result.cell_indices.empty()) return a;
            std::vector<uint8_t> keep(a.ncols, 0);
            for (uint32_t c : cc_result.cell_indices) if (c < a.ncols) keep[c] = 1;
            SparseAccumulator<uint16_t>::CSCMatrix o;
            o.nrows = a.nrows; o.ncols = a.ncols; o.indptr.assign(static_cast<size_t>(a.ncols) + 1, 0);
            for (uint32_t j = 0; j < a.ncols; ++j) {
                if (keep[j])
                    for (int32_t k = a.indptr[j]; k < a.indptr[j + 1]; ++k) {
                        o.indices.push_back(a.indices[k]);
                        o.data.push_back(a.data[k]);
                    }
                o.indptr[j + 1] = static_cast<int32_t>(o.indices.size());
            }
            return o;
        };
        write_threads.emplace_back([&, cells_only]() {
            auto m = cells_only(mt_csc);
            write_matrix("private/mt_allele_counts", m, mt_feat_names);
        });
        if (mt_rev_csc.data.size() > 0)
            write_threads.emplace_back([&, cells_only]() {
                auto m = cells_only(mt_rev_csc);
                write_matrix("private/mt_allele_counts_rev", m, mt_feat_names);
            });
    }

    if (export_cfg.pipeline_mode && mt_het.n_variants > 0) {
        write_threads.emplace_back([&]() {
            if (use_1pz) {
                std::vector<uint16_t> scaled(mt_het.data.size());
                for (size_t i = 0; i < mt_het.data.size(); ++i) {
                    float v = mt_het.data[i];
                    if (v < 0.f) v = 0.f;
                    if (v > 1.f) v = 1.f;
                    scaled[i] = static_cast<uint16_t>(v * 10000.f + 0.5f);
                }
                if (!atomic_write_1pz(out_prefix + "/mt_heteroplasmy.1pz",
                        mt_het.n_variants, mt_het.n_cells,
                        mt_het.indptr, mt_het.indices, scaled, het_names, barcodes,
                        3, 1024, 4, export_cfg.user_meta))
                    record_failed_write("mt_heteroplasmy.1pz");
            } else {
                write_mtx(out_prefix + "/mt_heteroplasmy.mtx.gz",
                          mt_het.n_variants, mt_het.n_cells,
                          mt_het.indptr, mt_het.indices, mt_het.data);
                write_names(out_prefix + "/mt_heteroplasmy_features.tsv.gz", het_names);
                write_names(out_prefix + "/mt_heteroplasmy_barcodes.tsv.gz", barcodes);
            }
        });
    }

    for (auto& t : write_threads) t.join();
    result.n_write_threads = static_cast<int>(write_threads.size());
    result.failed_writes = failed_writes;  // writer threads have all joined

    // G6 stats declared here so they are visible to the summary block below
    mt::MtDonorOutputStats g6_stats;

    // Wait for demux; then aggregate per-donor depths and write VCF + coverage maps
    if (demux_thread.joinable()) {
        demux_thread.join();
        write_donor_assignments(out_prefix + "/donor/donor_assignments.tsv",
                                demux_result.assignments, barcodes);
        // The top-level donor_assignments.tsv (a required output) was a header-only stub; it now holds the same
        // table. Donor labels are public; the genotypes behind them are not.
        write_donor_assignments(out_prefix + "/donor_assignments.tsv",
                                demux_result.assignments, barcodes);

        if (demux_result.n_donors_k > 0 && !demux_result.covered_to_original.empty()) {
            auto depths = aggregate_donor_depths(
                demux_result,
                snp_ad_csc.nrows, snp_ad_csc.ncols,
                snp_ad_csc.indptr.data(), snp_ad_csc.indices.data(), snp_ad_csc.data.data(),
                snp_dp_csc.indptr.data(), snp_dp_csc.indices.data(), snp_dp_csc.data.data());

            // Per-donor genotype VCFs and SNP coverage: private.
            std::filesystem::create_directories(out_prefix + "/private/donor");
            write_donor_vcfs(out_prefix + "/private/donor", demux_result, depths, engine.snp_names());
            write_donor_coverages(out_prefix + "/private/donor", demux_result, depths, engine.snp_names());
        }

        // ── G6: Donor-aware mitochondrial outputs ──
        // Runs when --snps was set (donor demux ran) and mt pileup data exists.
        // Writes mt/ directory with: donor{N}_mt_consensus.fa,
        //   donor{N}_mt_variants.vcf, mt_events.1pz, mt_summary.tsv.
        // Now passes CIGAR indel events (codes 5-7) and GTF path (codes 7/8).
        if (pileup_cfg.count_mt && mt_csc.data.size() > 0 &&
            demux_result.n_donors_k > 0) {
            auto t_g6_0 = std::chrono::high_resolution_clock::now();
            g6_stats = mt::write_mt_donor_outputs(
                out_prefix, mt_csc, demux_result, barcodes,
                export_cfg.user_meta,
                engine.mt_indels(),
                pileup_cfg.exon_gtf_path,
                mt_ref_seq,
                mt_rev_csc.data.empty() ? nullptr : &mt_rev_csc,
                nullptr, &mt_het);
            auto t_g6_1 = std::chrono::high_resolution_clock::now();
            std::cerr << "[mt_event_caller] G6 time: "
                      << std::chrono::duration<double>(t_g6_1 - t_g6_0).count()
                      << "s\n";
        }

        // Release SNP matrices — they were only needed for demux
        { auto _unused1 = std::move(snp_ad_csc); }
        { auto _unused2 = std::move(snp_dp_csc); }
    }

    auto export_t1 = std::chrono::high_resolution_clock::now();
    result.export_time_s = std::chrono::duration<double>(export_t1 - export_t0).count();
    std::cerr << "[export] Total: " << result.export_time_s << "s\n";

    // ── G-RRNA: rRNA contamination report ──
    nvexp.next("export: G-RRNA: rRNA contamination report");
    // Written whenever pileup_stats has rRNA sampling data (always, when stats are available).
    if (export_cfg.pileup_stats != nullptr
        && export_cfg.pileup_stats->rrna_reads_sampled > 0) {
        singlet::RrnaStats rrna;
        rrna.total_reads   = export_cfg.pileup_stats->rrna_reads_sampled;
        rrna.rrna_reads    = export_cfg.pileup_stats->rrna_reads;
        rrna.rrna_fraction = static_cast<double>(rrna.rrna_reads) /
                             static_cast<double>(rrna.total_reads);
        singlet::write_rrna_report(rrna, out_prefix + "/rrna_report.json");
        std::cerr << "[rrna] fraction=" << rrna.rrna_fraction
                  << " (" << rrna.rrna_reads << "/" << rrna.total_reads << " sampled)\n";
    }

    // ── N8: Provenance manifest ──
    nvexp.next("export: N8: Provenance manifest");
    if (!export_cfg.provenance.input_file.empty()) {
        uint64_t total_umis = 0;
        for (auto v : exon_csc.data) total_umis += static_cast<uint64_t>(v);
        // Inject the cell_caller field determined during Phase 1.7 (not in const ExportConfig)
        ProvenanceConfig prov_with_caller = export_cfg.provenance;
        if (!cell_caller_method.empty())
            prov_with_caller.cell_caller = cell_caller_method;
        else if (!export_cfg.provenance.cell_caller.empty())
            prov_with_caller.cell_caller = export_cfg.provenance.cell_caller;
        // N8-FIX: pass cc_result.cell_indices.size() as n_cells so provenance.json "cells"
        // reflects called cells (same as summary.json estimated_cells), not total barcodes
        // in the unfiltered matrix (exon_csc.ncols).  Before this fix, a sample with 40
        // auto-discovered barcodes but 0 EmptyDrops-called cells would write "cells": 40 in
        // provenance.json while summary.json correctly wrote "estimated_cells": 0.
        const uint32_t n_called_cells = static_cast<uint32_t>(cc_result.cell_indices.size());
        write_provenance_json(export_cfg.out_prefix,
                              prov_with_caller,
                              exon_csc.nrows,
                              n_called_cells,
                              total_umis);
        std::cerr << "[export] Provenance: " << export_cfg.out_prefix << "/provenance.json\n";
    }

    // Set in the summary block when a .1pz run called cells but
    // exon_counts.1pz is missing; the standardization pass then leaves that
    // stub out.
    bool skip_exon_stub = false;

    // ── Summary JSON ── (always written; VAL2-compatible structured output)
    nvexp.next("export: Summary JSON");
    {
        PipelineSummary summary;
        // Helper: look up user_meta key
        auto meta_get = [&](const std::string& key) -> std::string {
            auto it = export_cfg.user_meta.find(key);
            return it != export_cfg.user_meta.end() ? it->second : "";
        };
        summary.sample_id        = meta_get("gsm_id");
        summary.protocol         = meta_get("protocol");
        summary.organism         = meta_get("organism");
        summary.singlet_version = export_cfg.provenance.singlet_version;
        summary.track            = export_cfg.provenance.cascade_enabled ? "B" : "A";
        summary.cascade_used     = export_cfg.provenance.cascade_enabled;
        // Schema 1.1 traceability: source-tree git sha + reference-manifest hash from env.
        if (const char* env_sha = std::getenv("SINGLET_GIT_SHA"))
            summary.git_sha = env_sha;
        if (const char* env_rms = std::getenv("SINGLET_REF_MANIFEST_SHA"))
            summary.reference_manifest_sha256 = env_rms;
        // §3.6 required fields from ExportConfig
        summary.protocol_id     = export_cfg.protocol_id;
        summary.protocol_name   = export_cfg.protocol_name.empty()
            ? meta_get("protocol") : export_cfg.protocol_name;
        summary.species         = export_cfg.species.empty()
            ? meta_get("organism") : export_cfg.species;
        summary.reference_build = export_cfg.reference_build;
        summary.peak_rss_gb     = export_cfg.peak_rss_gb;
        summary.memory_tier     = export_cfg.memory_tier;
        // Donor block (filled after demux)
        summary.donor_n_donors_inferred = demux_result.n_donors_k;
        summary.donor_demux_method = (demux_result.n_donors_k > 0) ? "vb_binomial" : "";
        // mt block (from G6 stats)
        summary.mt_donors_with_consensus = g6_stats.n_donors_with_consensus;
        summary.mt_n_events_total        = g6_stats.n_mt_events_total;
        // nonhost block
        summary.nonhost_screened         = export_cfg.nonhost_was_screened;
        summary.nonhost_species_above_em = export_cfg.nonhost_species_above_em;
        // Read metrics
        if (export_cfg.pileup_stats) {
            const auto& ps = *export_cfg.pileup_stats;
            summary.total_reads  = ps.total_reads;
            summary.mapped_reads = ps.mapped_reads;
            summary.mapping_rate = ps.total_reads > 0
                ? static_cast<double>(ps.mapped_reads) / static_cast<double>(ps.total_reads)
                : 0.0;
            if (ps.mapped_reads > 0) {
                summary.exonic_fraction   = static_cast<double>(ps.exon_hits)
                                            / static_cast<double>(ps.mapped_reads);
                summary.intronic_fraction = static_cast<double>(ps.intron_hits)
                                            / static_cast<double>(ps.mapped_reads);
                double genic = static_cast<double>(ps.exon_hits + ps.intron_hits);
                summary.intergenic_fraction = std::max(0.0,
                    1.0 - genic / static_cast<double>(ps.mapped_reads));
            }
        }
        // Override total_reads / mapped_reads / mapping_rate from STAR's Log.final.out.
        // STAR does not write unmapped reads to the BAM by default, so the pileup BAM
        // read counter counts only mapped reads — making total_reads == mapped_reads == 1.0.
        // Log.final.out contains the authoritative input read count and uniquely mapped count.
        {
            std::string star_log = out_prefix + "/star_Log.final.out";
            if (std::FILE* lf = std::fopen(star_log.c_str(), "r")) {
                char buf[256];
                uint64_t input_reads = 0;
                uint64_t uniq_reads  = 0;
                bool got_input = false, got_uniq = false;
                while (std::fgets(buf, static_cast<int>(sizeof(buf)), lf)) {
                    if (!got_input && std::strstr(buf, "Number of input reads")) {
                        const char* pipe = std::strrchr(buf, '|');
                        if (pipe) {
                            input_reads = static_cast<uint64_t>(
                                std::strtoull(pipe + 1, nullptr, 10));
                            got_input = true;
                        }
                    }
                    if (!got_uniq && std::strstr(buf, "Uniquely mapped reads number")) {
                        const char* pipe = std::strrchr(buf, '|');
                        if (pipe) {
                            uniq_reads = static_cast<uint64_t>(
                                std::strtoull(pipe + 1, nullptr, 10));
                            got_uniq = true;
                        }
                    }
                    if (got_input && got_uniq) break;
                }
                std::fclose(lf);
                if (got_input && input_reads > 0) {
                    summary.total_reads  = input_reads;
                    summary.mapped_reads = uniq_reads;
                    summary.mapping_rate = static_cast<double>(uniq_reads)
                                          / static_cast<double>(input_reads);
                }
            }
        }
        // Cell metrics (only when cell calling ran)
        result.n_called_cells = static_cast<uint64_t>(cc_result.cell_indices.size());
        // FIX-ZEROCELL: flag zero-cells outcome when cell calling was configured to run
        // but produced no cells (0-cell sample or protocol mismatch).
        if (result.n_called_cells == 0 && export_cfg.run_cell_calling)
            result.exit_code = 4;  // zero_cells
        summary.estimated_cells = result.n_called_cells;
        if (!cc_result.cell_indices.empty() && !bc_totals.empty()) {
            // Median UMIs per cell
            std::vector<uint64_t> umis;
            umis.reserve(cc_result.cell_indices.size());
            for (uint32_t ci : cc_result.cell_indices)
                if (ci < static_cast<uint32_t>(bc_totals.size()))
                    umis.push_back(bc_totals[ci]);
            if (!umis.empty()) {
                std::nth_element(umis.begin(),
                                 umis.begin() + static_cast<std::ptrdiff_t>(umis.size() / 2),
                                 umis.end());
                summary.median_umis_per_cell = static_cast<double>(umis[umis.size() / 2]);
            }
            // Mean reads per cell
            if (export_cfg.pileup_stats && summary.estimated_cells > 0) {
                summary.mean_reads_per_cell =
                    static_cast<double>(export_cfg.pileup_stats->total_reads)
                    / static_cast<double>(summary.estimated_cells);
            }
            // Fraction reads (UMIs) in cells
            uint64_t in_cell_umi = 0;
            for (uint32_t ci : cc_result.cell_indices)
                if (ci < static_cast<uint32_t>(bc_totals.size()))
                    in_cell_umi += bc_totals[ci];
            uint64_t all_umi = 0;
            for (uint64_t v : bc_totals) all_umi += v;
            if (all_umi > 0)
                summary.fraction_reads_in_cells =
                    static_cast<double>(in_cell_umi) / static_cast<double>(all_umi);
            // Median genes per cell + total genes detected (requires gene model)
            if (!pileup_cfg.exon_gtf_path.empty()
                && engine.gene_model().has_gene_hierarchy()
                && exon_csc.ncols > 0) {
                const uint32_t ng = engine.gene_model().n_genes();
                std::vector<uint32_t> cell_genes;
                cell_genes.reserve(cc_result.cell_indices.size());
                std::vector<bool> gene_seen(ng, false);
                std::vector<bool> gene_any(ng, false);
                for (uint32_t ci : cc_result.cell_indices) {
                    if (ci >= exon_csc.ncols) continue;
                    uint32_t n_g = 0;
                    for (int32_t k = exon_csc.indptr[ci];
                             k < exon_csc.indptr[ci + 1]; ++k) {
                        uint32_t g = engine.gene_model().exon_to_gene(
                            static_cast<uint32_t>(exon_csc.indices[k]));
                        if (g < ng && !gene_seen[g]) {
                            gene_seen[g] = true;
                            gene_any[g]  = true;
                            ++n_g;
                        }
                    }
                    cell_genes.push_back(n_g);
                    // Reset per-cell gene_seen
                    for (int32_t k = exon_csc.indptr[ci];
                             k < exon_csc.indptr[ci + 1]; ++k) {
                        uint32_t g = engine.gene_model().exon_to_gene(
                            static_cast<uint32_t>(exon_csc.indices[k]));
                        if (g < ng) gene_seen[g] = false;
                    }
                }
                if (!cell_genes.empty()) {
                    std::nth_element(cell_genes.begin(),
                        cell_genes.begin() +
                            static_cast<std::ptrdiff_t>(cell_genes.size() / 2),
                        cell_genes.end());
                    summary.median_genes_per_cell =
                        static_cast<double>(cell_genes[cell_genes.size() / 2]);
                }
                summary.total_genes_detected = static_cast<uint64_t>(
                    std::count(gene_any.begin(), gene_any.end(), true));
            }
        }
        // Timing
        summary.wall_seconds   = export_cfg.provenance.wall_seconds + result.export_time_s;
        summary.pileup_seconds = export_cfg.provenance.pileup_seconds;
        summary.star_seconds   = std::max(0.0, export_cfg.provenance.wall_seconds
                                               - export_cfg.provenance.pileup_seconds);
        summary.export_seconds = result.export_time_s;
        // Status + warnings
        summary.user_meta = export_cfg.user_meta;
        std::string assay = meta_get("modality");
        if (assay.empty()) assay = "scrna";
        summary.status = classify_outcome(summary, assay);
        // Hollow-sample guard. When cells were called in a .1pz run the count
        // matrix must be on disk; otherwise the 0x0 exon_counts.1pz stub
        // written below would be packed as a sample that claims cells but
        // holds no data. A failed .1pz write, or a .1pz run that left no
        // exon_counts.1pz, is recorded as status=fail_export_matrix plus a
        // "write_failed:<file>" / "exon_counts_1pz_missing" warning. pack_gse
        // (python/singlet/bundle.py, _hollow_reason) reads summary.json and
        // leaves any such sample out of the bundle, listing it under
        // manifest.json "excluded_samples".
        //
        // --output-format mtx|h5ad|loom writes no .1pz by design, so a
        // missing exon_counts.1pz there is not a failure: the counts are in
        // the requested format. It gets a "no_1pz_output_format" warning and
        // keeps the standard 0x0 stub (validate_output requires the file);
        // pack_gse refuses that stub as hollow because cells were called.
        {
            std::error_code exon_ec;
            const bool exon_1pz_present =
                std::filesystem::exists(out_prefix + "/exon_counts.1pz", exon_ec);
            if (result.n_called_cells > 0 && !exon_1pz_present) {
                if (use_1pz) {
                    skip_exon_stub = true;
                    summary.warnings.push_back("exon_counts_1pz_missing");
                } else {
                    summary.warnings.push_back("no_1pz_output_format");
                }
            }
            for (const auto& name : failed_writes)
                summary.warnings.push_back("write_failed:" + name);
            if (skip_exon_stub || !failed_writes.empty()) {
                summary.status = "fail_export_matrix";
                std::cerr << "[export] ERROR: count matrices incomplete ("
                          << failed_writes.size() << " failed .1pz write(s)"
                          << (skip_exon_stub ? ", no exon_counts.1pz" : "")
                          << " with " << result.n_called_cells
                          << " called cells); summary.json status=fail_export_matrix\n";
            }
        }
        if (summary.mapping_rate > 0.0 && summary.mapping_rate < 0.50)
            summary.warnings.push_back("low_mapping_rate");
        if (summary.estimated_cells > 0 && summary.estimated_cells < 10)
            summary.warnings.push_back("few_cells_detected");
        bool ok = write_summary_json(summary, out_prefix + "/summary.json");
        std::cerr << "[export] summary.json status=" << summary.status
                  << " cells=" << summary.estimated_cells
                  << " mapping_rate=" << summary.mapping_rate
                  << (ok ? "" : " WRITE_FAILED") << "\n";
    }

    // ── MT reference genome extraction ──
    nvexp.next("export: MT reference genome extraction");
    // Always write mt_reference.fa — either the actual chrM sequence from the STAR
    // genome or a stub explaining why it's absent.
    if (!export_cfg.genome_dir.empty()) {
        auto mt_ref = singlet::extract_mt_reference(
            export_cfg.genome_dir, export_cfg.reference_build);
        if (mt_ref.found) {
            singlet::write_mt_reference_fasta(
                out_prefix + "/mt_reference.fa", mt_ref);
            std::cerr << "[mt_reference] Wrote " << mt_ref.contig_name
                      << " (" << mt_ref.length << "bp) from "
                      << export_cfg.reference_build << "\n";
        } else {
            singlet::write_mt_reference_stub(
                out_prefix + "/mt_reference.fa",
                "chrM not found in genome_dir=" + export_cfg.genome_dir);
            std::cerr << "[mt_reference] chrM not found in genome — wrote stub\n";
        }
    } else {
        singlet::write_mt_reference_stub(
            out_prefix + "/mt_reference.fa", "genome_dir not provided");
        std::cerr << "[mt_reference] No genome_dir — wrote stub\n";
    }

    // ── Standardization: write empty stubs for conditional outputs ──
    nvexp.next("export: Standardization: write empty stubs for conditional outputs");
    // Ensures every successful run produces the EXACT same set of files.
    // Files that were already written above are skipped (exists-check).
    {
        auto write_stub_if_missing = [&](const std::string& filename,
                                          const std::string& header) {
            std::string path = out_prefix + "/" + filename;
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                std::ofstream f(path);
                if (f) f << header << "\n";
            }
        };
        auto write_json_stub = [&](const std::string& filename,
                                    const std::string& content) {
            std::string path = out_prefix + "/" + filename;
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                std::ofstream f(path);
                if (f) f << content;
            }
        };
        auto write_1pz_stub = [&](const std::string& filename) {
            // Never stand a 0x0 stub in for a matrix that failed to write, or
            // for exon_counts when a .1pz run called cells but left no matrix
            // (see the hollow-sample guard in the summary block): the stub
            // would look like data to anything that does not read summary.json.
            if (std::find(failed_writes.begin(), failed_writes.end(), filename)
                    != failed_writes.end())
                return;
            if (skip_exon_stub && filename == "exon_counts.1pz")
                return;
            std::string path = out_prefix + "/" + filename;
            std::error_code ec;
            if (!std::filesystem::exists(path, ec)) {
                // Write minimal valid 1pz: 0 features × 0 cells, 0 nnz
                std::vector<int32_t> empty_ip = {0};
                std::vector<int32_t> empty_idx;
                std::vector<uint16_t> empty_data;
                std::vector<std::string> empty_names;
                pz::write_1pz(path, 0u, 0u, empty_ip, empty_idx, empty_data,
                              empty_names, empty_names, 3, 1024, 4,
                              export_cfg.user_meta);
            }
        };

        // 1pz stubs
        write_1pz_stub("sj_counts.1pz");
        write_1pz_stub("mt_heteroplasmy.1pz");
        write_1pz_stub("splice_psi.1pz");
        write_1pz_stub("vdj_gene_usage.1pz");
        write_1pz_stub("intron_counts.1pz");
        write_1pz_stub("ambiguous_counts.1pz");
        write_1pz_stub("spliced_counts.1pz");
        write_1pz_stub("unspliced_counts.1pz");
        write_1pz_stub("exon_counts.1pz");

        // TSV stubs (header-only)
        write_stub_if_missing("cell_qc_metrics.tsv",
            "barcode\tn_umi\tn_genes\tpct_mt\tpct_ribo\tintronic_pct"
            "\tn_reads\tduplication_rate\tsaturation\tcell_cycle_phase\tdoublet_score"
            "\tis_doublet\tmultimapped_reads\tmismatch_rate\tstar_cb_reads\tunmapped_reads"
            "\tstar_multimapped_reads\texonic_reads\tintronic_reads\tantisense_reads\tmito_reads\tintergenic_reads");
        write_stub_if_missing("cell_calls.tsv",
            "barcode\ttotal_umi\tdeviance\tfdr\tis_cell");
        write_stub_if_missing("cell_cycle_scores.tsv",
            "barcode\tphase\ts_score\tg2m_score");
        write_stub_if_missing("doublet_scores.tsv",
            "barcode\ttotal_umis\tdoublet_score\tis_doublet");
        write_stub_if_missing("ambient_contamination.tsv",
            "barcode\trho\tn_genes_used");
        write_stub_if_missing("ambient_profile.tsv",
            "feature\tambient_fraction");
        write_stub_if_missing("read_stats.tsv",
            "barcode\ttotal_reads\tunique_umis\tdup_reads\tdup_rate\test_complexity");
        write_stub_if_missing("donor_assignments.tsv",
            "cell\tdonor_id\tprob_max\tprob_doublet\tbest_singlet\tbest_doublet");
        write_stub_if_missing("auto_barcodes.tsv", "");
        write_stub_if_missing("saturation_curve.tsv",
            "fraction\tsampled_reads\tmedian_umis\tmedian_genes\tmean_umis\tmean_genes");
        write_stub_if_missing("mt_variants.tsv",
            "pos\tref\talt\tn_cells_covered\tn_cells_het\tmean_vaf");
        write_stub_if_missing("gene_expression.tsv",
            "gene_id\tgene_name\teffective_length\tcount\tTPM\tFPKM");
        write_stub_if_missing("splice_events.tsv",
            "event_id\tevent_type\tn_junctions\tjunction_names");
        write_stub_if_missing("metrics_summary.csv",
            "Metric,Value\nEstimated Number of Cells,0");

        // JSON stubs
        write_json_stub("sex_call.json",
            "{\"sex\": \"unknown\", \"confidence\": 0.0}\n");
        write_json_stub("ancestry_call.json",
            "{\"ancestry\": \"unknown\", \"confidence\": 0.0}\n");
        write_json_stub("rrna_report.json",
            "{\"total_reads\": 0, \"rrna_reads\": 0, \"rrna_fraction\": 0.0}\n");
        write_json_stub("pileup_stats.json",
            "{\"version\": \"stub\", \"total_reads\": 0}\n");
        write_json_stub("provenance.json",
            "{\"singlet_version\": \"unknown\", \"note\": \"stub\"}\n");

        // STAR log stubs
        write_stub_if_missing("star_Log.final.out", "# STAR Log.final.out (stub)");
        write_stub_if_missing("star_Log.out", "# STAR Log.out (stub)");

        std::cerr << "[export] Standardization stubs written for any missing files\n";
    }

    // ── Post-export validation ──
    nvexp.next("export: Post-export validation");
    // Verify all required files exist and are non-empty.
    {
        auto vr = singlet::validate_output(out_prefix);
        if (!vr.passed) {
            std::cerr << "[VALIDATION FAILED] " << vr.n_missing << " missing, "
                      << vr.n_empty << " empty out of " << vr.n_expected << " required files\n";
            for (const auto& f : vr.missing_files)
                std::cerr << "  MISSING: " << f << "\n";
            for (const auto& f : vr.empty_files)
                std::cerr << "  EMPTY: " << f << "\n";
            // Write validation result for auditing
            singlet::write_validation_json(
                out_prefix + "/validation.json", vr);
        } else {
            std::cerr << "[export] Validation passed: " << vr.n_found
                      << "/" << vr.n_expected << " files OK\n";
        }
    }

    return result;
}

/// Write pileup stats JSON to file.
inline void write_stats_json(const std::string& path,
                             const PileupStats& stats,
                             const std::string& version,
                             double wall_time_s,
                             const std::string& extra_json = "") {
    std::ofstream sf(path);
    if (!sf) return;
    sf << "{\n"
       << "  \"version\": \"" << version << "\",\n";
    if (!extra_json.empty()) sf << extra_json;
    sf << "  \"total_reads\": " << stats.total_reads << ",\n"
       << "  \"mapped_reads\": " << stats.mapped_reads << ",\n"
       << "  \"barcoded_reads\": " << stats.barcoded_reads << ",\n"
       << "  \"snp_hits\": " << stats.snp_hits << ",\n"
       << "  \"exon_hits\": " << stats.exon_hits << ",\n"
       << "  \"intron_hits\": " << stats.intron_hits << ",\n"
       << "  \"sj_hits\": " << stats.sj_hits << ",\n"
       << "  \"chrm_reads\": " << stats.chrm_reads << ",\n"
       << "  \"mt_pileup_bases\": " << stats.mt_pileup_bases << ",\n"
       << "  \"low_mapq\": " << stats.low_mapq << ",\n"
       << "  \"secondary_reads\": " << stats.secondary_reads << ",\n"
       << "  \"no_barcode\": " << stats.no_barcode << ",\n"
       << "  \"no_umi\": " << stats.no_umi << ",\n"
       << "  \"umi_unique\": " << stats.umi_unique << ",\n"
       << "  \"umi_duplicate\": " << stats.umi_duplicate << ",\n"
       << "  \"multimapper_reads\": " << stats.multimapper_reads << ",\n"
       << "  \"multigene_reads\": " << stats.multigene_reads << ",\n"
       << "  \"wrong_strand\": " << stats.wrong_strand << ",\n"
       << "  \"rrna_reads_sampled\": " << stats.rrna_reads_sampled << ",\n"
       << "  \"rrna_reads\": " << stats.rrna_reads << ",\n"
       << "  \"snp_molecules\": " << stats.snp_molecules << ",\n"
       << "  \"snp_molecules_ambiguous\": " << stats.snp_molecules_ambiguous << ",\n"
       << "  \"snp_molecules_other\": " << stats.snp_molecules_other << ",\n"
       << "  \"molecules_spliced\": " << stats.molecules_spliced << ",\n"
       << "  \"molecules_unspliced\": " << stats.molecules_unspliced << ",\n"
       << "  \"molecules_ambiguous\": " << stats.molecules_ambiguous << ",\n"
       << "  \"molecules_legacy_double\": " << stats.molecules_legacy_double << ",\n"
       << "  \"cross_gene_intron_reads\": " << stats.cross_gene_intron_reads << ",\n"
       << "  \"mt_numt_rescued\": " << stats.mt_numt_rescued << ",\n"
       << "  \"molecules_dropped_multigene_umi\": " << stats.molecules_dropped_multigene_umi << ",\n"
       << "  \"intron_rules\": " << stats.intron_rules << ",\n"
       << "  \"pileup_time_s\": " << stats.wall_time_s << ",\n"
       << "  \"wall_time_s\": " << wall_time_s << "\n"
       << "}\n";
}

} // namespace singlet
