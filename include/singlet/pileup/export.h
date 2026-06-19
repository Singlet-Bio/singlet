// SPDX-License-Identifier: MIT
#pragma once
// singlet-pileup: export.h
// Shared export logic for writing pileup results to disk.
// Used by both the singlet-pileup CLI and the unified singlet pipeline.
//
// Exports: COO→CSC conversion, mt heteroplasmy, donor demux, matrix I/O.
// All operations are parallelized where safe.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
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
    double fdr_threshold = 0.01;       ///< BH-FDR cutoff for cell calls (STARsolo EmptyDrops_CR default)
    uint64_t lower_umi = 100;          ///< Lower UMI bound for ambient pool estimation
    uint64_t min_umi_test = 500;       ///< Min UMI to be tested by EmptyDrops (STARsolo umiMin=500)
    // N11: Ambient RNA correction
    bool run_ambient_correction = false; ///< Write corrected exon_counts_corrected matrix
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

    const auto& barcodes = engine.barcodes();
    const bool use_1pz  = (export_cfg.output_format == "1pz" || export_cfg.output_format == "all");
    const bool use_h5ad = (export_cfg.output_format == "h5ad" || export_cfg.output_format == "all");
    const bool use_loom = (export_cfg.output_format == "loom");
    const auto& out_prefix = export_cfg.out_prefix;

    // ── Phase 1: COO → CSC (parallel) ──
    auto csc_t0 = std::chrono::high_resolution_clock::now();

    SparseAccumulator<uint8_t>::CSCMatrix snp_ad_csc, snp_dp_csc;
    SparseAccumulator<uint16_t>::CSCMatrix exon_csc, intron_csc, sj_csc, mt_csc, guide_csc;

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
        if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0)
            csc_threads.emplace_back([&]() { intron_csc = engine.introns().to_csc(); });
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

    auto csc_t1 = std::chrono::high_resolution_clock::now();
    result.csc_time_s = std::chrono::duration<double>(csc_t1 - csc_t0).count();
    result.n_csc_threads = static_cast<int>(csc_threads.size());
    std::cerr << "[export] CSC conversion: " << result.csc_time_s
              << "s (" << result.n_csc_threads << " threads)\n";

    // ── Phase 1.5: Cell QC metrics (compute only; write deferred to Phase 1.8) ──
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
    // Uses the DirectionalUmiStore groups built during run() — no extra BAM pass.
    std::vector<CellSaturation> sat_cells;
    if (pileup_cfg.umi_dedup_directional && !engine.dir_exon_store().empty()) {
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
    // Declare outside so Phase 1.75 (ambient correction) and N8 (provenance) can access them.
    std::vector<uint64_t> bc_totals;
    CellCallResult cc_result;
    CSCu16 exon_corrected_csc;
    std::string cell_caller_method;  ///< which cell caller ran (written to provenance.json)

    if (export_cfg.run_cell_calling && !pileup_cfg.exon_gtf_path.empty() && exon_csc.ncols > 0) {
        // Sum each column (barcode) of exon CSC to get per-barcode UMI totals.
        // These totals are used for barcode ranking (lower/test UMI thresholds).
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

        // ── STAR Solo.out preferred cell caller ───────────────────────────────────
        // When STAR ran in CB_UMI_Simple mode (10x-style) it writes
        // Solo.out/Gene/filtered/barcodes.tsv — a knee-called, well-calibrated cell
        // set.  Use it directly and skip EmptyDrops entirely.
        // Fall back to EmptyDrops when the file is absent (CB_samTagOut protocols:
        // Drop-seq / sci-RNA / etc. do not produce Solo.out).
        bool used_star_solo = false;
        {
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
        if (!used_star_solo) {
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
        singlet::write_cell_calls(out_prefix + "/cell_calls.tsv",
                                  cc_result, barcodes, bc_totals,
                                  export_cfg.fdr_threshold,
                                  cell_caller_method);
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

    // ── Phase 1.75: Ambient RNA correction (N11, SoupX-lite) ──
    // Runs whenever cell calling produced a non-empty cell set.
    // Always outputs ambient_profile.tsv + ambient_contamination.tsv.
    // Corrected matrix written only when run_ambient_correction=true.
    if (!cc_result.cell_indices.empty() && exon_csc.ncols > 0 && !bc_totals.empty()) {
        auto t_amb0 = std::chrono::high_resolution_clock::now();
        auto ambient = singlet::estimate_ambient(
            exon_csc, bc_totals, export_cfg.lower_umi);
        auto contam = singlet::estimate_contamination(
            exon_csc, ambient, cc_result.cell_indices, bc_totals);
        singlet::write_ambient_profile(
            out_prefix + "/ambient_profile.tsv", ambient, engine.exon_names());
        singlet::write_contamination_tsv(
            out_prefix + "/ambient_contamination.tsv",
            contam, cc_result.cell_indices, barcodes);
        // AUTOFIX-E2E-G-AMBIENT: write ambient_contamination.json summary
        {
            // Compute mean contamination fraction (rho) across called cells
            double sum_rho = 0.0;
            uint64_t n_rho = 0;
            for (const auto& cc : contam) {
                if (cc.n_genes_used > 0) { sum_rho += cc.rho; ++n_rho; }
            }
            const double est_contam = n_rho > 0 ? sum_rho / static_cast<double>(n_rho) : 0.0;
            // Median rho (already computed below for cerr; recompute here in-block)
            std::vector<double> rhos_j;
            rhos_j.reserve(contam.size());
            for (const auto& cc : contam) rhos_j.push_back(cc.rho);
            double med_rho_j = 0.0;
            if (!rhos_j.empty()) {
                std::nth_element(rhos_j.begin(),
                                 rhos_j.begin() + static_cast<std::ptrdiff_t>(rhos_j.size() / 2),
                                 rhos_j.end());
                med_rho_j = rhos_j[rhos_j.size() / 2];
            }
            std::ofstream acj(out_prefix + "/ambient_contamination.json");
            if (acj.is_open()) {
                acj << "{\n"
                    << "  \"estimated_contamination_fraction\": " << est_contam << ",\n"
                    << "  \"ambient_profile_n_barcodes\": "
                        << ambient.n_ambient_barcodes << ",\n"
                    << "  \"ambient_profile_total_umis\": "
                        << static_cast<uint64_t>(ambient.total_ambient_umi) << ",\n"
                    << "  \"method\": \"emptydrops_derived\",\n"
                    << "  \"rho\": " << med_rho_j << "\n"
                    << "}\n";
            }
        }
        if (export_cfg.run_ambient_correction)
            exon_corrected_csc = singlet::correct_counts(
                exon_csc, ambient, contam, cc_result.cell_indices, bc_totals);
        auto t_amb1 = std::chrono::high_resolution_clock::now();
        double amb_s = std::chrono::duration<double>(t_amb1 - t_amb0).count();
        double med_rho = 0.0;
        if (!contam.empty()) {
            std::vector<double> rhos;
            rhos.reserve(contam.size());
            for (const auto& cc : contam) rhos.push_back(cc.rho);
            std::nth_element(rhos.begin(), rhos.begin() + rhos.size() / 2, rhos.end());
            med_rho = rhos[rhos.size() / 2];
        }
        std::cerr << "[ambient] n_ambient_barcodes=" << ambient.n_ambient_barcodes
                  << " total_ambient_umi=" << static_cast<uint64_t>(ambient.total_ambient_umi)
                  << " median_rho=" << med_rho
                  << " n_cells=" << contam.size()
                  << " corrected=" << (export_cfg.run_ambient_correction ? "yes" : "no")
                  << " time=" << amb_s << "s\n";
    }

    // ── Phase 1.76: Doublet detection (N12) ──
    // Runs when we have called cells OR when --barcodes was provided (all barcodes
    // are treated as cells in that case, cc_result.cell_indices is empty).
    // Hybrid score: 0.6 × kNN-sim-fraction + 0.4 × UMI-ratio component.
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
            dbl_results = singlet::detect_doublets(exon_csc, dbl_cell_indices);
            singlet::write_doublet_tsv(
                out_prefix + "/doublet_scores.tsv",
                dbl_results, dbl_cell_indices, barcodes, exon_csc);
            auto t_dbl1 = std::chrono::high_resolution_clock::now();
            double dbl_s = std::chrono::duration<double>(t_dbl1 - t_dbl0).count();
            uint32_t n_doublets = 0;
            for (const auto& r : dbl_results) if (r.is_doublet) ++n_doublets;
            std::cerr << "[doublet] n_cells=" << dbl_cell_indices.size()
                      << " n_doublets=" << n_doublets
                      << " doublet_rate=" << (dbl_results.empty() ? 0.0 :
                         100.0 * n_doublets / dbl_results.size()) << "%"
                      << " time=" << dbl_s << "s\n";
        }  // if (!dbl_cell_indices.empty())
    }  // Phase 1.76 block

    // ── Phase 1.8: Write unified cell_qc_metrics.tsv ──
    // Merges data from Phase 1.5 (QC), 1.57 (cell cycle), 1.58 (read stats),
    // 1.6 (saturation), and 1.76 (doublet) into the schema-compliant TSV.
    if (have_cell_qc) {
        const uint32_t n = static_cast<uint32_t>(barcodes.size());
        std::ofstream f(out_prefix + "/cell_qc_metrics.tsv");
        if (f) {
            f << std::fixed << std::setprecision(4);
            f << "barcode\tn_umi\tn_genes\tpct_mt\tpct_ribo\tintronic_pct"
                 "\tn_reads\tduplication_rate\tsaturation\tcell_cycle_phase\tdoublet_score\n";

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
                f << '\n';
            }
            std::cerr << "[cell_qc] Wrote: " << out_prefix << "/cell_qc_metrics.tsv (" << n << " cells)\n";
        }
    }

    // ── Phase 2: mt heteroplasmy (pipeline mode) ──
    mt::MtHetResult mt_het;
    if (export_cfg.pipeline_mode && mt_csc.data.size() > 0) {
        mt_het = mt::compute_heteroplasmy(
            mt_csc.indptr.data(), mt_csc.indices.data(), mt_csc.data.data(),
            mt_csc.nrows, mt_csc.ncols);
        mt::write_mt_variants(out_prefix + "/mt_variants.tsv", mt_het);
    }

    // ── Phase 3: Demux + parallel writes ──
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

    // Write helper — uses atomic write-to-tmp-then-rename for crash safety
    auto write_matrix = [&](const std::string& prefix, auto& csc,
                            const std::vector<std::string>& feature_names) {
        if (use_1pz) {
            atomic_write_1pz(out_prefix + "/" + prefix + ".1pz",
                csc.nrows, csc.ncols, csc.indptr, csc.indices, csc.data,
                feature_names, barcodes, 3, 1024, 4, export_cfg.user_meta);
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
    if (!pileup_cfg.snp_path.empty()) {
        std::filesystem::create_directories(out_prefix + "/donor");
        write_threads.emplace_back([&]() { write_matrix("donor/snp_ad", snp_ad_csc, engine.snp_names()); });
        write_threads.emplace_back([&]() { write_matrix("donor/snp_dp", snp_dp_csc, engine.snp_names()); });
    }

    // N15: ASE — per-cell, per-SNP allele counts (both pipeline and non-pipeline).
    // Uses the already-computed snp_ad_csc / snp_dp_csc; no extra BAM pass.
    // Always written when --snps is provided (empty file if no het-SNP coverage).
    if (!pileup_cfg.snp_path.empty()) {
        write_threads.emplace_back([&]() {
            auto ase_entries = ase::compute_ase(snp_ad_csc, snp_dp_csc);
            ase::write_ase_tsv(out_prefix + "/ase_counts.tsv",
                               ase_entries, barcodes, engine.snp_names());
        });
    }

    // Gene model matrices
    if (!pileup_cfg.exon_gtf_path.empty()) {
        write_threads.emplace_back([&]() { write_matrix("exon_counts", exon_csc, engine.exon_names()); });
        // Gene-level aggregation thread: TPM/FPKM, barnyard, h5ad/loom/mtx (no gene_counts.1pz)
        // Gene-level 1pz matrices are NOT written — they are redundant with exon_counts +
        // intron_counts and can be computed on the fly by downstream tools.
        if (engine.gene_model().n_genes() > 0) {
            write_threads.emplace_back([&]() {
                const bool has_int = pileup_cfg.count_introns && intron_csc.data.size() > 0;
                auto gene_csc = has_int
                    ? collapse_to_gene_counts(exon_csc, intron_csc, engine.gene_model())
                    : collapse_exon_to_gene(exon_csc, engine.gene_model());
                const auto& gene_ids   = engine.gene_model().gene_ids();
                const auto& gene_names_vec = engine.gene_model().gene_names();

                // Non-1pz formats still write gene-level output for compatibility
                if (!use_1pz && !use_h5ad) {
                    // CellRanger-compatible output: filtered_feature_bc_matrix/
                    std::string mtx_dir = out_prefix + "/filtered_feature_bc_matrix";
                    std::filesystem::create_directories(mtx_dir);
                    write_mtx(mtx_dir + "/matrix.mtx.gz",
                              gene_csc.nrows, gene_csc.ncols,
                              gene_csc.indptr, gene_csc.indices, gene_csc.data);
                    write_features_10x(mtx_dir + "/features.tsv.gz", gene_ids, gene_names_vec);
                    write_barcodes_10x(mtx_dir + "/barcodes.tsv.gz", barcodes);
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
                        auto spliced_csc   = collapse_exon_to_gene(exon_csc, engine.gene_model());
                        auto unspliced_csc = collapse_intron_to_gene(intron_csc, engine.gene_model());
                        h5cfg.layers.push_back({"spliced",
                            spliced_csc.indptr.data(), spliced_csc.indices.data(),
                            spliced_csc.data.data(),
                            static_cast<uint64_t>(spliced_csc.data.size())});
                        h5cfg.layers.push_back({"unspliced",
                            unspliced_csc.indptr.data(), unspliced_csc.indices.data(),
                            unspliced_csc.data.data(),
                            static_cast<uint64_t>(unspliced_csc.data.size())});
                        bool ok = singlet::write_h5ad(h5cfg);
                        std::cerr << "[h5ad] " << (ok ? "wrote" : "FAILED") << " "
                                  << h5cfg.filepath << " n_genes=" << h5cfg.n_genes
                                  << " n_cells=" << h5cfg.n_cells
                                  << " nnz=" << h5cfg.nnz << " layers=2\n";
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
                        auto spl  = collapse_exon_to_gene(exon_csc, engine.gene_model());
                        auto uspl = collapse_intron_to_gene(intron_csc, engine.gene_model());
                        LoomWriteConfig::Layer spl_layer, uspl_layer;
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
                        bool ok = singlet::write_loom(lcfg);
                        std::cerr << "[loom] " << (ok ? "wrote" : "FAILED") << " "
                                  << lcfg.filepath << " n_genes=" << lcfg.n_genes
                                  << " n_cells=" << lcfg.n_cells << " layers=2\n";
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
                        auto by = singlet::classify_barnyard(gene_csc, gene_species);
                        singlet::write_barnyard_tsv(
                            out_prefix + "/barnyard_classification.tsv", by, barcodes);
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
        // N11: ambient-corrected matrix (written only when run_ambient_correction=true)
        if (exon_corrected_csc.data.size() > 0)
            write_threads.emplace_back([&]() {
                write_matrix("exon_counts_corrected", exon_corrected_csc, engine.exon_names());
            });
        if (pileup_cfg.count_introns && engine.gene_model().n_introns() > 0)
            write_threads.emplace_back([&]() { write_matrix("intron_counts", intron_csc, engine.intron_names()); });
        if (pileup_cfg.count_sj && !engine.sj_names().empty())
            write_threads.emplace_back([&]() { write_matrix("sj_counts", sj_csc, engine.sj_names()); });

        // G-PSI: per-cell splice junction PSI
        if (pileup_cfg.count_sj && !engine.sj_names().empty() && !sj_csc.data.empty()) {
            write_threads.emplace_back([&]() {
                auto psi = compute_psi(sj_csc, engine.sj_names());
                if (!psi.data.empty()) {
                    if (use_1pz) {
                        pz::write_1pz(out_prefix + "/splice_psi.1pz",
                                      psi.nrows, psi.ncols,
                                      psi.indptr, psi.indices, psi.data,
                                      engine.sj_names(), barcodes,
                                      3, 1024, 4, export_cfg.user_meta);
                    } else {
                        write_mtx(out_prefix + "/splice_psi.mtx",
                                  psi.nrows, psi.ncols,
                                  psi.indptr, psi.indices, psi.data);
                        write_names(out_prefix + "/splice_psi_features.tsv", engine.sj_names());
                        write_names(out_prefix + "/splice_psi_barcodes.tsv", barcodes);
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

    if (pileup_cfg.count_mt && mt_csc.data.size() > 0 && !export_cfg.pipeline_mode)
        write_threads.emplace_back([&]() { write_matrix("mt_allele_counts", mt_csc, mt_feat_names); });

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
                pz::write_1pz(out_prefix + "/mt_heteroplasmy.1pz",
                    mt_het.n_variants, mt_het.n_cells,
                    mt_het.indptr, mt_het.indices, scaled, het_names, barcodes,
                    3, 1024, 4, export_cfg.user_meta);
            } else {
                write_mtx(out_prefix + "/mt_heteroplasmy.mtx",
                          mt_het.n_variants, mt_het.n_cells,
                          mt_het.indptr, mt_het.indices, mt_het.data);
                write_names(out_prefix + "/mt_heteroplasmy_features.tsv", het_names);
                write_names(out_prefix + "/mt_heteroplasmy_barcodes.tsv", barcodes);
            }
        });
    }

    for (auto& t : write_threads) t.join();
    result.n_write_threads = static_cast<int>(write_threads.size());

    // G6 stats declared here so they are visible to the summary block below
    mt::MtDonorOutputStats g6_stats;

    // Wait for demux; then aggregate per-donor depths and write VCF + coverage maps
    if (demux_thread.joinable()) {
        demux_thread.join();
        write_donor_assignments(out_prefix + "/donor/donor_assignments.tsv",
                                demux_result.assignments, barcodes);

        if (demux_result.n_donors_k > 0 && !demux_result.covered_to_original.empty()) {
            auto depths = aggregate_donor_depths(
                demux_result,
                snp_ad_csc.nrows, snp_ad_csc.ncols,
                snp_ad_csc.indptr.data(), snp_ad_csc.indices.data(), snp_ad_csc.data.data(),
                snp_dp_csc.indptr.data(), snp_dp_csc.indices.data(), snp_dp_csc.data.data());

            write_donor_vcfs(out_prefix + "/donor", demux_result, depths, engine.snp_names());
            write_donor_coverages(out_prefix + "/donor", demux_result, depths, engine.snp_names());
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
                pileup_cfg.exon_gtf_path);
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

    // ── Summary JSON ── (always written; VAL2-compatible structured output)
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
        write_1pz_stub("exon_counts.1pz");

        // TSV stubs (header-only)
        write_stub_if_missing("cell_qc_metrics.tsv",
            "barcode\tn_umi\tn_genes\tpct_mt\tpct_ribo\tintronic_pct"
            "\tn_reads\tduplication_rate\tsaturation\tcell_cycle_phase\tdoublet_score");
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
        write_stub_if_missing("ase_counts.tsv",
            "barcode\tsnp_id\tref_count\talt_count\tallelic_ratio");
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
       << "  \"pileup_time_s\": " << stats.wall_time_s << ",\n"
       << "  \"wall_time_s\": " << wall_time_s << "\n"
       << "}\n";
}

} // namespace singlet
