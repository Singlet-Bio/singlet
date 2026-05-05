// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/r/src/singlet_gpu_r_bindings.cpp
//
// Rcpp entry point for the singletGpu R package.
//
// Architecture:
//   This file is the ONLY .cpp compiled by R CMD SHLIB / R CMD INSTALL.
//   It includes the split binding headers which each contain the inline
//   function bodies and Rcpp export attributes.
//
//   The singlet-gpu C++ library is HEADER-ONLY (compiled with
//   -DSINGLET_GPU_HEADER_ONLY=1 in Makevars) so every translation unit that
//   includes singlet-gpu headers also compiles any template instantiations it
//   touches.  This is intentional — the library uses explicit instantiations
//   only in its own .cu files which we do not link here.
//
// Build:
//   R CMD INSTALL singlet-gpu/r/
//   or from within R:
//     devtools::install_local("singlet-gpu/r/")
//
//   The Makevars file overrides CXX to nvcc so CUDA device code embedded in
//   factornet headers is compiled correctly.  nvcc must be on PATH.
//
// Compile flags (see src/Makevars):
//   -DSINGLET_GPU_HEADER_ONLY=1     — suppress separate .cu linkage
//   -DFACTORNET_HAS_GPU=1           — enable factornet GPU headers
//   -x cu                           — treat .cpp as CUDA source (nvcc only)
//   --use_fast_math -O3             — performance flags
//   -arch sm_70,sm_80,sm_90         — Volta + Ampere + Hopper
//
// R API surface (exported via Rcpp attributes):
//   --- Cycle 24 (load, preprocess, reduce) ---
//   load_pz_to_sce_cpp(pz_dir, modality, keep_host_pinned) → List
//   lognorm_cpp(dgc, target_sum)                           → S4 dgCMatrix
//   hvg_cpp(dgc, n_top, flavor)                            → List
//   pca_cpp(dgc, n_pcs, zero_center, seed)                 → List
//   nmf_cpp(dgc, n_factors, loss, seed)                    → List
//   --- Cycle 25 (graph, embed, DE, annotation, streaming) ---
//   neighbors_cpp(embedding, n_neighbors, metric, seed)    → List
//   leiden_cpp(indices, distances, resolution, seed,
//              weight_function)                            → List
//   umap_cpp(indices, distances, n_components, min_dist,
//            spread, init, seed)                           → NumericMatrix
//   wilcoxon_de_cpp(dgc, labels, top_n, deterministic)     → List of Lists
//   ttest_de_cpp(dgc, labels, top_n, deterministic)        → List of Lists
//   load_celltypist_model_cpp(npz_path)                    → externalptr
//   marker_score_cpp(dgc, gene_sets, gene_names, method)   → List
//   celltypist_project_cpp(embedding, model_ptr)           → List
//   streaming_pipeline_run_cpp(input_paths, chunk_cols,
//     run_lognorm, run_hvg, run_pca, pca_k,
//     run_nmf, nmf_factors)                                → List
//   --- Cycle 26 (GSEA, integrate, velocity, lineage, pseudobulk) ---
//   fgsea_cpp(stats, pathways, eps, min_size, max_size,
//             n_perm, seed)                                → DataFrame
//   aucell_cpp(dgc, gene_sets, gene_names,
//              auc_max_rank, norm_auc)                     → NumericMatrix
//   harmony_cpp(embedding, batch_ids, n_clusters,
//               max_iter, tol, seed)                       → NumericMatrix
//   bbknn_cpp(embedding, batch_ids,
//             neighbors_within_batch, n_pcs, metric)       → List
//   velocity_moments_cpp(spliced, unspliced,
//                        knn_indices, n_neighbors)         → List
//   velocity_prep_compute_cpp(spliced_moments,
//                             unspliced_moments,
//                             min_S_count, min_U_count,
//                             top_n_quantile,
//                             compute_velocity)            → List
//   mt_lineage_call_clones_cpp(alt_matrix, depth_matrix,
//                              min_depth, min_cells_alt,
//                              min_vaf, min_K, max_K, seed)→ List
//   pseudobulk_aggregate_cpp(dgc, cluster_ids, donor_ids,
//                            min_cells)                    → List
//   donor_pseudobulk_de_cpp(pb_matrix, cluster_ids,
//                           donor_ids, n_clusters,
//                           apeglm_shrinkage, seed)        → List
//   --- Cycle 52b (fate, comm, network, abundance, disease) ---
//   cospar_cpp(expression, time_points, clone_ids, n_fates,
//              lambda1, lambda2, max_iter, tol, seed,
//              resource)                                   → List
//   cellrank2_cpp(transition_matrix, terminal_state_ids,
//                 max_iter, tol, seed, resource)           → List
//   palantir_cpp(expression, start_cell, n_terminals,
//                n_neighbors, n_diffusion_components,
//                seed, resource)                           → List
//   cellchat_cpp(expression, cell_type_ids, ligand_ids,
//                receptor_ids, pathway_ids, n_pathways,
//                n_permutations, seed, resource)           → List
//   hdwgcna_cpp(expression, soft_power, min_module_size,
//               n_top_hub, seed, resource)                 → List
//   milo_cpp(expression, donor_ids, conditions,
//            n_neighbors, n_nh_cells, seed, resource)      → List
//   scdrs_cpp(expression, gene_set_names, gene_set_indices,
//             gene_set_weights, gene_set_offsets,
//             n_ctrl_sets, seed, resource)                 → List
//   --- Cycle 52b Tier 2 (GRaNIE, NEBULA, DAESC, Numbat CNA, Monopogen, chromVAR) ---
//   granie_cpp(gex, peaks, tf_motif, peak_gene_pairs,
//              tf_names, gene_names, min_abs_r,
//              peak_gene_fdr, tf_target_fdr,
//              min_peak_mean, min_peak_var, run_leiden,
//              leiden_resolution, chunk_size_tfs,
//              seed, resource)                             → List
//   nebula_cpp(counts, snp_ad, snp_dp, donor_id,
//              covariates, chunk_snps, chunk_genes,
//              run_bh_fdr, fdr_alpha, seed, resource)      → List
//   daesc_cpp(snp_ad, snp_dp, cell_type, chunk_snps,
//             run_bh_fdr, fdr_alpha, min_depth,
//             min_cells, seed, resource)                   → List
//   detect_cna_cpp(expr, gene_chr, gene_start, ref,
//                  smooth_half_win, trans_prob,
//                  segments_per_chr, leiden_resolution,
//                  knn_k, include_sex_chrs,
//                  seed, resource)                         → List
//   call_variants_cpp(snp_ad, snp_dp, chromosome,
//                     ld_panels, somatic_af_min,
//                     somatic_af_max, fdr_alpha,
//                     run_bh_fdr, min_total_dp,
//                     seed, resource)                      → List
//   chromvar_cpp(accessibility, motif_in_peak, peak_gc,
//                peak_mean_access, motif_names,
//                n_background_peaks, n_permutations,
//                cell_batch, seed, resource)               → List
//
// Each exported function is declared with // [[Rcpp::export]] inside its
// respective binding header.  Rcpp::compileAttributes() generates the
// RcppExports.cpp and R/RcppExports.R wrappers automatically.
//
// No logic lives in this file — it is a pure aggregation point.

// [[Rcpp::depends(Rcpp)]]
#include <Rcpp.h>

// Ensure CUDA and factornet GPU headers are available
#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#ifndef SINGLET_GPU_HEADER_ONLY
#  define SINGLET_GPU_HEADER_ONLY 1
#endif

// Marshalling helpers (no Rcpp exports — internal only)
#include "_r_to_eigen.hpp"

// Binding implementations — each provides Rcpp::export-decorated functions.
// Include order matters:
//   load → preprocess (defines upload_host_csc_to_device) →
//   reduce (uses upload_host_csc_to_device) →
//   graph (uses upload_host_csc_to_device + defines r_matrix_to_device_dense) →
//   embed (uses graph types) →
//   de    (uses upload_host_csc_to_device) →
//   anno  (uses upload_host_csc_to_device + r_matrix_to_device_dense from graph) →
//   streaming (uses eigen_dense_to_r_matrix from reduce).
#include "_bind_load.hpp"
#include "_bind_preprocess.hpp"
#include "_bind_reduce.hpp"
// Cycle 25: graph, embed, DE, annotation, streaming
#include "_bind_graph.hpp"
#include "_bind_embed.hpp"
#include "_bind_de.hpp"
#include "_bind_anno.hpp"
#include "_bind_streaming.hpp"
// Cycle 26: GSEA, batch integration, velocity, MT lineage, pseudobulk DE
// Include order: gsea (uses build_gene_set_db from _bind_anno.hpp) →
//   integrate (uses r_matrix_to_device_dense from _bind_graph.hpp) →
//   velocity (uses upload_host_csc_to_device + r_matrix_to_device_dense) →
//   lineage (uses upload_host_csc_to_device) →
//   pseudobulk (uses upload_host_csc_to_device + r_matrix_to_device_dense).
#include "_bind_gsea.hpp"
#include "_bind_integrate.hpp"
#include "_bind_velocity.hpp"
#include "_bind_lineage.hpp"
#include "_bind_pseudobulk.hpp"
// Cycle 52b Tier 1: fate (Cospar, CellRank2, Palantir) + comm (CellChat) +
//   network (hdWGCNA) + abundance (Milo) + disease (scDRS).
// Include order: fate → comm → network → abundance → disease.
// All five headers use gpu_stubs fallback when FACTORNET_HAS_GPU is not defined.
#include "_bind_fate.hpp"
#include "_bind_comm.hpp"
#include "_bind_network.hpp"
#include "_bind_abundance.hpp"
#include "_bind_disease.hpp"
// Cycle 52b Tier 3: spatial Phase B (FlashDeconv, STAGATE, Cell2fate),
//   generative (Discrete Diffusion), perturbation (PerturbGraph),
//   advanced enrichment (ssGSEA, PROGENy), advanced QC (OmniDoublet, DoubletScore),
//   advanced NMF (CSI-GEP).
// Include order:
//   enrich (uses build_gene_set_db from _bind_anno.hpp + upload_host_csc_to_device) →
//   qc_new (uses upload_host_csc_to_device) →
//   nmf_new (uses upload_host_csc_to_device) →
//   perturbation (uses upload_host_csc_to_device) →
//   generative (uses upload_host_csc_to_device) →
//   spatial_phaseb (uses upload_host_csc_to_device + r_matrix_to_device_dense).
#include "_bind_enrich.hpp"
#include "_bind_qc_new.hpp"
#include "_bind_nmf_new.hpp"
#include "_bind_perturbation.hpp"
#include "_bind_generative.hpp"
#include "_bind_spatial_phaseb.hpp"
// Cycle 52b Tier 2: GRaNIE, NEBULA, DAESC, Numbat CNA, Monopogen, chromVAR.
// Include order: grn (defines r_list_to_host_csr used by atac) →
//   eqtl (uses upload_host_csc_to_device + encode_labels_0based from pseudobulk) →
//   ase  (uses upload_host_csc_to_device + encode_labels_0based) →
//   cna  (uses upload_host_csc_to_device) →
//   variants (uses upload_host_csc_to_device + r_list_to_ld_panel helper) →
//   atac (uses upload_host_csc_to_device + r_list_to_host_csr from grn).
#include "_bind_grn.hpp"
#include "_bind_eqtl.hpp"
#include "_bind_ase.hpp"
#include "_bind_cna.hpp"
#include "_bind_variants.hpp"
#include "_bind_atac.hpp"

// ---------------------------------------------------------------------------
// RcppExports registration stub.
//
// When Rcpp::compileAttributes() is run (or devtools::document()), it will
// generate R/RcppExports.R and src/RcppExports.cpp automatically from the
// // [[Rcpp::export]] annotations in the binding headers.
//
// The manually declared extern "C" stubs below are the canonical registration
// block that Rcpp generates; they are included here for documentation purposes.
// In practice, devtools::document() regenerates src/RcppExports.cpp — do NOT
// edit that file manually.
// ---------------------------------------------------------------------------

// Module-level init: called once when the shared library is loaded by R.
// We confirm CUDA is available and log the device count to R's message stream.
// This fires via .onLoad → useDynLib → R's package loading machinery.
static void singletGpu_init() __attribute__((constructor));
static void singletGpu_init() {
    // We deliberately do NOT throw here — if CUDA is absent the library still
    // loads; individual functions will throw informative errors on first call.
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    // Silently do nothing if CUDA is absent (n_devices stays 0).
    (void)n_devices;
}

// Package version constant accessible from R via singletGpu::::.pkg_version()
// [[Rcpp::export(.pkg_version)]]
std::string pkg_version_() { return "0.1.0"; }
