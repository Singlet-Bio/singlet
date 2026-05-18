// SPDX-License-Identifier: MIT
// singlet/gpu/r/src/_bind_pseudobulk.hpp
//
// Rcpp bindings for cycle 17 donor pseudobulk DE kernels:
//   pseudobulk_aggregate_cpp  — wraps singlet::gpu::de::pseudobulk_aggregate
//   donor_pseudobulk_de_cpp   — wraps singlet::gpu::de::donor_pseudobulk_de
//
// Input format:
//   pseudobulk_aggregate_cpp:
//     dgc             — dgCMatrix [genes × cells] (counts layer)
//     cluster_ids     — IntegerVector [n_cells] (0-based cluster labels)
//     donor_ids       — IntegerVector [n_cells] (0-based donor labels)
//     min_cells       — int, minimum cells per (cluster, donor) pseudo-replicate
//
//   donor_pseudobulk_de_cpp:
//     pb_matrix       — NumericMatrix [genes × n_pseudobulks] (aggregated counts)
//     cluster_ids     — IntegerVector [n_pseudobulks] (0-based cluster of each pb)
//     donor_ids       — IntegerVector [n_pseudobulks] (0-based donor of each pb)
//     n_clusters      — int, total number of clusters
//     apeglm_shrinkage — bool, apply apeGLM-style log2FC shrinkage
//     seed            — int
//
// Host-copy overhead:
//   pseudobulk_aggregate_cpp: O(nnz) host→device (matrix) +
//                              2 × O(n_cells) host→device (labels) +
//                              O(n_genes × n_pseudobulks) device→host (aggregated).
//   donor_pseudobulk_de_cpp:  O(n_genes × n_pseudobulks) host→device (pb matrix) +
//                              O(n_clusters × n_genes × 5 result fields) device→host.
//
// Design note: donor_assignments come from singlet's donor_assignments.tsv
// (per-cell donor_id column, which maps to 0-based integer after encoding).
// The R wrapper (pseudobulk.R) handles this mapping from colData columns.

#pragma once

#include <Rcpp.h>

#ifndef FACTORNET_HAS_GPU
#  define FACTORNET_HAS_GPU 1
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>
#include <singlet/gpu/de/donor_pseudobulk.h>
#include <singlet/gpu/de/pseudobulk_types.h>

#include "_r_to_eigen.hpp"
#include "_bind_preprocess.hpp"   // upload_host_csc_to_device
#include "_bind_graph.hpp"        // r_matrix_to_device_dense

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace singlet_gpu_r {

// ---------------------------------------------------------------------------
// encode_labels_0based
//
// Convert an R vector (factor / character / integer) to a 0-based integer
// std::vector.  Also fills `levels_out` with the label strings for naming.
// ---------------------------------------------------------------------------
inline std::vector<int>
encode_labels_0based(const Rcpp::RObject& r_vec, int n, std::vector<std::string>& levels_out)
{
    std::vector<int> out(n);
    if (Rf_isFactor(r_vec)) {
        Rcpp::IntegerVector fv(r_vec);
        Rcpp::CharacterVector lvls = fv.attr("levels");
        levels_out.resize(lvls.size());
        for (int i = 0; i < (int)lvls.size(); ++i) levels_out[i] = Rcpp::as<std::string>(lvls[i]);
        for (int i = 0; i < n; ++i) out[i] = fv[i] - 1;
    } else if (TYPEOF(r_vec) == STRSXP) {
        Rcpp::CharacterVector cv(r_vec);
        std::unordered_map<std::string, int> lmap;
        for (int i = 0; i < n; ++i) {
            std::string s = Rcpp::as<std::string>(cv[i]);
            auto it = lmap.find(s);
            if (it == lmap.end()) {
                int idx = static_cast<int>(levels_out.size());
                lmap[s] = idx;
                levels_out.push_back(s);
                out[i] = idx;
            } else {
                out[i] = it->second;
            }
        }
    } else {
        Rcpp::IntegerVector iv(r_vec);
        int min_v = *std::min_element(iv.begin(), iv.end());
        int max_v = *std::max_element(iv.begin(), iv.end());
        for (int i = 0; i <= max_v - min_v; ++i)
            levels_out.push_back(std::to_string(i + min_v));
        for (int i = 0; i < n; ++i) out[i] = iv[i] - min_v;
    }
    return out;
}

// ---------------------------------------------------------------------------
// pseudobulk_aggregate_cpp
//
// Aggregates (sums) per-cell counts into (cluster, donor) pseudobulk profiles.
// Mirrors muscat::aggregateData(fun="sum").
//
// Parameters:
//   dgc          — dgCMatrix [genes × cells]
//   cluster_ids  — IntegerVector [n_cells] (0-based)
//   donor_ids    — IntegerVector [n_cells] (0-based)
//   min_cells    — int, minimum cells per pseudobulk to include
//
// Returns:
//   Rcpp::List{
//     "counts"          — NumericMatrix [genes × n_pseudobulks]
//     "pseudobulk_meta" — List{
//                            "cluster_id" — IntegerVector [n_pseudobulks]
//                            "donor_id"   — IntegerVector [n_pseudobulks]
//                            "n_cells"    — IntegerVector [n_pseudobulks]
//                         }
//     "n_pseudobulks"   — int
//   }
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List pseudobulk_aggregate_cpp(
        Rcpp::S4            dgc,
        Rcpp::IntegerVector cluster_ids,
        Rcpp::IntegerVector donor_ids,
        int                 min_cells)
{
    HostCsc h;
    try { h = r_sparse_to_host_csc(dgc); }
    catch (const std::exception& e) { Rcpp::stop("%s", e.what()); }

    const int n_genes = h.n_rows;
    const int n_cells = h.n_cols;

    if (cluster_ids.size() != n_cells)
        Rcpp::stop("pseudobulk_aggregate_cpp: cluster_ids length (%d) != n_cells (%d).",
                   (int)cluster_ids.size(), n_cells);
    if (donor_ids.size() != n_cells)
        Rcpp::stop("pseudobulk_aggregate_cpp: donor_ids length (%d) != n_cells (%d).",
                   (int)donor_ids.size(), n_cells);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceCSC d;
    try { d = upload_host_csc_to_device(h, stream); }
    catch (const std::exception& e) { Rcpp::stop("pseudobulk_aggregate_cpp upload: %s", e.what()); }

    // Upload labels
    std::vector<int> h_cluster(n_cells), h_donor(n_cells);
    int max_cluster = -1, max_donor = -1;
    for (int i = 0; i < n_cells; ++i) {
        h_cluster[i] = cluster_ids[i];
        h_donor  [i] = donor_ids  [i];
        if (h_cluster[i] > max_cluster) max_cluster = h_cluster[i];
        if (h_donor  [i] > max_donor  ) max_donor   = h_donor  [i];
    }
    const int n_clusters = max_cluster + 1;
    const int n_donors   = max_donor   + 1;

    singlet::gpu::core::DeviceMemory<int> d_cluster(n_cells), d_donor(n_cells);
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_cluster.get(), h_cluster.data(),
                                n_cells * sizeof(int), cudaMemcpyHostToDevice, stream), "cluster upload");
            chk(cudaMemcpyAsync(d_donor.data(),   h_donor.data(),
                                n_cells * sizeof(int), cudaMemcpyHostToDevice, stream), "donor upload");
            chk(cudaStreamSynchronize(stream), "upload sync");
        } catch (const std::exception& e) {
            Rcpp::stop("pseudobulk_aggregate_cpp label upload: %s", e.what());
        }
    }

    singlet::gpu::de::PseudobulkAggregateConfig cfg;
    cfg.min_cells  = min_cells;
    cfg.n_clusters = n_clusters;
    cfg.n_donors   = n_donors;

    singlet::gpu::de::PseudobulkAggregateResult res;
    try {
        res = singlet::gpu::de::pseudobulk_aggregate(d, d_cluster, d_donor,
                                                     n_cells, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("pseudobulk_aggregate_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    const int n_pb = res.n_pseudobulks;
    const std::size_t pb_sz = static_cast<std::size_t>(n_genes) * n_pb;

    // Copy aggregated counts device → host
    std::vector<float> h_counts(pb_sz > 0 ? pb_sz : 1);
    if (pb_sz > 0) {
        cudaError_t err = cudaMemcpy(h_counts.data(), res.counts.data(),
                                     pb_sz * sizeof(float), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess)
            Rcpp::stop("pseudobulk_aggregate_cpp counts copyback: %s", cudaGetErrorString(err));
    }

    // Metadata arrays (already host-side in result)
    Rcpp::IntegerVector r_pb_cluster(n_pb), r_pb_donor(n_pb), r_n_cells(n_pb);
    for (int i = 0; i < n_pb; ++i) {
        r_pb_cluster[i] = res.pseudobulk_cluster[i];
        r_pb_donor  [i] = res.pseudobulk_donor  [i];
        r_n_cells   [i] = res.pseudobulk_ncells  [i];
    }

    Rcpp::NumericMatrix r_counts(n_genes, n_pb);
    for (int g = 0; g < n_genes; ++g)
        for (int p = 0; p < n_pb; ++p)
            r_counts(g, p) = static_cast<double>(
                h_counts[static_cast<std::size_t>(g) * n_pb + p]);

    return Rcpp::List::create(
        Rcpp::Named("counts") = r_counts,
        Rcpp::Named("pseudobulk_meta") = Rcpp::List::create(
            Rcpp::Named("cluster_id") = r_pb_cluster,
            Rcpp::Named("donor_id")   = r_pb_donor,
            Rcpp::Named("n_cells")    = r_n_cells
        ),
        Rcpp::Named("n_pseudobulks") = n_pb
    );
}

// ---------------------------------------------------------------------------
// donor_pseudobulk_de_cpp
//
// Runs negative-binomial GLM DE on donor pseudobulk profiles per cluster.
// Mirrors muscat::pbDS(method="DESeq2") output shape:
//   list of data.frames, one per cluster, keyed by cluster name.
//
// Parameters:
//   pb_matrix        — NumericMatrix [genes × n_pseudobulks]
//   cluster_ids      — IntegerVector [n_pseudobulks] (0-based cluster of each pb)
//   donor_ids        — IntegerVector [n_pseudobulks] (0-based donor of each pb)
//   n_clusters       — int
//   apeglm_shrinkage — bool, apply apeGLM-style log2FC shrinkage
//   seed             — int
//
// Returns:
//   Named Rcpp::List (one element per cluster), each element is a List with:
//     "gene"        — CharacterVector (0-based gene index as string if no names)
//     "log2FC"      — NumericVector
//     "pvalue"      — NumericVector
//     "padj"        — NumericVector (BH)
//     "dispersion"  — NumericVector (NB dispersion estimate)
// ---------------------------------------------------------------------------
// [[Rcpp::export]]
inline Rcpp::List donor_pseudobulk_de_cpp(
        Rcpp::NumericMatrix pb_matrix,
        Rcpp::IntegerVector cluster_ids,
        Rcpp::IntegerVector donor_ids,
        int                 n_clusters,
        bool                apeglm_shrinkage,
        int                 seed)
{
    const int n_genes = pb_matrix.nrow();
    const int n_pb    = pb_matrix.ncol();

    if (n_genes <= 0 || n_pb <= 0)
        Rcpp::stop("donor_pseudobulk_de_cpp: empty pb_matrix.");
    if (cluster_ids.size() != n_pb)
        Rcpp::stop("donor_pseudobulk_de_cpp: cluster_ids length (%d) != n_pseudobulks (%d).",
                   (int)cluster_ids.size(), n_pb);
    if (donor_ids.size() != n_pb)
        Rcpp::stop("donor_pseudobulk_de_cpp: donor_ids length (%d) != n_pseudobulks (%d).",
                   (int)donor_ids.size(), n_pb);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    struct StreamGuard { cudaStream_t s; ~StreamGuard() { if (s) cudaStreamDestroy(s); } } guard{stream};

    singlet::gpu::core::DeviceDense d_pb;
    try { d_pb = r_matrix_to_device_dense(pb_matrix, stream); }
    catch (const std::exception& e) { Rcpp::stop("donor_pseudobulk_de_cpp upload: %s", e.what()); }

    // Upload cluster and donor IDs
    std::vector<int> h_cluster(n_pb), h_donor(n_pb);
    for (int i = 0; i < n_pb; ++i) {
        h_cluster[i] = cluster_ids[i];
        h_donor  [i] = donor_ids  [i];
    }
    int max_donor = *std::max_element(h_donor.begin(), h_donor.end());
    const int n_donors = max_donor + 1;

    singlet::gpu::core::DeviceMemory<int> d_cluster(n_pb), d_donor(n_pb);
    {
        auto chk = [](cudaError_t e, const char* t) {
            if (e != cudaSuccess)
                throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
        };
        try {
            chk(cudaMemcpyAsync(d_cluster.get(), h_cluster.data(),
                                n_pb * sizeof(int), cudaMemcpyHostToDevice, stream), "cluster");
            chk(cudaMemcpyAsync(d_donor.data(),   h_donor.data(),
                                n_pb * sizeof(int), cudaMemcpyHostToDevice, stream), "donor");
            chk(cudaStreamSynchronize(stream), "sync");
        } catch (const std::exception& e) {
            Rcpp::stop("donor_pseudobulk_de_cpp label upload: %s", e.what());
        }
    }

    singlet::gpu::de::DonorPseudobulkDeConfig cfg;
    cfg.n_clusters       = n_clusters;
    cfg.n_donors         = n_donors;
    cfg.apeglm_shrinkage = apeglm_shrinkage;
    cfg.seed             = static_cast<uint64_t>(seed);

    singlet::gpu::de::DonorPseudobulkDeResult res;
    try {
        res = singlet::gpu::de::donor_pseudobulk_de(d_pb, d_cluster, d_donor, cfg, stream);
    } catch (const std::exception& e) {
        Rcpp::stop("donor_pseudobulk_de_cpp kernel: %s", e.what());
    }
    cudaStreamSynchronize(stream);

    // Marshal per-cluster results into a named list of lists
    Rcpp::List out(n_clusters);
    Rcpp::CharacterVector out_names(n_clusters);

    auto chk = [](cudaError_t e, const char* t) {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(t) + ": " + cudaGetErrorString(e));
    };

    for (int cl = 0; cl < n_clusters; ++cl) {
        const auto& cr = res.per_cluster[cl];
        const int ng = static_cast<int>(cr.n_genes);

        std::vector<float> h_lfc (ng), h_pval(ng), h_padj(ng), h_disp(ng);
        try {
            chk(cudaMemcpy(h_lfc .data(), cr.log2FC.data(),     ng * sizeof(float), cudaMemcpyDeviceToHost), "log2FC");
            chk(cudaMemcpy(h_pval.data(), cr.pvalue.data(),     ng * sizeof(float), cudaMemcpyDeviceToHost), "pvalue");
            chk(cudaMemcpy(h_padj.data(), cr.padj.data(),       ng * sizeof(float), cudaMemcpyDeviceToHost), "padj");
            chk(cudaMemcpy(h_disp.data(), cr.dispersion.data(), ng * sizeof(float), cudaMemcpyDeviceToHost), "dispersion");
        } catch (const std::exception& e) {
            Rcpp::stop("donor_pseudobulk_de_cpp copyback cluster %d: %s", cl, e.what());
        }

        Rcpp::CharacterVector r_gene(ng);
        Rcpp::NumericVector   r_lfc (ng), r_pval(ng), r_padj(ng), r_disp(ng);
        for (int g = 0; g < ng; ++g) {
            r_gene[g] = std::to_string(g);   // gene index string; R caller replaces with gene names
            r_lfc [g] = static_cast<double>(h_lfc [g]);
            r_pval[g] = static_cast<double>(h_pval[g]);
            r_padj[g] = static_cast<double>(h_padj[g]);
            r_disp[g] = static_cast<double>(h_disp[g]);
        }

        out[cl] = Rcpp::List::create(
            Rcpp::Named("gene")        = r_gene,
            Rcpp::Named("log2FC")      = r_lfc,
            Rcpp::Named("pvalue")      = r_pval,
            Rcpp::Named("padj")        = r_padj,
            Rcpp::Named("dispersion")  = r_disp
        );
        out_names[cl] = std::to_string(cl);
    }
    out.names() = out_names;
    return out;
}

}  // namespace singlet_gpu_r
