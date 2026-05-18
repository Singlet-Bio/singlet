// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_results.hpp
//
// Py* result structs + pybind11 class<> registrations for cycles 7-17 kernels.
//
// Each Py* struct wraps the corresponding C++ result type, holding DeviceMemory<T>
// buffers (device-resident, RAII).  Registration follows the cycle 20 pattern:
//   - py::class_<Py*, std::shared_ptr<Py*>>
//   - scalar properties via def_property_readonly returning host scalars
//   - array properties via make_view_object<T> returning __cuda_array_interface__ dicts
//   - __repr__ returning a concise string
//
// No host data copies anywhere.  All array buffers are exposed zero-copy.
// FgseaResult and WilcoxonResult/TtestResult hold host-side std::vector data
// (pathway/marker structs) — those are returned as Python lists/dicts, not device views.
//
// Declared inside namespace singlet::gpu::python so bind_kernels.hpp can use the
// type names directly.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/core/types.h>
#include <singlet/gpu/streaming/streamed_pipeline.h>
#include <singlet/gpu/graph/knn.h>
#include <singlet/gpu/graph/leiden.h>
#include <singlet/gpu/embed/umap.h>
#include <singlet/gpu/de/types.h>
#include <singlet/gpu/anno/types.h>
#include <singlet/gpu/gsea/types.h>
#include <singlet/gpu/integrate/types.h>
#include <singlet/gpu/preprocess/velocity_prep.h>
#include <singlet/gpu/anno/mt_lineage.h>
#include <singlet/gpu/de/donor_pseudobulk.h>

#include "_cupy_interop.hpp"

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ===========================================================================
// Py* result wrappers
//
// Each struct holds the C++ result value (by move) plus a cudaStream_t for the
// __cuda_array_interface__ stream field.
// ===========================================================================

// ---------------------------------------------------------------------------
// PyStreamingPipelineResult — wraps streaming::PipelineResult (cycle 7).
// size_factors, qc_mask, hvg_indices, hvg_scores, gene_means, gene_vars live
// on host (std::vector) because the streaming pipeline aggregates them there.
// pca / nmf results are exposed via the existing PySvdResult / PyNmfResult
// indirectly; for this Python result we expose the raw host vectors as Python
// lists (no device views needed).
// ---------------------------------------------------------------------------
struct PyStreamingPipelineResult {
    int64_t n_cells         = 0;
    int64_t n_genes_total   = 0;
    int64_t n_nnz_total     = 0;
    float   lognorm_target_used = 0.0f;
    bool    pca_ran         = false;
    bool    nmf_ran         = false;
    double  wall_lognorm_s  = 0;
    double  wall_hvg_s      = 0;
    double  wall_pca_s      = 0;
    double  wall_nmf_s      = 0;
    int     n_chunks_processed = 0;

    // Host vectors from the streaming result.
    std::vector<float>   size_factors;
    std::vector<uint8_t> qc_mask;
    std::vector<int>     hvg_indices;
    std::vector<float>   hvg_scores;
    std::vector<float>   gene_means;
    std::vector<float>   gene_vars;

    // PCA / NMF results — kept as upstream C++ types for upload on demand.
    singlet::gpu::reduce::svd::SvdResult pca;
    singlet::gpu::reduce::nmf::NmfResult nmf;
};

// ---------------------------------------------------------------------------
// PyKnnResult — wraps graph::KnnResult (cycle 8).
// row_offsets, neighbors, distances are DeviceMemory<T> — exposed zero-copy.
// ---------------------------------------------------------------------------
struct PyKnnResult {
    singlet::gpu::core::DeviceMemory<int>   row_offsets;
    singlet::gpu::core::DeviceMemory<int>   neighbors;
    singlet::gpu::core::DeviceMemory<float> distances;
    int n  = 0;
    int k  = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyLeidenResult — wraps graph::LeidenResult (cycle 9).
// ---------------------------------------------------------------------------
struct PyLeidenResult {
    singlet::gpu::core::DeviceMemory<int> labels;
    int   n_clusters  = 0;
    float modularity  = 0.0f;
    int   iterations  = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyUmapResult — wraps embed::UmapResult (cycle 10).
// ---------------------------------------------------------------------------
struct PyUmapResult {
    singlet::gpu::core::DeviceMemory<float> embedding;
    int n           = 0;
    int n_components = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyClusterMarkers — wraps de::ClusterMarkers (cycle 11, used by Wilcoxon/Ttest).
// Holds per-cluster device buffers.
// ---------------------------------------------------------------------------
struct PyClusterMarkers {
    int cluster_id = 0;
    singlet::gpu::core::DeviceMemory<int>   gene_indices;
    singlet::gpu::core::DeviceMemory<float> z_scores;
    singlet::gpu::core::DeviceMemory<float> log2_fc;
    singlet::gpu::core::DeviceMemory<float> p_values;
    singlet::gpu::core::DeviceMemory<float> p_adj;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyWilcoxonResult / PyTtestResult — wrap de::WilcoxonResult / TtestResult.
// Hold a vector of PyClusterMarkers (one per cluster).
// ---------------------------------------------------------------------------
struct PyWilcoxonResult {
    std::vector<std::shared_ptr<PyClusterMarkers>> per_cluster;
};

struct PyTtestResult {
    std::vector<std::shared_ptr<PyClusterMarkers>> per_cluster;
};

// ---------------------------------------------------------------------------
// PyMarkerScoreResult — wraps anno::MarkerScoreResult (cycle 12).
// ---------------------------------------------------------------------------
struct PyMarkerScoreResult {
    singlet::gpu::core::DeviceMemory<float> scores;
    int n_sets  = 0;
    int n_cells = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyRefMapResult — wraps anno::RefMapResult (cycle 12).
// ---------------------------------------------------------------------------
struct PyRefMapResult {
    singlet::gpu::core::DeviceMemory<float> probabilities;
    singlet::gpu::core::DeviceMemory<int>   labels;
    int n_classes = 0;
    int n_cells   = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyCelltypistModel — wraps anno::CelltypistModel (cycle 12).
// weights and intercepts on device; class_names on host.
// ---------------------------------------------------------------------------
struct PyCelltypistModel {
    singlet::gpu::core::DeviceMemory<float> weights;
    singlet::gpu::core::DeviceMemory<float> intercepts;
    std::vector<std::string>               class_names;
    int n_classes = 0;
    int n_pcs     = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyFgseaResult — wraps gsea::FgseaResult (cycle 13).
// Pathway results are host-side structs — returned as Python list of dicts.
// ---------------------------------------------------------------------------
struct PyFgseaResult {
    std::vector<singlet::gpu::gsea::PathwayResult> pathways;
};

// ---------------------------------------------------------------------------
// PyAUCellResult — wraps gsea::AUCellResult (cycle 13).
// scores is DeviceMemory — exposed zero-copy.
// ---------------------------------------------------------------------------
struct PyAUCellResult {
    singlet::gpu::core::DeviceMemory<float> scores;
    int n_pathways = 0;
    int n_cells    = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyHarmonyResult — wraps integrate::HarmonyResult (cycle 14).
// ---------------------------------------------------------------------------
struct PyHarmonyResult {
    singlet::gpu::core::DeviceMemory<float> corrected;
    int   n_iters_used  = 0;
    float final_delta   = 0.0f;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyVelocityPrepResult — wraps preprocess::VelocityPrepResult (cycle 15).
// ---------------------------------------------------------------------------
struct PyVelocityPrepResult {
    singlet::gpu::core::DeviceMemory<float>   gamma;
    singlet::gpu::core::DeviceMemory<float>   gamma_se;
    singlet::gpu::core::DeviceMemory<float>   S_mean;
    singlet::gpu::core::DeviceMemory<float>   U_mean;
    singlet::gpu::core::DeviceMemory<uint8_t> filter_mask;
    singlet::gpu::core::DeviceMemory<float>   velocity;
    int n_genes_passing_filter = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyClonePrediction — wraps anno::ClonePrediction (cycle 16).
// ---------------------------------------------------------------------------
struct PyClonePrediction {
    singlet::gpu::core::DeviceMemory<int>   labels;
    int                                    n_clones = 0;
    singlet::gpu::core::DeviceMemory<int>   informative_sites;
    int                                    n_informative_sites = 0;
    singlet::gpu::core::DeviceMemory<float> clone_profiles;
    singlet::gpu::core::DeviceMemory<float> per_cell_heteroplasmy;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyDonorPseudobulkResult — wraps de::DonorPseudobulkResult (cycle 17).
// ---------------------------------------------------------------------------
struct PyDonorPseudobulkResult {
    singlet::gpu::core::DeviceMemory<float> log2_fc;
    singlet::gpu::core::DeviceMemory<float> p_values;
    singlet::gpu::core::DeviceMemory<float> p_adj;
    singlet::gpu::core::DeviceMemory<float> dispersion;
    singlet::gpu::core::DeviceMemory<float> pseudobulk;
    int n_genes_passing_filter = 0;
    cudaStream_t stream = nullptr;
};

// ===========================================================================
// Registration function — called from _singlet_gpu_core.cpp before
// bind_kernels() so that the result types are defined before their functions.
// ===========================================================================

inline void bind_results(py::module_& m)
{
    // -----------------------------------------------------------------------
    // ClusterMarkers — shared sub-type for Wilcoxon + Ttest.
    // -----------------------------------------------------------------------
    py::class_<PyClusterMarkers, std::shared_ptr<PyClusterMarkers>>(m, "ClusterMarkers",
        "Per-cluster top-N marker genes.  Device buffers exposed zero-copy.")
    .def_property_readonly("cluster_id",
        [](const PyClusterMarkers& r) { return r.cluster_id; })
    .def_property_readonly("gene_indices_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClusterMarkers&>();
            return make_view_object<int32_t>(
                r.gene_indices.get(), r.gene_indices.size(), r.stream, self_obj);
        }, "Gene row-indices int32 (top_n,) — __cuda_array_interface__ dict.")
    .def_property_readonly("z_scores_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClusterMarkers&>();
            return make_view_object<float>(
                r.z_scores.get(), r.z_scores.size(), r.stream, self_obj);
        }, "Test statistic float32 (top_n,) — __cuda_array_interface__ dict.")
    .def_property_readonly("log2_fc_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClusterMarkers&>();
            return make_view_object<float>(
                r.log2_fc.get(), r.log2_fc.size(), r.stream, self_obj);
        }, "log2 fold-change float32 (top_n,) — __cuda_array_interface__ dict.")
    .def_property_readonly("p_values_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClusterMarkers&>();
            return make_view_object<float>(
                r.p_values.get(), r.p_values.size(), r.stream, self_obj);
        }, "Two-sided p-values float32 (top_n,) — __cuda_array_interface__ dict.")
    .def_property_readonly("p_adj_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClusterMarkers&>();
            return make_view_object<float>(
                r.p_adj.get(), r.p_adj.size(), r.stream, self_obj);
        }, "BH-adjusted p-values float32 (top_n,) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyClusterMarkers& r) {
            return "<ClusterMarkers cluster_id=" + std::to_string(r.cluster_id)
                 + " top_n=" + std::to_string(r.gene_indices.size()) + ">";
        });

    // -----------------------------------------------------------------------
    // StreamingPipelineResult — cycle 7.
    // -----------------------------------------------------------------------
    py::class_<PyStreamingPipelineResult, std::shared_ptr<PyStreamingPipelineResult>>(
        m, "StreamingPipelineResult",
        "Result of streaming_pipeline_run.  Host vectors for scalars/HVG; "
        "see .pca / .nmf properties for device factor results.")
    .def_property_readonly("n_cells",
        [](const PyStreamingPipelineResult& r) { return r.n_cells; })
    .def_property_readonly("n_genes_total",
        [](const PyStreamingPipelineResult& r) { return r.n_genes_total; })
    .def_property_readonly("n_nnz_total",
        [](const PyStreamingPipelineResult& r) { return r.n_nnz_total; })
    .def_property_readonly("lognorm_target_used",
        [](const PyStreamingPipelineResult& r) { return r.lognorm_target_used; })
    .def_property_readonly("pca_ran",
        [](const PyStreamingPipelineResult& r) { return r.pca_ran; })
    .def_property_readonly("nmf_ran",
        [](const PyStreamingPipelineResult& r) { return r.nmf_ran; })
    .def_property_readonly("wall_lognorm_s",
        [](const PyStreamingPipelineResult& r) { return r.wall_lognorm_s; })
    .def_property_readonly("wall_hvg_s",
        [](const PyStreamingPipelineResult& r) { return r.wall_hvg_s; })
    .def_property_readonly("wall_pca_s",
        [](const PyStreamingPipelineResult& r) { return r.wall_pca_s; })
    .def_property_readonly("wall_nmf_s",
        [](const PyStreamingPipelineResult& r) { return r.wall_nmf_s; })
    .def_property_readonly("n_chunks_processed",
        [](const PyStreamingPipelineResult& r) { return r.n_chunks_processed; })
    .def_property_readonly("size_factors",
        [](const PyStreamingPipelineResult& r) { return r.size_factors; },
        "Per-cell size factors (host list[float]).")
    .def_property_readonly("qc_mask",
        [](const PyStreamingPipelineResult& r) { return r.qc_mask; },
        "Per-cell QC mask (host list[int], 1 = zero-count cell).")
    .def_property_readonly("hvg_indices",
        [](const PyStreamingPipelineResult& r) { return r.hvg_indices; },
        "Top-N HVG gene indices (host list[int]).")
    .def_property_readonly("hvg_scores",
        [](const PyStreamingPipelineResult& r) { return r.hvg_scores; },
        "HVG scores (host list[float]).")
    .def_property_readonly("gene_means",
        [](const PyStreamingPipelineResult& r) { return r.gene_means; },
        "Per-gene mean expression (host list[float]).")
    .def_property_readonly("gene_vars",
        [](const PyStreamingPipelineResult& r) { return r.gene_vars; },
        "Per-gene variance (host list[float]).")
    .def("__repr__",
        [](const PyStreamingPipelineResult& r) {
            return "<StreamingPipelineResult n_cells=" + std::to_string(r.n_cells)
                 + " n_genes=" + std::to_string(r.n_genes_total)
                 + " chunks=" + std::to_string(r.n_chunks_processed) + ">";
        });

    // -----------------------------------------------------------------------
    // KnnResult — cycle 8.
    // -----------------------------------------------------------------------
    py::class_<PyKnnResult, std::shared_ptr<PyKnnResult>>(m, "KnnResult",
        "k-nearest-neighbour graph (device CSR).  Zero-copy views.")
    .def_property_readonly("n",
        [](const PyKnnResult& r) { return r.n; }, "Number of cells.")
    .def_property_readonly("k",
        [](const PyKnnResult& r) { return r.k; }, "Neighbours per cell.")
    .def_property_readonly("row_offsets_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyKnnResult&>();
            return make_view_object<int32_t>(
                r.row_offsets.get(), r.row_offsets.size(), r.stream, self_obj);
        }, "CSR row offsets int32 (n+1,) — __cuda_array_interface__ dict.")
    .def_property_readonly("neighbors_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyKnnResult&>();
            return make_view_object<int32_t>(
                r.neighbors.get(), r.neighbors.size(), r.stream, self_obj);
        }, "Neighbour cell indices int32 (n*k,) — __cuda_array_interface__ dict.")
    .def_property_readonly("distances_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyKnnResult&>();
            return make_view_object<float>(
                r.distances.get(), r.distances.size(), r.stream, self_obj);
        }, "Distances float32 (n*k,) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyKnnResult& r) {
            return "<KnnResult n=" + std::to_string(r.n)
                 + " k=" + std::to_string(r.k) + ">";
        });

    // -----------------------------------------------------------------------
    // LeidenResult — cycle 9.
    // -----------------------------------------------------------------------
    py::class_<PyLeidenResult, std::shared_ptr<PyLeidenResult>>(m, "LeidenResult",
        "Leiden/Louvain community detection result.")
    .def_property_readonly("n_clusters",
        [](const PyLeidenResult& r) { return r.n_clusters; })
    .def_property_readonly("modularity",
        [](const PyLeidenResult& r) { return r.modularity; })
    .def_property_readonly("iterations",
        [](const PyLeidenResult& r) { return r.iterations; })
    .def_property_readonly("labels_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyLeidenResult&>();
            return make_view_object<int32_t>(
                r.labels.get(), r.labels.size(), r.stream, self_obj);
        }, "Per-cell community label int32 (n_cells,) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyLeidenResult& r) {
            return "<LeidenResult n_clusters=" + std::to_string(r.n_clusters)
                 + " modularity=" + std::to_string(r.modularity) + ">";
        });

    // -----------------------------------------------------------------------
    // UmapResult — cycle 10.
    // -----------------------------------------------------------------------
    py::class_<PyUmapResult, std::shared_ptr<PyUmapResult>>(m, "UmapResult",
        "UMAP embedding result (device float32).")
    .def_property_readonly("n",
        [](const PyUmapResult& r) { return r.n; }, "Number of cells.")
    .def_property_readonly("n_components",
        [](const PyUmapResult& r) { return r.n_components; }, "Embedding dimensionality.")
    .def_property_readonly("embedding_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyUmapResult&>();
            return make_view_object<float>(
                r.embedding.get(), r.embedding.size(), r.stream, self_obj);
        }, "Embedding float32 (n × n_components, row-major) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyUmapResult& r) {
            return "<UmapResult n=" + std::to_string(r.n)
                 + " n_components=" + std::to_string(r.n_components) + ">";
        });

    // -----------------------------------------------------------------------
    // WilcoxonResult — cycle 11.
    // -----------------------------------------------------------------------
    py::class_<PyWilcoxonResult, std::shared_ptr<PyWilcoxonResult>>(m, "WilcoxonResult",
        "Wilcoxon rank-sum DE result.  .per_cluster is a list of ClusterMarkers.")
    .def_property_readonly("per_cluster",
        [](const PyWilcoxonResult& r) { return r.per_cluster; },
        "list[ClusterMarkers] — one entry per cluster.")
    .def("__repr__",
        [](const PyWilcoxonResult& r) {
            return "<WilcoxonResult n_clusters=" + std::to_string(r.per_cluster.size()) + ">";
        });

    // -----------------------------------------------------------------------
    // TtestResult — cycle 11.
    // -----------------------------------------------------------------------
    py::class_<PyTtestResult, std::shared_ptr<PyTtestResult>>(m, "TtestResult",
        "Welch t-test DE result.  .per_cluster is a list of ClusterMarkers.")
    .def_property_readonly("per_cluster",
        [](const PyTtestResult& r) { return r.per_cluster; },
        "list[ClusterMarkers] — one entry per cluster.")
    .def("__repr__",
        [](const PyTtestResult& r) {
            return "<TtestResult n_clusters=" + std::to_string(r.per_cluster.size()) + ">";
        });

    // -----------------------------------------------------------------------
    // MarkerScoreResult — cycle 12.
    // -----------------------------------------------------------------------
    py::class_<PyMarkerScoreResult, std::shared_ptr<PyMarkerScoreResult>>(m, "MarkerScoreResult",
        "Marker activity scores (n_sets × n_cells, device float32).")
    .def_property_readonly("n_sets",
        [](const PyMarkerScoreResult& r) { return r.n_sets; })
    .def_property_readonly("n_cells",
        [](const PyMarkerScoreResult& r) { return r.n_cells; })
    .def_property_readonly("scores_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyMarkerScoreResult&>();
            return make_view_object<float>(
                r.scores.get(), r.scores.size(), r.stream, self_obj);
        }, "Activity scores float32 (n_sets × n_cells, col-major) — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyMarkerScoreResult& r) {
            return "<MarkerScoreResult n_sets=" + std::to_string(r.n_sets)
                 + " n_cells=" + std::to_string(r.n_cells) + ">";
        });

    // -----------------------------------------------------------------------
    // CelltypistModel — cycle 12.
    // -----------------------------------------------------------------------
    py::class_<PyCelltypistModel, std::shared_ptr<PyCelltypistModel>>(m, "CelltypistModel",
        "Loaded CellTypist model (weights on device).")
    .def_property_readonly("n_classes",
        [](const PyCelltypistModel& r) { return r.n_classes; })
    .def_property_readonly("n_pcs",
        [](const PyCelltypistModel& r) { return r.n_pcs; })
    .def_property_readonly("class_names",
        [](const PyCelltypistModel& r) { return r.class_names; },
        "list[str] cell-type class names.")
    .def_property_readonly("weights_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyCelltypistModel&>();
            return make_view_object<float>(
                r.weights.get(), r.weights.size(), r.stream, self_obj);
        }, "Weights float32 (n_classes × n_pcs, row-major) — __cuda_array_interface__.")
    .def_property_readonly("intercepts_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyCelltypistModel&>();
            return make_view_object<float>(
                r.intercepts.get(), r.intercepts.size(), r.stream, self_obj);
        }, "Intercepts float32 (n_classes,) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyCelltypistModel& r) {
            return "<CelltypistModel n_classes=" + std::to_string(r.n_classes)
                 + " n_pcs=" + std::to_string(r.n_pcs) + ">";
        });

    // -----------------------------------------------------------------------
    // RefMapResult — cycle 12.
    // -----------------------------------------------------------------------
    py::class_<PyRefMapResult, std::shared_ptr<PyRefMapResult>>(m, "RefMapResult",
        "CellTypist projection result (device float32 probabilities + int32 labels).")
    .def_property_readonly("n_classes",
        [](const PyRefMapResult& r) { return r.n_classes; })
    .def_property_readonly("n_cells",
        [](const PyRefMapResult& r) { return r.n_cells; })
    .def_property_readonly("probabilities_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyRefMapResult&>();
            return make_view_object<float>(
                r.probabilities.get(), r.probabilities.size(), r.stream, self_obj);
        }, "Class probabilities float32 (n_classes × n_cells, col-major) — __cuda_array_interface__.")
    .def_property_readonly("labels_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyRefMapResult&>();
            return make_view_object<int32_t>(
                r.labels.get(), r.labels.size(), r.stream, self_obj);
        }, "Argmax class index int32 (n_cells,) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyRefMapResult& r) {
            return "<RefMapResult n_classes=" + std::to_string(r.n_classes)
                 + " n_cells=" + std::to_string(r.n_cells) + ">";
        });

    // -----------------------------------------------------------------------
    // FgseaResult — cycle 13.
    // PathwayResult is host-side; return as Python list of dicts.
    // -----------------------------------------------------------------------
    py::class_<PyFgseaResult, std::shared_ptr<PyFgseaResult>>(m, "FgseaResult",
        "fGSEA preranked enrichment result.  .pathways is a list of dicts.")
    .def_property_readonly("pathways",
        [](const PyFgseaResult& r) {
            py::list lst;
            for (const auto& p : r.pathways) {
                py::dict d;
                d["name"]          = p.name;
                d["es"]            = p.es;
                d["nes"]           = p.nes;
                d["p_value"]       = p.p_value;
                d["q_value"]       = p.q_value;
                d["n_member_genes"]= p.n_member_genes;
                d["n_perms_used"]  = p.n_perms_used;
                lst.append(d);
            }
            return lst;
        }, "list[dict] — one dict per pathway with keys: name, es, nes, p_value, q_value, "
           "n_member_genes, n_perms_used.")
    .def("__repr__",
        [](const PyFgseaResult& r) {
            return "<FgseaResult n_pathways=" + std::to_string(r.pathways.size()) + ">";
        });

    // -----------------------------------------------------------------------
    // AUCellResult — cycle 13.
    // -----------------------------------------------------------------------
    py::class_<PyAUCellResult, std::shared_ptr<PyAUCellResult>>(m, "AUCellResult",
        "AUCell gene-set activity scores (device float32).")
    .def_property_readonly("n_pathways",
        [](const PyAUCellResult& r) { return r.n_pathways; })
    .def_property_readonly("n_cells",
        [](const PyAUCellResult& r) { return r.n_cells; })
    .def_property_readonly("scores_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyAUCellResult&>();
            return make_view_object<float>(
                r.scores.get(), r.scores.size(), r.stream, self_obj);
        }, "AUC scores float32 (n_pathways × n_cells, col-major) — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyAUCellResult& r) {
            return "<AUCellResult n_pathways=" + std::to_string(r.n_pathways)
                 + " n_cells=" + std::to_string(r.n_cells) + ">";
        });

    // -----------------------------------------------------------------------
    // HarmonyResult — cycle 14.
    // -----------------------------------------------------------------------
    py::class_<PyHarmonyResult, std::shared_ptr<PyHarmonyResult>>(m, "HarmonyResult",
        "Harmony batch-corrected embedding (device float32).")
    .def_property_readonly("n_iters_used",
        [](const PyHarmonyResult& r) { return r.n_iters_used; })
    .def_property_readonly("final_delta",
        [](const PyHarmonyResult& r) { return r.final_delta; })
    .def_property_readonly("corrected_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyHarmonyResult&>();
            return make_view_object<float>(
                r.corrected.get(), r.corrected.size(), r.stream, self_obj);
        }, "Corrected embedding float32 (n × d, row-major) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyHarmonyResult& r) {
            return "<HarmonyResult iters=" + std::to_string(r.n_iters_used)
                 + " final_delta=" + std::to_string(r.final_delta) + ">";
        });

    // -----------------------------------------------------------------------
    // VelocityPrepResult — cycle 15.
    // -----------------------------------------------------------------------
    py::class_<PyVelocityPrepResult, std::shared_ptr<PyVelocityPrepResult>>(
        m, "VelocityPrepResult",
        "RNA velocity preparation result (device float32 / uint8 buffers).")
    .def_property_readonly("n_genes_passing_filter",
        [](const PyVelocityPrepResult& r) { return r.n_genes_passing_filter; })
    .def_property_readonly("gamma_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<float>(
                r.gamma.get(), r.gamma.size(), r.stream, self_obj);
        }, "Per-gene γ (n_genes,) float32 — __cuda_array_interface__ dict.")
    .def_property_readonly("gamma_se_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<float>(
                r.gamma_se.get(), r.gamma_se.size(), r.stream, self_obj);
        }, "Per-gene γ SE (n_genes,) float32 — __cuda_array_interface__ dict.")
    .def_property_readonly("S_mean_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<float>(
                r.S_mean.get(), r.S_mean.size(), r.stream, self_obj);
        }, "Per-gene mean spliced (n_genes,) float32 — __cuda_array_interface__ dict.")
    .def_property_readonly("U_mean_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<float>(
                r.U_mean.get(), r.U_mean.size(), r.stream, self_obj);
        }, "Per-gene mean unspliced (n_genes,) float32 — __cuda_array_interface__ dict.")
    .def_property_readonly("filter_mask_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<uint8_t>(
                r.filter_mask.get(), r.filter_mask.size(), r.stream, self_obj);
        }, "Gene filter mask uint8 (n_genes,): 1 = passes count filter — __cuda_array_interface__.")
    .def_property_readonly("velocity_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyVelocityPrepResult&>();
            return make_view_object<float>(
                r.velocity.get(), r.velocity.size(), r.stream, self_obj);
        }, "Velocity matrix float32 (n_pass × n_cells, row-major) — __cuda_array_interface__. "
           "Empty if compute_velocity=False.")
    .def("__repr__",
        [](const PyVelocityPrepResult& r) {
            return "<VelocityPrepResult n_genes_passing=" +
                   std::to_string(r.n_genes_passing_filter) + ">";
        });

    // -----------------------------------------------------------------------
    // ClonePrediction — cycle 16.
    // -----------------------------------------------------------------------
    py::class_<PyClonePrediction, std::shared_ptr<PyClonePrediction>>(m, "ClonePrediction",
        "MT lineage clone prediction result (device int32 / float32 buffers).")
    .def_property_readonly("n_clones",
        [](const PyClonePrediction& r) { return r.n_clones; })
    .def_property_readonly("n_informative_sites",
        [](const PyClonePrediction& r) { return r.n_informative_sites; })
    .def_property_readonly("labels_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClonePrediction&>();
            return make_view_object<int32_t>(
                r.labels.get(), r.labels.size(), r.stream, self_obj);
        }, "Per-cell clone index int32 (n_cells,) — __cuda_array_interface__ dict.")
    .def_property_readonly("informative_sites_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClonePrediction&>();
            return make_view_object<int32_t>(
                r.informative_sites.get(), r.informative_sites.size(), r.stream, self_obj);
        }, "Informative MT site indices int32 (n_informative_sites,) — __cuda_array_interface__.")
    .def_property_readonly("clone_profiles_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClonePrediction&>();
            return make_view_object<float>(
                r.clone_profiles.get(), r.clone_profiles.size(), r.stream, self_obj);
        }, "Clone VAF profiles float32 (n_clones × n_informative_sites, row-major) — "
           "__cuda_array_interface__ dict.")
    .def_property_readonly("per_cell_heteroplasmy_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyClonePrediction&>();
            return make_view_object<float>(
                r.per_cell_heteroplasmy.get(), r.per_cell_heteroplasmy.size(), r.stream, self_obj);
        }, "Per-cell per-site heteroplasmy float32 (n_cells × n_informative_sites, row-major) — "
           "__cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyClonePrediction& r) {
            return "<ClonePrediction n_clones=" + std::to_string(r.n_clones)
                 + " n_informative_sites=" + std::to_string(r.n_informative_sites) + ">";
        });

    // -----------------------------------------------------------------------
    // DonorPseudobulkResult — cycle 17.
    // -----------------------------------------------------------------------
    py::class_<PyDonorPseudobulkResult, std::shared_ptr<PyDonorPseudobulkResult>>(
        m, "DonorPseudobulkResult",
        "Donor-aware pseudobulk NB GLM DE result (device float32 buffers).")
    .def_property_readonly("n_genes_passing_filter",
        [](const PyDonorPseudobulkResult& r) { return r.n_genes_passing_filter; })
    .def_property_readonly("log2_fc_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyDonorPseudobulkResult&>();
            return make_view_object<float>(
                r.log2_fc.get(), r.log2_fc.size(), r.stream, self_obj);
        }, "log2 fold-change float32 (m × n_clusters × (n_donors-1), row-major) — "
           "__cuda_array_interface__ dict.")
    .def_property_readonly("p_values_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyDonorPseudobulkResult&>();
            return make_view_object<float>(
                r.p_values.get(), r.p_values.size(), r.stream, self_obj);
        }, "p-values float32 (m × n_clusters) — __cuda_array_interface__ dict.")
    .def_property_readonly("p_adj_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyDonorPseudobulkResult&>();
            return make_view_object<float>(
                r.p_adj.get(), r.p_adj.size(), r.stream, self_obj);
        }, "BH-adjusted p-values float32 (m × n_clusters) — __cuda_array_interface__ dict.")
    .def_property_readonly("dispersion_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyDonorPseudobulkResult&>();
            return make_view_object<float>(
                r.dispersion.get(), r.dispersion.size(), r.stream, self_obj);
        }, "NB dispersion α float32 (m × n_clusters) — __cuda_array_interface__ dict.")
    .def_property_readonly("pseudobulk_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyDonorPseudobulkResult&>();
            return make_view_object<float>(
                r.pseudobulk.get(), r.pseudobulk.size(), r.stream, self_obj);
        }, "Raw pseudobulk sums float32 (m × n_clusters × n_donors, row-major) — "
           "__cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyDonorPseudobulkResult& r) {
            return "<DonorPseudobulkResult n_genes_passing="
                 + std::to_string(r.n_genes_passing_filter) + ">";
        });
}

}  // namespace python
}  // namespace singlet::gpu
