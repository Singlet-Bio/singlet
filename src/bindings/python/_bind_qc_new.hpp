// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_qc_new.hpp
//
// Cycle 52a — QC pybind11 bindings (cycles 31 + 39).
// Covers:
//   doublet_score  (qc/doublet_score.h,  cycle 31) — RNA-only Scrublet-style
//   omni_doublet   (qc/omnidoublet.h,    cycle 39) — multimodal CITE-seq
//
// doublet_score takes a PCA embedding (device float32 dense matrix).
// omni_doublet  takes two CSC matrices (RNA + ADT); handles PCA internally.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/qc/doublet_score.h>
// OmniDoublet (cycle 39) is in the deferred-indefinitely scope per
// state/roadmap.md and still uses factornet::SVDConfig + factornet::svd::
// randomized_svd_gpu_dense. Gated to keep the foundational _core.so build
// independent of factornet.
#ifdef SINGLET_GPU_BUILD_DEFERRED
#  include <singlet/gpu/qc/omnidoublet.h>
#endif
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyDoubletScoreResult
// ---------------------------------------------------------------------------
struct PyDoubletScoreResult {
    singlet::gpu::core::DeviceMemory<float>   score;
    singlet::gpu::core::DeviceMemory<uint8_t> doublet_call;
    float threshold_used         = 0.0f;
    int   n_predicted_doublets   = 0;
    int   n_cells                = 0;
    cudaStream_t stream          = nullptr;
};

// ---------------------------------------------------------------------------
// PyOmniDoubletResult — gated; OmniDoublet is in deferred-indefinitely scope.
// ---------------------------------------------------------------------------
#ifdef SINGLET_GPU_BUILD_DEFERRED
struct PyOmniDoubletResult {
    singlet::gpu::core::DeviceMemory<float>   doublet_score;
    singlet::gpu::core::DeviceMemory<uint8_t> doublet_call;
    singlet::gpu::core::DeviceMemory<float>   simulated_scores;
    float threshold_used          = 0.0f;
    int   n_predicted_doublets    = 0;
    int   n_cells                 = 0;
    int   n_sim                   = 0;
    cudaStream_t stream           = nullptr;
};
#endif

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_qc(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_qc_new — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_qc_new(py::module_& m) {

    // ── DoubletScoreResult ─────────────────────────────────────────────────
    py::class_<PyDoubletScoreResult, std::shared_ptr<PyDoubletScoreResult>>(
        m, "DoubletScoreResult",
        "Result of doublet_score (Scrublet-style, RNA-only).")
    .def_property_readonly("n_cells",
        [](const PyDoubletScoreResult& r){ return r.n_cells; })
    .def_property_readonly("threshold_used",
        [](const PyDoubletScoreResult& r){ return r.threshold_used; })
    .def_property_readonly("n_predicted_doublets",
        [](const PyDoubletScoreResult& r){ return r.n_predicted_doublets; })
    .def_property_readonly("score_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyDoubletScoreResult&>();
            return make_view_object<float>(r.score.get(), r.score.size(),
                                           r.stream, self_obj);
        }, "[n_cells] float32 doublet scores — __cuda_array_interface__.")
    .def_property_readonly("doublet_call_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyDoubletScoreResult&>();
            return make_view_object<uint8_t>(r.doublet_call.get(),
                                             r.doublet_call.size(),
                                             r.stream, self_obj);
        }, "[n_cells] uint8 binary doublet call — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyDoubletScoreResult& r){
            return "<DoubletScoreResult n_cells=" + std::to_string(r.n_cells)
                 + " n_doublets=" + std::to_string(r.n_predicted_doublets)
                 + " threshold=" + std::to_string(r.threshold_used) + ">";
        });

    // ── OmniDoubletResult — gated; deferred-indefinitely scope ─────────────
#ifdef SINGLET_GPU_BUILD_DEFERRED
    py::class_<PyOmniDoubletResult, std::shared_ptr<PyOmniDoubletResult>>(
        m, "OmniDoubletResult",
        "Result of omni_doublet (multimodal CITE-seq doublet detection).")
    .def_property_readonly("n_cells",
        [](const PyOmniDoubletResult& r){ return r.n_cells; })
    .def_property_readonly("n_sim",
        [](const PyOmniDoubletResult& r){ return r.n_sim; })
    .def_property_readonly("threshold_used",
        [](const PyOmniDoubletResult& r){ return r.threshold_used; })
    .def_property_readonly("n_predicted_doublets",
        [](const PyOmniDoubletResult& r){ return r.n_predicted_doublets; })
    .def_property_readonly("doublet_score_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyOmniDoubletResult&>();
            return make_view_object<float>(r.doublet_score.get(),
                                           r.doublet_score.size(),
                                           r.stream, self_obj);
        }, "[n_cells] float32 doublet scores — __cuda_array_interface__.")
    .def_property_readonly("doublet_call_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyOmniDoubletResult&>();
            return make_view_object<uint8_t>(r.doublet_call.get(),
                                             r.doublet_call.size(),
                                             r.stream, self_obj);
        }, "[n_cells] uint8 binary doublet call — __cuda_array_interface__.")
    .def_property_readonly("simulated_scores_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyOmniDoubletResult&>();
            return make_view_object<float>(r.simulated_scores.get(),
                                           r.simulated_scores.size(),
                                           r.stream, self_obj);
        }, "[n_sim] float32 scores for simulated doublets (calibration) — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyOmniDoubletResult& r){
            return "<OmniDoubletResult n_cells=" + std::to_string(r.n_cells)
                 + " n_doublets=" + std::to_string(r.n_predicted_doublets)
                 + " threshold=" + std::to_string(r.threshold_used) + ">";
        });

    // ── doublet_score function ─────────────────────────────────────────────
    m.def("doublet_score",
        [](py::object embedding_obj,
           float    n_synth_frac,
           int      k,
           float    manual_threshold,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_qc(stream);

            // Unwrap embedding via cuda_array_interface.
            auto ei = embedding_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto e_ptr = reinterpret_cast<const float*>(
                py::cast<uintptr_t>(ei["data"].cast<py::tuple>()[0]));
            auto e_shape = ei["shape"].cast<py::tuple>();
            int n_real = py::cast<int>(e_shape[0]);
            int n_pcs  = (int)e_shape.size() > 1 ? py::cast<int>(e_shape[1]) : 1;

            // Build non-owning DeviceDense view.
            singlet::gpu::core::DeviceDense emb;
            emb.d_data = const_cast<float*>(e_ptr);
            emb.nrows  = n_real;
            emb.ncols  = n_pcs;
            emb.ld     = n_pcs;   // row-major

            singlet::gpu::qc::DoubletScoreConfig cfg;
            cfg.n_synth_frac     = n_synth_frac;
            cfg.k                = k;
            cfg.manual_threshold = manual_threshold;
            cfg.seed             = seed;

            auto r = singlet::gpu::qc::doublet_score(emb, cfg, s);

            auto py_r = std::make_shared<PyDoubletScoreResult>();
            py_r->score               = std::move(r.score);
            py_r->doublet_call        = std::move(r.doublet_call);
            py_r->threshold_used      = r.threshold_used;
            py_r->n_predicted_doublets = r.n_predicted_doublets;
            py_r->n_cells             = n_real;
            py_r->stream              = s;
            return py::cast(py_r);
        },
        py::arg("embedding"),
        py::arg("n_synth_frac")     = 0.25f,
        py::arg("k")                = 20,
        py::arg("manual_threshold") = 0.0f,
        py::arg("stream")           = py::none(),
        py::arg("seed")             = 0ULL,
        R"doc(
        Scrublet-style RNA-only doublet detection on a PCA embedding (cycle 31).

        Parameters
        ----------
        embedding : cuda array (float32, n_cells × n_pcs)
            PCA-projected cells.  Must be row-major.
        n_synth_frac : float
            Fraction of real cells to use as synthetic doublets.
        k : int
            kNN neighbours.
        manual_threshold : float
            0 = auto-threshold (histogram knee); >0 = use as-is.
        stream : int or None
        seed : int

        Returns
        -------
        DoubletScoreResult
        )doc");

    // ── omni_doublet function ── (still gated by SINGLET_GPU_BUILD_DEFERRED)
    m.def("omni_doublet",
        [](const PyDeviceCsc& rna_counts,
           const PyDeviceCsc& adt_counts,
           int   n_sim_mult,
           int   n_pcs_rna,
           int   n_pcs_adt,
           int   n_hvg,
           int   k,
           int   irls_max_iter,
           float irls_tol,
           float target_doublet_rate,
           bool  deterministic,
           py::object stream,
           uint64_t seed) -> py::object
        {
            cudaStream_t s = resolve_stream_qc(stream);
            const singlet::gpu::core::DeviceCSC& rna = *rna_counts.csc;
            const singlet::gpu::core::DeviceCSC& adt = *adt_counts.csc;

            singlet::gpu::qc::OmniDoubletConfig cfg;
            cfg.n_sim_mult          = n_sim_mult;
            cfg.seed                = seed;
            cfg.n_pcs_rna           = n_pcs_rna;
            cfg.n_pcs_adt           = n_pcs_adt;
            cfg.n_hvg               = n_hvg;
            cfg.k                   = k;
            cfg.irls_max_iter       = irls_max_iter;
            cfg.irls_tol            = irls_tol;
            cfg.target_doublet_rate = target_doublet_rate;
            cfg.deterministic       = deterministic;

            auto r = singlet::gpu::qc::omni_doublet(rna, adt, cfg, s);

            int n_cells = (int)rna.ncols;
            auto py_r = std::make_shared<PyOmniDoubletResult>();
            py_r->doublet_score       = std::move(r.doublet_score);
            py_r->doublet_call        = std::move(r.doublet_call);
            py_r->simulated_scores    = std::move(r.simulated_scores);
            py_r->threshold_used      = r.threshold_used;
            py_r->n_predicted_doublets = r.n_predicted_doublets;
            py_r->n_cells             = n_cells;
            py_r->n_sim               = n_cells * n_sim_mult;
            py_r->stream              = s;
            return py::cast(py_r);
        },
        py::arg("rna_counts"),
        py::arg("adt_counts"),
        py::arg("n_sim_mult")          = 2,
        py::arg("n_pcs_rna")           = 30,
        py::arg("n_pcs_adt")           = 20,
        py::arg("n_hvg")               = 2000,
        py::arg("k")                   = 50,
        py::arg("irls_max_iter")       = 20,
        py::arg("irls_tol")            = 1e-5f,
        py::arg("target_doublet_rate") = 0.10f,
        py::arg("deterministic")       = false,
        py::arg("stream")              = py::none(),
        py::arg("seed")                = 0ULL,
        R"doc(
        Multimodal CITE-seq doublet detection (cycle 39, OmniDoublet).

        For RNA-only datasets use doublet_score instead.

        Parameters
        ----------
        rna_counts : DeviceCsc
            genes × cells raw RNA count matrix.
        adt_counts : DeviceCsc
            tags × cells raw ADT count matrix.  Same number of cells as rna_counts.
        target_doublet_rate : float
            Expected doublet fraction for FDR threshold.
        deterministic : bool
            Use segmented scan instead of atomicAdd for feature computation.
        stream : int or None
        seed : int

        Returns
        -------
        OmniDoubletResult
        )doc");
#endif // SINGLET_GPU_BUILD_DEFERRED — closes both PyOmniDoubletResult class + omni_doublet fn
}

}  // namespace python
}  // namespace singlet::gpu
