// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_spatial_phaseb.hpp
//
// Cycle 52a — Phase B spatial pybind11 bindings.
// Covers:
//   flash_deconv  (spatial/flash_deconv.h, cycle 33)
//   stagate       (spatial/stagate.h,       cycle 29)
//   cell2fate     (embed/cell2fate.h,        cycle 27)
//
// Pattern: every function takes explicit stream=None and seed=0.
// Neural-net features (STAGATE, Cell2fate) expose fit() + predict() surfaces
// via separate C++ entry points.  Discrete-state functions expose the main
// compute function directly.
//
// Result types (Py*) are registered in bind_spatial_phaseb(m).
// The C++ kernel namespace:
//   singlet_gpu::spatial::flash_deconv / stagate
//   singlet_gpu::embed::cell2fate_fit

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet-gpu/spatial/flash_deconv.h>
#include <singlet-gpu/spatial/stagate.h>
#include <singlet-gpu/embed/cell2fate.h>
#include <singlet-gpu/core/types.h>
#include <singlet-gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyFlashDeconvResult
// ---------------------------------------------------------------------------
struct PyFlashDeconvResult {
    singlet_gpu::core::DeviceMemory<float> abundance;    // n_spots × n_types
    singlet_gpu::core::DeviceMemory<float> uncertainty;  // n_spots × n_types
    int   n_admm_iters_used = 0;
    int   n_spots = 0;
    int   n_types = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyStagateResult
// ---------------------------------------------------------------------------
struct PyStagateResult {
    singlet_gpu::core::DeviceMemory<float> embedding;       // n_spots × d_embed
    singlet_gpu::core::DeviceMemory<int>   spatial_domain;  // n_spots (Leiden labels)
    std::vector<float>                     loss_history;    // host, one per epoch
    int n_epochs_used = 0;
    int n_spots       = 0;
    int d_embed       = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyCell2FateResult
// ---------------------------------------------------------------------------
struct PyCell2FateResult {
    singlet_gpu::core::DeviceMemory<float> alpha;            // [n_genes]
    singlet_gpu::core::DeviceMemory<float> beta;             // [n_genes]
    singlet_gpu::core::DeviceMemory<float> gamma;            // [n_genes]
    singlet_gpu::core::DeviceMemory<float> alpha_var;        // [n_genes]
    singlet_gpu::core::DeviceMemory<float> beta_var;         // [n_genes]
    singlet_gpu::core::DeviceMemory<float> gamma_var;        // [n_genes]
    singlet_gpu::core::DeviceMemory<float> module_loadings;  // [K × n_genes]
    singlet_gpu::core::DeviceMemory<float> cell_activities;  // [n_cells × K]
    singlet_gpu::core::DeviceMemory<float> velocity;         // [n_cells × n_genes] or empty
    singlet_gpu::core::DeviceMemory<int>   module_topgenes;  // [K × top_n]
    std::vector<float>                     elbo_history;
    int  n_iters_used = 0;
    int  K_used       = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python object (None → nullptr)
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    // Accept an integer handle (user cast from cupy stream pointer).
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// flash_deconv Python wrapper
// ---------------------------------------------------------------------------
inline py::object py_flash_deconv(
    const PyDeviceCsc& spatial_counts,
    const PyDeviceCsc& reference_counts,
    py::object         ref_cell_types_obj,   // 1-D int32 device array (cuda_array_interface)
    int                n_types,
    py::object         spatial_coords_obj,   // 2-D float32 device array or None
    int                sketch_size,
    int                max_admm_iter,
    float              admm_tol,
    float              l1_weight,
    float              spatial_lambda,
    int                n_bootstrap,
    int                spot_batch,
    py::object         stream,
    uint64_t           seed)
{
    cudaStream_t s = resolve_stream(stream);

    // Build DeviceCSC views from PyDeviceCsc.
    const singlet_gpu::core::DeviceCSC& sc = *spatial_counts.csc;
    const singlet_gpu::core::DeviceCSC& rc = *reference_counts.csc;

    // Unwrap reference_cell_types via cuda_array_interface.
    auto rct_iface = ref_cell_types_obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto rct_data_ptr = reinterpret_cast<const int*>(
        py::cast<uintptr_t>(rct_iface["data"].cast<py::tuple>()[0]));
    auto rct_shape = rct_iface["shape"].cast<py::tuple>();
    size_t n_ref_cells = py::cast<size_t>(rct_shape[0]);
    // Wrap in DeviceMemory (non-owning view).
    singlet_gpu::core::DeviceMemory<int> d_ref_types(n_ref_cells);
    cudaMemcpy(d_ref_types.get(), rct_data_ptr,
               n_ref_cells * sizeof(int), cudaMemcpyDeviceToDevice);

    // Optional spatial_coords.
    singlet_gpu::core::DeviceMemory<float> d_coords_buf;
    const singlet_gpu::core::DeviceMemory<float>* coords_ptr = nullptr;
    if (!spatial_coords_obj.is_none()) {
        auto ci = spatial_coords_obj.attr("__cuda_array_interface__").cast<py::dict>();
        auto ci_ptr = reinterpret_cast<const float*>(
            py::cast<uintptr_t>(ci["data"].cast<py::tuple>()[0]));
        auto ci_shape = ci["shape"].cast<py::tuple>();
        size_t n_coord_elems = py::cast<size_t>(ci_shape[0]) *
                               (ci_shape.size() > 1 ? py::cast<size_t>(ci_shape[1]) : 1u);
        d_coords_buf = singlet_gpu::core::DeviceMemory<float>(n_coord_elems);
        cudaMemcpy(d_coords_buf.get(), ci_ptr,
                   n_coord_elems * sizeof(float), cudaMemcpyDeviceToDevice);
        coords_ptr = &d_coords_buf;
    }

    singlet_gpu::spatial::FlashDeconvConfig cfg;
    cfg.sketch_size    = sketch_size;
    cfg.max_admm_iter  = max_admm_iter;
    cfg.admm_tol       = admm_tol;
    cfg.l1_weight      = l1_weight;
    cfg.spatial_lambda = spatial_lambda;
    cfg.n_bootstrap    = n_bootstrap;
    cfg.spot_batch     = spot_batch;
    cfg.seed           = seed;

    auto r = singlet_gpu::spatial::flash_deconv(sc, rc, d_ref_types, n_types, coords_ptr, cfg, s);

    auto py_r = std::make_shared<PyFlashDeconvResult>();
    py_r->abundance    = std::move(r.abundance);
    py_r->uncertainty  = std::move(r.uncertainty);
    py_r->n_admm_iters_used = r.n_admm_iters_used;
    py_r->n_spots = (int)sc.ncols;
    py_r->n_types = n_types;
    py_r->stream  = s;
    return py::cast(py_r);
}

// ---------------------------------------------------------------------------
// stagate Python wrapper
// ---------------------------------------------------------------------------
inline py::object py_stagate(
    const PyDeviceCsc& counts,
    py::object         spatial_coords_obj,   // n_spots × 2 float32 device array
    int    n_neighbors,
    int    d_hidden,
    int    d_embed,
    int    n_epochs,
    float  learning_rate,
    bool   sparse_mask_recon,
    int    spot_tile,
    bool   run_post_leiden,
    float  leiden_resolution,
    py::object stream,
    uint64_t seed)
{
    cudaStream_t s = resolve_stream(stream);
    const singlet_gpu::core::DeviceCSC& mat = *counts.csc;

    // Unwrap spatial coords.
    auto ci = spatial_coords_obj.attr("__cuda_array_interface__").cast<py::dict>();
    auto ci_ptr = reinterpret_cast<const float*>(
        py::cast<uintptr_t>(ci["data"].cast<py::tuple>()[0]));
    auto ci_shape = ci["shape"].cast<py::tuple>();
    size_t n_coord_elems = py::cast<size_t>(ci_shape[0]) *
                           (ci_shape.size() > 1 ? py::cast<size_t>(ci_shape[1]) : 1u);
    singlet_gpu::core::DeviceMemory<float> d_coords(n_coord_elems);
    cudaMemcpy(d_coords.get(), ci_ptr,
               n_coord_elems * sizeof(float), cudaMemcpyDeviceToDevice);

    singlet_gpu::spatial::StagateConfig cfg;
    cfg.n_neighbors       = n_neighbors;
    cfg.d_hidden          = d_hidden;
    cfg.d_embed           = d_embed;
    cfg.n_epochs          = n_epochs;
    cfg.learning_rate     = learning_rate;
    cfg.sparse_mask_recon = sparse_mask_recon;
    cfg.spot_tile         = spot_tile;
    cfg.run_post_leiden   = run_post_leiden;
    cfg.leiden_resolution = leiden_resolution;
    cfg.seed              = seed;

    auto r = singlet_gpu::spatial::stagate(mat, d_coords, cfg, s);

    auto py_r = std::make_shared<PyStagateResult>();
    py_r->embedding      = std::move(r.embedding);
    py_r->spatial_domain = std::move(r.spatial_domain);
    py_r->loss_history   = std::move(r.loss_history);
    py_r->n_epochs_used  = r.n_epochs_used;
    py_r->n_spots = (int)mat.ncols;
    py_r->d_embed = d_embed;
    py_r->stream  = s;
    return py::cast(py_r);
}

// ---------------------------------------------------------------------------
// cell2fate Python wrapper (fit)
// ---------------------------------------------------------------------------
inline py::object py_cell2fate_fit(
    const PyDeviceCsc& spliced,
    const PyDeviceCsc& unspliced,
    int     n_modules,
    int     max_iter,
    int     batch_size,
    float   learning_rate,
    int     n_posterior_draws,
    int     top_n_per_module,
    float   beta_kl_max,
    float   kl_anneal_fraction,
    float   convergence_tol,
    int     convergence_patience,
    bool    compute_velocity,
    int     gene_tile,
    py::object stream,
    uint64_t seed)
{
    cudaStream_t s = resolve_stream(stream);
    const singlet_gpu::core::DeviceCSC& sp = *spliced.csc;
    const singlet_gpu::core::DeviceCSC& un = *unspliced.csc;

    singlet_gpu::embed::Cell2FateConfig cfg;
    cfg.n_modules            = n_modules;
    cfg.max_iter             = max_iter;
    cfg.batch_size           = batch_size;
    cfg.learning_rate        = learning_rate;
    cfg.n_posterior_draws    = n_posterior_draws;
    cfg.top_n_per_module     = top_n_per_module;
    cfg.beta_kl_max          = beta_kl_max;
    cfg.kl_anneal_fraction   = kl_anneal_fraction;
    cfg.convergence_tol      = convergence_tol;
    cfg.convergence_patience = convergence_patience;
    cfg.seed                 = seed;
    cfg.compute_velocity     = compute_velocity;
    cfg.gene_tile            = gene_tile;

    auto r = singlet_gpu::embed::cell2fate_fit(sp, un, nullptr, cfg, s);

    auto py_r = std::make_shared<PyCell2FateResult>();
    py_r->alpha           = std::move(r.alpha);
    py_r->beta            = std::move(r.beta);
    py_r->gamma           = std::move(r.gamma);
    py_r->alpha_var       = std::move(r.alpha_var);
    py_r->beta_var        = std::move(r.beta_var);
    py_r->gamma_var       = std::move(r.gamma_var);
    py_r->module_loadings = std::move(r.module_loadings);
    py_r->cell_activities = std::move(r.cell_activities);
    py_r->velocity        = std::move(r.velocity);
    py_r->module_topgenes = std::move(r.module_topgenes);
    py_r->elbo_history    = std::move(r.elbo_history);
    py_r->n_iters_used    = r.n_iters_used;
    py_r->K_used          = r.K_used;
    py_r->stream          = s;
    return py::cast(py_r);
}

// ---------------------------------------------------------------------------
// bind_spatial_phaseb — called from PYBIND11_MODULE
// ---------------------------------------------------------------------------
inline void bind_spatial_phaseb(py::module_& m) {

    // ── FlashDeconvResult ──────────────────────────────────────────────────
    py::class_<PyFlashDeconvResult, std::shared_ptr<PyFlashDeconvResult>>(
        m, "FlashDeconvResult",
        "Result of flash_deconv.  abundance + uncertainty device float32 buffers.")
    .def_property_readonly("n_admm_iters_used",
        [](const PyFlashDeconvResult& r){ return r.n_admm_iters_used; })
    .def_property_readonly("n_spots",
        [](const PyFlashDeconvResult& r){ return r.n_spots; })
    .def_property_readonly("n_types",
        [](const PyFlashDeconvResult& r){ return r.n_types; })
    .def_property_readonly("abundance_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyFlashDeconvResult&>();
            return make_view_object<float>(r.abundance.get(), r.abundance.size(),
                                           r.stream, self_obj);
        }, "[n_spots × n_types] row-major float32 — __cuda_array_interface__.")
    .def_property_readonly("uncertainty_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyFlashDeconvResult&>();
            return make_view_object<float>(r.uncertainty.get(), r.uncertainty.size(),
                                           r.stream, self_obj);
        }, "[n_spots × n_types] bootstrap stddev — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyFlashDeconvResult& r){
            return "<FlashDeconvResult n_spots=" + std::to_string(r.n_spots)
                 + " n_types=" + std::to_string(r.n_types)
                 + " admm_iters=" + std::to_string(r.n_admm_iters_used) + ">";
        });

    // ── StagateResult ──────────────────────────────────────────────────────
    py::class_<PyStagateResult, std::shared_ptr<PyStagateResult>>(
        m, "StagateResult",
        "Result of stagate.  embedding + spatial_domain device buffers.")
    .def_property_readonly("n_spots",
        [](const PyStagateResult& r){ return r.n_spots; })
    .def_property_readonly("d_embed",
        [](const PyStagateResult& r){ return r.d_embed; })
    .def_property_readonly("n_epochs_used",
        [](const PyStagateResult& r){ return r.n_epochs_used; })
    .def_property_readonly("loss_history",
        [](const PyStagateResult& r){ return r.loss_history; },
        "Host-side float32 list, one entry per epoch.")
    .def_property_readonly("embedding_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyStagateResult&>();
            return make_view_object<float>(r.embedding.get(), r.embedding.size(),
                                           r.stream, self_obj);
        }, "[n_spots × d_embed] float32 — __cuda_array_interface__.")
    .def_property_readonly("spatial_domain_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyStagateResult&>();
            return make_view_object<int>(r.spatial_domain.get(),
                                         r.spatial_domain.size(),
                                         r.stream, self_obj);
        }, "[n_spots] int32 Leiden cluster labels — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyStagateResult& r){
            return "<StagateResult n_spots=" + std::to_string(r.n_spots)
                 + " d_embed=" + std::to_string(r.d_embed)
                 + " epochs=" + std::to_string(r.n_epochs_used) + ">";
        });

    // ── Cell2FateResult ────────────────────────────────────────────────────
    py::class_<PyCell2FateResult, std::shared_ptr<PyCell2FateResult>>(
        m, "Cell2FateResult",
        "Result of cell2fate_fit.  Kinetic rates + module decomposition on device.")
    .def_property_readonly("K_used",
        [](const PyCell2FateResult& r){ return r.K_used; })
    .def_property_readonly("n_iters_used",
        [](const PyCell2FateResult& r){ return r.n_iters_used; })
    .def_property_readonly("elbo_history",
        [](const PyCell2FateResult& r){ return r.elbo_history; })
    .def_property_readonly("alpha_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.alpha.get(),r.alpha.size(),r.stream,s); })
    .def_property_readonly("beta_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.beta.get(),r.beta.size(),r.stream,s); })
    .def_property_readonly("gamma_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.gamma.get(),r.gamma.size(),r.stream,s); })
    .def_property_readonly("alpha_var_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.alpha_var.get(),r.alpha_var.size(),r.stream,s); })
    .def_property_readonly("beta_var_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.beta_var.get(),r.beta_var.size(),r.stream,s); })
    .def_property_readonly("gamma_var_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.gamma_var.get(),r.gamma_var.size(),r.stream,s); })
    .def_property_readonly("module_loadings_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.module_loadings.get(),r.module_loadings.size(),r.stream,s); },
        "[K × n_genes] row-major — __cuda_array_interface__.")
    .def_property_readonly("cell_activities_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<float>(r.cell_activities.get(),r.cell_activities.size(),r.stream,s); },
        "[n_cells × K] row-major — __cuda_array_interface__.")
    .def_property_readonly("velocity_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            if(r.velocity.size()==0) return py::cast(py::none());
            return py::cast(make_view_object<float>(r.velocity.get(),r.velocity.size(),r.stream,s)); },
        "[n_cells × n_genes] velocity or None if compute_velocity=False.")
    .def_property_readonly("module_topgenes_view",
        [](py::object s){ auto& r=s.cast<PyCell2FateResult&>();
            return make_view_object<int>(r.module_topgenes.get(),r.module_topgenes.size(),r.stream,s); },
        "[K × top_n_per_module] int32 gene indices — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyCell2FateResult& r){
            return "<Cell2FateResult K=" + std::to_string(r.K_used)
                 + " iters=" + std::to_string(r.n_iters_used) + ">";
        });

    // ── flash_deconv function ──────────────────────────────────────────────
    m.def("flash_deconv",
        [](const PyDeviceCsc& sc, const PyDeviceCsc& rc,
           py::object rct, int n_types, py::object coords,
           int sketch_size, int max_admm_iter, float admm_tol, float l1_weight,
           float spatial_lambda, int n_bootstrap, int spot_batch,
           py::object stream, uint64_t seed){
            return py_flash_deconv(sc, rc, rct, n_types, coords,
                                   sketch_size, max_admm_iter, admm_tol, l1_weight,
                                   spatial_lambda, n_bootstrap, spot_batch, stream, seed);
        },
        py::arg("spatial_counts"),
        py::arg("reference_counts"),
        py::arg("reference_cell_types"),
        py::arg("n_types"),
        py::arg("spatial_coords")   = py::none(),
        py::arg("sketch_size")      = 500,
        py::arg("max_admm_iter")    = 50,
        py::arg("admm_tol")         = 1e-4f,
        py::arg("l1_weight")        = 0.01f,
        py::arg("spatial_lambda")   = 0.0f,
        py::arg("n_bootstrap")      = 20,
        py::arg("spot_batch")       = 100000,
        py::arg("stream")           = py::none(),
        py::arg("seed")             = 0ULL,
        R"doc(
        GPU-native spatial deconvolution via FlashDeconv (cycle 33).

        Parameters
        ----------
        spatial_counts : DeviceCsc
            genes × spots matrix.
        reference_counts : DeviceCsc
            genes × ref_cells matrix.
        reference_cell_types : cuda array (int32, n_ref_cells)
            Integer cell-type label per reference cell (0-based).
        n_types : int
            Number of distinct cell types.
        spatial_coords : cuda array (float32, n_spots × 2) or None
            XY spot coordinates for spatial regularization.
        stream : int or None
            CUDA stream handle.  None uses the default stream.
        seed : int
            Philox4x32 base seed for leverage-score sketching + bootstrap.

        Returns
        -------
        FlashDeconvResult
        )doc");

    // ── stagate function ───────────────────────────────────────────────────
    m.def("stagate",
        [](const PyDeviceCsc& counts, py::object coords,
           int n_neighbors, int d_hidden, int d_embed, int n_epochs,
           float learning_rate, bool sparse_mask_recon, int spot_tile,
           bool run_post_leiden, float leiden_resolution,
           py::object stream, uint64_t seed){
            return py_stagate(counts, coords, n_neighbors, d_hidden, d_embed,
                              n_epochs, learning_rate, sparse_mask_recon, spot_tile,
                              run_post_leiden, leiden_resolution, stream, seed);
        },
        py::arg("counts"),
        py::arg("spatial_coords"),
        py::arg("n_neighbors")        = 6,
        py::arg("d_hidden")           = 256,
        py::arg("d_embed")            = 64,
        py::arg("n_epochs")           = 500,
        py::arg("learning_rate")      = 1e-3f,
        py::arg("sparse_mask_recon")  = true,
        py::arg("spot_tile")          = 10000,
        py::arg("run_post_leiden")    = true,
        py::arg("leiden_resolution")  = 0.5f,
        py::arg("stream")             = py::none(),
        py::arg("seed")               = 0ULL,
        R"doc(
        GPU-native spatial domain segmentation via STAGATE (cycle 29).

        Parameters
        ----------
        counts : DeviceCsc
            genes × spots expression matrix.
        spatial_coords : cuda array (float32, n_spots × 2)
            XY spot coordinates; used for kNN graph construction.
        stream : int or None
        seed : int
            Philox4x32 seed for Xavier initialization.

        Returns
        -------
        StagateResult
        )doc");

    // ── cell2fate_fit function ─────────────────────────────────────────────
    m.def("cell2fate_fit",
        [](const PyDeviceCsc& spliced, const PyDeviceCsc& unspliced,
           int n_modules, int max_iter, int batch_size,
           float learning_rate, int n_posterior_draws, int top_n_per_module,
           float beta_kl_max, float kl_anneal_fraction,
           float convergence_tol, int convergence_patience,
           bool compute_velocity, int gene_tile,
           py::object stream, uint64_t seed){
            return py_cell2fate_fit(spliced, unspliced, n_modules, max_iter,
                                    batch_size, learning_rate, n_posterior_draws,
                                    top_n_per_module, beta_kl_max, kl_anneal_fraction,
                                    convergence_tol, convergence_patience,
                                    compute_velocity, gene_tile, stream, seed);
        },
        py::arg("spliced"),
        py::arg("unspliced"),
        py::arg("n_modules")            = 10,
        py::arg("max_iter")             = 2000,
        py::arg("batch_size")           = 1024,
        py::arg("learning_rate")        = 0.01f,
        py::arg("n_posterior_draws")    = 100,
        py::arg("top_n_per_module")     = 20,
        py::arg("beta_kl_max")          = 1.0f,
        py::arg("kl_anneal_fraction")   = 0.5f,
        py::arg("convergence_tol")      = 1e-4f,
        py::arg("convergence_patience") = 10,
        py::arg("compute_velocity")     = false,
        py::arg("gene_tile")            = 1024,
        py::arg("stream")               = py::none(),
        py::arg("seed")                 = 0ULL,
        R"doc(
        GPU-native Cell2fate velocity-module decomposition (cycle 27).

        Parameters
        ----------
        spliced : DeviceCsc
            Spliced count matrix (genes × cells).
        unspliced : DeviceCsc
            Unspliced count matrix (genes × cells).  Must match spliced dimensions.
        stream : int or None
        seed : int

        Returns
        -------
        Cell2FateResult
        )doc");
}

}  // namespace python
}  // namespace singlet_gpu
