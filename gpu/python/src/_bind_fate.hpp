// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_fate.hpp
//
// pybind11 bindings for fate trajectory kernels:
//   - cospar        (cycle 41) singlet_gpu::fate::run_cospar
//   - cellrank2     (cycle 43) singlet_gpu::fate::run_cellrank2
//   - palantir      (cycle 45) singlet_gpu::fate::run_palantir
//
// Pattern mirrors _bind_kernels.hpp / _bind_results.hpp:
//   Py* structs hold result data; make_view_object<T> exposes device buffers
//   via __cuda_array_interface__ for zero-copy cupy interop.
//   Host-side result vectors (std::vector<float>) are returned as numpy arrays
//   via py::array_t.
//
// No host↔device copies inside this file except final result export.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/fate/cospar.h>
#include <singlet-gpu/fate/cellrank2.h>
#include <singlet-gpu/fate/palantir.h>
#include <singlet-gpu/core/types.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"   // PyDeviceCsc

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// PyCosparResult
// ---------------------------------------------------------------------------
struct PyCosparResult {
    // transition_maps[i]: device fp32 dense matrix n_t0[i] × n_t1[i]
    std::vector<singlet_gpu::core::DeviceMemory<float>> transition_maps;
    std::vector<int>   n_cells_t0;
    std::vector<int>   n_cells_t1;

    // Host arrays — small enough to return as numpy.
    std::vector<float> fate_bias;          // n_cells × n_fates row-major
    std::vector<float> potency_score;      // n_cells
    std::vector<float> driver_genes;       // n_fates × n_genes row-major
    std::vector<float> final_residuals;

    int n_cells_total   = 0;
    int n_terminal_fates = 0;
    int n_genes         = 0;
    int iters_run       = 0;
    bool state_only_mode = false;

    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyCellRank2Result
// ---------------------------------------------------------------------------
struct PyCellRank2Result {
    std::vector<float>  absorption_prob;  // n_cells × n_terminals row-major (host)
    std::vector<float>  driver_genes;     // n_terminals × n_genes row-major (host)
    std::vector<double> final_residuals;
    std::vector<int>    restarts_used;

    int  n_cells     = 0;
    int  n_terminals = 0;
    int  n_genes     = 0;
    bool converged_all = false;

    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// PyPalantirResult
// ---------------------------------------------------------------------------
struct PyPalantirResult {
    std::vector<float> pseudotime;         // n_cells (host)
    std::vector<float> branch_prob;        // n_cells × n_terminals row-major (host)
    std::vector<float> eigenvalues;        // k_eig (host)
    std::vector<int>   terminal_indices;   // host

    // Eigenvectors stay device-resident for downstream use.
    singlet_gpu::core::DeviceMemory<float> eigenvectors_dev;  // k_eig × n_cells

    int n_cells     = 0;
    int n_terminals = 0;
    int k_eig       = 0;

    cudaStream_t stream = nullptr;
};

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// bind_fate — register Py* classes and Python functions into submodule m.
// Called from bind_fate_module(m) which creates/gets the "fate" submodule.
// ---------------------------------------------------------------------------

inline void bind_fate(py::module_& m) {

    // --- PyCosparResult ----------------------------------------------------
    py::class_<PyCosparResult, std::shared_ptr<PyCosparResult>>(m, "CosparResult",
        "Result of cospar().  Host arrays returned as numpy; device transition maps "
        "exposed via __cuda_array_interface__.")
        .def_property_readonly("n_cells",
            [](const PyCosparResult& r) { return r.n_cells_total; })
        .def_property_readonly("n_terminal_fates",
            [](const PyCosparResult& r) { return r.n_terminal_fates; })
        .def_property_readonly("n_genes",
            [](const PyCosparResult& r) { return r.n_genes; })
        .def_property_readonly("iters_run",
            [](const PyCosparResult& r) { return r.iters_run; })
        .def_property_readonly("state_only_mode",
            [](const PyCosparResult& r) { return r.state_only_mode; })
        .def_property_readonly("fate_bias",
            [](const PyCosparResult& r) {
                return py::array_t<float>(
                    {r.n_cells_total, r.n_terminal_fates},
                    r.fate_bias.data());
            },
            "Fate bias matrix (n_cells × n_fates), numpy float32.")
        .def_property_readonly("potency_score",
            [](const PyCosparResult& r) {
                return py::array_t<float>({r.n_cells_total}, r.potency_score.data());
            },
            "Per-cell potency (Shannon entropy of fate_bias), numpy float32.")
        .def_property_readonly("driver_genes",
            [](const PyCosparResult& r) {
                return py::array_t<float>(
                    {r.n_terminal_fates, r.n_genes},
                    r.driver_genes.data());
            },
            "Driver gene Pearson correlations (n_fates × n_genes), numpy float32.")
        .def_property_readonly("final_residuals",
            [](const PyCosparResult& r) {
                return py::array_t<float>(
                    {(int)r.final_residuals.size()}, r.final_residuals.data());
            })
        .def("transition_map_view",
            [](py::object self_obj, int pair_idx) {
                auto& r = self_obj.cast<PyCosparResult&>();
                if (pair_idx < 0 || pair_idx >= (int)r.transition_maps.size())
                    throw std::out_of_range("pair_idx out of range");
                int n0 = r.n_cells_t0[pair_idx];
                int n1 = r.n_cells_t1[pair_idx];
                return make_view_object<float>(
                    r.transition_maps[pair_idx].data(),
                    static_cast<std::size_t>(n0) * n1,
                    r.stream, self_obj);
            },
            py::arg("pair_idx"),
            "Return __cuda_array_interface__ view of transition map for time-pair pair_idx.")
        .def("__repr__",
            [](const PyCosparResult& r) {
                return "<CosparResult n_cells=" + std::to_string(r.n_cells_total)
                     + " n_fates=" + std::to_string(r.n_terminal_fates)
                     + " iters=" + std::to_string(r.iters_run) + ">";
            });

    // --- PyCellRank2Result -------------------------------------------------
    py::class_<PyCellRank2Result, std::shared_ptr<PyCellRank2Result>>(m, "CellRank2Result",
        "Result of cellrank2().  All arrays host-side numpy.")
        .def_property_readonly("n_cells",
            [](const PyCellRank2Result& r) { return r.n_cells; })
        .def_property_readonly("n_terminals",
            [](const PyCellRank2Result& r) { return r.n_terminals; })
        .def_property_readonly("converged_all",
            [](const PyCellRank2Result& r) { return r.converged_all; })
        .def_property_readonly("absorption_prob",
            [](const PyCellRank2Result& r) {
                return py::array_t<float>(
                    {r.n_cells, r.n_terminals},
                    r.absorption_prob.data());
            },
            "Absorption probability matrix (n_cells × n_terminals), numpy float32.")
        .def_property_readonly("driver_genes",
            [](const PyCellRank2Result& r) {
                return py::array_t<float>(
                    {r.n_terminals, r.n_genes},
                    r.driver_genes.data());
            },
            "Driver gene Pearson r (n_terminals × n_genes); empty if not requested.")
        .def_property_readonly("final_residuals",
            [](const PyCellRank2Result& r) {
                return py::array_t<double>(
                    {(int)r.final_residuals.size()}, r.final_residuals.data());
            })
        .def("__repr__",
            [](const PyCellRank2Result& r) {
                return "<CellRank2Result n_cells=" + std::to_string(r.n_cells)
                     + " n_terminals=" + std::to_string(r.n_terminals)
                     + " converged=" + (r.converged_all ? "True" : "False") + ">";
            });

    // --- PyPalantirResult --------------------------------------------------
    py::class_<PyPalantirResult, std::shared_ptr<PyPalantirResult>>(m, "PalantirResult",
        "Result of palantir().  Pseudotime + branch probs on host; eigenvectors device.")
        .def_property_readonly("n_cells",
            [](const PyPalantirResult& r) { return r.n_cells; })
        .def_property_readonly("n_terminals",
            [](const PyPalantirResult& r) { return r.n_terminals; })
        .def_property_readonly("k_eig",
            [](const PyPalantirResult& r) { return r.k_eig; })
        .def_property_readonly("pseudotime",
            [](const PyPalantirResult& r) {
                return py::array_t<float>({r.n_cells}, r.pseudotime.data());
            },
            "Diffusion pseudotime per cell (n_cells,) normalised to [0,1], numpy float32.")
        .def_property_readonly("branch_prob",
            [](const PyPalantirResult& r) {
                return py::array_t<float>(
                    {r.n_cells, r.n_terminals},
                    r.branch_prob.data());
            },
            "Branch probabilities (n_cells × n_terminals), numpy float32.")
        .def_property_readonly("eigenvalues",
            [](const PyPalantirResult& r) {
                return py::array_t<float>(
                    {r.k_eig}, r.eigenvalues.data());
            })
        .def_property_readonly("terminal_indices",
            [](const PyPalantirResult& r) {
                return py::array_t<int>(
                    {(int)r.terminal_indices.size()}, r.terminal_indices.data());
            })
        .def_property_readonly("eigenvectors_view",
            [](py::object self_obj) {
                auto& r = self_obj.cast<PyPalantirResult&>();
                return make_view_object<float>(
                    r.eigenvectors_dev.get(),
                    static_cast<std::size_t>(r.k_eig) * r.n_cells,
                    r.stream, self_obj);
            },
            "Diffusion eigenvectors (k_eig × n_cells, device) — __cuda_array_interface__.")
        .def("__repr__",
            [](const PyPalantirResult& r) {
                return "<PalantirResult n_cells=" + std::to_string(r.n_cells)
                     + " n_terminals=" + std::to_string(r.n_terminals)
                     + " k_eig=" + std::to_string(r.k_eig) + ">";
            });

    // --- cospar() -----------------------------------------------------------
    m.def("cospar",
        [](py::object mat_obj,
           py::array_t<int32_t>  clone_ids,
           py::array_t<int32_t>  time_labels,
           py::array_t<float>    embedding,
           int   k_neighbors,
           int   max_iters,
           float lambda1,
           float lambda2,
           float knn_sigma,
           float convergence_tol,
           int   n_fate_bias_steps,
           int   max_cells_per_timepoint,
           py::object stream_obj,
           uint64_t   seed) -> std::shared_ptr<PyCosparResult>
        {
            (void)seed;   // deterministic; kept for API consistency
            auto& py_csc = mat_obj.cast<PyDeviceCsc&>();

            singlet_gpu::fate::CosparConfig cfg;
            cfg.k_neighbors              = k_neighbors;
            cfg.max_iters                = max_iters;
            cfg.lambda1                  = lambda1;
            cfg.lambda2                  = lambda2;
            cfg.knn_sigma                = knn_sigma;
            cfg.convergence_tol          = convergence_tol;
            cfg.n_fate_bias_steps        = n_fate_bias_steps;
            cfg.max_cells_per_timepoint  = max_cells_per_timepoint;

            cudaStream_t stream = py_csc.producer_stream;
            if (!stream_obj.is_none()) {
                uintptr_t sptr = stream_obj.cast<uintptr_t>();
                stream = reinterpret_cast<cudaStream_t>(sptr);
            }

            // Build non-owning CSC view for the expression matrix.
            singlet_gpu::io::PzDeviceMatrix pz;
            pz.mat = singlet_gpu::core::DeviceCSC::from_device_ptrs(
                (int)py_csc.csc->rows, (int)py_csc.csc->cols, (int)py_csc.csc->nnz,
                py_csc.csc->col_ptr.data(),
                py_csc.csc->row_indices.data(),
                py_csc.csc->values.data());
            pz.producer_stream = stream;
            pz.host_retained = false;

            // Unpack numpy arrays (contiguous host buffers).
            auto ci = clone_ids.unchecked<1>();
            auto tl = time_labels.unchecked<1>();
            auto emb = embedding.unchecked<2>();  // (n_cells, n_pcs)

            const int n_cells = (int)tl.shape(0);
            const int n_pcs   = (int)emb.shape(1);
            (void)n_pcs;

            singlet_gpu::fate::CosparResult r =
                singlet_gpu::fate::run_cospar(
                    pz,
                    ci.data(0), (int)ci.shape(0),
                    tl.data(0), n_cells,
                    emb.data(0, 0), n_cells,
                    cfg, stream);

            auto out = std::make_shared<PyCosparResult>();
            out->transition_maps   = std::move(r.transition_maps);
            out->n_cells_t0        = std::move(r.n_cells_t0);
            out->n_cells_t1        = std::move(r.n_cells_t1);
            out->fate_bias         = std::move(r.fate_bias);
            out->potency_score     = std::move(r.potency_score);
            out->driver_genes      = std::move(r.driver_genes);
            out->final_residuals   = std::move(r.final_residuals);
            out->n_cells_total     = r.n_cells_total;
            out->n_terminal_fates  = r.n_terminal_fates;
            out->n_genes           = r.n_genes;
            out->iters_run         = r.iters_run;
            out->state_only_mode   = r.state_only_mode;
            out->stream            = stream;
            return out;
        },
        py::arg("mat"),
        py::arg("clone_ids"),
        py::arg("time_labels"),
        py::arg("embedding"),
        py::arg("k_neighbors")              = 50,
        py::arg("max_iters")                = 50,
        py::arg("lambda1")                  = 1e-3f,
        py::arg("lambda2")                  = 1.0f,
        py::arg("knn_sigma")                = 0.1f,
        py::arg("convergence_tol")          = 1e-4f,
        py::arg("n_fate_bias_steps")        = 5,
        py::arg("max_cells_per_timepoint")  = 30000,
        py::arg("stream")                   = py::none(),
        py::arg("seed")                     = 0ULL,
        R"doc(
        GPU-native CoSpar cell fate mapping (cycle-41 kernel).

        Parameters
        ----------
        mat : DeviceCsc
            Expression matrix (genes × cells) on device.
        clone_ids : np.ndarray int32, shape (n_cells,)
            Clone identifier per cell (-1 = no clone).
        time_labels : np.ndarray int32, shape (n_cells,)
            Integer time-point label per cell (0, 1, 2, ...).
        embedding : np.ndarray float32, shape (n_cells, n_pcs)
            Pre-computed PCA embedding.
        stream : int or None
            CUDA stream pointer (from cupy stream.ptr); None uses the mat's stream.
        seed : int
            Unused (algorithm is deterministic); kept for API consistency.

        Returns
        -------
        CosparResult
        )doc");

    // --- cellrank2() --------------------------------------------------------
    m.def("cellrank2",
        [](py::array_t<float>   transition_indptr,
           py::array_t<int32_t> transition_indices,
           py::array_t<float>   transition_data,
           int                  n_cells,
           py::array_t<int32_t> terminal_indices,
           int   gmres_m,
           int   gmres_max_restarts,
           double convergence_tol,
           int   ooc_batch_terminals,
           bool  compute_driver_genes,
           py::object mat_obj,
           py::object stream_obj,
           uint64_t   seed) -> std::shared_ptr<PyCellRank2Result>
        {
            (void)seed;

            cudaStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }
            if (!stream_obj.is_none() && mat_obj.is_none()) {
                // stream was given via stream= kwarg; nothing else to do.
            } else if (!mat_obj.is_none()) {
                auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
                if (stream == nullptr) stream = py_csc.producer_stream;
            }

            singlet_gpu::fate::CellRank2Config cfg;
            cfg.gmres_m              = gmres_m;
            cfg.gmres_max_restarts   = gmres_max_restarts;
            cfg.convergence_tol      = convergence_tol;
            cfg.ooc_batch_terminals  = ooc_batch_terminals;
            cfg.compute_driver_genes = compute_driver_genes;

            auto t_indptr  = transition_indptr.unchecked<1>();
            auto t_indices = transition_indices.unchecked<1>();
            auto t_data    = transition_data.unchecked<1>();
            auto t_idx     = terminal_indices.unchecked<1>();
            int n_terminals = (int)t_idx.shape(0);

            // Expression matrix (optional; needed for driver genes).
            singlet_gpu::io::PzDeviceMatrix* pz_ptr = nullptr;
            singlet_gpu::io::PzDeviceMatrix pz;
            if (!mat_obj.is_none()) {
                auto& py_csc = mat_obj.cast<PyDeviceCsc&>();
                pz.mat = singlet_gpu::core::DeviceCSC::from_device_ptrs(
                    (int)py_csc.csc->rows, (int)py_csc.csc->cols, (int)py_csc.csc->nnz,
                    py_csc.csc->col_ptr.data(),
                    py_csc.csc->row_indices.data(),
                    py_csc.csc->values.data());
                pz.producer_stream = stream;
                pz.host_retained   = false;
                pz_ptr = &pz;
            }

            singlet_gpu::fate::CellRank2Result r =
                singlet_gpu::fate::run_cellrank2(
                    t_indptr.data(0), t_indices.data(0), t_data.data(0),
                    n_cells, (int)(t_indptr.shape(0) - 1),
                    t_idx.data(0), n_terminals,
                    cfg, stream, pz_ptr);

            auto out = std::make_shared<PyCellRank2Result>();
            out->absorption_prob  = std::move(r.absorption_prob);
            out->driver_genes     = std::move(r.driver_genes);
            out->final_residuals  = std::move(r.final_residuals);
            out->restarts_used    = std::move(r.restarts_used);
            out->n_cells          = r.n_cells;
            out->n_terminals      = r.n_terminals;
            out->n_genes          = r.n_genes;
            out->converged_all    = r.converged_all;
            out->stream           = stream;
            return out;
        },
        py::arg("transition_indptr"),
        py::arg("transition_indices"),
        py::arg("transition_data"),
        py::arg("n_cells"),
        py::arg("terminal_indices"),
        py::arg("gmres_m")               = 30,
        py::arg("gmres_max_restarts")    = 10,
        py::arg("convergence_tol")       = 1e-6,
        py::arg("ooc_batch_terminals")   = -1,
        py::arg("compute_driver_genes")  = false,
        py::arg("mat")                   = py::none(),
        py::arg("stream")                = py::none(),
        py::arg("seed")                  = 0ULL,
        R"doc(
        GPU-native CellRank 2 absorption probability solver (cycle-43 kernel).

        Parameters
        ----------
        transition_indptr : np.ndarray int32, shape (n_cells+1,)
            CSR row pointers of the row-stochastic transition matrix.
        transition_indices : np.ndarray int32, shape (nnz,)
            CSR column indices.
        transition_data : np.ndarray float32, shape (nnz,)
            CSR values.
        n_cells : int
            Total cell count (rows = cols of T).
        terminal_indices : np.ndarray int32
            Indices of absorbing (terminal) states.
        mat : DeviceCsc or None
            Expression matrix for driver gene computation (optional).
        stream : int or None
            CUDA stream pointer.
        seed : int
            Unused (deterministic); kept for API consistency.

        Returns
        -------
        CellRank2Result
        )doc");

    // --- palantir() ---------------------------------------------------------
    m.def("palantir",
        [](py::array_t<float>   embedding,
           int                  start_cell,
           py::array_t<int32_t> terminal_indices,
           int   k_neighbors,
           int   n_pcs,
           int   k_eig,
           bool  auto_terminals,
           int   n_terminals_auto,
           int   gmres_m,
           int   gmres_max_restarts,
           double gmres_convergence_tol,
           int   gmres_ooc_batch_terminals,
           bool  normalize_pseudotime,
           py::object stream_obj,
           uint64_t   seed) -> std::shared_ptr<PyPalantirResult>
        {
            cudaStream_t stream = nullptr;
            if (!stream_obj.is_none()) {
                stream = reinterpret_cast<cudaStream_t>(
                    stream_obj.cast<uintptr_t>());
            }

            singlet_gpu::fate::PalantirConfig cfg;
            cfg.k_neighbors              = k_neighbors;
            cfg.n_pcs                    = n_pcs;
            cfg.k_eig                    = k_eig;
            cfg.seed                     = seed;
            cfg.auto_terminals           = auto_terminals;
            cfg.n_terminals_auto         = n_terminals_auto;
            cfg.gmres_m                  = gmres_m;
            cfg.gmres_max_restarts       = gmres_max_restarts;
            cfg.gmres_convergence_tol    = gmres_convergence_tol;
            cfg.gmres_ooc_batch_terminals = gmres_ooc_batch_terminals;
            cfg.normalize_pseudotime     = normalize_pseudotime;

            auto emb = embedding.unchecked<2>();
            int n_cells = (int)emb.shape(0);

            auto t_idx = terminal_indices.unchecked<1>();
            int n_terminals = (int)t_idx.shape(0);

            singlet_gpu::fate::PalantirResult r =
                singlet_gpu::fate::run_palantir(
                    emb.data(0, 0), n_cells, (int)emb.shape(1),
                    start_cell,
                    t_idx.data(0), n_terminals,
                    cfg, stream);

            auto out = std::make_shared<PyPalantirResult>();
            out->pseudotime       = std::move(r.pseudotime);
            out->branch_prob      = std::move(r.branch_prob);
            out->eigenvalues      = std::move(r.eigenvalues);
            out->terminal_indices = std::move(r.terminal_indices);
            out->eigenvectors_dev = std::move(r.eigenvectors_dev);
            out->n_cells          = r.n_cells;
            out->n_terminals      = r.n_terminals;
            out->k_eig            = r.k_eig;
            out->stream           = stream;
            return out;
        },
        py::arg("embedding"),
        py::arg("start_cell"),
        py::arg("terminal_indices")           = py::array_t<int32_t>(),
        py::arg("k_neighbors")                = 30,
        py::arg("n_pcs")                      = 30,
        py::arg("k_eig")                      = 20,
        py::arg("auto_terminals")             = false,
        py::arg("n_terminals_auto")           = 5,
        py::arg("gmres_m")                    = 30,
        py::arg("gmres_max_restarts")         = 10,
        py::arg("gmres_convergence_tol")      = 1e-6,
        py::arg("gmres_ooc_batch_terminals")  = -1,
        py::arg("normalize_pseudotime")       = true,
        py::arg("stream")                     = py::none(),
        py::arg("seed")                       = 42ULL,
        R"doc(
        GPU-native Palantir diffusion pseudotime + branch probabilities (cycle-45 kernel).

        Parameters
        ----------
        embedding : np.ndarray float32, shape (n_cells, n_pcs)
            PCA embedding.
        start_cell : int
            Index of the root / start cell.
        terminal_indices : np.ndarray int32
            Terminal cell indices.  If empty and auto_terminals=False, auto-detects.
        seed : int
            Seed for Lanczos randomised initialisation.
        stream : int or None
            CUDA stream pointer.

        Returns
        -------
        PalantirResult
        )doc");
}

// ---------------------------------------------------------------------------
// bind_fate_module — creates "fate" submodule and delegates to bind_fate.
// ---------------------------------------------------------------------------
inline void bind_fate_module(py::module_& parent) {
    auto m = parent.def_submodule("fate",
        "GPU-native cell fate trajectory kernels: cospar, cellrank2, palantir.");
    bind_fate(m);
}

} // namespace python
} // namespace singlet_gpu
