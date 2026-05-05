// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_singlet_gpu_core.cpp
//
// pybind11 entry point for the singlet_gpu._core extension module.
//
// Exposes the minimal public API needed to:
//   1. Load a .1pz file to GPU device (load_pz → PzDeviceMatrix).
//   2. Inspect the embedded GEO metadata (Metadata).
//   3. Build zero-copy cupy views of device arrays (DeviceCsc.data_view, etc.).
//   4. Optionally copy device data to host for scipy/numpy users (to_host()).
//
// Architecture:
//   - This file contains only PYBIND11_MODULE and the module docstring.
//   - Actual bindings are split across _bind_loader.hpp, _bind_metadata.hpp,
//     and _cupy_interop.hpp for clarity and manageable diff size per cycle.
//   - The Python package singlet_gpu/ re-exports from _core via __init__.py.
//
// Compile flags:
//   -fvisibility=hidden  (pybind11 requirement — set in python/CMakeLists.txt)
//   FACTORNET_HAS_GPU=1  (set by singlet-gpu::singlet-gpu INTERFACE target)
//
// Build: cmake -S singlet-gpu/ -B build -DSINGLET_GPU_BUILD_PYTHON=ON
//        cmake --build build -j --target _core
//
// Usage after install:
//   from singlet_gpu._core import DeviceCsc, Metadata, PzDeviceMatrix, load_pz

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Binding helpers — each registers one class / function group.
#include "_bind_metadata.hpp"
#include "_bind_loader.hpp"

// _cupy_interop.hpp is included transitively through _bind_loader.hpp;
// no direct include needed here.

// Cycle 20 extensions: cupy ingest + all per-kernel bindings.
// _bind_loader.hpp must be included first (defines PyDeviceCsc).
#include "_bind_cupy_ingest.hpp"
#include "_bind_kernels.hpp"

// Cycle 22 extensions: result classes for cycles 7-17 + extended kernel bindings.
// _bind_results.hpp must be included after _bind_kernels.hpp (uses Py* structs
// defined in that file) and before PYBIND11_MODULE so that the py::class_<>
// registrations in bind_results() run before bind_kernels_ext().
// Note: _bind_results.hpp is already #included transitively through
// _bind_kernels.hpp (which includes it via its own #include).
// bind_kernels_ext is defined at file scope inside _bind_kernels.hpp.

// Cycle 52a: genomics/SNP domain bindings (grn, eqtl, ase, cna, variants, atac).
// CYCLE-107 (2026-04-29): all of these are deferred-indefinitely per state/roadmap.md
// and still pull factornet headers transitively. Gated behind SINGLET_GPU_BUILD_DEFERRED
// to keep the foundational `_core.so` build factornet-independent.
#ifdef SINGLET_GPU_BUILD_DEFERRED
#  include "_bind_grn.hpp"        // GRaNIE
#  include "_bind_eqtl.hpp"       // NEBULA
#  include "_bind_ase.hpp"        // DAESC
#  include "_bind_cna.hpp"        // Numbat
#  include "_bind_variants.hpp"   // Monopogen
#  include "_bind_atac.hpp"       // chromVAR
#endif

// Cycle 52a Tier 1: fate, comm, network, abundance, disease (also deferred).
#ifdef SINGLET_GPU_BUILD_DEFERRED
#  include "_bind_fate.hpp"          // Cospar, CellRank2, Palantir
#  include "_bind_comm.hpp"          // CellChat
#  include "_bind_network.hpp"       // hdWGCNA
#  include "_bind_abundance.hpp"     // Milo
#  include "_bind_disease.hpp"       // scDRS
#endif

// Cycle 52a Tier 3: spatial Phase B, generative, perturbation, enrich (also deferred).
#ifdef SINGLET_GPU_BUILD_DEFERRED
#  include "_bind_spatial_phaseb.hpp"  // FlashDeconv, STAGATE, Cell2Fate
#  include "_bind_generative.hpp"      // DiscreteDiffusion
#  include "_bind_perturbation.hpp"    // PerturbGraph
#  include "_bind_enrich.hpp"          // ssGSEA, PROGENy
#endif

#include "_bind_qc_new.hpp"          // doublet_score (foundational); OmniDoublet gated within
// CSI-GEP (cycle 28) is in the deferred-indefinitely scope per state/roadmap.md.
// Its binding pulls factornet/gpu/loss.cuh which uses unqualified `min` — fails
// under g++ with C++20. Gated behind SINGLET_GPU_BUILD_DEFERRED until either
// factornet upstream qualifies the calls or scope reopens.
#ifdef SINGLET_GPU_BUILD_DEFERRED
#  include "_bind_nmf_new.hpp"       // CsiGepResult
#endif

// Cycle 103: QC metrics + filter + preprocess scale/regress_out bindings.
#include "_bind_qc_metrics.hpp"      // QcResult, calculate_qc_metrics, filter_cells, filter_genes
#include "_bind_preprocess.hpp"      // DenseResult, scale, regress_out

// Cycle 186: score_genes pybind11 binding (enrich/score_genes.h, cycle 129).
// Not gated behind SINGLET_GPU_BUILD_DEFERRED — score_genes.h is factornet-free
// and scored Phase E benchmark; Rule 26 requires wrappers within 2 cycles.
#include "_bind_score_genes.hpp"     // ScoreGenesResult, score_genes

namespace py = pybind11;

// ---------------------------------------------------------------------------
// PYBIND11_MODULE(_core, m)
//
// All public API lives at the _core module level.  The singlet_gpu Python
// package re-exports these names in __init__.py so users only need:
//   import singlet_gpu
//   m = singlet_gpu.load_pz("counts.1pz")
// ---------------------------------------------------------------------------
PYBIND11_MODULE(_core, m) {
    m.doc() = R"doc(
    singlet_gpu._core — GPU-native single-cell analysis, pybind11 extension.

    This module is compiled against singlet-gpu's header-only C++ library and
    exposes zero-copy access to `.1pz` sparse matrices on the GPU device.

    Public API
    ----------
    load_pz(path, keep_host_pinned=False) -> PzDeviceMatrix
        Load a singlify `.1pz` file to GPU device.  Returns a :class:`PzDeviceMatrix`
        containing a :class:`DeviceCsc` and a :class:`Metadata` object.

    DeviceCsc
        A CSC sparse matrix resident on the GPU.  Provides:
        - ``rows``, ``cols``, ``nnz``, ``dtype``, ``device`` properties.
        - ``data_view``, ``indices_view``, ``indptr_view``: __cuda_array_interface__
          dicts for zero-copy cupy integration.
        - ``to_host()``: explicit device → scipy.sparse.csr_matrix copy.

    Metadata
        GEO metadata embedded in the .1pz file (gsm_id, gse_id, organism, etc.).
        Also provides ``rownames`` (genes) and ``colnames`` (barcodes).

    PzDeviceMatrix
        Combined result: ``.mat`` (DeviceCsc) + ``.meta`` (Metadata).
        Convenience pass-throughs: ``.rows``, ``.cols``, ``.nnz``.

    Zero-copy cupy usage
    --------------------
    ::

        import singlet_gpu, cupy.sparse as csp
        m = singlet_gpu.load_pz("exon_counts.1pz")
        csr = csp.csr_matrix(
            (m.mat.data_view, m.mat.indices_view, m.mat.indptr_view),
            shape=(m.rows, m.cols)
        )

    Lifetime / memory management
    ----------------------------
    Device memory is managed by factornet's RAII pool (``DeviceMemory<T>``).
    The Python GC calls the C++ destructor when the last reference is dropped.
    cupy views hold a reference back to the ``DeviceCsc`` object via the
    ``_owner`` key in the ``__cuda_array_interface__`` dict, keeping device
    memory alive until all views are collected.

    No host data is ever copied implicitly.  Only ``DeviceCsc.to_host()``
    issues a ``cudaMemcpy`` to the host.
    )doc";

    // Module version — mirrors singlet_gpu/version.py.
    m.attr("__version__") = "0.1.0";

    // -----------------------------------------------------------------------
    // Register Metadata class.
    // -----------------------------------------------------------------------
    bind_metadata(m);

    // -----------------------------------------------------------------------
    // Register DeviceCsc class.
    // -----------------------------------------------------------------------
    bind_device_csc(m);

    // -----------------------------------------------------------------------
    // Register PzDeviceMatrix class + load_pz function.
    // -----------------------------------------------------------------------
    bind_pz_device_matrix(m);

    // -----------------------------------------------------------------------
    // CYCLE-275: PzDataLoader (streaming loader for nmf_chunked + future
    // streaming kernels).
    // -----------------------------------------------------------------------
    bind_pz_data_loader(m);

    // -----------------------------------------------------------------------
    // Cycle 20: cupy ingest functions.
    // -----------------------------------------------------------------------
    m.def("from_cupy_csr",
        &singlet_gpu::python::from_cupy_csr,
        py::arg("csr_matrix"),
        R"doc(
        Build a DeviceCsc from a cupy.sparse.csr_matrix (zero-copy).

        The resulting DeviceCsc holds non-owning device pointers into the
        cupy matrix's device buffers.  The source csr_matrix is kept alive
        by the DeviceCsc's internal reference; it will NOT be garbage-collected
        until all Python references to the DeviceCsc (and any cupy views built
        from it) are also dropped.

        Parameters
        ----------
        csr_matrix : cupy.sparse.csr_matrix or csc_matrix
            Source sparse matrix (int32 indptr/indices, float32 data).

        Returns
        -------
        DeviceCsc
            Zero-copy device view of the input matrix.
        )doc");

    m.def("to_cupy_csr",
        [](py::object csc_obj) { return singlet_gpu::python::to_cupy_csr(csc_obj); },
        py::arg("device_csc"),
        R"doc(
        Return a Python dict with __cuda_array_interface__ views of a DeviceCsc.

        Usage::

            import cupy.sparse as csp
            d = _core.to_cupy_csr(dev_csc)
            csc = csp.csc_matrix((d["data"], d["indices"], d["indptr"]),
                                  shape=d["shape"])

        Returns
        -------
        dict
            Keys: "indptr", "indices", "data" (__cuda_array_interface__ dicts),
            "shape" (rows, cols) tuple, "_owner" lifetime anchor.
        )doc");

    // -----------------------------------------------------------------------
    // Cycle 20: per-kernel function bindings (cycles 1-6).
    // -----------------------------------------------------------------------
    singlet_gpu::python::bind_kernels(m);

    // -----------------------------------------------------------------------
    // Cycle 22: result classes for cycles 7-17.
    // Must be registered BEFORE bind_kernels_ext so that Python can resolve
    // the result types when the m.def(...)  return-type annotations are built.
    // -----------------------------------------------------------------------
    singlet_gpu::python::bind_results(m);

    // -----------------------------------------------------------------------
    // Cycle 22: per-kernel function bindings for cycles 7-17.
    // -----------------------------------------------------------------------
    singlet_gpu::python::bind_kernels_ext(m);

    // -----------------------------------------------------------------------
    // CYCLE-107: deferred-indefinitely domain bindings — gated.
    // Per state/roadmap.md, all of these features are deferred indefinitely
    // and still pull factornet headers transitively. Build them only when
    // SINGLET_GPU_BUILD_DEFERRED is set (default OFF).
    // -----------------------------------------------------------------------
#ifdef SINGLET_GPU_BUILD_DEFERRED
    singlet_gpu::python::bind_cna(m);      // Numbat
    singlet_gpu::python::bind_atac(m);     // chromVAR
    singlet_gpu::python::bind_grn(m);      // GRaNIE
    singlet_gpu::python::bind_eqtl(m);     // NEBULA
    singlet_gpu::python::bind_ase(m);      // DAESC
    singlet_gpu::python::bind_variants(m); // Monopogen

    singlet_gpu::python::bind_fate_module(m);       // Cospar, CellRank2, Palantir
    singlet_gpu::python::bind_comm_module(m);       // CellChat
    singlet_gpu::python::bind_network_module(m);    // hdWGCNA
    singlet_gpu::python::bind_abundance_module(m);  // Milo
    singlet_gpu::python::bind_disease_module(m);    // scDRS

    singlet_gpu::python::bind_spatial_phaseb(m);    // FlashDeconv, STAGATE, Cell2Fate
    singlet_gpu::python::bind_generative(m);        // DiscreteDiffusion
    singlet_gpu::python::bind_perturbation(m);      // PerturbGraph
    singlet_gpu::python::bind_enrich(m);            // ssGSEA, PROGENy
#endif

    singlet_gpu::python::bind_qc_new(m);           // doublet_score (foundational); OmniDoublet gated within
#ifdef SINGLET_GPU_BUILD_DEFERRED
    singlet_gpu::python::bind_nmf_new(m);          // csi_gep — deferred-indefinitely scope
#endif

    // -----------------------------------------------------------------------
    // Cycle 103: QC metrics + filtering + scale/regress_out.
    // bind_qc_metrics must come before bind_preprocess (no dependency, but
    // QcResult class registration should precede DenseResult for stable
    // Python help() output ordering).
    // -----------------------------------------------------------------------
    singlet_gpu::python::bind_qc_metrics(m);   // QcResult, calculate_qc_metrics, filter_cells, filter_genes
    singlet_gpu::python::bind_preprocess(m);   // DenseResult, scale, regress_out

    // -----------------------------------------------------------------------
    // Cycle 186: score_genes (enrich/score_genes.h, cycle 129 kernel).
    // Ungated — score_genes.h carries no factornet dependency.
    // -----------------------------------------------------------------------
    singlet_gpu::python::bind_score_genes(m);  // ScoreGenesResult, score_genes

    // -----------------------------------------------------------------------
    // Cycle 20: result classes (cycles 1-6).
    // Each result class exposes device buffers via _view properties returning
    // __cuda_array_interface__ dicts (same pattern as DeviceCsc.data_view).
    // -----------------------------------------------------------------------

    // NormalizeResult — returned by normalize_total.
    py::class_<PyNormalizeResult, std::shared_ptr<PyNormalizeResult>>(m, "NormalizeResult",
        "Result of normalize_total.  Device float32 buffers exposed zero-copy.")
    .def_property_readonly("target_used",
        [](const PyNormalizeResult& r) { return r.target_used; },
        "The library-size target T actually applied.")
    .def_property_readonly("size_factors_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyNormalizeResult&>();
            return make_view_object<float>(
                r.size_factors.data(),
                r.size_factors.size(),
                r.stream, self_obj);
        },
        "Device float32 per-cell size factors (n_cells,) — __cuda_array_interface__ dict.")
    .def_property_readonly("qc_mask_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyNormalizeResult&>();
            return make_view_object<uint8_t>(
                r.qc_mask.data(),
                r.qc_mask.size(),
                r.stream, self_obj);
        },
        "Device uint8 QC mask (n_cells,): 1 = zero-count cell — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyNormalizeResult& r) {
            return "<NormalizeResult target_used=" + std::to_string(r.target_used)
                 + " n_cells=" + std::to_string(r.size_factors.size()) + ">";
        });

    // HvgResult — returned by highly_variable_genes.
    py::class_<PyHvgResult, std::shared_ptr<PyHvgResult>>(m, "HvgResult",
        "Result of highly_variable_genes.  Device buffers exposed zero-copy.")
    .def_property_readonly("indices_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyHvgResult&>();
            return make_view_object<int32_t>(
                r.indices.data(), r.indices.size(), r.stream, self_obj);
        },
        "Top-N gene indices (int32) — __cuda_array_interface__ dict.")
    .def_property_readonly("scores_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyHvgResult&>();
            return make_view_object<float>(
                r.scores.data(), r.scores.size(), r.stream, self_obj);
        },
        "HVG scores (float32, top_n) — __cuda_array_interface__ dict.")
    .def_property_readonly("mean_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyHvgResult&>();
            return make_view_object<float>(
                r.mean.data(), r.mean.size(), r.stream, self_obj);
        },
        "Per-gene mean expression (float32, n_genes) — __cuda_array_interface__ dict.")
    .def_property_readonly("var_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyHvgResult&>();
            return make_view_object<float>(
                r.var.data(), r.var.size(), r.stream, self_obj);
        },
        "Per-gene variance (float32, n_genes) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyHvgResult& r) {
            return "<HvgResult top_n=" + std::to_string(r.indices.size()) + ">";
        });

    // SvdResult — returned by pca, svd_auto_select, svd_lanczos, etc.
    py::class_<PySvdResult, std::shared_ptr<PySvdResult>>(m, "SvdResult",
        "Result of SVD/PCA.  U (m×k), d (k), V (n×k) on device.")
    .def_property_readonly("k_selected",
        [](const PySvdResult& r) { return r.k_selected; },
        "Number of components selected/computed.")
    .def_property_readonly("U_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PySvdResult&>();
            return make_view_object<float>(
                r.d_U.data(),
                static_cast<std::size_t>(r.rows * r.k_selected),
                r.stream, self_obj);
        },
        "Left singular vectors U (rows × k, col-major float32) — __cuda_array_interface__.")
    .def_property_readonly("d_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PySvdResult&>();
            return make_view_object<float>(
                r.d_d.data(),
                static_cast<std::size_t>(r.k_selected),
                r.stream, self_obj);
        },
        "Singular values d (k float32) — __cuda_array_interface__ dict.")
    .def_property_readonly("V_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PySvdResult&>();
            return make_view_object<float>(
                r.d_V.data(),
                static_cast<std::size_t>(r.cols * r.k_selected),
                r.stream, self_obj);
        },
        "Right singular vectors V (cols × k, col-major float32) — __cuda_array_interface__.")
    .def("__repr__",
        [](const PySvdResult& r) {
            return "<SvdResult k=" + std::to_string(r.k_selected)
                 + " rows=" + std::to_string(r.rows)
                 + " cols=" + std::to_string(r.cols) + ">";
        });

    // NmfResult — returned by nmf, nmf_chunked, nmf_graph_factorize.
    py::class_<PyNmfResult, std::shared_ptr<PyNmfResult>>(m, "NmfResult",
        "Result of NMF.  W (m×k), d (k), H (k×n) on device.")
    .def_property_readonly("k_used",
        [](const PyNmfResult& r) { return r.k_used; })
    .def_property_readonly("iterations",
        [](const PyNmfResult& r) { return r.iterations; })
    .def_property_readonly("converged",
        [](const PyNmfResult& r) { return r.converged; })
    .def_property_readonly("W_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyNmfResult&>();
            return make_view_object<float>(
                r.d_W.data(), r.d_W.size(), r.stream, self_obj);
        },
        "W factor matrix (m × k, col-major float32) — __cuda_array_interface__ dict.")
    .def_property_readonly("d_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyNmfResult&>();
            return make_view_object<float>(
                r.d_d.data(), r.d_d.size(), r.stream, self_obj);
        },
        "Scale factors d (k float32) — __cuda_array_interface__ dict.")
    .def_property_readonly("H_view",
        [](py::object self_obj) {
            auto& r = self_obj.cast<PyNmfResult&>();
            return make_view_object<float>(
                r.d_H.data(), r.d_H.size(), r.stream, self_obj);
        },
        "H factor matrix (k × n float32) — __cuda_array_interface__ dict.")
    .def("__repr__",
        [](const PyNmfResult& r) {
            return "<NmfResult k=" + std::to_string(r.k_used)
                 + " iter=" + std::to_string(r.iterations)
                 + " converged=" + (r.converged ? "True" : "False") + ">";
        });

    // -----------------------------------------------------------------------
    // Module-level docstring aliases (convenience re-exports).
    // The actual load_pz function is registered inside bind_pz_device_matrix.
    // -----------------------------------------------------------------------

    // Expose CUDA compute capability and device count at import time so users
    // can do a quick sanity check without importing cupy.
    m.def("device_count", []() -> int {
        int n = 0;
        cudaError_t err = cudaGetDeviceCount(&n);
        if (err == cudaErrorNoDevice) return 0;
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("cudaGetDeviceCount: ") + cudaGetErrorString(err));
        return n;
    }, "Return the number of CUDA-capable devices visible to the process.");

    m.def("cuda_device_name", [](int device) -> std::string {
        cudaDeviceProp prop{};
        cudaError_t err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("cudaGetDeviceProperties: ") + cudaGetErrorString(err));
        return std::string(prop.name);
    }, py::arg("device") = 0,
       "Return the name of the specified CUDA device (default: device 0).");

    m.def("current_device", []() -> int {
        int dev = 0;
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("cudaGetDevice: ") + cudaGetErrorString(err));
        return dev;
    }, "Return the index of the current CUDA device.");

    m.def("set_device", [](int device) {
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess)
            throw std::runtime_error(
                std::string("cudaSetDevice: ") + cudaGetErrorString(err));
    }, py::arg("device"),
       "Set the current CUDA device.  Must be called before load_pz if you "
       "want data on a specific device.");
}
