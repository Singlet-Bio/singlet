// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_qc_metrics.hpp
//
// CYCLE-103 — pybind11 bindings for:
//   qc::calculate_qc_metrics  (metrics.h:468, DeviceCSC overload)
//   qc::filter_cells          (metrics.h:490)
//   qc::filter_genes          (metrics.h:585)
//
// PyQcResult owns the 7 device arrays returned by calculate_qc_metrics and
// exposes them as __cuda_array_interface__ views (zero-copy cupy protocol).
//
// filter_cells / filter_genes return a new DeviceCsc (PyDeviceCsc), allowing
// the result to be passed back through the same cupy interop chain as load_pz.
//
// WHY a dedicated file rather than extending _bind_qc_new.hpp:
//   _bind_qc_new.hpp already handles doublet detection (Tier-3 Cycle-52a);
//   mixing filter / metrics bindings there would grow it past 400 LOC and
//   risk conflicting include-order with _bind_loader.hpp's PyDeviceCsc.
//   A clean split keeps each file under ~250 LOC and independently auditable.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <singlet/gpu/qc/metrics.h>
#include <singlet/gpu/core/types.h>
#include <singlet/gpu/core/memory.h>

#include "_cupy_interop.hpp"
#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <memory>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyQcResult — mirrors QcResult's 7 DeviceMemory fields + scalar metadata.
// ---------------------------------------------------------------------------
struct PyQcResult {
    singlet::gpu::core::DeviceMemory<float> n_umis;        // [n_cells]
    singlet::gpu::core::DeviceMemory<int>   n_genes;       // [n_cells]
    singlet::gpu::core::DeviceMemory<float> pct_mt;        // [n_cells]
    singlet::gpu::core::DeviceMemory<float> pct_ribo;      // [n_cells]
    singlet::gpu::core::DeviceMemory<float> gene_mean;     // [n_genes]
    singlet::gpu::core::DeviceMemory<float> gene_var;      // [n_genes]
    singlet::gpu::core::DeviceMemory<int>   gene_n_cells;  // [n_genes]
    int n_cells       = 0;
    int n_genes_total = 0;
    cudaStream_t stream = nullptr;
};

// ---------------------------------------------------------------------------
// Helper: build a non-owning QcResult view from a PyQcResult.
//
// WHY DeviceMemory::wrap: filter_cells / filter_genes only READ from QcResult;
// we need to pass a const QcResult& without moving memory out of PyQcResult
// (which must keep its fields alive for the duration of the Python object).
// DeviceMemory::wrap creates a non-owning DeviceMemory that will NOT free the
// pointer on destruction — exactly what we need for a temporary view.
// ---------------------------------------------------------------------------
inline singlet::gpu::qc::QcResult make_qc_view(const PyQcResult& py_r) {
    singlet::gpu::qc::QcResult v;
    // Non-owning wraps: destructor is a no-op (owns_=false in factornet impl).
    v.n_umis       = singlet::gpu::core::DeviceMemory<float>::wrap(
                         const_cast<float*>(py_r.n_umis.get()), py_r.n_umis.size());
    v.n_genes      = singlet::gpu::core::DeviceMemory<int>::wrap(
                         const_cast<int*>(py_r.n_genes.get()), py_r.n_genes.size());
    v.pct_mt       = singlet::gpu::core::DeviceMemory<float>::wrap(
                         const_cast<float*>(py_r.pct_mt.get()), py_r.pct_mt.size());
    v.pct_ribo     = singlet::gpu::core::DeviceMemory<float>::wrap(
                         const_cast<float*>(py_r.pct_ribo.get()), py_r.pct_ribo.size());
    v.gene_mean    = singlet::gpu::core::DeviceMemory<float>::wrap(
                         const_cast<float*>(py_r.gene_mean.get()), py_r.gene_mean.size());
    v.gene_var     = singlet::gpu::core::DeviceMemory<float>::wrap(
                         const_cast<float*>(py_r.gene_var.get()), py_r.gene_var.size());
    v.gene_n_cells = singlet::gpu::core::DeviceMemory<int>::wrap(
                         const_cast<int*>(py_r.gene_n_cells.get()), py_r.gene_n_cells.size());
    v.n_cells       = py_r.n_cells;
    v.n_genes_total = py_r.n_genes_total;
    return v;
}

// ---------------------------------------------------------------------------
// Helper: resolve cudaStream_t from Python int or None.
// ---------------------------------------------------------------------------
inline cudaStream_t resolve_stream_metrics(py::object stream_obj) {
    if (stream_obj.is_none()) return nullptr;
    return reinterpret_cast<cudaStream_t>(py::cast<uintptr_t>(stream_obj));
}

// ---------------------------------------------------------------------------
// bind_qc_metrics — called from PYBIND11_MODULE(_core, m)
// ---------------------------------------------------------------------------
inline void bind_qc_metrics(py::module_& m) {

    // ── QcResult Python class ──────────────────────────────────────────────
    py::class_<PyQcResult, std::shared_ptr<PyQcResult>>(
        m, "QcResult",
        "Per-cell and per-gene QC metrics (cycle-103, qc::calculate_qc_metrics).")
    .def_property_readonly("n_cells",
        [](const PyQcResult& r){ return r.n_cells; })
    .def_property_readonly("n_genes_total",
        [](const PyQcResult& r){ return r.n_genes_total; })
    // per-cell views
    .def_property_readonly("n_umis_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<float>(
                r.n_umis.get(), r.n_umis.size(), r.stream, self_obj);
        }, "[n_cells] float32 total UMI per cell — __cuda_array_interface__.")
    .def_property_readonly("n_genes_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<int>(
                r.n_genes.get(), r.n_genes.size(), r.stream, self_obj);
        }, "[n_cells] int32 detected genes per cell — __cuda_array_interface__.")
    .def_property_readonly("pct_mt_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<float>(
                r.pct_mt.get(), r.pct_mt.size(), r.stream, self_obj);
        }, "[n_cells] float32 %% mitochondrial UMIs — __cuda_array_interface__.")
    .def_property_readonly("pct_ribo_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<float>(
                r.pct_ribo.get(), r.pct_ribo.size(), r.stream, self_obj);
        }, "[n_cells] float32 %% ribosomal UMIs — __cuda_array_interface__.")
    // per-gene views
    .def_property_readonly("gene_mean_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<float>(
                r.gene_mean.get(), r.gene_mean.size(), r.stream, self_obj);
        }, "[n_genes] float32 per-gene mean expression — __cuda_array_interface__.")
    .def_property_readonly("gene_var_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<float>(
                r.gene_var.get(), r.gene_var.size(), r.stream, self_obj);
        }, "[n_genes] float32 per-gene variance (N-1 denom) — __cuda_array_interface__.")
    .def_property_readonly("gene_n_cells_view",
        [](py::object self_obj){
            auto& r = self_obj.cast<PyQcResult&>();
            return make_view_object<int>(
                r.gene_n_cells.get(), r.gene_n_cells.size(), r.stream, self_obj);
        }, "[n_genes] int32 cells expressing each gene — __cuda_array_interface__.")
    .def("__repr__",
        [](const PyQcResult& r){
            return "<QcResult n_cells=" + std::to_string(r.n_cells)
                 + " n_genes=" + std::to_string(r.n_genes_total) + ">";
        });

    // ── calculate_qc_metrics ────────────────────────────────────────────────
    m.def("calculate_qc_metrics",
        [](const PyDeviceCsc& mat,
           py::object is_mt_obj,
           py::object is_ribo_obj,
           py::object stream_obj,
           bool deterministic) -> py::object
        {
            cudaStream_t s = resolve_stream_metrics(stream_obj);

            // Unwrap is_mt and is_ribo via __cuda_array_interface__ (uint8 masks).
            auto mt_iface = is_mt_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto mt_ptr = reinterpret_cast<const uint8_t*>(
                py::cast<uintptr_t>(mt_iface["data"].cast<py::tuple>()[0]));
            auto mt_len = py::cast<std::size_t>(mt_iface["shape"].cast<py::tuple>()[0]);

            auto rb_iface = is_ribo_obj.attr("__cuda_array_interface__").cast<py::dict>();
            auto rb_ptr = reinterpret_cast<const uint8_t*>(
                py::cast<uintptr_t>(rb_iface["data"].cast<py::tuple>()[0]));

            int n_genes = mat.csc->rows;
            if (static_cast<int>(mt_len) != n_genes)
                throw std::runtime_error(
                    "calculate_qc_metrics: is_mt length (" +
                    std::to_string(mt_len) + ") != mat.rows (" +
                    std::to_string(n_genes) + ")");

            singlet::gpu::qc::QcConfig cfg;
            cfg.deterministic = deterministic;

            // Call the pointer-level overload (avoiding DeviceMemory copy issues).
            auto r = singlet::gpu::qc::calculate_qc_metrics(
                mat.csc->col_ptr.get(),
                mat.csc->row_indices.get(),
                mat.csc->values.get(),
                mat.csc->cols,   // n_cells
                mat.csc->rows,   // n_genes
                mt_ptr, rb_ptr,
                s, cfg);

            auto py_r = std::make_shared<PyQcResult>();
            py_r->n_umis       = std::move(r.n_umis);
            py_r->n_genes      = std::move(r.n_genes);
            py_r->pct_mt       = std::move(r.pct_mt);
            py_r->pct_ribo     = std::move(r.pct_ribo);
            py_r->gene_mean    = std::move(r.gene_mean);
            py_r->gene_var     = std::move(r.gene_var);
            py_r->gene_n_cells = std::move(r.gene_n_cells);
            py_r->n_cells      = r.n_cells;
            py_r->n_genes_total = r.n_genes_total;
            py_r->stream       = s;
            return py::cast(py_r);
        },
        py::arg("mat"),
        py::arg("is_mt"),
        py::arg("is_ribo"),
        py::arg("stream")        = py::none(),
        py::arg("deterministic") = false,
        R"doc(
        Compute per-cell and per-gene QC metrics entirely on device (cycle-103).

        Parameters
        ----------
        mat : DeviceCsc
            genes × cells count matrix (int/float values).
        is_mt : cuda array (uint8, n_genes)
            Mitochondrial gene mask — 1 = MT gene, 0 = not.
        is_ribo : cuda array (uint8, n_genes)
            Ribosomal gene mask — 1 = ribo gene, 0 = not.
        stream : int or None
        deterministic : bool
            True = two-pass gene variance path (bit-identical, ~1.5× slower).

        Returns
        -------
        QcResult
            Holds n_umis, n_genes, pct_mt, pct_ribo (per-cell) and
            gene_mean, gene_var, gene_n_cells (per-gene).  Access via
            ``*_view`` properties returning __cuda_array_interface__ dicts.
        )doc");

    // ── filter_cells ────────────────────────────────────────────────────────
    m.def("filter_cells",
        [](const PyDeviceCsc& mat,
           const PyQcResult&  qc,
           float min_genes,
           float max_genes,
           float min_umis,
           float max_umis,
           float max_pct_mt,
           py::object stream_obj) -> py::object
        {
            cudaStream_t s = resolve_stream_metrics(stream_obj);

            singlet::gpu::qc::FilterConfig cfg;
            cfg.min_genes  = min_genes;
            cfg.max_genes  = max_genes;
            cfg.min_umis   = min_umis;
            cfg.max_umis   = max_umis;
            cfg.max_pct_mt = max_pct_mt;

            // Build non-owning QcResult view so filter_cells can read the device arrays.
            singlet::gpu::qc::QcResult qc_view = make_qc_view(qc);

            auto out_csc = singlet::gpu::qc::filter_cells(*mat.csc, qc_view, cfg, s);

            auto py_out = std::make_shared<PyDeviceCsc>(
                std::make_shared<singlet::gpu::core::DeviceCSC>(std::move(out_csc)),
                s);
            return py::cast(py_out);
        },
        py::arg("mat"),
        py::arg("qc"),
        py::arg("min_genes") = 200.0f,
        py::arg("max_genes") = std::numeric_limits<float>::infinity(),
        py::arg("min_umis")  = 0.0f,
        py::arg("max_umis")  = std::numeric_limits<float>::infinity(),
        py::arg("max_pct_mt") = 100.0f,
        py::arg("stream")    = py::none(),
        R"doc(
        Filter cells by QC thresholds; returns a new DeviceCsc (cycle-103).

        One-time D2H transfer of the indptr / mask for column compaction — not
        in any hot loop; compliant with the no-hot-loop-D2H rule (metrics.h).

        Parameters
        ----------
        mat : DeviceCsc
        qc : QcResult  (from calculate_qc_metrics on the same matrix)
        min_genes, max_genes : float
        min_umis, max_umis : float
        max_pct_mt : float  (0–100)
        stream : int or None

        Returns
        -------
        DeviceCsc  (filtered, new allocation)
        )doc");

    // ── filter_genes ────────────────────────────────────────────────────────
    m.def("filter_genes",
        [](const PyDeviceCsc& mat,
           const PyQcResult&  qc,
           int min_cells,
           py::object stream_obj) -> py::object
        {
            cudaStream_t s = resolve_stream_metrics(stream_obj);

            singlet::gpu::qc::FilterConfig cfg;
            cfg.min_cells_per_gene = min_cells;

            singlet::gpu::qc::QcResult qc_view = make_qc_view(qc);

            auto out_csc = singlet::gpu::qc::filter_genes(*mat.csc, qc_view, cfg, s);

            auto py_out = std::make_shared<PyDeviceCsc>(
                std::make_shared<singlet::gpu::core::DeviceCSC>(std::move(out_csc)),
                s);
            return py::cast(py_out);
        },
        py::arg("mat"),
        py::arg("qc"),
        py::arg("min_cells") = 1,
        py::arg("stream")    = py::none(),
        R"doc(
        Filter genes expressed in fewer than min_cells cells (cycle-103).

        Row index relabeling is O(nnz) on device; one-time D2H for the gene
        mask download is not in any hot loop.

        Parameters
        ----------
        mat : DeviceCsc
        qc : QcResult  (from calculate_qc_metrics on the same matrix)
        min_cells : int  — keep genes detected in >= min_cells cells
        stream : int or None

        Returns
        -------
        DeviceCsc  (filtered, new allocation; rows renumbered 0..n_kept-1)
        )doc");
}

}  // namespace python
}  // namespace singlet::gpu
