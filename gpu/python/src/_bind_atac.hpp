// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_atac.hpp
//
// pybind11 binding for singlet_gpu::atac::chromvar (cycle 34).
//
// Architecture:
//   ChromVarResult holds DeviceMemory<float> for deviation, variability,
//   p_values, and p_adj — all device-resident.
//   Exposed via __cuda_array_interface__ dicts for zero-copy cupy access.
//
// Input marshalling:
//   accessibility  : PyDeviceCsc  (n_peaks × n_cells, CSC, float32)
//   motif_in_peak  : PyDeviceCsc  (n_motifs × n_peaks, CSC, float32 binary)
//   peak_gc        : numpy float32 [n_peaks]  — per-peak GC content ∈ [0, 1]
//   peak_mean_access : numpy float32 [n_peaks] — per-peak mean accessibility
//
// peak_gc and peak_mean_access are uploaded once to device memory and passed
// as DeviceMemory<float> to the chromvar() call.
//
// Result marshalling (ChromVarResult — device buffers):
//   Returns a dict with __cuda_array_interface__ views:
//     "deviation"   : float32 [n_motifs × n_cells] — bias-corrected deviation
//     "variability" : float32 [n_motifs]            — variance of deviation across cells
//     "p_values"    : float32 [n_motifs × n_cells]  — raw permutation p-value
//     "p_adj"       : float32 [n_motifs × n_cells]  — BH-adjusted per-motif
//     "n_motifs"    : int
//     "n_cells"     : int
//     "_result_owner" : lifetime anchor

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/atac/chromvar.h>
#include <singlet-gpu/core/memory.h>

#include "_bind_loader.hpp"
#include "_cupy_interop.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// PyChromVarResult — device buffers + metadata for Python lifetime management
// ---------------------------------------------------------------------------
struct PyChromVarResult {
    core::DeviceMemory<float> deviation;
    core::DeviceMemory<float> variability;
    core::DeviceMemory<float> p_values;
    core::DeviceMemory<float> p_adj;
    int          n_motifs = 0;
    int          n_cells  = 0;
    cudaStream_t stream   = nullptr;
};

// ---------------------------------------------------------------------------
// py_chromvar
// ---------------------------------------------------------------------------
inline py::dict py_chromvar(
    py::object    accessibility_obj,
    py::object    motif_in_peak_obj,
    py::object    peak_gc_obj,           // numpy float32 [n_peaks]
    py::object    peak_mean_access_obj,  // numpy float32 [n_peaks]
    int           n_background_peaks  = 50,
    int           n_permutations      = 1000,
    int           gc_bins             = 25,
    int           access_bins         = 25,
    int           cell_batch          = 4096,
    int           bg_batch            = 50,
    int           perm_inner_batch    = 50,
    float         deviation_epsilon   = 1e-6f,
    py::object    stream              = py::none(),
    uint64_t      seed                = 0)
{
    // ── Unwrap DeviceCsc inputs ───────────────────────────────────────────────
    const core::DeviceCSC& access_csc = *accessibility_obj.cast<PyDeviceCsc&>().csc;
    const core::DeviceCSC& motif_csc  = *motif_in_peak_obj.cast<PyDeviceCsc&>().csc;

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none())
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));

    // ── Upload peak_gc and peak_mean_access to device ─────────────────────────
    auto gc_arr    = peak_gc_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto mean_arr  = peak_mean_access_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
    auto gc_buf    = gc_arr.request();
    auto mean_buf  = mean_arr.request();

    const int n_peaks = static_cast<int>(gc_buf.size);
    if (static_cast<int>(mean_buf.size) != n_peaks)
        throw std::runtime_error("py_chromvar: peak_gc and peak_mean_access must have the same length");

    core::DeviceMemory<float> d_peak_gc(n_peaks);
    core::DeviceMemory<float> d_peak_mean_access(n_peaks);
    cudaMemcpyAsync(d_peak_gc.get(),           static_cast<const float*>(gc_buf.ptr),
                    n_peaks * sizeof(float), cudaMemcpyHostToDevice, cu_stream);
    cudaMemcpyAsync(d_peak_mean_access.get(),  static_cast<const float*>(mean_buf.ptr),
                    n_peaks * sizeof(float), cudaMemcpyHostToDevice, cu_stream);

    // ── Build config ──────────────────────────────────────────────────────────
    atac::ChromVarConfig cfg;
    cfg.n_background_peaks  = n_background_peaks;
    cfg.n_permutations      = n_permutations;
    cfg.gc_bins             = gc_bins;
    cfg.access_bins         = access_bins;
    cfg.cell_batch          = cell_batch;
    cfg.bg_batch            = bg_batch;
    cfg.perm_inner_batch    = perm_inner_batch;
    cfg.deviation_epsilon   = deviation_epsilon;
    cfg.seed                = seed;

    // ── Run kernel ────────────────────────────────────────────────────────────
    atac::ChromVarResult cpp_result = atac::chromvar(
        access_csc, motif_csc,
        d_peak_gc, d_peak_mean_access,
        cfg, cu_stream);

    // ── Package for Python lifetime management ────────────────────────────────
    auto py_result = std::make_shared<PyChromVarResult>();
    py_result->deviation   = std::move(cpp_result.deviation);
    py_result->variability = std::move(cpp_result.variability);
    py_result->p_values    = std::move(cpp_result.p_values);
    py_result->p_adj       = std::move(cpp_result.p_adj);
    py_result->n_motifs    = motif_csc.rows;
    py_result->n_cells     = access_csc.cols;
    py_result->stream      = cu_stream;

    py::object owner = py::cast(py_result, py::return_value_policy::take_ownership);

    const int n_motifs = py_result->n_motifs;
    const int n_cells  = py_result->n_cells;

    auto make_2d_view = [&](core::DeviceMemory<float>& buf,
                             const char* typestr,
                             int rows, int cols) -> py::dict {
        py::dict v;
        v["typestr"] = typestr;
        v["version"] = 3;
        v["shape"]   = py::make_tuple(rows, cols);
        v["data"]    = py::make_tuple(reinterpret_cast<uintptr_t>(buf.data()), true);
        v["stream"]  = reinterpret_cast<uintptr_t>(cu_stream);
        v["_owner"]  = owner;
        return v;
    };

    auto make_1d_view = [&](core::DeviceMemory<float>& buf,
                             const char* typestr,
                             int size) -> py::dict {
        py::dict v;
        v["typestr"] = typestr;
        v["version"] = 3;
        v["shape"]   = py::make_tuple(size);
        v["data"]    = py::make_tuple(reinterpret_cast<uintptr_t>(buf.data()), true);
        v["stream"]  = reinterpret_cast<uintptr_t>(cu_stream);
        v["_owner"]  = owner;
        return v;
    };

    py::dict out;
    out["deviation"]    = make_2d_view(py_result->deviation,   "<f4", n_motifs, n_cells);
    out["variability"]  = make_1d_view(py_result->variability, "<f4", n_motifs);
    out["p_values"]     = make_2d_view(py_result->p_values,    "<f4", n_motifs, n_cells);
    out["p_adj"]        = make_2d_view(py_result->p_adj,       "<f4", n_motifs, n_cells);
    out["n_motifs"]     = n_motifs;
    out["n_cells"]      = n_cells;
    out["_result_owner"] = owner;
    return out;
}

// ---------------------------------------------------------------------------
// bind_atac
// ---------------------------------------------------------------------------
inline void bind_atac(py::module_& m) {
    // Register PyChromVarResult so shared_ptr can be held via py::cast.
    py::class_<PyChromVarResult, std::shared_ptr<PyChromVarResult>>(m, "_ChromVarResultOwner",
        "Internal: keeps ChromVarResult device buffers alive for cupy views.")
    .def_property_readonly("n_motifs", [](const PyChromVarResult& r) { return r.n_motifs; })
    .def_property_readonly("n_cells",  [](const PyChromVarResult& r) { return r.n_cells;  })
    .def("__repr__", [](const PyChromVarResult& r) {
        return "<_ChromVarResultOwner n_motifs=" + std::to_string(r.n_motifs)
             + " n_cells=" + std::to_string(r.n_cells) + ">";
    });

    m.def("chromvar",
        &py_chromvar,
        py::arg("accessibility"),
        py::arg("motif_in_peak"),
        py::arg("peak_gc"),
        py::arg("peak_mean_access"),
        py::arg("n_background_peaks") = 50,
        py::arg("n_permutations")     = 1000,
        py::arg("gc_bins")            = 25,
        py::arg("access_bins")        = 25,
        py::arg("cell_batch")         = 4096,
        py::arg("bg_batch")           = 50,
        py::arg("perm_inner_batch")   = 50,
        py::arg("deviation_epsilon")  = 1e-6f,
        py::arg("stream")             = py::none(),
        py::arg("seed")               = 0,
        R"doc(
GPU-native chromVAR motif enrichment for scATAC-seq (cycle 34).

Computes bias-corrected motif deviation scores across cells, with
permutation-based p-values, following Schep et al. 2017 Nature Methods.

Parameters
----------
accessibility : DeviceCsc
    Peak accessibility matrix (n_peaks × n_cells, CSC, float32).
motif_in_peak : DeviceCsc
    Binary motif-in-peak matrix (n_motifs × n_peaks, CSC, float32).
peak_gc : ndarray float32 [n_peaks]
    Per-peak GC content ∈ [0, 1], used for background peak matching.
peak_mean_access : ndarray float32 [n_peaks]
    Per-peak mean accessibility, used for background peak matching.
n_background_peaks : int, default 50
    Background matched peaks per peak (K_bg in paper).
n_permutations : int, default 1000
    Cell-label permutations for p-value estimation (K_perm).
gc_bins : int, default 25
    Bins along GC-content axis for background matching.
access_bins : int, default 25
    Bins along mean-accessibility axis for background matching.
cell_batch : int, default 4096
    Cells processed per OOC batch (controls peak memory at 200k peaks × 1M cells).
bg_batch : int, default 50
    Background SpMM batch size.
perm_inner_batch : int, default 50
    Permutations per SpMM batch.
deviation_epsilon : float, default 1e-6
    Floor on std_bg to avoid divide-by-zero.
stream : int or None
seed : int, default 0

Returns
-------
dict
    "deviation"    — __cuda_array_interface__ dict (float32, n_motifs × n_cells)
    "variability"  — __cuda_array_interface__ dict (float32, n_motifs)
    "p_values"     — __cuda_array_interface__ dict (float32, n_motifs × n_cells)
    "p_adj"        — __cuda_array_interface__ dict (float32, n_motifs × n_cells)
    "n_motifs", "n_cells" — int
    "_result_owner" — internal lifetime anchor (keep alive with cupy views)

Notes
-----
Use ``cupy.ndarray.view`` or construct ``cupy.ndarray`` from
``__cuda_array_interface__`` dict to access results without host copies::

    import cupy as cp
    dev = result["deviation"]
    dev_cp = cp.ndarray(dev["shape"], dtype=cp.float32,
                        memptr=cp.cuda.MemoryPointer(
                            cp.cuda.UnownedMemory(dev["data"][0], 0, result["_result_owner"]), 0))
)doc");
}

} // namespace python
} // namespace singlet_gpu
