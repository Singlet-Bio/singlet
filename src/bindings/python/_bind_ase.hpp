// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_ase.hpp
//
// pybind11 binding for singlet::gpu::ase::run_daesc (cycle 40).
//
// Architecture:
//   DAESC takes: snp_ad CSC + snp_dp CSC + optional cell_type array + config.
//   All result arrays are host-side (chunked kernel accumulates directly).
//
// Input marshalling:
//   snp_ad, snp_dp : PyDeviceCsc
//   cell_type      : optional list[int] or numpy int32 [n_cells] (0-indexed), or None
//
// Result marshalling (DaescResult — host vectors):
//   Returns a dict:
//     "beta"       : float32 ndarray [n_snps]
//     "se"         : float32 ndarray [n_snps]
//     "phi"        : float32 ndarray [n_snps]
//     "p_value"    : float32 ndarray [n_snps]
//     "p_adj"      : float32 ndarray [n_snps]
//     "converged"  : uint8   ndarray [n_snps]
//     "n_cells"    : int32   ndarray [n_snps]  — qualifying cells per SNP
//     "strat_beta" : float32 ndarray [n_snps × n_types] (empty if n_types == 0)
//     "strat_se"   : float32 ndarray [n_snps × n_types]
//     "strat_phi"  : float32 ndarray [n_snps × n_types]
//     "n_snps"     : int
//     "n_types"    : int

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet/gpu/ase/daesc.h>
#include <singlet/gpu/core/handles.h>

#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet::gpu {
namespace python {

// ---------------------------------------------------------------------------
// py_run_daesc
// ---------------------------------------------------------------------------
inline py::dict py_run_daesc(
    py::object    snp_ad_obj,
    py::object    snp_dp_obj,
    py::object    cell_type_obj,      // list[int] or numpy int32 [n_cells], or None
    int           chunk_snps    = ase::DEFAULT_CHUNK_SNPS,
    int           max_iters     = ase::DAESC_MAX_ITERS,
    float         fisher_tol    = ase::DAESC_FISHER_TOL,
    int           min_depth     = ase::DAESC_MIN_DEPTH,
    int           min_cells     = ase::DAESC_MIN_CELLS,
    bool          run_bh_fdr    = true,
    float         fdr_alpha     = 0.05f,
    int           n_types       = 0,
    py::object    stream        = py::none(),
    uint64_t      seed          = 0)
{
    // ── Unwrap DeviceCsc inputs ───────────────────────────────────────────────
    const core::DeviceCSC& snp_ad_csc = *snp_ad_obj.cast<PyDeviceCsc&>().csc;
    const core::DeviceCSC& snp_dp_csc = *snp_dp_obj.cast<PyDeviceCsc&>().csc;

    // ── Unwrap cell_type (optional) ───────────────────────────────────────────
    std::vector<int> cell_type_vec;
    const int* cell_type_ptr = nullptr;
    if (!cell_type_obj.is_none() && n_types > 0) {
        if (py::isinstance<py::list>(cell_type_obj) || py::isinstance<py::tuple>(cell_type_obj)) {
            cell_type_vec = cell_type_obj.cast<std::vector<int>>();
        } else {
            auto arr = cell_type_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
            auto buf = arr.request();
            const int32_t* raw = static_cast<const int32_t*>(buf.ptr);
            cell_type_vec.assign(raw, raw + buf.size);
        }
        cell_type_ptr = cell_type_vec.data();
    }

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none())
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));

    // ── Build config ──────────────────────────────────────────────────────────
    ase::DaescConfig cfg;
    cfg.chunk_snps  = chunk_snps;
    cfg.max_iters   = max_iters;
    cfg.fisher_tol  = fisher_tol;
    cfg.min_depth   = min_depth;
    cfg.min_cells   = min_cells;
    cfg.run_bh_fdr  = run_bh_fdr;
    cfg.fdr_alpha   = fdr_alpha;
    cfg.n_types     = (cell_type_ptr != nullptr) ? n_types : 0;

    // ── Build GPUContext ───────────────────────────────────────────────────────
    core::GPUContext ctx;
    ctx.stream = cu_stream;
    cudaStream_t stream2 = cu_stream;

    // ── Run kernel ────────────────────────────────────────────────────────────
    ase::DaescResult result = ase::run_daesc(
        snp_ad_csc, snp_dp_csc, cell_type_ptr, cfg, ctx, stream2);

    // ── Marshal to Python dict ────────────────────────────────────────────────
    const py::ssize_t n_snps = static_cast<py::ssize_t>(result.n_snps);
    const int n_t = cfg.n_types;

    py::dict out;
    out["beta"]      = py::array_t<float>(n_snps, result.beta.data());
    out["se"]        = py::array_t<float>(n_snps, result.se.data());
    out["phi"]       = py::array_t<float>(n_snps, result.phi.data());
    out["p_value"]   = py::array_t<float>(n_snps, result.p_value.data());
    out["p_adj"]     = py::array_t<float>(n_snps, result.p_adj.data());
    out["converged"] = py::array_t<uint8_t>(n_snps, result.converged.data());
    out["n_cells"]   = py::array_t<int32_t>(n_snps,
        reinterpret_cast<const int32_t*>(result.n_cells.data()));
    out["n_snps"]    = static_cast<int>(n_snps);
    out["n_types"]   = n_t;

    if (n_t > 0 && !result.strat_beta.empty()) {
        const py::ssize_t nt = static_cast<py::ssize_t>(n_t);
        out["strat_beta"] = py::array_t<float>({n_snps, nt}, result.strat_beta.data());
        out["strat_se"]   = py::array_t<float>({n_snps, nt}, result.strat_se.data());
        out["strat_phi"]  = py::array_t<float>({n_snps, nt}, result.strat_phi.data());
    } else {
        out["strat_beta"] = py::array_t<float>({0L, 0L});
        out["strat_se"]   = py::array_t<float>({0L, 0L});
        out["strat_phi"]  = py::array_t<float>({0L, 0L});
    }
    return out;
}

// ---------------------------------------------------------------------------
// bind_ase
// ---------------------------------------------------------------------------
inline void bind_ase(py::module_& m) {
    m.def("run_daesc",
        &py_run_daesc,
        py::arg("snp_ad"),
        py::arg("snp_dp"),
        py::arg("cell_type")  = py::none(),
        py::arg("chunk_snps") = ase::DEFAULT_CHUNK_SNPS,
        py::arg("max_iters")  = ase::DAESC_MAX_ITERS,
        py::arg("fisher_tol") = ase::DAESC_FISHER_TOL,
        py::arg("min_depth")  = ase::DAESC_MIN_DEPTH,
        py::arg("min_cells")  = ase::DAESC_MIN_CELLS,
        py::arg("run_bh_fdr") = true,
        py::arg("fdr_alpha")  = 0.05f,
        py::arg("n_types")    = 0,
        py::arg("stream")     = py::none(),
        py::arg("seed")       = 0,
        R"doc(
GPU-native DAESC beta-binomial ASE calling (cycle 40).

Fits a per-SNP beta-binomial MLE via GPU Fisher scoring to detect
allele-specific expression from singlet snp_ad / snp_dp outputs.

Parameters
----------
snp_ad : DeviceCsc
    Alternate-allele counts (SNP sites × cells, CSC, from snp_ad.1pz).
snp_dp : DeviceCsc
    Total depth (SNP sites × cells, CSC, from snp_dp.1pz).
cell_type : list[int] or ndarray int32 [n_cells], optional
    Per-cell cell-type labels (0-indexed).  Enables stratified ASE
    alongside the global (unstratified) model.  Requires n_types > 0.
chunk_snps : int, default 10000
max_iters : int, default 50
fisher_tol : float, default 1e-5
min_depth : int, default 5
    Minimum DP per cell to include in the MLE.
min_cells : int, default 5
    Minimum qualifying cells per SNP.
run_bh_fdr : bool, default True
fdr_alpha : float, default 0.05
n_types : int, default 0
    Number of cell types for stratified analysis.  0 = no stratification.
stream : int or None
seed : int, default 0 (algorithm is deterministic)

Returns
-------
dict
    "beta", "se", "phi" — float32 ndarray [n_snps]
    "p_value", "p_adj"  — float32 ndarray [n_snps]
    "converged"         — uint8 ndarray [n_snps]
    "n_cells"           — int32 ndarray [n_snps]  qualifying cells per SNP
    "strat_beta", "strat_se", "strat_phi" — float32 [n_snps × n_types] (empty if n_types==0)
    "n_snps", "n_types" — int
)doc");
}

} // namespace python
} // namespace singlet::gpu
