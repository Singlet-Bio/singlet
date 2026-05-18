// SPDX-License-Identifier: MIT
// singlet/gpu/python/src/_bind_eqtl.hpp
//
// pybind11 binding for singlet::gpu::eqtl::run_nebula (cycle 38).
//
// Architecture:
//   NEBULA takes: counts CSC + snp_ad CSC + snp_dp CSC + donor_id array +
//                 optional covariate matrix + config.
//
// Input marshalling:
//   counts, snp_ad, snp_dp : PyDeviceCsc (already on device)
//   donor_id_list          : list[int] or numpy int32 array, host-side [n_cells]
//   covariates             : optional numpy float32 array [n_donors × n_cov] or None
//   n_donors               : int (must match length of donor_id domain)
//
// Result marshalling (NebulaResult — all host vectors):
//   Returns a dict of numpy arrays:
//     "beta"      : float32 [n_snps × n_genes]
//     "se"        : float32 [n_snps × n_genes]
//     "sigma_sq"  : float32 [n_snps × n_genes]
//     "p_value"   : float32 [n_snps × n_genes]
//     "p_adj"     : float32 [n_snps × n_genes]
//     "converged" : uint8   [n_snps × n_genes]
//     "n_snps"    : int
//     "n_genes"   : int
//
// NOTE: NebulaResult holds host-side std::vector data (chunk results accumulated).
//       The result dict holds plain numpy arrays (host copies), not device views.
//       This is consistent with other chunked/large-output kernels in this codebase.

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet/gpu/eqtl/nebula.h>
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
// py_run_nebula
// ---------------------------------------------------------------------------
inline py::dict py_run_nebula(
    py::object    counts_obj,
    py::object    snp_ad_obj,
    py::object    snp_dp_obj,
    py::object    donor_id_obj,       // list[int] or numpy int32 [n_cells]
    int           n_donors,
    py::object    covariates_obj,     // numpy float32 [n_donors × n_cov] or None
    int           n_cov,
    int           chunk_snps    = eqtl::DEFAULT_CHUNK_SNPS,
    int           chunk_genes   = eqtl::DEFAULT_CHUNK_GENES,
    int           max_iters     = eqtl::FISHER_MAX_ITERS,
    float         fisher_tol    = eqtl::FISHER_TOL,
    bool          run_bh_fdr    = true,
    float         fdr_alpha     = 0.05f,
    py::object    stream        = py::none(),
    uint64_t      seed          = 0)
{
    // ── Unwrap DeviceCsc inputs ───────────────────────────────────────────────
    const core::DeviceCSC& counts_csc = *counts_obj.cast<PyDeviceCsc&>().csc;
    const core::DeviceCSC& snp_ad_csc = *snp_ad_obj.cast<PyDeviceCsc&>().csc;
    const core::DeviceCSC& snp_dp_csc = *snp_dp_obj.cast<PyDeviceCsc&>().csc;

    // ── Unwrap donor_id to host vector ────────────────────────────────────────
    std::vector<int> donor_id_vec;
    if (py::isinstance<py::list>(donor_id_obj) || py::isinstance<py::tuple>(donor_id_obj)) {
        donor_id_vec = donor_id_obj.cast<std::vector<int>>();
    } else {
        // numpy array path
        auto arr = donor_id_obj.cast<py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
        auto buf = arr.request();
        const int32_t* ptr = static_cast<const int32_t*>(buf.ptr);
        donor_id_vec.assign(ptr, ptr + buf.size);
    }

    // ── Unwrap covariates (optional) ──────────────────────────────────────────
    std::vector<float> cov_vec;
    const float* cov_ptr = nullptr;
    if (!covariates_obj.is_none() && n_cov > 0) {
        auto arr = covariates_obj.cast<py::array_t<float, py::array::c_style | py::array::forcecast>>();
        auto buf = arr.request();
        const float* raw = static_cast<const float*>(buf.ptr);
        cov_vec.assign(raw, raw + buf.size);
        cov_ptr = cov_vec.data();
    }

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none())
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));

    // ── Build config ──────────────────────────────────────────────────────────
    eqtl::NebulaConfig cfg;
    cfg.chunk_snps  = chunk_snps;
    cfg.chunk_genes = chunk_genes;
    cfg.max_iters   = max_iters;
    cfg.fisher_tol  = fisher_tol;
    cfg.run_bh_fdr  = run_bh_fdr;
    cfg.fdr_alpha   = fdr_alpha;

    // ── Build GPUContext ───────────────────────────────────────────────────────
    core::GPUContext ctx;
    ctx.stream = cu_stream;
    // stream2 for prefetch overlap (same as primary stream if caller only has one)
    cudaStream_t stream2 = cu_stream;

    // ── Run kernel ────────────────────────────────────────────────────────────
    eqtl::NebulaResult result = eqtl::run_nebula(
        counts_csc, snp_ad_csc, snp_dp_csc,
        donor_id_vec.data(), n_donors,
        cov_ptr, n_cov,
        cfg, ctx, stream2,
        nullptr);  // no chunk callback — accumulate full result

    // ── Marshal to Python dict ────────────────────────────────────────────────
    const py::ssize_t n_snps  = static_cast<py::ssize_t>(result.n_snps);
    const py::ssize_t n_genes = static_cast<py::ssize_t>(result.n_genes);
    const py::ssize_t total   = n_snps * n_genes;

    auto make_f32 = [&](const std::vector<float>& v) {
        return py::array_t<float>({n_snps, n_genes}, v.data());
    };

    py::dict out;
    out["beta"]      = make_f32(result.beta);
    out["se"]        = make_f32(result.se);
    out["sigma_sq"]  = make_f32(result.sigma_sq);
    out["p_value"]   = make_f32(result.p_value);
    out["p_adj"]     = make_f32(result.p_adj);
    out["converged"] = py::array_t<uint8_t>({n_snps, n_genes}, result.converged.data());
    out["n_snps"]    = static_cast<int>(n_snps);
    out["n_genes"]   = static_cast<int>(n_genes);
    return out;
}

// ---------------------------------------------------------------------------
// bind_eqtl
// ---------------------------------------------------------------------------
inline void bind_eqtl(py::module_& m) {
    m.def("run_nebula",
        &py_run_nebula,
        py::arg("counts"),
        py::arg("snp_ad"),
        py::arg("snp_dp"),
        py::arg("donor_id"),
        py::arg("n_donors"),
        py::arg("covariates")  = py::none(),
        py::arg("n_cov")       = 0,
        py::arg("chunk_snps")  = eqtl::DEFAULT_CHUNK_SNPS,
        py::arg("chunk_genes") = eqtl::DEFAULT_CHUNK_GENES,
        py::arg("max_iters")   = eqtl::FISHER_MAX_ITERS,
        py::arg("fisher_tol")  = eqtl::FISHER_TOL,
        py::arg("run_bh_fdr")  = true,
        py::arg("fdr_alpha")   = 0.05f,
        py::arg("stream")      = py::none(),
        py::arg("seed")        = 0,
        R"doc(
GPU-native NEBULA single-cell eQTL mapping (cycle 38).

Fits a Negative Binomial mixed model via GPU Fisher scoring to find
genotype–expression associations at single-cell resolution.

Parameters
----------
counts : DeviceCsc
    Count matrix (cells × genes, CSC, float32).
snp_ad : DeviceCsc
    Alt-allele counts (SNP sites × cells, CSC, uint32→float32).
snp_dp : DeviceCsc
    Total depth (SNP sites × cells, CSC, uint32→float32).
donor_id : list[int] or ndarray of int32, shape (n_cells,)
    Per-cell donor assignment (0-indexed, range [0, n_donors)).
n_donors : int
    Number of unique donors.  Must be ≤ 200 (shared-memory budget limit).
covariates : ndarray of float32, shape (n_donors, n_cov), optional
    Donor-level covariates (e.g. age, sex, PCs).  None for intercept-only.
n_cov : int, default 0
chunk_snps : int, default 10000
chunk_genes : int, default 500
max_iters : int, default 50
fisher_tol : float, default 1e-5
run_bh_fdr : bool, default True
fdr_alpha : float, default 0.05
stream : int or None
seed : int, default 0 (algorithm is deterministic; seed not used)

Returns
-------
dict
    "beta"      — float32 ndarray (n_snps × n_genes) eQTL effect sizes
    "se"        — float32 ndarray standard errors
    "sigma_sq"  — float32 ndarray random-effect variances
    "p_value"   — float32 ndarray Wald p-values (uncorrected)
    "p_adj"     — float32 ndarray BH-adjusted q-values
    "converged" — uint8 ndarray (1=converged)
    "n_snps", "n_genes" — int
)doc");
}

} // namespace python
} // namespace singlet::gpu
