// SPDX-License-Identifier: GPL-2.0-or-later
// singlet-gpu/python/src/_bind_variants.hpp
//
// pybind11 binding for singlet_gpu::variants::call_variants (cycle 42, Monopogen).
//
// Architecture:
//   MonopogenResult holds host-side std::vector data (per-chromosome results
//   accumulated into global arrays).  All arrays are returned as plain numpy
//   arrays (host copies, not device views).
//
// Input marshalling:
//   snp_ad, snp_dp   : PyDeviceCsc (SNP sites × cells, CSC)
//   chromosome       : list[int] per-SNP chromosome (1-based, ≤ MAX_CHROMOSOMES)
//   ld_panels        : list of dicts per chromosome:
//                        "n_snps"  : int
//                        "indptr"  : list[int]   [n_snps_chr + 1]
//                        "col_idx" : list[int]   [nnz]
//                        "val"     : list[float] [nnz]  LD r-values
//                      Empty list → no-LD mode (call_variants_no_ld).
//
// Result marshalling (MonopogenResult — host vectors):
//   Returns a dict:
//     "germline_genotypes" : float32 ndarray [n_snps × 3] — P(genotype|data)
//     "germline_calls"     : int8 ndarray [n_snps]        — 0=0/0, 1=0/1, 2=1/1, -1=filtered
//     "qual_scores"        : float32 ndarray [n_snps]
//     "somatic_candidates" : list of dicts {snp_idx, alt_freq, ld_score, lrt_stat, p_value, q_value}
//     "n_snps"             : int

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <singlet-gpu/variants/monopogen.h>
#include <singlet-gpu/core/handles.h>

#include "_bind_loader.hpp"

#include <cuda_runtime.h>
#include <cusparse.h>
#include <stdexcept>
#include <vector>
#include <cstdint>

namespace py = pybind11;

namespace singlet_gpu {
namespace python {

// ---------------------------------------------------------------------------
// py_call_variants
// ---------------------------------------------------------------------------
inline py::dict py_call_variants(
    py::object    snp_ad_obj,
    py::object    snp_dp_obj,
    py::list      chromosome_list,    // per-SNP chromosome (1-based)
    py::list      ld_panels_list,     // list of per-chr LD panel dicts; empty = no-LD
    float         somatic_af_min   = variants::SOMATIC_AF_MIN,
    float         somatic_af_max   = variants::SOMATIC_AF_MAX,
    float         fdr_alpha        = 0.05f,
    float         ld_score_min     = 0.05f,
    int           min_total_dp     = variants::MIN_TOTAL_DP,
    bool          run_bh_fdr       = true,
    py::object    stream           = py::none(),
    uint64_t      seed             = 0)
{
    // ── Unwrap DeviceCsc inputs ───────────────────────────────────────────────
    const core::DeviceCSC& snp_ad_csc = *snp_ad_obj.cast<PyDeviceCsc&>().csc;
    const core::DeviceCSC& snp_dp_csc = *snp_dp_obj.cast<PyDeviceCsc&>().csc;

    // ── Resolve CUDA stream ───────────────────────────────────────────────────
    cudaStream_t cu_stream = nullptr;
    if (!stream.is_none())
        cu_stream = reinterpret_cast<cudaStream_t>(
            static_cast<uintptr_t>(stream.cast<py::int_>()));

    // ── Build per-SNP chromosome array ────────────────────────────────────────
    std::vector<int> chromosome = chromosome_list.cast<std::vector<int>>();

    // ── Build MonopogenConfig ─────────────────────────────────────────────────
    variants::MonopogenConfig cfg;
    cfg.somatic_af_min  = somatic_af_min;
    cfg.somatic_af_max  = somatic_af_max;
    cfg.fdr_alpha       = fdr_alpha;
    cfg.ld_score_min    = ld_score_min;
    cfg.min_total_dp    = min_total_dp;
    cfg.run_bh_fdr      = run_bh_fdr;
    cfg.deterministic   = true;

    // ── Build GPUContext ───────────────────────────────────────────────────────
    core::GPUContext ctx;
    ctx.stream = cu_stream;

    // We need a cuSPARSE handle for the LD-panel SpMV.
    cusparseHandle_t sp_handle = ctx.cusparse;

    variants::MonopogenResult result;

    if (ld_panels_list.empty()) {
        // No-LD path: call_germline_only builds empty LD panels internally.
        result = variants::call_germline_only(
            snp_ad_csc, snp_dp_csc, chromosome, cfg, cu_stream, sp_handle);
    } else {
        // Build per-chromosome LdPanelCsr list matching ld_panels_list order.
        // ld_panels_list[i] corresponds to the i-th entry in the caller's
        // per-chromosome ordering.  The caller must supply panels covering all
        // chromosomes in `chromosome` (indexed by their list position).
        std::vector<variants::LdPanelCsr> ld_panels;
        ld_panels.reserve(ld_panels_list.size());
        for (auto item : ld_panels_list) {
            auto d = item.cast<py::dict>();
            variants::LdPanelCsr lp;
            lp.n_snps   = d["n_snps"].cast<int>();
            lp.indptr   = d["indptr"].cast<std::vector<int32_t>>();
            lp.col_idx  = d["col_idx"].cast<std::vector<int32_t>>();
            lp.val      = d["val"].cast<std::vector<float>>();
            ld_panels.push_back(std::move(lp));
        }
        result = variants::call_variants(
            snp_ad_csc, snp_dp_csc, chromosome, ld_panels, cfg, cu_stream, sp_handle);
    }

    // ── Marshal to Python dict ────────────────────────────────────────────────
    const py::ssize_t n_snps = static_cast<py::ssize_t>(result.n_snps);

    // germline_genotypes: [n_snps × 3]
    py::array_t<float> geno_arr;
    if (!result.germline_genotypes.empty()) {
        geno_arr = py::array_t<float>({n_snps, py::ssize_t(3)},
                                      result.germline_genotypes.data());
    } else {
        geno_arr = py::array_t<float>({0L, 3L});
    }

    // germline_calls: [n_snps] int8
    py::array_t<int8_t> calls_arr;
    if (!result.germline_calls.empty()) {
        calls_arr = py::array_t<int8_t>(n_snps, result.germline_calls.data());
    } else {
        calls_arr = py::array_t<int8_t>(0L);
    }

    // qual_scores: [n_snps] float32
    py::array_t<float> qual_arr;
    if (!result.qual_scores.empty()) {
        qual_arr = py::array_t<float>(n_snps, result.qual_scores.data());
    } else {
        qual_arr = py::array_t<float>(0L);
    }

    // somatic_candidates: list of dicts
    py::list sc_list;
    for (const auto& sc : result.somatic_candidates) {
        py::dict sd;
        sd["snp_idx"]  = sc.snp_idx;
        sd["alt_freq"] = sc.alt_freq;
        sd["ld_score"] = sc.ld_score;
        sd["lrt_stat"] = sc.lrt_stat;
        sd["p_value"]  = sc.p_value;
        sd["q_value"]  = sc.q_value;
        sc_list.append(sd);
    }

    py::dict out;
    out["germline_genotypes"] = geno_arr;
    out["germline_calls"]     = calls_arr;
    out["qual_scores"]        = qual_arr;
    out["somatic_candidates"] = sc_list;
    out["n_snps"]             = static_cast<int>(n_snps);
    return out;
}

// ---------------------------------------------------------------------------
// bind_variants
// ---------------------------------------------------------------------------
inline void bind_variants(py::module_& m) {
    m.def("call_variants",
        &py_call_variants,
        py::arg("snp_ad"),
        py::arg("snp_dp"),
        py::arg("chromosome"),
        py::arg("ld_panels")     = py::list(),
        py::arg("somatic_af_min") = variants::SOMATIC_AF_MIN,
        py::arg("somatic_af_max") = variants::SOMATIC_AF_MAX,
        py::arg("fdr_alpha")     = 0.05f,
        py::arg("ld_score_min")  = 0.05f,
        py::arg("min_total_dp")  = variants::MIN_TOTAL_DP,
        py::arg("run_bh_fdr")    = true,
        py::arg("stream")        = py::none(),
        py::arg("seed")          = 0,
        R"doc(
GPU-native Monopogen somatic SNV calling from scRNA pileup (cycle 42).

Germline genotyping + somatic variant discovery from singlify snp_ad / snp_dp
outputs.  Rule-based filter (alt_freq band + LD correction + LRT) without
the ML refinement layer (deferred to v2).

Parameters
----------
snp_ad : DeviceCsc
    Alternate-allele counts (SNP sites × cells, CSC, from snp_ad.1pz).
snp_dp : DeviceCsc
    Total depth (SNP sites × cells, CSC, from snp_dp.1pz).
chromosome : list[int]
    Per-SNP chromosome assignment (1-based, ≤ 26).  Length = n_snps.
ld_panels : list of dicts, default []
    Per-chromosome LD panel (CSR).  Each dict:
      "n_snps"  — int
      "indptr"  — list[int]   [n_snps_chr + 1]
      "col_idx" — list[int]   [nnz]
      "val"     — list[float] [nnz] LD r-values ∈ [-1, 1]
    Empty list disables LD correction (call_variants_no_ld path).
somatic_af_min : float, default 0.01
somatic_af_max : float, default 0.30
fdr_alpha : float, default 0.05
ld_score_min : float, default 0.05
min_total_dp : int, default 10
run_bh_fdr : bool, default True
stream : int or None
seed : int, default 0 (algorithm is deterministic)

Returns
-------
dict
    "germline_genotypes" — float32 ndarray [n_snps × 3]
    "germline_calls"     — int8 ndarray [n_snps]  0=0/0, 1=0/1, 2=1/1, -1=filtered
    "qual_scores"        — float32 ndarray [n_snps]  Phred-scaled quality
    "somatic_candidates" — list of dicts {snp_idx, alt_freq, ld_score, lrt_stat, p_value, q_value}
    "n_snps"             — int
)doc");
}

} // namespace python
} // namespace singlet_gpu
