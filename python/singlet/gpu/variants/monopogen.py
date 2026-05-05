# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.variants.monopogen — GPU-native Monopogen somatic SNV calling.

Underlying C++ kernel: singlet_gpu::variants::call_variants (cycle 42,
variants/monopogen.h).

Germline genotyping + somatic SNV discovery from singlify snp_ad / snp_dp
outputs.  Pipeline:
  1. Per-SNP pileup (cub::DeviceSegmentedReduce).
  2. Binomial likelihood germline genotyping (0/0, 0/1, 1/1).
  3. Somatic candidate filter (alt-freq band + LD correction).
  4. Likelihood ratio test.
  5. BH FDR correction per chromosome tile.

Note: ML refinement layer (random forest) is deferred to v2.

Reference: Dou et al. (Monopogen, Nature Biotechnology 2023).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict

import numpy as np

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def call(
    snp_ad,
    snp_dp,
    chromosome: List[int],
    *,
    ld_panels: Optional[List[Dict]] = None,
    somatic_af_min: float = 0.01,
    somatic_af_max: float = 0.30,
    fdr_alpha: float = 0.05,
    ld_score_min: float = 0.05,
    min_total_dp: int = 10,
    run_bh_fdr: bool = True,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native Monopogen somatic SNV calling from scRNA pileup (cycle 42).

    Performs germline genotyping and somatic variant discovery from singlify
    pileup data.  Returns germline calls for all SNPs and a sparse list of
    somatic candidates that pass the rule-based filter.

    Parameters
    ----------
    snp_ad : DeviceCsc
        Alternate-allele counts (SNP sites × cells, CSC).
        From singlify ``snp_ad.1pz``.
    snp_dp : DeviceCsc
        Total depth at SNP sites (SNP sites × cells, CSC).
        From singlify ``snp_dp.1pz``.
    chromosome : list[int]
        Per-SNP chromosome assignment (1-based integers, 1-26).
        Length = number of SNP columns in snp_ad/snp_dp.
    ld_panels : list of dicts, optional
        Per-chromosome LD panels in CSR format.  Each dict has:

        ``"n_snps"``  — int, SNPs on this chromosome.
        ``"indptr"``  — list[int], length n_snps + 1.
        ``"col_idx"`` — list[int], length nnz.
        ``"val"``     — list[float], LD r-values ∈ [-1, 1].

        Chromosomes not in the list are processed without LD correction.
        None / empty list → no-LD mode for all chromosomes.
    somatic_af_min : float, default 0.01
        Minimum alt-freq for somatic candidates (below = noise).
    somatic_af_max : float, default 0.30
        Maximum alt-freq for somatic candidates (above = likely germline).
    fdr_alpha : float, default 0.05
        BH FDR threshold for somatic calls.
    ld_score_min : float, default 0.05
        Minimum LD-corrected somatic score to retain a candidate.
    min_total_dp : int, default 10
        Minimum total depth across all cells to enter germline calling.
    run_bh_fdr : bool, default True
    stream : int or None
    seed : int, default 0 (algorithm is deterministic)

    Returns
    -------
    dict
        ``"germline_genotypes"`` — float32 ndarray (n_snps × 3),
            per-genotype posteriors P(0/0|data), P(0/1|data), P(1/1|data).
        ``"germline_calls"``     — int8 ndarray (n_snps),
            0=0/0, 1=0/1, 2=1/1, -1=filtered (dp < min_total_dp).
        ``"qual_scores"``        — float32 ndarray (n_snps),
            Phred-scaled posterior quality score.
        ``"somatic_candidates"`` — list of dicts, each with keys:
            ``snp_idx``, ``alt_freq``, ``ld_score``, ``lrt_stat``,
            ``p_value``, ``q_value``.
        ``"n_snps"``             — int.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    ValueError
        If chromosome array length does not match snp_dp.cols.

    Notes
    -----
    **Somatic precision vs recall**: v1 ships a rule-based filter optimised for
    high precision (conservative).  The ML refinement layer (random forest on
    trinucleotide context) is planned for v2.

    **LD panel source**: pre-compute from HapMap3 / 1000 Genomes Phase 3 at
    r² > 0.5, top-100 LD neighbors per SNP.  400 MB per chromosome at 1M SNPs.

    **No-LD mode**: passing ``ld_panels=None`` or ``ld_panels=[]`` skips LD
    correction.  Recommended when LD data is unavailable; slightly lower
    somatic precision.

    Examples
    --------
    Without LD panels (fast, lower precision)::

        result = singlet_gpu.variants.monopogen.call(
            snp_ad, snp_dp, chromosome=snp_chromosomes
        )
        print(f"Germline SNPs called: {result['n_snps']}")
        print(f"Somatic candidates: {len(result['somatic_candidates'])}")

    With LD panels::

        ld_panels = [
            {"n_snps": n_chr1, "indptr": ..., "col_idx": ..., "val": ...},
            # one entry per chromosome
        ]
        result = singlet_gpu.variants.monopogen.call(
            snp_ad, snp_dp, chromosome=snp_chromosomes, ld_panels=ld_panels
        )
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "call_variants"):
        raise ImportError(
            "_core.call_variants is not available.  "
            "Compile the singlet-gpu pybind11 extension on a CUDA node."
        )

    # Default ld_panels to empty list (no-LD mode).
    if ld_panels is None:
        ld_panels = []

    return _core.call_variants(
        snp_ad,
        snp_dp,
        list(chromosome),
        list(ld_panels),
        float(somatic_af_min),
        float(somatic_af_max),
        float(fdr_alpha),
        float(ld_score_min),
        int(min_total_dp),
        bool(run_bh_fdr),
        stream,
        int(seed),
    )


__all__ = ["call"]
