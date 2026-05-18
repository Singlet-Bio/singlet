# SPDX-License-Identifier: MIT
"""
singlet.gpu.ase.daesc — GPU-native DAESC beta-binomial ASE calling.

Underlying C++ kernel: singlet::gpu::ase::run_daesc (cycle 40, ase/daesc.h).

DAESC fits per-SNP beta-binomial MLE (intercept-only model) to detect
allele-specific expression at single-cell resolution.

Reference: Hu et al. (DAESC, Genome Biology 2023); DAESC+ (bioRxiv 2025).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from singlet.gpu._coreutil import require_core

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    snp_ad,
    snp_dp,
    *,
    cell_type=None,
    chunk_snps: int = 10000,
    max_iters: int = 50,
    fisher_tol: float = 1e-5,
    min_depth: int = 5,
    min_cells: int = 5,
    run_bh_fdr: bool = True,
    fdr_alpha: float = 0.05,
    n_types: int = 0,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native DAESC beta-binomial ASE calling (cycle 40).

    Fits a per-SNP beta-binomial MLE via GPU Fisher scoring (shared memory,
    50-iteration loop) to detect allele-specific expression.  Optionally
    runs stratified MLE per cell type in addition to the global model.

    Parameters
    ----------
    snp_ad : DeviceCsc
        Alternate-allele counts (SNP sites × cells, CSC).
        From singlet ``snp_ad.1pz``.
    snp_dp : DeviceCsc
        Total depth at SNP sites (SNP sites × cells, CSC).
        From singlet ``snp_dp.1pz``.
    cell_type : list[int] or ndarray int32, shape (n_cells,), optional
        Per-cell cell-type labels (0-indexed, range [0, n_types)).
        Required if n_types > 0.  None = no stratification.
    chunk_snps : int, default 10000
        SNPs per device batch (controls peak device memory ~400 MB/chunk).
    max_iters : int, default 50
        Fisher scoring iterations.
    fisher_tol : float, default 1e-5
        Convergence threshold on |Δbeta|.
    min_depth : int, default 5
        Minimum per-cell DP to include a cell in the MLE.
    min_cells : int, default 5
        Minimum qualifying cells per SNP.  SNPs with fewer cells are skipped.
    run_bh_fdr : bool, default True
        Apply BH FDR correction across all SNPs.
    fdr_alpha : float, default 0.05
    n_types : int, default 0
        Number of cell types for stratified analysis.  0 = global only.
    stream : int or None
    seed : int, default 0 (algorithm is deterministic)

    Returns
    -------
    dict
        ``"beta"``       — float32 ndarray [n_snps] allelic log-odds (MLE).
        ``"se"``         — float32 ndarray [n_snps] standard error.
        ``"phi"``        — float32 ndarray [n_snps] beta-binomial overdispersion.
        ``"p_value"``    — float32 ndarray [n_snps] Wald test p (uncorrected).
        ``"p_adj"``      — float32 ndarray [n_snps] BH-adjusted q-value.
        ``"converged"``  — uint8 ndarray [n_snps] (1=converged).
        ``"n_cells"``    — int32 ndarray [n_snps] qualifying cells per SNP.
        ``"strat_beta"`` — float32 ndarray [n_snps × n_types] (empty if n_types==0).
        ``"strat_se"``   — float32 ndarray [n_snps × n_types].
        ``"strat_phi"``  — float32 ndarray [n_snps × n_types].
        ``"n_snps"``     — int.
        ``"n_types"``    — int.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    ValueError
        If cell_type is provided but n_types == 0, or vice versa.

    Notes
    -----
    **Beta interpretation**: ``beta`` is the logit-scale allelic fraction.
    Positive beta → reference allele preferred; negative → alt allele.
    Convert to allele fraction: ``p = 1 / (1 + exp(-beta))``.

    **SNP indexing**: result arrays are indexed by SNP position in the CSC
    column order of snp_dp.  Align with your SNP annotation accordingly.

    Examples
    --------
    Global ASE only::

        result = singlet.gpu.ase.daesc.run(snp_ad, snp_dp)
        sig_mask = result["p_adj"] < 0.05
        print(f"Significant ASE SNPs: {sig_mask.sum()}")

    With cell-type stratification::

        cell_types = np.array([0, 1, 0, 2, ...], dtype=np.int32)  # n_cells
        result = singlet.gpu.ase.daesc.run(
            snp_ad, snp_dp,
            cell_type=cell_types, n_types=3
        )
        # result["strat_beta"] has shape (n_snps, 3)
    """
    _core = require_core("run_daesc")

    if (cell_type is not None) != (n_types > 0):
        raise ValueError(
            "cell_type and n_types must be provided together: "
            "pass both or neither."
        )

    # Coerce cell_type to int32 numpy array.
    if cell_type is not None:
        if not isinstance(cell_type, np.ndarray):
            cell_type = np.asarray(cell_type, dtype=np.int32)
        elif cell_type.dtype != np.int32:
            cell_type = cell_type.astype(np.int32)

    return _core.run_daesc(
        snp_ad,
        snp_dp,
        cell_type,
        int(chunk_snps),
        int(max_iters),
        float(fisher_tol),
        int(min_depth),
        int(min_cells),
        bool(run_bh_fdr),
        float(fdr_alpha),
        int(n_types),
        stream,
        int(seed),
    )


__all__ = ["run"]
