# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.eqtl.nebula — GPU-native NEBULA single-cell eQTL mapping.

Underlying C++ kernel: singlet::gpu::eqtl::run_nebula (cycle 38, eqtl/nebula.h).

NEBULA fits a Negative Binomial mixed model at each (SNP, gene) pair via
GPU Fisher scoring in shared memory, with BH FDR correction per gene.

Reference: He et al. (NEBULA, Nature Commun Biol 2021).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Constants mirrored from C++ header for documentation
# ---------------------------------------------------------------------------
MAX_DONORS          = 200    # eqtl::MAX_DONORS_SHMEM
MAX_COVARIATES      = 7      # eqtl::MAX_COV (age + sex + PC1-5)
DEFAULT_CHUNK_SNPS  = 10000  # eqtl::DEFAULT_CHUNK_SNPS
DEFAULT_CHUNK_GENES = 500    # eqtl::DEFAULT_CHUNK_GENES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run(
    counts,
    snp_ad,
    snp_dp,
    donor_id,
    n_donors: int,
    *,
    covariates=None,
    n_cov: int = 0,
    chunk_snps: int = DEFAULT_CHUNK_SNPS,
    chunk_genes: int = DEFAULT_CHUNK_GENES,
    max_iters: int = 50,
    fisher_tol: float = 1e-5,
    run_bh_fdr: bool = True,
    fdr_alpha: float = 0.05,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native NEBULA single-cell eQTL mapping (cycle 38).

    Fits Negative Binomial mixed models for every (SNP chunk × gene chunk) pair
    using GPU Fisher scoring in shared memory.

    Parameters
    ----------
    counts : DeviceCsc
        Count matrix (cells × genes, CSC, float32).
    snp_ad : DeviceCsc
        Alternate-allele counts (SNP sites × cells, CSC).
        From singlify ``snp_ad.1pz``.
    snp_dp : DeviceCsc
        Total depth at SNP sites (SNP sites × cells, CSC).
        From singlify ``snp_dp.1pz``.
    donor_id : list[int] or ndarray int32, shape (n_cells,)
        Per-cell donor assignment.  Must be 0-indexed integers in [0, n_donors).
        Donors must be sorted (ascending) for correct cub::DeviceSegmentedReduce —
        either sort here or pass a pre-sorted array.
    n_donors : int
        Number of unique donors.  Must be ≤ 200 (shared-memory budget limit).
    covariates : ndarray float32, shape (n_donors, n_cov), optional
        Donor-level covariates (e.g., age=col0, sex=col1, PC1-5=cols2-6).
        Pass None for intercept-only model (default).
    n_cov : int, default 0
        Number of columns in *covariates*.  Ignored if covariates is None.
    chunk_snps : int, default 10000
        SNPs processed per device batch.
    chunk_genes : int, default 500
        Genes processed per device batch.
    max_iters : int, default 50
        Fisher scoring maximum iterations per (SNP, gene) pair.
    fisher_tol : float, default 1e-5
        Fisher scoring convergence threshold on |delta_beta|.
    run_bh_fdr : bool, default True
        Apply BH FDR correction per gene across all SNPs.
    fdr_alpha : float, default 0.05
    stream : int or None
    seed : int, default 0
        Not used (algorithm is deterministic).

    Returns
    -------
    dict
        ``"beta"``     — float32 ndarray (n_snps × n_genes) eQTL effect sizes.
        ``"se"``       — float32 ndarray standard errors.
        ``"sigma_sq"`` — float32 ndarray random-effect variances.
        ``"p_value"``  — float32 ndarray Wald p-values (uncorrected).
        ``"p_adj"``    — float32 ndarray BH-adjusted q-values.
        ``"converged"``— uint8 ndarray (1=converged within max_iters).
        ``"n_snps"``, ``"n_genes"`` — int.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    ValueError
        If n_donors > 200 or n_cov > 7.

    Notes
    -----
    **Chunked output warning**: at 1M SNPs × 20k genes, the output is ~240 GB.
    Use ``chunk_snps`` and ``chunk_genes`` to control peak device memory (~320 MB
    per chunk).  For large datasets, consider using the C++ callback API directly
    (``_core.run_nebula`` + chunk callback) to stream results to disk.

    **donor_id must be sorted**: NEBULA uses ``cub::DeviceSegmentedReduce`` which
    requires sorted donor IDs.  Sort on host before calling::

        import numpy as np
        sort_idx = np.argsort(donor_id)
        # reorder cells in counts, snp_ad, snp_dp accordingly

    Examples
    --------
    >>> result = singlet.gpu.eqtl.nebula.run(
    ...     counts, snp_ad, snp_dp,
    ...     donor_id=cell_donor_ids,
    ...     n_donors=80,
    ...     covariates=donor_covariates,  # (80, 7) float32
    ...     n_cov=7,
    ... )
    >>> import pandas as pd
    >>> beta_df = pd.DataFrame(result["beta"], columns=gene_names)
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "run_nebula"):
        raise ImportError(
            "_core.run_nebula is not available.  "
            "Install with: pip install singlet[gpu]"
        )

    if n_donors > MAX_DONORS:
        raise ValueError(
            f"n_donors={n_donors} exceeds MAX_DONORS={MAX_DONORS} "
            "(shared-memory budget limit).  Downsample donors."
        )
    if n_cov > MAX_COVARIATES:
        raise ValueError(
            f"n_cov={n_cov} exceeds MAX_COVARIATES={MAX_COVARIATES}."
        )

    # Coerce donor_id to int32 numpy array for consistent C++ path.
    if not isinstance(donor_id, np.ndarray):
        donor_id = np.asarray(donor_id, dtype=np.int32)
    elif donor_id.dtype != np.int32:
        donor_id = donor_id.astype(np.int32)

    # Coerce covariates to float32 c-contiguous numpy array.
    if covariates is not None and n_cov > 0:
        if not isinstance(covariates, np.ndarray):
            covariates = np.asarray(covariates, dtype=np.float32)
        elif covariates.dtype != np.float32:
            covariates = covariates.astype(np.float32)
        if not covariates.flags["C_CONTIGUOUS"]:
            covariates = np.ascontiguousarray(covariates)
    else:
        covariates = None
        n_cov = 0

    return _core.run_nebula(
        counts,
        snp_ad,
        snp_dp,
        donor_id,
        int(n_donors),
        covariates,
        int(n_cov),
        int(chunk_snps),
        int(chunk_genes),
        int(max_iters),
        float(fisher_tol),
        bool(run_bh_fdr),
        float(fdr_alpha),
        stream,
        int(seed),
    )


__all__ = ["run", "MAX_DONORS", "MAX_COVARIATES", "DEFAULT_CHUNK_SNPS", "DEFAULT_CHUNK_GENES"]
