# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.de.pseudobulk — GPU-native donor-aware pseudobulk DE.

Underlying C++ cycle: cycle 17 (de/donor_pseudobulk.h —
``singlet::gpu::de::donor_pseudobulk_de`` kernel).

Pseudobulk DE aggregates per-cell counts into per-donor pseudo-samples,
then fits a Negative Binomial GLM (with optional apeglm shrinkage) to
identify differentially expressed genes across conditions or cell types.

Donor labels come from singlify's ``donor_assignments.tsv`` — load with
``singlet.gpu.io.donor.load_donor_assignments(path)`` and add to
``adata.obs['donor_id']``.

Result location
---------------
``adata.uns['donor_pseudobulk']`` — dict mapping cell_type → per-gene
    DataFrame with columns: lfc, pvalue, padj, dispersion.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.donor_pseudobulk_de``
must be exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata
    import pandas as pd


# ---------------------------------------------------------------------------
# Valid parameter sets
# ---------------------------------------------------------------------------

_VALID_MODES = ("sum", "mean")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_obs_column(
    adata: "anndata.AnnData",
    col: str,
) -> None:
    """Raise KeyError with helpful message if obs column is missing."""
    if col not in adata.obs.columns:
        raise KeyError(
            f"obs column '{col}' not found in adata.obs.  "
            f"Available columns: {list(adata.obs.columns)}.  "
            f"Hint: load donor assignments with "
            f"singlet.gpu.io.donor.load_donor_assignments(path) and "
            f"assign to adata.obs['donor_id']."
        )


def _build_pseudobulk_config(
    *,
    mode: str,
    min_cells_per_pseudobulk: int,
    apeglm_shrinkage: bool,
    seed: int,
) -> dict:
    """Assemble config dict passed to the C++ binding."""
    return {
        "mode": str(mode),
        "min_cells_per_pseudobulk": int(min_cells_per_pseudobulk),
        "apeglm_shrinkage": bool(apeglm_shrinkage),
        "seed": int(seed),
    }


def _extract_matrix(adata: "anndata.AnnData") -> object:
    """Return adata.X (cells × genes), device-resident if available."""
    return adata.X


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def pseudobulk_de(
    adata: "anndata.AnnData",
    *,
    sample_col: str = "donor_id",
    groupby: str = "cell_type",
    mode: str = "sum",
    min_cells_per_pseudobulk: int = 10,
    apeglm_shrinkage: bool = True,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native donor-aware pseudobulk DE (cycle-17 NB GLM kernel).

    Aggregates per-cell counts into per-donor pseudo-samples within each
    cell-type group, then fits a Negative Binomial GLM to detect genes
    differentially expressed across conditions.  Optional apeglm shrinkage
    is applied to log-fold-change estimates (DESeq2-style).

    Donor labels are expected in ``adata.obs[sample_col]`` — load them
    from singlify's ``donor_assignments.tsv`` using
    ``singlet.gpu.io.donor.load_donor_assignments(path)``.

    Parameters
    ----------
    adata : AnnData
        Must contain:

        - ``adata.X`` — raw count matrix (cells × genes), integer dtype
          preferred; float32 also accepted (kernel casts internally).
        - ``adata.obs[sample_col]`` — per-cell donor label (str or int).
        - ``adata.obs[groupby]``    — per-cell cluster/cell-type label.
    sample_col : str, default ``"donor_id"``
        Observation column identifying donor / pseudo-sample.
    groupby : str, default ``"cell_type"``
        Observation column defining groups (e.g., cell types) within which
        DE is run independently.
    mode : str, default ``"sum"``
        Aggregation method when building pseudo-samples.
        ``'sum'``  — sum raw counts (recommended for NB GLM).
        ``'mean'`` — mean counts (not recommended for NB).
    min_cells_per_pseudobulk : int, default 10
        Minimum number of cells a donor must contribute to a group to be
        included in that group's DE analysis.  Donors below this threshold
        are dropped from that group.
    apeglm_shrinkage : bool, default True
        Apply apeglm adaptive shrinkage to LFC estimates.  Reduces
        variance for genes with low counts (DESeq2 / apeglm convention).
        This is a singlet-gpu addition (not present in base DESeq2 by
        default, but available via the ``apeglm`` R package).
    seed : int, default 0
        Random seed for IRLS initialisation.
    copy : bool, default False
        Return a modified copy of *adata*.  When ``False`` (default),
        *adata* is modified in-place and ``None`` is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.donor_pseudobulk_de`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If *sample_col* or *groupby* is not in ``adata.obs``.
    ValueError
        If *mode* is not ``'sum'`` or ``'mean'``.

    Notes
    -----
    **Result location**:

    ``adata.uns['donor_pseudobulk']`` — dict with structure::

        {
            "<cell_type>": pd.DataFrame(
                index=gene_names,
                columns=['lfc', 'pvalue', 'padj', 'dispersion']
            ),
            ...
        }

    where ``lfc`` is the shrunken log2-fold-change (if apeglm),
    ``pvalue`` the Wald-test p-value, ``padj`` the BH-adjusted q-value,
    and ``dispersion`` the NB dispersion estimate per gene.

    **Donor label loading**: singlify writes ``donor_assignments.tsv``
    with columns ``barcode``, ``donor_id``, ``prob_max``, ``prob_doublet``.
    Use ``singlet.gpu.io.donor.load_donor_assignments(path)`` to load as a
    ``pd.Series`` indexed by barcode, then assign::

        adata.obs['donor_id'] = load_donor_assignments(tsv_path)

    Examples
    --------
    Default (sum aggregation, apeglm shrinkage)::

        import singlet.gpu.de as sgde
        sgde.pseudobulk_de(adata)
        print(list(adata.uns['donor_pseudobulk'].keys()))  # cell types

    Access per-cell-type results::

        df = adata.uns['donor_pseudobulk']['T cell']
        print(df.nsmallest(10, 'padj'))

    No shrinkage::

        sgde.pseudobulk_de(adata, apeglm_shrinkage=False)
    """
    if mode not in _VALID_MODES:
        raise ValueError(
            f"mode='{mode}' not recognised.  "
            f"Choose from: {_VALID_MODES}."
        )

    _validate_obs_column(adata, sample_col)
    _validate_obs_column(adata, groupby)

    import singlet.gpu._core as _core

    if not hasattr(_core, "donor_pseudobulk_de"):
        raise AttributeError(
            "_core.donor_pseudobulk_de is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose donor_pseudobulk_de "
            "before singlet.gpu.de.pseudobulk_de() is callable."
        )

    import pandas as pd

    working = copy_module.copy(adata) if copy else adata

    mat = _extract_matrix(working)                           # (cells × genes)
    gene_names  = list(working.var_names)
    donor_codes = working.obs[sample_col].astype("category").cat.codes.to_numpy(
        dtype=np.int32
    )
    group_codes = working.obs[groupby].astype("category").cat.codes.to_numpy(
        dtype=np.int32
    )
    group_names = list(
        working.obs[groupby].astype("category").cat.categories
    )
    donor_names = list(
        working.obs[sample_col].astype("category").cat.categories
    )
    n_groups  = len(group_names)
    n_donors  = len(donor_names)

    # _core.donor_pseudobulk_de signature (py::kw_only after n_donors):
    #   donor_pseudobulk_de(mat, cluster_labels, n_clusters, donor_labels,
    #                        n_donors, *, min_cells_per_pseudobulk=10,
    #                        max_irls_iters=50, irls_tol=1e-5,
    #                        max_dispersion_iters=10, apeglm_shrinkage=True,
    #                        top_n=100, seed=0)
    # mat must be PyDeviceCsc; upload via the same pattern as score_genes /
    # lineage wrappers (CYCLE-194 / CYCLE-204).
    import cupy as cp
    try:
        import cupyx.scipy.sparse as csp
    except ImportError:
        import cupy.sparse as csp
    import scipy.sparse as sp

    # CYCLE-256 (CYCLE-211 root cause): _core.donor_pseudobulk_de expects a
    # (genes × cells) CSC — its kernel reads `mat.rows` as n_genes and
    # `mat.cols` as n_cells (donor_pseudobulk.h:873-874, "m genes × n cells").
    # `_extract_matrix` returns AnnData (cells × genes); transpose before
    # building the cupy CSC so the C++ axis convention is satisfied.
    if csp.issparse(mat):
        cu_csc = mat.T
        cu_csc = cu_csc if isinstance(cu_csc, csp.csc_matrix) else cu_csc.tocsc()
        if cu_csc.dtype != cp.float32:
            cu_csc = cu_csc.astype(cp.float32)
    elif sp.issparse(mat):
        cu_csc = csp.csc_matrix(mat.T.tocsc().astype(np.float32))
    elif hasattr(mat, "__cuda_array_interface__"):
        cu_csc = csp.csc_matrix(cp.asarray(mat).T.astype(cp.float32))
    else:
        cu_csc = csp.csc_matrix(cp.asarray(mat, dtype=cp.float32).T)
    device_mat = _core.from_cupy_csr(cu_csc)

    # cluster_labels and donor_labels must expose __cuda_array_interface__
    # (per the binding contract).  Upload the int32 numpy arrays to device.
    group_codes_d = cp.asarray(group_codes, dtype=cp.int32)
    donor_codes_d = cp.asarray(donor_codes, dtype=cp.int32)

    raw_results = _core.donor_pseudobulk_de(
        device_mat,
        group_codes_d,         # cluster_labels (device int32)
        int(n_groups),         # n_clusters
        donor_codes_d,         # donor_labels (device int32)
        int(n_donors),         # n_donors
        min_cells_per_pseudobulk=int(min_cells_per_pseudobulk),
        apeglm_shrinkage=bool(apeglm_shrinkage),
        seed=int(seed),
    )

    # _core.DonorPseudobulkResult is a SINGLE struct (not subscriptable) with
    # per-gene per-cluster CAI views:
    #   log2_fc_view:  float32 (m × n_clusters × (n_donors-1))  row-major
    #   p_values_view: float32 (m × n_clusters)
    #   p_adj_view:    float32 (m × n_clusters)
    #   dispersion_view: float32 (m × n_clusters)
    #   pseudobulk_view: float32 (m × n_clusters × n_donors)
    # Each per-gene index excludes filtered-out genes (n_genes_passing_filter).
    class _CaiView:  # see CYCLE-193 / §J.13
        def __init__(self, d): self.__cuda_array_interface__ = d

    n_pass    = int(raw_results.n_genes_passing_filter)
    n_donors_ = n_donors  # convenience alias

    log2_fc    = cp.asarray(_CaiView(raw_results.log2_fc_view)).get() \
                    .reshape(n_pass, n_groups, max(n_donors_ - 1, 1))
    p_values   = cp.asarray(_CaiView(raw_results.p_values_view)).get() \
                    .reshape(n_pass, n_groups)
    p_adj      = cp.asarray(_CaiView(raw_results.p_adj_view)).get() \
                    .reshape(n_pass, n_groups)
    dispersion = cp.asarray(_CaiView(raw_results.dispersion_view)).get() \
                    .reshape(n_pass, n_groups)

    # Test contract: uns['donor_pseudobulk'] = {'lfc', 'pvals', 'qvals',
    # 'dispersion'} dict of arrays.  We expose the average donor contrast for
    # lfc (mean across n_donors-1 columns).
    working.uns["donor_pseudobulk"] = {
        "lfc":        log2_fc.mean(axis=2).astype(np.float32),  # (n_pass × n_groups)
        "pvals":      p_values.astype(np.float32),
        "qvals":      p_adj.astype(np.float32),
        "dispersion": dispersion.astype(np.float32),
        "groups":     group_names,
        "n_genes_passing_filter": n_pass,
    }
    working.uns["donor_pseudobulk_params"] = {
        "sample_col": sample_col,
        "groupby": groupby,
        "mode": mode,
        "min_cells_per_pseudobulk": min_cells_per_pseudobulk,
        "apeglm_shrinkage": apeglm_shrinkage,
        "seed": seed,
        "n_groups": n_groups,
        "groups": group_names,
        "n_donors": len(donor_names),
    }

    return working if copy else None


__all__ = ["pseudobulk_de"]
