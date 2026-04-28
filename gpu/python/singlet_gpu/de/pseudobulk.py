# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.de.pseudobulk — GPU-native donor-aware pseudobulk DE.

Underlying C++ cycle: cycle 17 (de/donor_pseudobulk.h —
``singlet_gpu::de::donor_pseudobulk_de`` kernel).

Pseudobulk DE aggregates per-cell counts into per-donor pseudo-samples,
then fits a Negative Binomial GLM (with optional apeglm shrinkage) to
identify differentially expressed genes across conditions or cell types.

Donor labels come from singlify's ``donor_assignments.tsv`` — load with
``singlet_gpu.io.donor.load_donor_assignments(path)`` and add to
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
            f"singlet_gpu.io.donor.load_donor_assignments(path) and "
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
    ``singlet_gpu.io.donor.load_donor_assignments(path)``.

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
    Use ``singlet_gpu.io.donor.load_donor_assignments(path)`` to load as a
    ``pd.Series`` indexed by barcode, then assign::

        adata.obs['donor_id'] = load_donor_assignments(tsv_path)

    Examples
    --------
    Default (sum aggregation, apeglm shrinkage)::

        import singlet_gpu.de as sgde
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

    import singlet_gpu._core as _core

    if not hasattr(_core, "donor_pseudobulk_de"):
        raise AttributeError(
            "_core.donor_pseudobulk_de is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose donor_pseudobulk_de "
            "before singlet_gpu.de.pseudobulk_de() is callable."
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

    config = _build_pseudobulk_config(
        mode=mode,
        min_cells_per_pseudobulk=min_cells_per_pseudobulk,
        apeglm_shrinkage=apeglm_shrinkage,
        seed=seed,
    )

    # C++ kernel: donor_pseudobulk_de(mat, donor_codes, group_codes,
    #                                   n_groups, config)
    # Returns a list (one per group) of structs with fields:
    #   .lfc         — (n_genes,) float32 log2-fold-change
    #   .pvalue      — (n_genes,) float32 Wald-test p-value
    #   .padj        — (n_genes,) float32 BH-adjusted q-value
    #   .dispersion  — (n_genes,) float32 NB dispersion
    raw_results = _core.donor_pseudobulk_de(
        mat,
        donor_codes,
        group_codes,
        int(n_groups),
        config,
    )

    pb_dict: dict = {}
    for i, group_name in enumerate(group_names):
        r = raw_results[i]
        pb_dict[group_name] = pd.DataFrame(
            {
                "lfc":        np.asarray(r.lfc,        dtype=np.float32),
                "pvalue":     np.asarray(r.pvalue,     dtype=np.float32),
                "padj":       np.asarray(r.padj,       dtype=np.float32),
                "dispersion": np.asarray(r.dispersion, dtype=np.float32),
            },
            index=gene_names,
        )

    working.uns["donor_pseudobulk"] = pb_dict
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
