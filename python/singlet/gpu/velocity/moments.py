# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.velocity.moments — GPU-native first/second-order moment smoothing.

Underlying C++ cycle: cycle 15 (preprocess/velocity_prep.h —
``singlet::gpu::preprocess::velocity_moments`` kernel).

Drop-in for ``scvelo.pp.moments``.  All parameter names and defaults match
the scvelo 0.3 API.

Moment smoothing averages spliced and unspliced counts over each cell's
local neighbourhood to reduce measurement noise before velocity estimation.
The smoothed matrices are the inputs to ``singlet.gpu.velocity.velocity``.

Result location (matches scvelo)
---------------------------------
``adata.layers['Ms']`` — smoothed spliced counts (cells × genes), float32.
``adata.layers['Mu']`` — smoothed unspliced counts (cells × genes), float32.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.velocity_moments`` must
be exposed by the cycle-22 pybind11 binding extension.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_layer(
    adata: anndata.AnnData,
    layer_key: str,
) -> object:
    """
    Return the matrix at adata.layers[layer_key] (cupy or scipy sparse).
    """
    if layer_key not in adata.layers:
        raise KeyError(
            f"Layer '{layer_key}' not found in adata.layers.  "
            f"Available layers: {list(adata.layers.keys())}."
        )
    return adata.layers[layer_key]


def _get_connectivities(
    adata: anndata.AnnData,
    mode: str,
) -> object:
    """
    Extract the kNN connectivity matrix from adata.obsp for smoothing.

    Parameters
    ----------
    adata : AnnData
    mode : str
        ``'connectivities'`` — use the UMAP connectivities (scvelo default).
        ``'distances'``      — use raw distances.

    Returns
    -------
    sparse matrix (cupy or scipy), shape (n_cells × n_cells).
    """
    nbrs_key = "connectivities" if mode == "connectivities" else "distances"

    # Try to read the canonical key from uns['neighbors'] first.
    if "neighbors" in adata.uns:
        nbrs_key = adata.uns["neighbors"].get(f"{mode}_key", nbrs_key)

    if nbrs_key not in adata.obsp:
        raise KeyError(
            f"Connectivity key '{nbrs_key}' not found in adata.obsp.  "
            "Run singlet.gpu.pp.neighbors() (or sc.pp.neighbors()) first."
        )
    return adata.obsp[nbrs_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def moments(
    adata: anndata.AnnData,
    *,
    n_neighbors: int = 30,
    n_pcs: Optional[int] = None,
    mode: str = "connectivities",
    use_rep: Optional[str] = None,
    layer_spliced: str = "spliced",
    layer_unspliced: str = "unspliced",
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    GPU-native first/second-order moment smoothing for RNA velocity (cycle-15).

    Mirrors ``scvelo.pp.moments`` — parameter names and defaults are
    identical so this function is a drop-in replacement.

    For each cell, the smoothed expression is computed as the weighted
    average of the cell's own counts plus those of its *n_neighbors*
    nearest neighbours, using the kNN graph stored in ``adata.obsp``.
    This reduces measurement noise before velocity estimation.

    Parameters
    ----------
    adata : AnnData
        Must contain:

        - ``adata.layers[layer_spliced]``   — spliced count matrix.
        - ``adata.layers[layer_unspliced]`` — unspliced count matrix.
        - ``adata.obsp['connectivities']`` (or ``'distances'``) —
          precomputed kNN graph (from ``singlet.gpu.pp.neighbors`` or
          ``sc.pp.neighbors``).
    n_neighbors : int, default 30
        Number of nearest neighbours to use for smoothing.  The kNN
        graph must already be computed with at least *n_neighbors*
        neighbours per cell (scvelo convention: the graph is re-used,
        not recomputed).
    n_pcs : int or None, default None
        If provided, the PCA embedding is re-used directly — included
        for scvelo signature parity but not used when a precomputed
        kNN graph is present (scvelo 0.3 ignores it in that case too).
    mode : str, default ``"connectivities"``
        Which stored kNN graph to use for smoothing.
        ``'connectivities'`` (default) — UMAP fuzzy-graph weights.
        ``'distances'``               — raw distances (unnormalised).
    use_rep : str or None, default None
        Embedding key in ``adata.obsm`` to pass to ``neighbors`` if a kNN
        graph is not yet present.  ``None`` → ``'X_pca'``.
        Included for scvelo signature parity.
    layer_spliced : str, default ``"spliced"``
        Layer key for spliced counts.  Must exist in ``adata.layers``.
        Set to ``"exon"`` when loading singlify ``exon_counts.1pz`` output.
    layer_unspliced : str, default ``"unspliced"``
        Layer key for unspliced counts.  Must exist in ``adata.layers``.
        Set to ``"intron"`` when loading singlify ``intron_counts.1pz``.
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
        If ``_core.velocity_moments`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If the required layers or connectivity matrix are missing.

    Notes
    -----
    **Result location** (matches scvelo):

    - ``adata.layers['Ms']`` — smoothed spliced matrix, float32.
    - ``adata.layers['Mu']`` — smoothed unspliced matrix, float32.

    **Singlify note**: singlify outputs ``exon_counts.1pz`` (spliced) and
    ``intron_counts.1pz`` (unspliced).  Load them as::

        adata.layers['spliced'] = load_pz_to_anndata(sample_dir, modality='exon').X
        adata.layers['unspliced'] = load_pz_to_anndata(sample_dir, modality='intron').X

    or pass ``layer_spliced='exon'`` / ``layer_unspliced='intron'`` directly.

    Examples
    --------
    Default (scvelo-style)::

        import singlet.gpu.velocity as sgv
        sgv.moments(adata)
        print(adata.layers['Ms'].shape)  # (n_cells, n_genes)

    Using singlify layer names::

        sgv.moments(adata, layer_spliced='exon', layer_unspliced='intron')
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "velocity_moments"):
        raise AttributeError(
            "_core.velocity_moments is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose velocity_moments before "
            "singlet.gpu.velocity.moments() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    spliced_mat = _get_layer(working, layer_spliced)
    unspliced_mat = _get_layer(working, layer_unspliced)
    connectivity = _get_connectivities(working, mode)

    # C++ kernel: velocity_moments(spliced, unspliced, connectivity, n_neighbors)
    # Returns (Ms, Mu) — smoothed matrices, same shape as inputs.
    result = _core.velocity_moments(
        spliced_mat,
        unspliced_mat,
        connectivity,
        int(n_neighbors),
    )

    working.layers["Ms"] = np.asarray(result.Ms, dtype=np.float32)
    working.layers["Mu"] = np.asarray(result.Mu, dtype=np.float32)

    return working if copy else None


__all__ = ["moments"]
