# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.tools.umap — GPU-native UMAP dimensionality reduction.

Wraps ``singlet_gpu::embed::umap`` (cycle 10, ``embed/umap.h``).

Drop-in replacement for ``sc.tl.umap``.  All parameter names and
defaults match the scanpy 1.10 signature verified by the cycle-19
code-reader, with one intentional deviation:

    init_pos defaults to 'random' (not 'spectral') for reproducibility.
    The cycle-10 lit-scout identified cuML spectral init as non-deterministic
    (pitfall #1).  The 'spectral' value IS accepted and forwarded to C++, but
    users should prefer 'random' when bit-exact reproducibility is needed.

Result location (matches scanpy):
    adata.obsm['X_umap']   (or the key specified by key_added)

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.umap_embed`` is not yet
exposed by the cycle-20 binding extension.  The wrapper raises
``AttributeError`` with this tag until the binding is added.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Literal, Optional, Union

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# RNG helper
# ---------------------------------------------------------------------------

def _resolve_seed(rng: Optional[Union[int, "np.random.Generator"]]) -> int:
    """Convert scanpy-style ``rng`` to a uint64 seed for C++."""
    if rng is None:
        return 0
    if isinstance(rng, int):
        return rng & 0xFFFF_FFFF_FFFF_FFFF
    return int(np.random.default_rng(rng).integers(0, 2**63))


# ---------------------------------------------------------------------------
# Internal: extract the neighbors representation
# ---------------------------------------------------------------------------

def _get_neighbors_data(
    adata: "anndata.AnnData",
    neighbors_key: str,
) -> tuple:
    """
    Extract the kNN graph data needed by the UMAP kernel.

    Returns
    -------
    (distances, connectivities)  — both sparse (cupy or scipy)
    """
    if neighbors_key not in adata.uns:
        raise KeyError(
            f"Neighbors key '{neighbors_key}' not found in adata.uns.  "
            "Run singlet_gpu.pp.neighbors() (or sc.pp.neighbors()) first."
        )

    nbrs = adata.uns[neighbors_key]
    dist_key = nbrs.get("distances_key", "distances")
    conn_key = nbrs.get("connectivities_key", "connectivities")

    if dist_key not in adata.obsp:
        raise KeyError(
            f"Distances key '{dist_key}' not found in adata.obsp.  "
            "Run singlet_gpu.pp.neighbors() first."
        )
    if conn_key not in adata.obsp:
        raise KeyError(
            f"Connectivities key '{conn_key}' not found in adata.obsp.  "
            "Run singlet_gpu.pp.neighbors() first."
        )

    return adata.obsp[dist_key], adata.obsp[conn_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def umap(
    adata: "anndata.AnnData",
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: Optional[int] = None,
    alpha: float = 1.0,
    gamma: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: Union[Literal["spectral", "random"], "np.ndarray", None] = "random",
    rng: Optional[Union[int, "np.random.Generator"]] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    method: Literal["umap"] = "umap",
    key_added: Optional[str] = None,
    neighbors_key: str = "neighbors",
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native UMAP dimensionality reduction (cycle-10 kernel).

    Mirrors ``scanpy.tl.umap`` exactly — parameter names and defaults
    are identical so this function is a drop-in replacement, with ONE
    intentional deviation: ``init_pos`` defaults to ``'random'`` instead
    of ``'spectral'`` because cuML spectral initialisation is
    non-deterministic (cycle-10 lit-scout pitfall #1).  Pass
    ``init_pos='spectral'`` explicitly to match scanpy's default.

    Parameters
    ----------
    adata : AnnData
        Must contain a precomputed kNN graph in ``adata.obsp`` and
        ``adata.uns[neighbors_key]`` (from ``singlet_gpu.pp.neighbors``
        or ``sc.pp.neighbors``).
    min_dist : float, default 0.5
        Minimum distance between embedded points.  Controls compactness
        of clusters.  Maps to UMAP's ``min_dist``.
    spread : float, default 1.0
        Effective scale of embedded points.  Together with *min_dist*
        determines how clustered vs spread the embedding is.
    n_components : int, default 2
        Dimensionality of the embedding.
    maxiter : int or None, default None
        Number of optimisation iterations.  ``None`` → UMAP default
        (500 for datasets < 10k cells, 200 otherwise).
    alpha : float, default 1.0
        Initial learning rate for the embedding optimisation.
    gamma : float, default 1.0
        Weighting applied to negative samples in the loss.
    negative_sample_rate : int, default 5
        Number of negative samples per positive sample in each
        optimisation step.
    init_pos : 'spectral', 'random', ndarray, or None, default 'random'
        Initialisation of the embedding.
        - ``'random'``: random Gaussian (reproducible with *rng*).
        - ``'spectral'``: spectral embedding of the kNN graph.
          **Note**: cuML spectral init is non-deterministic regardless
          of *rng*.  Use ``'random'`` for reproducible results.
        - ndarray of shape (n_cells, n_components): user-provided init.
    rng : int or np.random.Generator or None, default None
        Random seed or generator.  ``None`` → seed 0.
    a : float or None, default None
        More specific control of the embedding parameters.  When set,
        overrides *spread* in the UMAP curve fit.
    b : float or None, default None
        More specific control of the embedding parameters.  When set,
        overrides *min_dist* in the UMAP curve fit.
    method : {'umap'}, default 'umap'
        Embedding method.  Only ``'umap'`` is supported; included for
        signature parity with scanpy.
    key_added : str or None, default None
        Key under which the embedding is written in ``adata.obsm``.
        ``None`` → ``'X_umap'``.
    neighbors_key : str, default 'neighbors'
        Key in ``adata.uns`` that holds the precomputed neighbors.
    copy : bool, default False
        Return a copy of *adata* with results written to it.  When
        ``False`` (default), *adata* is modified in-place and ``None``
        is returned.

    Returns
    -------
    None
        When ``copy=False`` (default, in-place).
    AnnData
        When ``copy=True`` — the modified copy.

    Raises
    ------
    AttributeError
        If ``_core.umap_embed`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    KeyError
        If the kNN graph is not present in ``adata.uns`` / ``adata.obsp``.
    ValueError
        If *method* is not ``'umap'`` or *n_components* < 2.

    Notes
    -----
    **Result location** (matches scanpy):
    ``adata.obsm['X_umap']`` — float32 array of shape (n_cells, n_components).

    **No host copies in the hot path.**  The kNN graph is already
    device-resident (cupy sparse).  The UMAP optimisation and the output
    embedding all remain on device until the cupy array is stored in
    ``obsm``.

    Examples
    --------
    Default UMAP (2D, random init)::

        import singlet_gpu.tools as sgt
        sgt.umap(adata)
        print(adata.obsm['X_umap'].shape)  # (n_cells, 2)

    3D embedding with explicit seed::

        sgt.umap(adata, n_components=3, rng=42, key_added='X_umap_3d')
    """
    if method != "umap":
        raise ValueError(f"Only method='umap' is supported, got {method!r}.")
    if n_components < 2:
        raise ValueError(f"n_components must be ≥2, got {n_components!r}.")

    import singlet_gpu._core as _core

    if not hasattr(_core, "umap_embed"):
        raise AttributeError(
            "_core.umap_embed is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose umap_embed before "
            "singlet_gpu.tools.umap() is callable."
        )

    working = copy_module.copy(adata) if copy else adata
    seed = _resolve_seed(rng)
    obsm_key = key_added if key_added is not None else "X_umap"

    distances, connectivities = _get_neighbors_data(working, neighbors_key)

    # Handle init_pos: convert ndarray to the raw __cuda_array_interface__ dict
    # so C++ can accept it without a host copy.
    init_arr = None
    init_mode = "random"
    if isinstance(init_pos, np.ndarray):
        init_mode = "provided"
        init_arr = init_pos
    elif init_pos is not None:
        init_mode = init_pos  # 'spectral' or 'random'

    embedding = _core.umap_embed(
        distances=distances,
        connectivities=connectivities,
        n_components=int(n_components),
        min_dist=float(min_dist),
        spread=float(spread),
        maxiter=int(maxiter) if maxiter is not None else -1,
        alpha=float(alpha),
        gamma=float(gamma),
        negative_sample_rate=int(negative_sample_rate),
        init_mode=init_mode,
        init_arr=init_arr,
        a=float(a) if a is not None else -1.0,
        b=float(b) if b is not None else -1.0,
        seed=seed,
    )

    working.obsm[obsm_key] = embedding

    return working if copy else None


__all__ = ["umap"]
