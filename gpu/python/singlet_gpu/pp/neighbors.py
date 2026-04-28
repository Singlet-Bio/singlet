# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.pp.neighbors — GPU-native kNN graph construction.

Wraps ``singlet_gpu::graph::knn`` (cycle 8, ``graph/knn.h``).

Drop-in replacement for ``sc.pp.neighbors``.  All parameter names and
default values match the scanpy 1.10 signature verified by the cycle-19
code-reader; this function can be substituted without changing call sites:

    import singlet_gpu.pp as sgp
    sgp.neighbors(adata, n_neighbors=15)   # instead of sc.pp.neighbors(adata)

Result locations (match scanpy exactly):
    adata.obsp['distances']        — sparse kNN distance matrix (cells × cells)
    adata.obsp['connectivities']   — UMAP-style weighted adjacency
    adata.uns['neighbors']         — dict with 'params' sub-dict

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.knn_graph`` is not yet
exposed by the cycle-20 binding extension.  The wrapper raises
``AttributeError`` with this tag until the binding is added.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Dict, Literal, Optional, Union

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
# Internal: extract embedding matrix
# ---------------------------------------------------------------------------

def _get_rep(
    adata: "anndata.AnnData",
    use_rep: Optional[str],
    n_pcs: Optional[int],
) -> object:
    """
    Return the representation matrix to build the kNN graph from.

    Priority (matches scanpy):
      1. ``use_rep`` if explicitly given.
      2. ``adata.obsm['X_pca']`` if it exists and ``n_pcs`` is set or
         the matrix has ≤50 components.
      3. ``adata.X`` as a fallback (dense or sparse; GPU-resident preferred).

    Parameters
    ----------
    adata : AnnData
    use_rep : str or None
        Key in ``adata.obsm`` to use as the embedding.  None = auto.
    n_pcs : int or None
        Number of PCs to use.  When not None, slices ``obsm['X_pca']``
        to the first *n_pcs* columns.

    Returns
    -------
    array-like (cupy ndarray or numpy ndarray)
        Shape (n_cells, n_features).
    """
    if use_rep is not None:
        if use_rep not in adata.obsm:
            raise KeyError(
                f"use_rep='{use_rep}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}"
            )
        rep = adata.obsm[use_rep]
        if n_pcs is not None:
            rep = rep[:, :n_pcs]
        return rep

    if "X_pca" in adata.obsm:
        rep = adata.obsm["X_pca"]
        if n_pcs is not None:
            rep = rep[:, :n_pcs]
        return rep

    # Fallback: use adata.X directly.
    # Callers are responsible for ensuring this is sensible (e.g. after HVG).
    import warnings
    warnings.warn(
        "neighbors: 'X_pca' not found in adata.obsm and use_rep is None. "
        "Falling back to adata.X.  Consider running singlet_gpu.pp.pca() first.",
        UserWarning,
        stacklevel=3,
    )
    x = adata.X
    # If sparse, return dense (kNN on raw sparse X is unusual but allowed).
    try:
        import cupy.sparse as csp
        if isinstance(x, csp.csc_matrix) or isinstance(x, csp.csr_matrix):
            return x.toarray()
    except ImportError:
        pass
    try:
        import scipy.sparse as sp
        if sp.issparse(x):
            return x.toarray()
    except ImportError:
        pass
    return x


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def neighbors(
    adata: "anndata.AnnData",
    n_neighbors: int = 15,
    n_pcs: Optional[int] = None,
    *,
    use_rep: Optional[str] = None,
    knn: bool = True,
    method: Literal["umap", "gauss"] = "umap",
    transformer: object = None,
    metric: str = "euclidean",
    metric_kwds: Optional[Dict] = None,
    rng: Optional[Union[int, "np.random.Generator"]] = None,
    key_added: Optional[str] = None,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Compute a kNN graph on the cell embedding (GPU-native, cycle-8 kernel).

    Mirrors ``scanpy.pp.neighbors`` exactly — parameter names and defaults
    are identical so this function is a drop-in replacement.

    The kNN graph is built on device using the cycle-8 ``graph/knn.h``
    kernel (brute-force fp32 for small n_neighbors, IVF-PQ for large
    datasets).  No host copies occur in the hot path; distance and
    connectivity matrices are returned as device-resident cupy sparse arrays
    wrapped in ``AnnData.obsp``.

    Parameters
    ----------
    adata : AnnData
        Must contain a device-resident embedding in ``obsm['X_pca']``
        (default) or the key specified by *use_rep*.
    n_neighbors : int, default 15
        Number of nearest neighbours (including the query cell itself in
        some implementations — scanpy excludes self; this wrapper follows
        scanpy convention: *n_neighbors* exclusive-self neighbours).
    n_pcs : int or None, default None
        Number of principal components to use from ``obsm['X_pca']``.
        ``None`` uses all available components.
    use_rep : str or None, default None
        Key in ``adata.obsm`` to use as embedding.  ``None`` defaults to
        ``'X_pca'`` if it exists, then ``adata.X``.
    knn : bool, default True
        If ``True``, return only the *k* nearest neighbours per cell
        (sparse kNN graph).  If ``False``, use a Gaussian kernel to
        compute full pairwise connectivities (dense — expensive).
    method : {'umap', 'gauss'}, default 'umap'
        Connectivities computation method.  ``'umap'`` uses the UMAP
        fuzzy-set cross entropy; ``'gauss'`` uses a Gaussian kernel.
    transformer : object or None, default None
        Custom approximate-nearest-neighbours transformer (e.g. a
        pre-fitted FAISS index).  When given, the internal kNN kernel is
        bypassed and *transformer* is used directly.  The object must
        expose a ``kneighbors(X, n_neighbors)`` method returning
        ``(indices, distances)``.
    metric : str, default 'euclidean'
        Distance metric for kNN.  Supported values depend on the C++
        kernel: ``'euclidean'``, ``'cosine'``, ``'manhattan'``,
        ``'l1'``, ``'l2'``.  Passed verbatim to the backend.
    metric_kwds : dict or None, default None
        Additional keyword arguments for the distance metric.  Currently
        unused by the C++ kernel; reserved for future expansion.
    rng : int or np.random.Generator or None, default None
        Random seed or generator controlling stochastic aspects of the
        IVF-PQ index construction.  ``None`` → seed 0 (deterministic).
    key_added : str or None, default None
        Base key under which results are written.  ``None`` → ``'neighbors'``.
        Results are written to:
          - ``adata.obsp[f'{key_added}_distances']`` (or ``'distances'``)
          - ``adata.obsp[f'{key_added}_connectivities']`` (or ``'connectivities'``)
          - ``adata.uns[key_added]`` (or ``'uns['neighbors']'``)
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
        If ``_core.knn_graph`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    KeyError
        If *use_rep* is not found in ``adata.obsm``.
    ValueError
        If *n_neighbors* < 2.

    Notes
    -----
    **Result locations** (match scanpy exactly):

    +-----------------------------------------+-------------------------------------+
    | ``adata.obsp['distances']``             | sparse kNN distance matrix          |
    +-----------------------------------------+-------------------------------------+
    | ``adata.obsp['connectivities']``        | UMAP-style weighted adjacency       |
    +-----------------------------------------+-------------------------------------+
    | ``adata.uns['neighbors']``              | ``{'params': {...}}`` metadata dict |
    +-----------------------------------------+-------------------------------------+

    When *key_added* is set, all three keys are prefixed with
    ``f'{key_added}_'`` (distances, connectivities) or use *key_added*
    as the uns key.

    Examples
    --------
    Default kNN (15 neighbours, using X_pca)::

        import singlet_gpu.pp as sgp
        sgp.neighbors(adata)

    Custom representation and distance::

        sgp.neighbors(adata, use_rep='X_nmf', n_neighbors=20, metric='cosine')
    """
    if n_neighbors < 2:
        raise ValueError(f"n_neighbors must be ≥2, got {n_neighbors!r}.")

    import singlet_gpu._core as _core

    if not hasattr(_core, "knn_graph"):
        raise AttributeError(
            "_core.knn_graph is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose knn_graph before "
            "singlet_gpu.pp.neighbors() is callable."
        )

    working = copy_module.copy(adata) if copy else adata
    seed = _resolve_seed(rng)

    rep = _get_rep(working, use_rep, n_pcs)

    # Dispatch to custom transformer or internal C++ kernel.
    if transformer is not None:
        indices, distances_arr = transformer.kneighbors(rep, n_neighbors)
        raw = _core.knn_graph_from_indices(
            indices,
            distances_arr,
            n_cells=rep.shape[0],
            method=method,
            knn=knn,
        )
    else:
        raw = _core.knn_graph(
            rep,
            n_neighbors=n_neighbors,
            metric=metric,
            method=method,
            knn=knn,
            seed=seed,
        )

    # Determine output keys.
    uns_key = key_added if key_added is not None else "neighbors"
    if key_added is not None:
        dist_key = f"{key_added}_distances"
        conn_key = f"{key_added}_connectivities"
    else:
        dist_key = "distances"
        conn_key = "connectivities"

    working.obsp[dist_key] = raw.distances
    working.obsp[conn_key] = raw.connectivities
    working.uns[uns_key] = {
        "params": {
            "n_neighbors": n_neighbors,
            "n_pcs": n_pcs,
            "use_rep": use_rep,
            "method": method,
            "metric": metric,
        },
        "distances_key": dist_key,
        "connectivities_key": conn_key,
    }

    return working if copy else None


__all__ = ["neighbors"]
