# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.tools.leiden — GPU-native Leiden community detection.

Wraps ``singlet_gpu::graph::leiden`` (cycle 9, ``graph/leiden.h``).

Drop-in replacement for ``sc.tl.leiden``.  All parameter names and
defaults match the scanpy 1.10 signature verified by the cycle-19
code-reader:

    import singlet_gpu.tools as sgt
    sgt.leiden(adata, resolution=0.5)   # instead of sc.tl.leiden(adata, 0.5)

Result location (matches scanpy):
    adata.obs['leiden']   (or the key specified by *key_added*)

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.leiden_partition`` is not
yet exposed by the cycle-20 binding extension.  The wrapper raises
``AttributeError`` with this tag until the binding is added.
"""

from __future__ import annotations

import copy as copy_module
from typing import TYPE_CHECKING, Iterable, Optional, Sequence, Tuple, Type, Union

import numpy as np

if TYPE_CHECKING:
    import anndata
    import scipy.sparse


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
# Internal: extract connectivities
# ---------------------------------------------------------------------------

def _get_connectivities(
    adata: "anndata.AnnData",
    neighbors_key: Optional[str],
    obsp: Optional[str],
) -> object:
    """
    Return the connectivities matrix for Leiden.

    Mirrors scanpy's resolution priority (obsp > uns['neighbors'] > default).

    Parameters
    ----------
    adata : AnnData
    neighbors_key : str or None
        Key in ``adata.uns`` that holds the neighbors dict.
        ``None`` defaults to ``'neighbors'``.
    obsp : str or None
        Direct key in ``adata.obsp`` to use.  Overrides *neighbors_key*.

    Returns
    -------
    sparse matrix (cupy.sparse.csr_matrix or scipy.sparse.csr_matrix)
    """
    if obsp is not None:
        if obsp not in adata.obsp:
            raise KeyError(
                f"obsp='{obsp}' not found in adata.obsp. "
                f"Available keys: {list(adata.obsp.keys())}"
            )
        return adata.obsp[obsp]

    nbrs_key = neighbors_key if neighbors_key is not None else "neighbors"

    if nbrs_key in adata.uns:
        conn_key = adata.uns[nbrs_key].get("connectivities_key", "connectivities")
    else:
        conn_key = "connectivities"

    if conn_key not in adata.obsp:
        raise KeyError(
            f"Connectivities key '{conn_key}' not found in adata.obsp. "
            "Run singlet_gpu.pp.neighbors() (or sc.pp.neighbors()) first."
        )
    return adata.obsp[conn_key]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def leiden(
    adata: "anndata.AnnData",
    resolution: float = 1,
    *,
    restrict_to: Optional[Tuple[str, Sequence[str]]] = None,
    rng: Optional[Union[int, "np.random.Generator"]] = None,
    key_added: str = "leiden",
    adjacency: Optional[object] = None,
    directed: Optional[bool] = None,
    use_weights: bool = True,
    n_iterations: int = -1,
    partition_type: Optional[Type] = None,
    neighbors_key: Optional[str] = None,
    obsp: Optional[str] = None,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    Leiden community detection on the kNN graph (GPU-native, cycle-9 kernel).

    Mirrors ``scanpy.tl.leiden`` exactly — parameter names and defaults are
    identical so this function is a drop-in replacement.

    The Leiden algorithm is run via the cycle-9 ``graph/leiden.h`` GPU
    kernel (cuGraph backend).  Community labels are written as a
    categorical Pandas Series to ``adata.obs[key_added]``.

    Parameters
    ----------
    adata : AnnData
        Must contain a connectivities matrix (from ``singlet_gpu.pp.neighbors``
        or ``sc.pp.neighbors``).
    resolution : float, default 1
        Resolution parameter controlling partition granularity.  Higher
        values produce more (smaller) communities.  This is the first
        positional argument, matching scanpy convention.
    restrict_to : (str, list of str) or None, default None
        Restrict clustering to a subset of cells.  Tuple of
        ``(obs_column, values)`` — only cells whose *obs_column* value
        is in *values* are considered.  ``None`` → cluster all cells.
    rng : int or np.random.Generator or None, default None
        Random seed or generator for Leiden's stochastic initialisation.
        ``None`` → seed 0.  Matches scanpy's ``rng`` convention.
    key_added : str, default 'leiden'
        Key under which community labels are written in ``adata.obs``.
    adjacency : sparse matrix or None, default None
        Custom adjacency matrix to use instead of reading from
        ``adata.obsp``.  Must be a square (cells × cells) sparse matrix.
    directed : bool or None, default None
        Whether the graph is directed.  ``None`` → auto-detect from the
        adjacency matrix symmetry.
    use_weights : bool, default True
        Use edge weights from the connectivities matrix when optimising
        the modularity.  Setting ``False`` treats the graph as unweighted.
    n_iterations : int, default -1
        Number of Leiden iterations.  ``-1`` → run until convergence.
    partition_type : type or None, default None
        Leiden partition type class.  ``None`` → ``RBConfigurationVertexPartition``
        (the scanpy default corresponding to the CPM / Reichardt–Bornholdt
        objective with resolution parameter).  The C++ binding accepts the
        string representation of the class name.
    neighbors_key : str or None, default None
        Key in ``adata.uns`` that holds the neighbors metadata dict.
        ``None`` → ``'neighbors'``.
    obsp : str or None, default None
        Direct key in ``adata.obsp`` for the connectivities matrix.
        Overrides *neighbors_key*.
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
        If ``_core.leiden_partition`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    KeyError
        If the connectivities matrix cannot be found in ``adata.obsp``.
    ValueError
        If *resolution* ≤ 0.

    Notes
    -----
    **Result location** (matches scanpy):
    ``adata.obs[key_added]`` — ``pandas.Categorical`` of string community
    labels (``'0'``, ``'1'``, …).

    When *restrict_to* is set, cells outside the restriction mask keep
    their existing ``adata.obs[key_added]`` value (or ``NaN`` if the
    column did not previously exist), matching scanpy's behaviour.

    Examples
    --------
    Default Leiden (resolution=1)::

        import singlet_gpu.tools as sgt
        sgt.leiden(adata)
        print(adata.obs['leiden'].value_counts())

    Higher resolution (more communities)::

        sgt.leiden(adata, resolution=2.0, rng=42)

    Restrict to a T-cell cluster::

        sgt.leiden(adata, restrict_to=('cell_type', ['T cell']),
                   key_added='leiden_Tcell')
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be >0, got {resolution!r}.")

    import singlet_gpu._core as _core

    if not hasattr(_core, "leiden_partition"):
        raise AttributeError(
            "_core.leiden_partition is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose leiden_partition before "
            "singlet_gpu.tools.leiden() is callable."
        )

    import pandas as pd

    working = copy_module.copy(adata) if copy else adata
    seed = _resolve_seed(rng)

    # Resolve adjacency matrix.
    if adjacency is not None:
        conn_mat = adjacency
    else:
        conn_mat = _get_connectivities(working, neighbors_key, obsp)

    # Build restriction mask if needed.
    restrict_mask = None
    if restrict_to is not None:
        restrict_col, restrict_vals = restrict_to
        if restrict_col not in working.obs.columns:
            raise KeyError(
                f"restrict_to column '{restrict_col}' not found in adata.obs."
            )
        restrict_mask = working.obs[restrict_col].isin(restrict_vals).values

    # Resolve partition type name for C++ binding.
    part_type_name = (
        partition_type.__name__
        if partition_type is not None
        else "RBConfigurationVertexPartition"
    )

    # Dispatch to C++ Leiden kernel.
    raw_labels = _core.leiden_partition(
        conn_mat,
        resolution=float(resolution),
        n_iterations=int(n_iterations),
        use_weights=bool(use_weights),
        directed=bool(directed) if directed is not None else None,
        partition_type=part_type_name,
        seed=seed,
        restrict_mask=restrict_mask,
    )

    # Write results: labels as pandas Categorical (matches scanpy).
    labels = np.asarray(raw_labels, dtype=str)
    if restrict_to is not None:
        # Merge back: cells outside the mask keep existing or NaN.
        full_labels = (
            working.obs[key_added].values.astype(object)
            if key_added in working.obs.columns
            else np.full(working.n_obs, None, dtype=object)
        )
        full_labels[restrict_mask] = labels[restrict_mask]
        labels = full_labels

    working.obs[key_added] = pd.Categorical(labels.astype(str))

    return working if copy else None


__all__ = ["leiden"]
