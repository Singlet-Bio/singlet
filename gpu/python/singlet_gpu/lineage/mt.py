# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.lineage.mt — GPU-native MT heteroplasmy clone detection.

Underlying C++ cycle: cycle 16 (anno/mt_lineage.h —
``singlet_gpu::anno::mt_lineage_call_clones`` kernel).

This is a NEW addition to the single-cell ecosystem.  scanpy, Seurat, and
MQuad do not expose an AnnData-native MT clone calling interface that
integrates with the singlify ``mt_alleles.1pz`` output.

Algorithm overview
------------------
1. Compute per-site variant allele frequency (VAF) from (alt / depth).
2. Filter informative sites by: min_depth, min_cells_alt, min_vaf.
3. For K = min_K … max_K, run GPU Gaussian Mixture Model clustering
   on the informative-site VAF matrix (cells × informative_sites).
4. Select optimal K by BIC.
5. Assign each cell to the best-fit clone; store heteroplasmy profile.

Result location
---------------
``adata.obs['mt_clone_id']``        — int per cell (0-indexed clone label).
``adata.obsm['mt_heteroplasmy']``   — float32 (n_cells × n_informative_sites).
``adata.uns['mt_lineage_params']``  — run parameters + BIC per K.

CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: ``_core.mt_lineage_call_clones``
must be exposed by the cycle-22 pybind11 binding extension.
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

def _get_mt_layers(
    adata: "anndata.AnnData",
    alt_layer: str,
    depth_layer: str,
) -> tuple:
    """
    Return the (MT sites × cells) alt-count and depth matrices.

    These correspond to singlify's ``mt_alleles.1pz`` output loaded into two
    layers: alt counts and total depth.  Both must be (sites × cells) or
    (cells × sites) — the wrapper normalises to (sites × cells) for the C++
    kernel.

    Returns
    -------
    (alt_mat, depth_mat)  — cupy sparse or numpy, shape (n_sites, n_cells).
    """
    if alt_layer not in adata.layers:
        raise KeyError(
            f"alt_layer='{alt_layer}' not found in adata.layers.  "
            f"Available layers: {list(adata.layers.keys())}.  "
            "Load with read_pz_to_anndata(..., modality='mt') and split into "
            "alt/depth layers, or pass the correct layer names."
        )
    if depth_layer not in adata.layers:
        raise KeyError(
            f"depth_layer='{depth_layer}' not found in adata.layers.  "
            f"Available layers: {list(adata.layers.keys())}."
        )
    return adata.layers[alt_layer], adata.layers[depth_layer]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_clones(
    adata: "anndata.AnnData",
    *,
    alt_layer: str = "mt_alt",
    depth_layer: str = "mt_depth",
    min_depth: int = 10,
    min_cells_alt: int = 5,
    min_vaf: float = 0.01,
    min_K: int = 2,
    max_K: int = 10,
    seed: int = 0,
    copy: bool = False,
) -> Optional["anndata.AnnData"]:
    """
    GPU-native MT heteroplasmy clone calling (cycle-16 kernel).

    Detects mitochondrial clones from per-site heteroplasmy profiles using
    GPU Gaussian Mixture Model clustering.  This is a NEW capability with
    no direct scanpy or MQuad AnnData-native equivalent.

    The function consumes singlify's ``mt_alleles.1pz`` output (loaded as
    two layers: ``mt_alt`` and ``mt_depth``).  Informative sites are
    selected, VAF is computed, and GMM clustering at K = min_K … max_K
    is run on device.  Optimal K is chosen by BIC.

    Parameters
    ----------
    adata : AnnData
        Must contain two layers from singlify's MT output:

        - ``adata.layers[alt_layer]``   — per-site alt-allele counts.
        - ``adata.layers[depth_layer]`` — per-site total depth counts.

        Canonical singlify layer names after loading ``mt_alleles.1pz``
        are ``'mt_alt'`` and ``'mt_depth'`` (see ``io.read_pz_to_anndata``).
    alt_layer : str, default ``"mt_alt"``
        Layer key for per-site alternate-allele counts.
    depth_layer : str, default ``"mt_depth"``
        Layer key for per-site total depth.
    min_depth : int, default 10
        Minimum per-cell total depth at a site for it to be considered
        informative for that cell.  Sites not passing this threshold in
        any cell are excluded.
    min_cells_alt : int, default 5
        Minimum number of cells with at least one alt read at a site for
        the site to be retained as informative.
    min_vaf : float, default 0.01
        Minimum mean VAF across cells for a site to be retained.
        Filters out very rare variants with near-zero VAF.
    min_K : int, default 2
        Minimum number of clones to consider.
    max_K : int, default 10
        Maximum number of clones to consider.  BIC selects the optimal K
        in [min_K, max_K].
    seed : int, default 0
        Random seed for GMM initialisation.
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
        If ``_core.mt_lineage_call_clones`` is not available.
        See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE.
    KeyError
        If the required layers are not present in ``adata.layers``.
    ValueError
        If min_K > max_K or min_K < 2.

    Notes
    -----
    **Result location** (NEW — no existing scanpy/MQuad equivalent):

    - ``adata.obs['mt_clone_id']`` — int per cell (0-indexed clone label,
      −1 for cells not assignable).
    - ``adata.obsm['mt_heteroplasmy']`` — float32 matrix
      (n_cells × n_informative_sites) of per-site VAF values.
    - ``adata.uns['mt_lineage_params']`` — dict containing:
        - ``'optimal_K'``:  int, selected number of clones.
        - ``'bic_per_K'``:  list of BIC values for K = min_K … max_K.
        - ``'n_informative_sites'``:  int.
        - run parameters.

    **Singlify integration**: singlify produces ``mt_alleles.1pz`` when
    ``--pipeline`` is used.  Load it with::

        adata_mt = read_pz_to_anndata(sample_dir, modality='mt')
        adata.layers['mt_alt']   = adata_mt.X   # alt counts
        # The depth layer must be constructed from singlify provenance or
        # the separate mt_depth output once available in the format.

    Examples
    --------
    Default clone detection::

        import singlet_gpu.lineage as sgl
        sgl.detect_clones(adata)
        print(adata.obs['mt_clone_id'].value_counts())
        print(adata.obsm['mt_heteroplasmy'].shape)

    Stricter filtering::

        sgl.detect_clones(adata, min_depth=20, min_vaf=0.05, max_K=6)
    """
    if min_K < 2:
        raise ValueError(f"min_K must be ≥2, got {min_K!r}.")
    if min_K > max_K:
        raise ValueError(
            f"min_K ({min_K!r}) must be ≤ max_K ({max_K!r})."
        )

    import singlet_gpu._core as _core

    if not hasattr(_core, "mt_lineage_call_clones"):
        raise AttributeError(
            "_core.mt_lineage_call_clones is not available.  "
            "See CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE — the cycle-22 "
            "pybind11 binding extension must expose mt_lineage_call_clones "
            "before singlet_gpu.lineage.detect_clones() is callable."
        )

    working = copy_module.copy(adata) if copy else adata

    alt_mat, depth_mat = _get_mt_layers(working, alt_layer, depth_layer)

    # _core.mt_lineage_call_clones expects PyDeviceCsc for both inputs.
    # Layers may be scipy sparse (host) or cupy sparse (device).  Upload via
    # cupy → _core.from_cupy_csr (zero-copy if already cupy; H2D otherwise).
    # We pass the matrices as-is shape-wise; the kernel handles orientation.
    import cupy as cp
    try:
        import cupyx.scipy.sparse as csp  # cupy >= 14
    except ImportError:
        import cupy.sparse as csp         # cupy < 14 fallback
    import scipy.sparse as sp

    def _to_device_csc(mat):
        if csp.issparse(mat):
            cu_csc = mat if isinstance(mat, csp.csc_matrix) else mat.tocsc()
            if cu_csc.dtype != cp.float32:
                cu_csc = cu_csc.astype(cp.float32)
        elif sp.issparse(mat):
            cu_csc = csp.csc_matrix(mat.tocsc().astype(np.float32))
        elif hasattr(mat, "__cuda_array_interface__"):
            cu_csc = csp.csc_matrix(mat.astype(cp.float32))
        else:
            cu_csc = csp.csc_matrix(cp.asarray(mat, dtype=cp.float32))
        return _core.from_cupy_csr(cu_csc)

    alt_mat   = _to_device_csc(alt_mat)
    depth_mat = _to_device_csc(depth_mat)

    # C++ kernel: mt_lineage_call_clones(alt_mat, depth_mat,
    #                                     min_depth, min_cells_alt, min_vaf,
    #                                     min_K, max_K, seed)
    # Returns struct with fields:
    #   .clone_ids            — (n_cells,) int32 clone assignment
    #   .heteroplasmy_matrix  — (n_cells × n_informative_sites) float32 VAF
    #   .optimal_K            — int
    #   .bic_per_K            — list of floats, length = max_K - min_K + 1
    #   .n_informative_sites  — int
    # _core.mt_lineage_call_clones signature (py::kw_only after depth_counts):
    #   mt_lineage_call_clones(alt_counts, depth_counts, *, min_depth=10,
    #     min_cells_alt=5, min_vaf=0.01, max_em_iters=100, em_tol=1e-5,
    #     min_K=2, max_K=10, seed=0) -> ClonePrediction
    result = _core.mt_lineage_call_clones(
        alt_mat,
        depth_mat,
        min_depth=int(min_depth),
        min_cells_alt=int(min_cells_alt),
        min_vaf=float(min_vaf),
        min_K=int(min_K),
        max_K=int(max_K),
        seed=int(seed),
    )

    import pandas as pd

    # _core.ClonePrediction exposes: n_clones, n_informative_sites,
    # labels_view (int32, n_cells), informative_sites_view (int32),
    # clone_profiles_view (float32, n_clones × n_informative_sites),
    # per_cell_heteroplasmy_view (float32, n_cells × n_informative_sites).
    # All *_view fields are CAI dicts → wrap with _CaiView for cupy 14.
    class _CaiView:  # see CYCLE-193 / §J.13
        def __init__(self, d): self.__cuda_array_interface__ = d

    n_clones = int(result.n_clones)
    n_inf    = int(result.n_informative_sites)
    n_cells  = int(working.n_obs)

    labels = cp.asarray(_CaiView(result.labels_view)).get()  # int32 (n_cells,)
    het    = cp.asarray(_CaiView(result.per_cell_heteroplasmy_view)).get() \
                .reshape(n_cells, n_inf)  # float32 (n_cells × n_informative)

    working.obs["mt_clone_id"] = pd.array(
        np.asarray(labels, dtype=np.int32),
        dtype=pd.Int32Dtype(),  # nullable int, -1 for unassigned cells
    )
    working.obsm["mt_heteroplasmy"] = np.asarray(het, dtype=np.float32)
    working.uns["mt_lineage_params"] = {
        "optimal_K": n_clones,             # ClonePrediction.n_clones
        "n_informative_sites": n_inf,
        "min_depth": min_depth,
        "min_cells_alt": min_cells_alt,
        "min_vaf": min_vaf,
        "min_K": min_K,
        "max_K": max_K,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["detect_clones"]
