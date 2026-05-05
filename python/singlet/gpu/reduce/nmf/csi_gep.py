# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.reduce.nmf.csi_gep — Consensus NMF gene expression programs.

Underlying C++: cycle 28, ``reduce/nmf/csi_gep.h``.
Algorithm: bootstrap NMF + spherical k-means + elbow auto-rank
           (Kotliar et al. 2019 eLife, CSI-GEP / cNMF).

API
---
``run_csi_gep`` — fit consensus programs; writes to AnnData.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import anndata


def run_csi_gep(
    adata: anndata.AnnData,
    *,
    k_range: Optional[List[int]] = None,
    n_runs: int = 100,
    subsample_frac: float = 0.8,
    top_n_genes_jaccard: int = 30,
    elbow_gap_threshold: float = 0.05,
    max_kmeans_iters: int = 100,
    kmeans_tol: float = 1e-4,
    nmf_max_iter: int = 200,
    nmf_tol: float = 1e-4,
    varm_key: str = "csi_gep_programs",
    obsm_key: str = "csi_gep_usage",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    Consensus NMF gene expression programs (cycle 28).

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` is cells × genes (raw counts recommended).
        Must have been loaded via ``singlet.gpu.io.load_pz`` (host-pinned CSC).
    k_range : list[int] or None
        Ranks to evaluate.  Default: [5, 10, 15, 20, 25, 30].
    n_runs : int
        Bootstrap NMF runs per rank.
    subsample_frac : float
        Cell subsample fraction per run.
    top_n_genes_jaccard : int
        Top-N genes for Jaccard reproducibility score.
    elbow_gap_threshold : float
        Gap threshold for auto-rank selection.
    varm_key : str
        Key written to ``adata.varm`` with shape (n_genes, chosen_k).
    obsm_key : str
        Key written to ``adata.obsm`` with shape (n_cells, chosen_k).
    stream : int or None
    seed : int
    copy : bool

    Returns
    -------
    None or AnnData
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "csi_gep"):
        raise AttributeError(
            "_core.csi_gep is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    if k_range is None:
        k_range = [5, 10, 15, 20, 25, 30]

    working = adata.copy() if copy else adata

    try:
        import cupy as cp

        try:
            import cupyx.scipy.sparse as csp  # cupy >= 14
        except ImportError:
            import cupy.sparse as csp  # cupy < 14 fallback
        import scipy.sparse as sp

        X = working.X
        if X.shape[0] == working.n_obs:
            X = X.T  # genes × cells
        device_mat = csp.csc_matrix(X) if sp.issparse(X) else csp.csc_matrix(cp.array(X))
    except ImportError as e:
        raise ImportError(f"singlet.gpu.reduce.nmf.run_csi_gep requires cupy.  Original error: {e}") from e

    result = _core.csi_gep(
        device_mat,
        k_range=k_range,
        n_runs=n_runs,
        subsample_frac=subsample_frac,
        top_n_genes_jaccard=top_n_genes_jaccard,
        elbow_gap_threshold=elbow_gap_threshold,
        max_kmeans_iters=max_kmeans_iters,
        kmeans_tol=kmeans_tol,
        nmf_max_iter=nmf_max_iter,
        nmf_tol=nmf_tol,
        stream=stream,
        seed=seed,
    )

    chosen_k = result.chosen_k
    n_genes = result.n_genes
    n_cells = result.n_cells

    programs = cp.asarray(result.consensus_programs_view).reshape(chosen_k, n_genes).get()
    usage = cp.asarray(result.program_usage_view).reshape(n_cells, chosen_k).get()

    working.varm[varm_key] = programs.T  # n_genes × chosen_k
    working.obsm[obsm_key] = usage  # n_cells × chosen_k
    working.uns["csi_gep_params"] = {
        "chosen_k": chosen_k,
        "k_range": k_range,
        "k_used": result.k_used,
        "reproducibility_curve": result.reproducibility_curve,
        "n_runs": n_runs,
        "subsample_frac": subsample_frac,
        "seed": seed,
    }

    return working if copy else None


__all__ = ["run_csi_gep"]
