# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.enrich.progeny — GPU-native PROGENy pathway activity scoring.

Underlying C++: cycle 44, ``enrich/progeny.h``.
Algorithm: weighted sum of expression × pre-trained pathway weights
           (Schubert et al. Nat Commun 2018; Holland et al. Genome Biol 2020).

decoupleR compatibility
-----------------------
``run_progeny`` mirrors ``decoupler.run_progeny``.
Results written to ``adata.obsm`` with decoupleR-compatible key names.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata


def run_progeny(
    adata: anndata.AnnData,
    weights_tsv: str,
    *,
    gene_names_key: Optional[str] = None,
    normalize: bool = True,
    norm_eps: float = 1e-6,
    cell_batch: int = 0,
    obsm_key: str = "progeny_activity",
    stream=None,
    seed: int = 0,
    copy: bool = False,
) -> Optional[anndata.AnnData]:
    """
    PROGENy pathway activity scoring (cycle 44).

    Parameters
    ----------
    adata : AnnData
        ``adata.X`` is cells × genes (log-normalized recommended).
    weights_tsv : str
        Path to PROGENy weight file in TSV format (columns: gene, pathway, weight).
        Holland et al. 2020 human top-100 file: available from the decoupleR/PROGENy R package.
    gene_names_key : str or None
        Column in ``adata.var`` to use as gene names.  None = use ``var_names``.
    normalize : bool
        Apply per-pathway z-score normalization across cells.
    cell_batch : int
        0 = all cells at once; >0 = batch size.
    obsm_key : str
        Key written to ``adata.obsm`` with shape (n_cells, n_pathways).
    stream : int or None
    seed : int
        Unused (PROGENy is deterministic).
    copy : bool

    Returns
    -------
    None or AnnData
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "progeny"):
        raise AttributeError(
            "_core.progeny is not available.  "
            "Ensure the cycle-52a binding extension has been compiled."
        )

    if not os.path.exists(weights_tsv):
        raise FileNotFoundError(
            f"PROGENy weight file not found: {weights_tsv}.  "
            "Download from the decoupleR/PROGENy R package or use "
            "`singlet.gpu.enrich.progeny.download_progeny_weights()`."
        )

    working = adata.copy() if copy else adata

    if gene_names_key is not None:
        gene_names = list(working.var[gene_names_key])
    else:
        gene_names = list(working.var_names)

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
        raise ImportError(
            f"singlet.gpu.enrich.run_progeny requires cupy.  Original error: {e}"
        ) from e

    result = _core.progeny(
        device_mat,
        weights_tsv,
        gene_names,
        normalize=normalize,
        norm_eps=norm_eps,
        cell_batch=cell_batch,
        stream=stream,
        seed=seed,
    )

    n_cells = result.n_cells
    n_pathways = result.n_pathways
    pw_names = result.pathway_names

    activity = cp.asarray(result.activity_view).reshape(n_cells, n_pathways).get()
    working.obsm[obsm_key] = activity
    working.uns["progeny_pathway_names"] = pw_names
    working.uns["progeny_params"] = {
        "weights_tsv": weights_tsv,
        "normalize": normalize,
        "n_pathways": n_pathways,
    }

    return working if copy else None


__all__ = ["run_progeny"]
