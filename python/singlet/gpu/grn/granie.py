# SPDX-License-Identifier: MIT
"""
singlet.gpu.grn.granie — GPU-native GRaNIE gene regulatory network inference.

Underlying C++ kernel: singlet::gpu::grn::run_granie (cycle 36, grn/granie.h).

GRaNIE reconstructs enhancer-mediated GRNs from multiome (scATAC + scRNA) data:
  1. Peak filtering (Welford mean + variance).
  2. Peak–gene Pearson correlation (cis window).
  3. TF activity: cuSPARSE SpMM(tf_motif_in_peak, peak_accessibility).
  4. TF–target scoring: corr(TF_activity, GEX) × mean |r_peak_gene|.
  5. BH FDR correction.
  6. Optional Leiden community detection on the filtered GRN.

Reference: Gaumondo et al. (2024) GRaNIE, Nature Methods.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

from singlet.gpu._coreutil import require_core

if TYPE_CHECKING:
    import anndata
    import scipy.sparse


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_from_csc(
    gex,
    peaks,
    tf_motif: dict,
    peak_gene_pairs: list,
    *,
    min_peak_mean: float = 0.01,
    min_peak_var: float = 0.01,
    min_abs_r: float = 0.3,
    peak_gene_fdr: float = 0.05,
    tf_target_fdr: float = 0.05,
    min_combined_score: float = 0.0,
    run_leiden: bool = False,
    leiden_resolution: float = 1.0,
    leiden_max_iter: int = 100,
    chunk_size_tfs: int = 100,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native GRaNIE GRN inference from multiome DeviceCsc matrices.

    Parameters
    ----------
    gex : DeviceCsc
        GEX matrix (cells × genes, CSC, float32).
    peaks : DeviceCsc
        ATAC peaks matrix (cells × peaks, CSC, float32).
        Must have the same number of rows (cells) as *gex*.
    tf_motif : dict
        TF motif-in-peak sparse matrix in CSR format.  Keys:

        ``n_rows``      — int, number of TFs.
        ``n_cols``      — int, number of peaks (must match peaks.cols).
        ``row_ptr``     — list[int], length n_rows + 1.
        ``col_indices`` — list[int], length nnz.
        ``values``      — list[float], length nnz (1.0 for binary motif hits).

        Build with e.g. scipy::

            from scipy.sparse import csr_matrix
            M = csr_matrix(jaspar_tf_by_peak_matrix)
            tf_motif = {"n_rows": M.shape[0], "n_cols": M.shape[1],
                        "row_ptr": M.indptr.tolist(),
                        "col_indices": M.indices.tolist(),
                        "values": M.data.astype(float).tolist()}

    peak_gene_pairs : list of [peak_idx, gene_idx]
        Cis-window candidate pairs.  Pre-compute from genomic coordinates,
        e.g. all peaks within 500 kbp of each gene's TSS.
    min_peak_mean : float, default 0.01
        Minimum mean accessibility across cells; peaks below are filtered.
    min_peak_var : float, default 0.01
        Minimum variance; constant peaks are filtered.
    min_abs_r : float, default 0.3
        Minimum |Pearson r| threshold for peak–gene links.
    peak_gene_fdr : float, default 0.05
        BH FDR threshold for peak–gene link significance.
    tf_target_fdr : float, default 0.05
        BH FDR threshold for TF–target edge significance.
    min_combined_score : float, default 0.0
        Minimum combined score (Pearson × |r|) post-FDR.
    run_leiden : bool, default False
        Run Leiden community detection on the filtered GRN graph.
    leiden_resolution : float, default 1.0
    leiden_max_iter : int, default 100
    chunk_size_tfs : int, default 100
        Number of TFs processed per scoring chunk (OOC VRAM control).
    stream : CUDA stream handle (int) or None
        Device stream to run on.  None = default stream.
    seed : int, default 0
        Random seed (passed to Leiden if run_leiden=True).

    Returns
    -------
    dict
        ``"edges"`` — list of dicts, each with:

            ``"tf_idx"``, ``"gene_idx"`` — 0-based indices.
            ``"combined_score"`` — Pearson(TF_activity, GEX) × mean |r_peak_gene|.
            ``"p_value"`` — nominal p-value.
            ``"fdr"`` — BH-adjusted FDR.

        ``"tf_activity"`` — numpy float32 (n_tfs × n_cells, row-major).
        ``"n_edges"`` — int.
        ``"n_peak_gene_links"`` — int, significant peak–gene links.
        ``"n_tfs"``, ``"n_cells"`` — int.
        ``"community_labels"`` — list[int], empty if run_leiden=False.
        ``"n_communities"`` — int.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    RuntimeError
        If gex.rows != peaks.rows or tf_motif shape mismatches n_peaks.

    Examples
    --------
    >>> result = singlet.gpu.grn.granie.run_from_csc(
    ...     gex, peaks, tf_motif, peak_gene_pairs,
    ...     min_abs_r=0.3, tf_target_fdr=0.05,
    ... )
    >>> print(f"GRN edges: {result['n_edges']}")
    >>> import pandas as pd
    >>> edges_df = pd.DataFrame(result["edges"])
    """
    _core = require_core("run_granie")

    return _core.run_granie(
        gex,
        peaks,
        tf_motif,
        peak_gene_pairs,
        min_peak_mean=float(min_peak_mean),
        min_peak_var=float(min_peak_var),
        min_abs_r=float(min_abs_r),
        peak_gene_fdr=float(peak_gene_fdr),
        tf_target_fdr=float(tf_target_fdr),
        min_combined_score=float(min_combined_score),
        run_leiden=bool(run_leiden),
        leiden_resolution=float(leiden_resolution),
        leiden_max_iter=int(leiden_max_iter),
        chunk_size_tfs=int(chunk_size_tfs),
        stream=stream,
        seed=int(seed),
    )


def run_from_anndata(
    adata_rna: "anndata.AnnData",
    adata_atac: "anndata.AnnData",
    tf_motif: dict,
    peak_gene_pairs: list,
    **kwargs,
) -> dict:
    """
    Convenience wrapper: load GEX + ATAC from AnnData objects.

    Extracts ``adata_rna.X`` and ``adata_atac.X`` as DeviceCsc via
    ``singlet.gpu.load_pz`` (or directly from cupy if already on device),
    then delegates to :func:`run_from_csc`.

    Parameters
    ----------
    adata_rna : AnnData
        RNA assay.  ``adata_rna.X`` must be a count matrix (cells × genes).
    adata_atac : AnnData
        ATAC assay.  ``adata_atac.X`` must be peak accessibility (cells × peaks).
        Must have the same ``obs_names`` (cells) as *adata_rna*.
    tf_motif : dict
        See :func:`run_from_csc`.
    peak_gene_pairs : list
        See :func:`run_from_csc`.
    **kwargs
        Forwarded to :func:`run_from_csc`.

    Returns
    -------
    dict
        Same as :func:`run_from_csc`.
    """
    _core = require_core("from_cupy_csr")

    def _to_device_csc(X):
        # X may be scipy sparse, cupy sparse, or dense numpy.
        try:
            import scipy.sparse as sp
            try:
                import cupyx.scipy.sparse as csp  # cupy >= 14
            except ImportError:
                import cupy.sparse as csp         # cupy < 14 fallback
            if sp.issparse(X):
                X_csr = X.tocsr().astype(np.float32)
                # Upload to device via cupy
                X_cupy = csp.csr_matrix(X_csr)
                return _core.from_cupy_csr(X_cupy)
        except ImportError:
            pass
        raise TypeError(
            "adata.X must be a scipy sparse matrix or cupy sparse matrix. "
            "Load your data with singlet.gpu.load_pz() for direct device access."
        )

    gex   = _to_device_csc(adata_rna.X)
    peaks = _to_device_csc(adata_atac.X)
    return run_from_csc(gex, peaks, tf_motif, peak_gene_pairs, **kwargs)


__all__ = ["run_from_csc", "run_from_anndata"]
