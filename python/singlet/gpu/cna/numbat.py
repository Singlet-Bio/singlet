# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.cna.numbat — GPU-native Numbat copy-number alteration detection.

Underlying C++ kernel: singlet_gpu::cna::detect_cna (cycle 35, cna/numbat.h).

Detects CNAs at single-cell resolution:
  1. Log-ratio vs reference expression.
  2. Per-chromosome rolling-window smoothing (direct kernel, window ≤ 15 genes).
  3. HMM forward-backward per cell per chromosome (log-space fp32).
     States: loss=0, neutral=1, gain=2.
  4. Leiden clone clustering on CNA patterns.

Reference: Gao et al. (Numbat, Genome Biology 2024).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, List, Dict

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# HMM state constants (mirrors C++ HMM_LOSS / HMM_NEUTRAL / HMM_GAIN)
# ---------------------------------------------------------------------------
STATE_LOSS    = 0
STATE_NEUTRAL = 1
STATE_GAIN    = 2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect(
    expr,
    chr: List[int],
    start_bp: List[int],
    *,
    ref=None,
    smooth_half_win: int = 5,
    min_genes_per_chr: int = 10,
    loss_mean: float = -1.0,
    loss_sigma: float = 1.0,
    neutral_mean: float = 0.0,
    neutral_sigma: float = 0.5,
    gain_mean: float = 0.5,
    gain_sigma: float = 1.0,
    trans_prob: float = 0.01,
    segments_per_chr: int = 50,
    include_sex_chrs: bool = False,
    knn_k: int = 15,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native Numbat CNA detection (cycle 35).

    Detects copy-number alterations from single-cell gene expression via
    log-ratio smoothing, HMM forward-backward, and Leiden clone clustering.

    Parameters
    ----------
    expr : DeviceCsc
        Gene expression matrix (genes × cells, CSC, float32).
        Rows = genes, columns = cells.
    chr : list[int]
        Per-gene chromosome assignment (1-based: 1-22, 23=X, 24=Y, 25=MT).
        Must have length == expr.rows.
    start_bp : list[int]
        Per-gene genomic start position (bp).  Used for sorting within chromosomes.
        Must have the same length as *chr*.
    ref : ndarray float32 [n_genes], optional
        Reference expression profile (mean log expression of normal cells).
        None → reference computed as mean expression across all input cells.
    smooth_half_win : int, default 5
        Rolling-window half-width; full window = 2 × smooth_half_win + 1.
        Must satisfy (2 × smooth_half_win + 1) ≤ 15 (MAX_SMOOTH_WINDOW).
    min_genes_per_chr : int, default 10
        Chromosomes with fewer genes than this are skipped.
    loss_mean, loss_sigma : float, default -1.0, 1.0
        Gaussian emission mean and σ for the CNA loss (deletion) state.
    neutral_mean, neutral_sigma : float, default 0.0, 0.5
        Gaussian emission parameters for the neutral diploid state.
    gain_mean, gain_sigma : float, default 0.5, 1.0
        Gaussian emission parameters for the CNA gain (amplification) state.
    trans_prob : float, default 0.01
        HMM off-diagonal transition probability per segment boundary.
        p(stay) = 1 - 2 × trans_prob.  Must be in (0, 0.5).
    segments_per_chr : int, default 50
        Number of HMM segments per chromosome (genes binned uniformly).
    include_sex_chrs : bool, default False
        Whether to include X (chr 23) and Y (chr 24) in the analysis.
    knn_k : int, default 15
        kNN k for clone clustering (Leiden input graph).
    stream : int or None
        CUDA stream handle (as Python int).  None = default stream.
    seed : int, default 0
        Random seed for Leiden clone clustering.

    Returns
    -------
    dict
        ``"cna_states"``        — ``__cuda_array_interface__`` dict (uint8,
                                   n_cells × n_total_segments).
                                   Values: 0=loss, 1=neutral, 2=gain.
        ``"hmm_posteriors"``    — ``__cuda_array_interface__`` dict (float32,
                                   n_cells × n_total_segments × 3).
        ``"clone_labels"``      — ``__cuda_array_interface__`` dict (int32, n_cells).
        ``"n_clones"``          — int, number of CNA-based clones detected.
        ``"leiden_modularity"`` — float, Leiden modularity score.
        ``"n_total_segments"``  — int, total HMM segments across all chromosomes.
        ``"segments"``          — list of dicts with keys:
                                   ``chr``, ``gene_start``, ``gene_end``, ``seg_idx``.
        ``"_result_owner"``     — internal lifetime anchor; keep a reference to
                                   prevent premature device memory deallocation.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    ValueError
        If smooth_half_win is too large (full window > 15) or trans_prob out of range.

    Notes
    -----
    **Zero-copy cupy access**::

        import cupy as cp
        d = result["cna_states"]
        owner = result["_result_owner"]
        states_cp = cp.ndarray(
            d["shape"], dtype=cp.uint8,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(d["data"][0], 0, owner), 0)
        )

    **Segment indexing**: ``result["cna_states"][cell_idx, seg_idx]`` gives the
    CNA call for that cell at that segment.  Segment metadata is in
    ``result["segments"][seg_idx]`` → chr, gene_start, gene_end.

    **Reference expression**: for tumor analysis, provide the mean expression of
    matched normal cells as *ref* to obtain more accurate log-ratios.  Without a
    reference, the kernel uses the global mean which may be biased in highly
    aneuploid samples.

    Examples
    --------
    >>> import numpy as np
    >>> result = singlet_gpu.cna.numbat.detect(
    ...     expr_csc,
    ...     chr=gene_chr_list,        # list of ints, length n_genes
    ...     start_bp=gene_start_list, # list of ints, length n_genes
    ... )
    >>> print(f"Detected {result['n_clones']} clones across "
    ...       f"{result['n_total_segments']} segments")
    """
    import singlet_gpu._core as _core

    if not hasattr(_core, "detect_cna"):
        raise ImportError(
            "_core.detect_cna is not available.  "
            "Install with: pip install singlet[gpu]"
        )

    full_window = 2 * smooth_half_win + 1
    if full_window > 15:
        raise ValueError(
            f"smooth_half_win={smooth_half_win} gives full_window={full_window} "
            "which exceeds MAX_SMOOTH_WINDOW=15."
        )
    if not (0.0 < trans_prob < 0.5):
        raise ValueError(
            f"trans_prob={trans_prob} must be in (0, 0.5)."
        )

    # Coerce ref to float32 c-contiguous numpy array if provided.
    if ref is not None:
        if not isinstance(ref, np.ndarray):
            ref = np.asarray(ref, dtype=np.float32)
        elif ref.dtype != np.float32:
            ref = ref.astype(np.float32)
        if not ref.flags["C_CONTIGUOUS"]:
            ref = np.ascontiguousarray(ref)

    return _core.detect_cna(
        expr,
        list(chr),
        list(start_bp),
        ref,
        int(smooth_half_win),
        int(min_genes_per_chr),
        float(loss_mean),
        float(loss_sigma),
        float(neutral_mean),
        float(neutral_sigma),
        float(gain_mean),
        float(gain_sigma),
        float(trans_prob),
        int(segments_per_chr),
        bool(include_sex_chrs),
        int(knn_k),
        stream,
        int(seed),
    )


__all__ = [
    "detect",
    "STATE_LOSS",
    "STATE_NEUTRAL",
    "STATE_GAIN",
]
