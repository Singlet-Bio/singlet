# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet.gpu.atac.chromvar — GPU-native chromVAR motif enrichment.

Underlying C++ kernel: singlet::gpu::atac::chromvar (cycle 34, atac/chromvar.h).

chromVAR computes bias-corrected motif deviation scores per cell from
scATAC-seq peak accessibility, with permutation-based p-values.

Reference: Schep et al. (2017) Nature Methods — chromVAR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    import anndata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute(
    accessibility,
    motif_in_peak,
    peak_gc,
    peak_mean_access,
    *,
    n_background_peaks: int = 50,
    n_permutations: int = 1000,
    gc_bins: int = 25,
    access_bins: int = 25,
    cell_batch: int = 4096,
    bg_batch: int = 50,
    perm_inner_batch: int = 50,
    deviation_epsilon: float = 1e-6,
    stream=None,
    seed: int = 0,
) -> dict:
    """
    GPU-native chromVAR motif enrichment for scATAC-seq (cycle 34).

    Computes bias-corrected motif deviation scores and variability across
    cells, following the chromVAR algorithm (Schep et al. 2017).  Scores are
    zero-copy device-resident via ``__cuda_array_interface__`` dicts.

    Parameters
    ----------
    accessibility : DeviceCsc
        Peak accessibility matrix (n_peaks × n_cells, CSC, float32).
        Peaks are columns in the original orientation; this CSC has
        peaks as columns and cells as rows — i.e., n_peaks × n_cells CSC.
    motif_in_peak : DeviceCsc
        Binary motif-in-peak matrix (n_motifs × n_peaks, CSC, float32).
        ``motif_in_peak.cols`` must equal ``accessibility.rows`` (n_peaks).
    peak_gc : ndarray float32, shape (n_peaks,)
        Per-peak GC content ∈ [0, 1].  Used for background peak matching.
    peak_mean_access : ndarray float32, shape (n_peaks,)
        Per-peak mean accessibility ∈ [0, ∞).  Used for background matching.
        Compute as: ``np.array(accessibility_csc.mean(axis=1)).ravel()``.
    n_background_peaks : int, default 50
        Number of matched background peaks per peak (K_bg in the paper).
    n_permutations : int, default 1000
        Number of cell-label permutations for p-value estimation (K_perm).
    gc_bins : int, default 25
        Bins along GC-content axis for 2D background matching.
    access_bins : int, default 25
        Bins along mean-accessibility axis for 2D background matching.
    cell_batch : int, default 4096
        Cells processed per OOC batch.  Reduces peak memory for large datasets.
        At 200k peaks × 1M cells: 10 batches × 100k cells.
    bg_batch : int, default 50
        Background SpMM batch size (K_bg draws per round).
    perm_inner_batch : int, default 50
        Permutations per SpMM batch call.
    deviation_epsilon : float, default 1e-6
        Floor on std_bg to avoid division by zero in normalization.
    stream : int or None
        CUDA stream handle (as Python int).  None = default stream.
    seed : int, default 0
        Random seed for background peak sampling and permutation shuffles.

    Returns
    -------
    dict
        All array outputs are zero-copy device-resident via
        ``__cuda_array_interface__`` dicts.  Keep ``result["_result_owner"]``
        alive to prevent device memory deallocation.

        ``"deviation"``    — float32 (n_motifs × n_cells), bias-corrected deviation.
        ``"variability"``  — float32 (n_motifs,), variance of deviation across cells.
        ``"p_values"``     — float32 (n_motifs × n_cells), permutation p-values.
        ``"p_adj"``        — float32 (n_motifs × n_cells), BH-adjusted per-motif.
        ``"n_motifs"``     — int.
        ``"n_cells"``      — int.
        ``"_result_owner"``— lifetime anchor for device memory.

    Raises
    ------
    ImportError
        If the C++ extension is not compiled.
    RuntimeError
        If motif_in_peak.cols != accessibility.rows, or peak_gc / peak_mean_access
        have wrong size.

    Notes
    -----
    **Zero-copy cupy access**::

        import cupy as cp
        d = result["deviation"]
        owner = result["_result_owner"]
        dev_cp = cp.ndarray(
            d["shape"], dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(d["data"][0], 0, owner), 0)
        )

    **Memory budget**: at CELL_BATCH=4096 and 200k peaks, peak device memory
    is ~800 MB + input CSC sizes.  Reduce ``cell_batch`` on devices with <8 GB VRAM.

    **n_permutations tradeoff**: 1000 permutations give stable p-values.  For
    quick exploratory runs, 100–200 permutations are often sufficient.

    Examples
    --------
    >>> result = singlet.gpu.atac.chromvar.compute(
    ...     access_csc,
    ...     motif_csc,
    ...     peak_gc=peak_gc_arr,
    ...     peak_mean_access=peak_mean_arr,
    ... )
    >>> print(f"n_motifs={result['n_motifs']}, n_cells={result['n_cells']}")

    Access deviation on host::

        import cupy as cp
        dev = result["deviation"]
        owner = result["_result_owner"]
        deviation_cp = cp.ndarray(
            dev["shape"], dtype=cp.float32,
            memptr=cp.cuda.MemoryPointer(
                cp.cuda.UnownedMemory(dev["data"][0], 0, owner), 0)
        )
        deviation_np = deviation_cp.get()  # host copy
    """
    import singlet.gpu._core as _core

    if not hasattr(_core, "chromvar"):
        raise ImportError(
            "_core.chromvar is not available.  "
            "Install with: pip install singlet[gpu]"
        )

    # Coerce peak_gc and peak_mean_access to float32 c-contiguous numpy arrays.
    def _to_f32(arr):
        if not isinstance(arr, np.ndarray):
            arr = np.asarray(arr, dtype=np.float32)
        elif arr.dtype != np.float32:
            arr = arr.astype(np.float32)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr)
        return arr

    peak_gc_np   = _to_f32(peak_gc)
    peak_mean_np = _to_f32(peak_mean_access)

    return _core.chromvar(
        accessibility,
        motif_in_peak,
        peak_gc_np,
        peak_mean_np,
        int(n_background_peaks),
        int(n_permutations),
        int(gc_bins),
        int(access_bins),
        int(cell_batch),
        int(bg_batch),
        int(perm_inner_batch),
        float(deviation_epsilon),
        stream,
        int(seed),
    )


__all__ = ["compute"]
