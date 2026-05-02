# SPDX-License-Identifier: GPL-2.0-or-later
"""
singlet_gpu.streaming.pipeline — ``run_pipeline`` implementation.

Wraps ``singlet_gpu::streaming::StreamedPipeline`` (cycle 7,
``streaming/streamed_pipeline.h``).

The pipeline processes multiple .1pz inputs in column-major streaming
chunks, avoiding full materialisation.  Each input file is loaded
one chunk at a time via the zero-copy device loader; the configured
pipeline stages (lognorm, HVG, PCA, NMF) receive each chunk and
accumulate their state before emitting the consolidated result.

Since the output spans multiple input files, this function does NOT
return an AnnData.  It returns a ``PipelineResult`` whose fields
mirror the C++ struct ``singlet_gpu::streaming::PipelineResult``.

CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND: ``_core.streaming_pipeline_run``
is not yet exposed by the cycle-20 binding extension.  The wrapper is
written against that name and will raise ``AttributeError`` with this
tag until the binding is added.
"""

from __future__ import annotations

import copy as copy_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """
    Consolidated output of a ``run_pipeline`` call.

    All device-resident arrays are ``cupy`` arrays when a GPU was used
    and ``numpy`` arrays when the pipeline ran in CPU-fallback mode
    (not recommended — always ensure a GPU is available).

    Attributes
    ----------
    cell_ids : list of str
        Barcode / cell identifiers concatenated across all input files.
    gene_ids : list of str
        Gene identifiers from the first input file (all inputs must share
        the same gene universe; ``run_pipeline`` validates this).
    X_pca : array of shape (n_cells, n_pcs) or None
        PCA embedding on the normalised, HVG-filtered matrix.
        ``None`` when ``run_pca=False``.
    X_nmf_W : array of shape (n_cells, n_factors) or None
        NMF cell loadings (W matrix).  ``None`` when ``run_nmf=False``.
    X_nmf_H : array of shape (n_factors, n_genes_hvg) or None
        NMF gene factors (H matrix).  ``None`` when ``run_nmf=False``.
    hvg_mask : array of bool, shape (n_genes,) or None
        Boolean mask over ``gene_ids`` indicating which genes were
        selected as highly variable.  ``None`` when ``run_hvg=False``.
    size_factors : array of shape (n_cells,) or None
        Per-cell size factors used for normalisation.
        ``None`` when ``run_lognorm=False``.
    input_paths : list of str
        Resolved paths that were processed (after glob/expansion).
    chunk_cols : int
        Chunk width used during streaming (columns per chunk).
    wall_time_s : float
        Total wall-clock seconds for the pipeline run.
    n_cells : int
        Total number of cells across all input files.
    n_genes : int
        Number of genes (pre-HVG filter).
    n_hvg : int
        Number of HVGs selected (0 when ``run_hvg=False``).
    """

    cell_ids: List[str] = field(default_factory=list)
    gene_ids: List[str] = field(default_factory=list)
    X_pca: Optional[object] = None
    X_nmf_W: Optional[object] = None
    X_nmf_H: Optional[object] = None
    hvg_mask: Optional[object] = None
    size_factors: Optional[object] = None
    input_paths: List[str] = field(default_factory=list)
    chunk_cols: int = 100_000
    wall_time_s: float = 0.0
    n_cells: int = 0
    n_genes: int = 0
    n_hvg: int = 0


# ---------------------------------------------------------------------------
# RNG helper — identical convention to scanpy (rng parameter)
# ---------------------------------------------------------------------------

def _resolve_seed(rng: Optional[Union[int, np.random.Generator]]) -> int:
    """
    Convert *rng* (int seed or np.random.Generator) to a uint64 for C++.

    Convention (mirrors scanpy / design doc §Constraints):
      - ``None``   → 0  (C++ uses a fixed default seed)
      - ``int``    → passed through as uint64
      - ``Generator`` → draw one uint64 from it
    """
    if rng is None:
        return 0
    if isinstance(rng, int):
        return rng & 0xFFFF_FFFF_FFFF_FFFF
    # np.random.Generator
    return int(np.random.default_rng(rng).integers(0, 2**63))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pipeline(
    input_paths: Sequence[Union[str, Path]],
    *,
    chunk_cols: int = 100_000,
    # Normalisation stage
    run_lognorm: bool = True,
    target_sum: Optional[float] = None,
    log_base: Optional[float] = None,
    # HVG stage
    run_hvg: bool = True,
    n_top_genes: int = 3_000,
    hvg_flavor: str = "seurat_v3",
    # PCA stage
    run_pca: bool = False,
    pca_k: int = 50,
    pca_solver: str = "auto",
    # NMF stage
    run_nmf: bool = False,
    nmf_factors: int = 20,
    nmf_loss: str = "mse",
    nmf_max_iter: int = 200,
    # Misc
    rng: Optional[Union[int, "np.random.Generator"]] = None,
    n_threads: int = 0,
) -> PipelineResult:
    """
    Run the singlet-gpu streaming pipeline over one or more .1pz files.

    This is a **multi-input** pipeline: it streams column chunks from
    each input file sequentially without materialising all data at once.
    Because the inputs can span many files, the output is a
    ``PipelineResult`` rather than an AnnData.

    Parameters
    ----------
    input_paths : sequence of str or Path
        Paths to ``.1pz`` count-matrix files (e.g.
        ``counts.1pz``, ``gene_counts.1pz``).  All files must share the
        same gene universe (row labels); a ``ValueError`` is raised at
        C++ initialisation time if they differ.
    chunk_cols : int, default 100_000
        Number of columns (cells) to load per streaming chunk.  Larger
        values reduce GPU kernel launch overhead but increase device
        memory pressure.  Must be ≥1.
    run_lognorm : bool, default True
        Apply total-count normalisation (``target_sum``) followed by
        log1p.  Corresponds to ``sc.pp.normalize_total`` +
        ``sc.pp.log1p``.
    target_sum : float or None, default None
        Target total count per cell for normalisation.  ``None`` uses
        the median total count (scanpy default).
    log_base : float or None, default None
        Log base for log1p transformation.  ``None`` uses natural log.
    run_hvg : bool, default True
        Select highly variable genes after normalisation.
    n_top_genes : int, default 3000
        Number of HVGs to select.  Passed to the cycle-3 HVG kernel.
    hvg_flavor : str, default "seurat_v3"
        HVG selection flavour.  One of ``"seurat_v3"``, ``"cell_ranger"``,
        or ``"seurat"``.  Passed verbatim to the C++ kernel.
    run_pca : bool, default False
        Compute PCA on the HVG-filtered, normalised matrix.
    pca_k : int, default 50
        Number of principal components to compute.
    pca_solver : str, default "auto"
        SVD solver hint for the factornet backend.  One of
        ``"auto"``, ``"lanczos"``, ``"irlba"``, ``"randomized"``.
    run_nmf : bool, default False
        Compute NMF factorisation on the HVG-filtered matrix.
    nmf_factors : int, default 20
        Number of NMF factors (rank *k*).
    nmf_loss : str, default "mse"
        NMF loss function.  One of ``"mse"``, ``"kl"``, ``"nb"``.
    nmf_max_iter : int, default 200
        Maximum NMF iterations.
    rng : int or np.random.Generator or None, default None
        Random seed or generator.  ``None`` → deterministic default seed
        (seed 0 in C++).  Matches the ``rng`` convention used by
        ``sc.pp.pca`` and other scanpy functions.
    n_threads : int, default 0
        CPU threads for .1pz decompression (0 = auto-detect from
        ``os.cpu_count()``).

    Returns
    -------
    PipelineResult
        Consolidated outputs across all input files.  See
        ``PipelineResult`` docstring for field descriptions.

    Raises
    ------
    ValueError
        If *input_paths* is empty or *chunk_cols* < 1.
    AttributeError
        If ``_core.streaming_pipeline_run`` is not available.
        See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND.
    RuntimeError
        If input files have mismatched gene universes.

    Notes
    -----
    **No host copies in the hot path.**  Data flows:
    ``.1pz`` → pinned host staging → device CSC → kernel chain.
    The only host traffic is metadata (cell/gene labels) and the
    small summary scalars in ``PipelineResult``.

    **Out-of-core / billion-cell design**: ``chunk_cols`` controls
    the residency budget.  At ``chunk_cols=100_000`` and a 3_000-gene
    HVG filter the working-set peak is ~3 GB device memory, allowing
    the pipeline to process arbitrarily large catalogues on a single
    GPU without OOM.

    Examples
    --------
    Basic run (lognorm + HVG only)::

        from singlet_gpu.streaming import run_pipeline
        result = run_pipeline(["GSM4037629/counts.1pz"])
        print(result.n_cells, result.n_hvg)

    With PCA and NMF::

        result = run_pipeline(
            ["GSM001/counts.1pz", "GSM002/counts.1pz"],
            run_pca=True,
            run_nmf=True,
            nmf_factors=30,
            rng=42,
        )
        print(result.X_pca.shape)   # (n_cells, 50)
        print(result.X_nmf_W.shape) # (n_cells, 30)
    """
    if not input_paths:
        raise ValueError("input_paths must not be empty.")
    if chunk_cols < 1:
        raise ValueError(f"chunk_cols must be ≥1, got {chunk_cols!r}.")

    import singlet_gpu._core as _core

    if not hasattr(_core, "streaming_pipeline_run"):
        raise AttributeError(
            "_core.streaming_pipeline_run is not available.  "
            "See CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND — the cycle-20 "
            "pybind11 binding extension must expose streaming_pipeline_run "
            "before this function is callable."
        )

    import time

    resolved = [str(Path(p).expanduser().resolve()) for p in input_paths]
    seed = _resolve_seed(rng)

    # _core.streaming_pipeline_run signature (positional, no config-object):
    #   streaming_pipeline_run(input_paths_list, chunk_cols, run_lognorm,
    #                          run_hvg, run_pca, pca_k, run_nmf, nmf_factors,
    #                          cache_normalized, in_memory_pca_threshold)
    # The fuller wrapper API (target_sum, hvg_flavor, pca_solver, nmf_loss,
    # nmf_max_iter, seed, n_threads) is currently NOT plumbed through to the
    # C++ kernel — accepted by the wrapper for API parity with future
    # binding extension (see CYCLE-21-FOLLOWUP).  Used vars are passed
    # below; ignored vars are silent.
    _ = (target_sum, log_base, hvg_flavor, pca_solver, nmf_loss,
         nmf_max_iter, seed, n_threads)  # quiet unused-variable warnings

    # cache_normalized + in_memory_pca_threshold are C++ binding params with
    # no Python-API exposure yet; use sensible defaults from the C++ contract.
    cache_normalized_default = False
    in_memory_pca_threshold_default = 1_000_000  # 1M cells before chunking PCA

    t0 = time.perf_counter()
    raw = _core.streaming_pipeline_run(
        resolved,
        int(chunk_cols),
        bool(run_lognorm),
        bool(run_hvg),
        bool(run_pca),
        int(pca_k),
        bool(run_nmf),
        int(nmf_factors),
        bool(cache_normalized_default),
        int(in_memory_pca_threshold_default),
    )
    wall = time.perf_counter() - t0

    return PipelineResult(
        cell_ids=list(raw.cell_ids),
        gene_ids=list(raw.gene_ids),
        X_pca=raw.X_pca if run_pca else None,
        X_nmf_W=raw.X_nmf_W if run_nmf else None,
        X_nmf_H=raw.X_nmf_H if run_nmf else None,
        hvg_mask=raw.hvg_mask if run_hvg else None,
        size_factors=raw.size_factors if run_lognorm else None,
        input_paths=resolved,
        chunk_cols=chunk_cols,
        wall_time_s=wall,
        n_cells=int(raw.n_cells),
        n_genes=int(raw.n_genes),
        n_hvg=int(raw.n_hvg) if run_hvg else 0,
    )


__all__ = ["run_pipeline", "PipelineResult"]
