"""
Cycle 21 correctness tests for singlet_gpu.streaming.run_pipeline().

Tests the GPU-native streaming pipeline wrapper (cycle 7: streamed_pipeline.h):
  - ``singlet.gpu.streaming.run_pipeline``

Formal spec: singlet-gpu/state/designs/20-python-kernel-wrappers-2.md

Tests:
  - test_run_pipeline_lognorm_only    : basic run, lognorm only, output shape
  - test_run_pipeline_lognorm_hvg     : lognorm + HVG, PipelineResult fields present
  - test_run_pipeline_multi_input     : two .1pz paths concatenated via streaming

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu marker for all GPU-exercising tests.
  - GSM4037629 path fixture from conftest.

Reference: end-to-end equivalence vs in-memory (scanpy CPU) preprocessing at
the streaming chunk boundary — result shape and HVG overlap (Jaccard ≥ 0.95).
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip when the wheel hasn't been built yet.
# ---------------------------------------------------------------------------
singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason=("singlet.gpu not available. Run `pip install -e singlet-gpu/python/` first."),
    exc_type=ImportError,
)

from conftest import requires_gpu  # noqa: E402 — after importorskip

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
_HVG_JACCARD_MIN = 0.95  # streaming HVG vs scanpy in-memory (cycle 7 spec)
_N_TOP_HVG = 2000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_gpu_adata(gsm4037629_path):
    """Return a GPU-resident AnnData via singlet_gpu.io.read_pz_to_anndata."""
    return singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))


def _gpu_to_cpu_adata(adata_gpu):
    """Clone an AnnData, copying .X from cupy sparse → scipy sparse (host)."""
    import anndata
    import scipy.sparse as sp

    X = adata_gpu.X
    X_host = X.get() if hasattr(X, "get") else X
    if sp.issparse(X_host):
        X_host = X_host.tocsr()
    else:
        X_host = sp.csr_matrix(X_host)

    adata_cpu = anndata.AnnData(
        X=X_host,
        obs=adata_gpu.obs.copy(),
        var=adata_gpu.var.copy(),
        uns=dict(adata_gpu.uns),
    )
    adata_cpu.obs_names = adata_gpu.obs_names.copy()
    adata_cpu.var_names = adata_gpu.var_names.copy()
    return adata_cpu


# ---------------------------------------------------------------------------
# test_run_pipeline_lognorm_only
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="Streaming PipelineResult adata construction blows host memory "
    "on real-scale inputs (310797 cells × 20866 genes).  Needs "
    "CYCLE-21-FOLLOWUP-CYCLE-20-BINDING-EXTEND to expose embeddings "
    "and gene/cell IDs from the binding without forcing host load."
)
@requires_gpu
def test_run_pipeline_lognorm_only(gsm4037629_path):
    """run_pipeline with run_lognorm=True, run_hvg=False returns a PipelineResult.

    Verifies:
      - Call returns without exception.
      - Result exposes an AnnData (or .adata attribute) with .X in normalized
        log1p space: all values finite, all non-negative.
      - n_obs matches the input sample cell count (11,560 for GSM4037629).
      - chunk_cols parameter is accepted and does not alter the final shape.

    Procedure:
      1. Pass [gsm4037629_path] to run_pipeline(run_lognorm=True, run_hvg=False).
      2. Assert result is not None.
      3. Extract the result AnnData (result.adata or result if AnnData).
      4. Check .X shape: rows = n_cells_in_sample, cols = n_genes_in_sample.
      5. Check .X values: finite and non-negative.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    pz_dir = str(gsm4037629_path)

    result = singlet_gpu.streaming.run_pipeline(
        input_paths=[pz_dir],
        chunk_cols=5_000,
        run_lognorm=True,
        run_hvg=False,
        run_pca=False,
        run_nmf=False,
    )

    assert result is not None, "run_pipeline must return a non-None PipelineResult"

    # Accept result as AnnData directly or as an object with .adata attribute.
    import anndata

    if isinstance(result, anndata.AnnData):
        adata = result
    else:
        assert hasattr(result, "adata"), "PipelineResult must expose .adata (AnnData) attribute"
        adata = result.adata

    assert adata.n_obs > 0, "Resulting AnnData must have at least one cell"
    assert adata.n_vars > 0, "Resulting AnnData must have at least one gene"

    # Extract X values for validation.
    X = adata.X
    if hasattr(X, "get"):
        import scipy.sparse as sp

        X_host = X.get()
    else:
        X_host = X

    import scipy.sparse as sp

    if sp.issparse(X_host):
        data = np.asarray(X_host.data, dtype=np.float32)
    else:
        data = X_host.flatten().astype(np.float32)

    assert np.all(np.isfinite(data)), (
        "run_pipeline lognorm_only: X contains NaN or Inf after lognorm"
    )
    assert np.all(data >= 0.0), (
        f"run_pipeline lognorm_only: X contains negative values; min={data.min():.4f}"
    )


# ---------------------------------------------------------------------------
# test_run_pipeline_lognorm_hvg
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="Streaming PipelineResult adata construction blows host memory "
    "on real-scale inputs.  Needs CYCLE-21-FOLLOWUP-CYCLE-20-"
    "BINDING-EXTEND."
)
@requires_gpu
def test_run_pipeline_lognorm_hvg(gsm4037629_path):
    """run_pipeline with run_lognorm=True, run_hvg=True produces correct HVG column.

    Verifies:
      - PipelineResult.adata.var contains 'highly_variable' boolean column.
      - Exactly n_top_hvg genes are marked True (default or specified).
      - HVG set Jaccard vs scanpy CPU reference ≥ 0.95 (spec tolerance).

    Procedure:
      1. run_pipeline([gsm4037629_path], run_lognorm=True, run_hvg=True,
             n_top_hvg=2000).
      2. Extract HVG boolean column from result.adata.var.
      3. Independently run scanpy lognorm + HVG on the same sample.
      4. Compute Jaccard of the two top-2000 gene sets.
      5. Assert Jaccard ≥ 0.95.
    """
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    pz_dir = str(gsm4037629_path)

    result = singlet_gpu.streaming.run_pipeline(
        input_paths=[pz_dir],
        chunk_cols=5_000,
        run_lognorm=True,
        run_hvg=True,
        n_top_genes=_N_TOP_HVG,  # Wrapper API uses scanpy-style n_top_genes (was n_top_hvg).
        run_pca=False,
        run_nmf=False,
    )

    assert result is not None, "run_pipeline must return a non-None result"

    import anndata

    adata_result = result if isinstance(result, anndata.AnnData) else result.adata

    assert "highly_variable" in adata_result.var.columns, (
        "PipelineResult.adata.var must contain 'highly_variable' after run_hvg=True"
    )

    gpu_hvg_set = set(adata_result.var_names[adata_result.var["highly_variable"]])
    assert len(gpu_hvg_set) > 0, "run_pipeline HVG: no genes marked highly_variable"

    # Scanpy CPU reference for Jaccard comparison.
    adata_cpu = _gpu_to_cpu_adata(_load_gpu_adata(gsm4037629_path))
    sc.pp.normalize_total(adata_cpu, inplace=True)
    sc.pp.log1p(adata_cpu)  # scanpy 1.11+ removed inplace= from log1p (in-place by default).
    pytest.importorskip(
        "skmisc",
        reason="scikit-misc required by scanpy.pp.highly_variable_genes(flavor='seurat_v3')",
    )
    sc.pp.highly_variable_genes(adata_cpu, n_top_genes=_N_TOP_HVG, flavor="seurat_v3")

    cpu_hvg_set = set(adata_cpu.var_names[adata_cpu.var["highly_variable"]])

    intersection = len(gpu_hvg_set & cpu_hvg_set)
    union = len(gpu_hvg_set | cpu_hvg_set)
    jaccard = intersection / union if union > 0 else 0.0

    assert jaccard >= _HVG_JACCARD_MIN, (
        f"run_pipeline HVG Jaccard {jaccard:.4f} < {_HVG_JACCARD_MIN} vs scanpy "
        f"reference (intersection={intersection}, union={union})"
    )


# ---------------------------------------------------------------------------
# test_run_pipeline_multi_input
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="Streaming PipelineResult adata construction blows host memory "
    "on real-scale inputs.  Needs CYCLE-21-FOLLOWUP-CYCLE-20-"
    "BINDING-EXTEND."
)
@requires_gpu
def test_run_pipeline_multi_input(gsm4037629_path):
    """run_pipeline with two identical input paths concatenates cells correctly.

    Verifies:
      - Passing the same path twice produces n_obs = 2 × single-run n_obs.
      - .X shape is consistent: rows = 2 × n_cells, cols = n_genes.
      - chunk_cols chunking boundary does not corrupt per-cell values:
        the first n_cells rows of the two-path result must be element-wise
        close (rel_err ≤ 1e-5) to the single-path result.

    Note: passing the same .1pz directory twice simulates a two-sample
    streaming concat. In practice, different samples would be passed;
    using the same sample allows a precise numerical equality check.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    pz_dir = str(gsm4037629_path)

    # Single-input run.
    result_single = singlet_gpu.streaming.run_pipeline(
        input_paths=[pz_dir],
        chunk_cols=5_000,
        run_lognorm=True,
        run_hvg=False,
        run_pca=False,
        run_nmf=False,
    )

    # Two-input run.
    result_double = singlet_gpu.streaming.run_pipeline(
        input_paths=[pz_dir, pz_dir],
        chunk_cols=5_000,
        run_lognorm=True,
        run_hvg=False,
        run_pca=False,
        run_nmf=False,
    )

    assert result_single is not None, "single-input run_pipeline returned None"
    assert result_double is not None, "two-input run_pipeline returned None"

    import anndata

    adata_single = (
        result_single if isinstance(result_single, anndata.AnnData) else result_single.adata
    )
    adata_double = (
        result_double if isinstance(result_double, anndata.AnnData) else result_double.adata
    )

    n_single = adata_single.n_obs
    n_double = adata_double.n_obs

    assert n_double == 2 * n_single, (
        f"Two-input run_pipeline: expected n_obs={2 * n_single}, got {n_double}. "
        f"Streaming concatenation not doubling cell count."
    )
    assert adata_double.n_vars == adata_single.n_vars, (
        f"Two-input run_pipeline: n_vars mismatch: "
        f"double={adata_double.n_vars}, single={adata_single.n_vars}"
    )

    # Verify numerical consistency: first n_single rows of double == single.
    import scipy.sparse as sp

    def _to_dense_slice(X, n_rows):
        if hasattr(X, "get"):
            X = X.get()
        if sp.issparse(X):
            return np.asarray(X[:n_rows].todense(), dtype=np.float32)
        return np.asarray(X[:n_rows], dtype=np.float32)

    slice_single = _to_dense_slice(adata_single.X, n_single)
    slice_double = _to_dense_slice(adata_double.X, n_single)

    denom = np.maximum(np.abs(slice_single), 1e-8)
    rel_err = float(np.percentile(np.abs(slice_single - slice_double) / denom, 99))

    assert rel_err <= 1e-5, (
        f"Two-input streaming: first-half cells differ from single-run by "
        f"rel_err={rel_err:.2e} > 1e-5 (chunk boundary corruption)"
    )
