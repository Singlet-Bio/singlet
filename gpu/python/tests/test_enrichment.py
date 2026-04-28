"""
Cycle 23 correctness tests for singlet_gpu.enrichment.

Tests the GPU-native enrichment wrappers (cycle 13: gsea/{fgsea,aucell}.h):
  - ``singlet_gpu.enrichment.run_gsea``
  - ``singlet_gpu.enrichment.run_aucell``

Formal spec: singlet-gpu/state/designs/22-python-kernel-wrappers-3.md

API (from spec):
  run_gsea(mat, net, *, source='source', target='target', times=1000,
           min_n=5, seed=42, use_raw=False) -> AnnData | pd.DataFrame
  run_aucell(mat, net, *, source='source', target='target', min_n=5,
             seed=42, use_raw=False) -> AnnData | pd.DataFrame

Result locations (spec):
  - GSEA : adata.obs['gsea_norm_es_<set>'] columns (one per pathway)
  - AUCell: adata.obsm['X_aucell']  (n_cells × n_pathways)

Tolerances (from spec):
  - run_gsea  vs decoupler.run_gsea : Spearman ρ on enrichment scores ≥ 0.95
  - run_aucell vs decoupler.run_aucell : Spearman ρ on AUC scores ≥ 0.95

Skip strategy:
  - Module-level skip if singlet_gpu wheel not built.
  - requires_gpu for all GPU-exercising tests.
  - vs-decoupler tests additionally skip if decoupleR is not importable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

singlet_gpu = pytest.importorskip(
    "singlet_gpu",
    reason="singlet_gpu wheel not built. Run `pip install -e singlet-gpu/python/` first.",
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
_GSEA_SPEARMAN_MIN = 0.95    # Spearman ρ of normalized ES vs decoupler.run_gsea
_AUCELL_SPEARMAN_MIN = 0.95  # Spearman ρ of AUC scores vs decoupler.run_aucell
_SEED = 42
_MIN_N = 5
_TIMES = 100                 # reduced permutations for speed in harness

# ---------------------------------------------------------------------------
# Minimal synthetic gene-set net fixture
# ---------------------------------------------------------------------------

def _make_net(var_names: list[str], n_sets: int = 5, set_size: int = 10) -> pd.DataFrame:
    """Build a minimal decoupleR-compatible net DataFrame.

    Columns: source (set name), target (gene name), weight (float).
    Gene sets are drawn deterministically from var_names.
    """
    rng = np.random.default_rng(0xC0FFEE)
    rows = []
    for i in range(n_sets):
        set_name = f"PATHWAY_{i:02d}"
        # non-overlapping slices; wrap around if gene list is short
        start = (i * set_size) % max(1, len(var_names) - set_size)
        genes = var_names[start : start + set_size]
        for g in genes:
            rows.append({"source": set_name, "target": g, "weight": float(rng.uniform(0.5, 1.5))})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_gpu_adata(gsm4037629_path):
    return singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))


def _gpu_to_cpu_adata(adata_gpu):
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


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr
    res = spearmanr(a, b)
    return float(res.correlation if hasattr(res, "correlation") else res[0])


def _preprocess_gpu(adata):
    """Normalize + log1p (GPU path) for enrichment input."""
    singlet_gpu.preprocess.normalize_total(adata, inplace=True)
    singlet_gpu.preprocess.log1p(adata, inplace=True)


def _preprocess_cpu(adata):
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata, inplace=True)


# ---------------------------------------------------------------------------
# test_run_gsea_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_run_gsea_basic(gsm4037629_path):
    """run_gsea() completes and writes per-pathway obs columns.

    Procedure:
      1. Load GSM4037629, normalize + log1p.
      2. Build a synthetic 5-pathway net from var_names.
      3. Call singlet_gpu.enrichment.run_gsea(adata, net, seed=42).
      4. Assert result is either:
         (a) None (inplace) with adata.obs columns 'gsea_norm_es_PATHWAY_*', OR
         (b) a pd.DataFrame with one row per cell and one column per pathway.
      5. Verify at least one enrichment score column is present.
      6. Verify values are finite.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    net = _make_net(list(adata.var_names), n_sets=5, set_size=_MIN_N)

    result = singlet_gpu.enrichment.run_gsea(
        adata, net, times=_TIMES, min_n=_MIN_N, seed=_SEED
    )

    if result is None:
        # inplace: scores written to adata.obs
        es_cols = [c for c in adata.obs.columns if c.startswith("gsea_norm_es_")]
        assert len(es_cols) >= 1, (
            "run_gsea(inplace) wrote no 'gsea_norm_es_*' columns to adata.obs"
        )
        scores = adata.obs[es_cols[0]].values.astype(np.float64)
    else:
        assert isinstance(result, (pd.DataFrame,)), (
            f"run_gsea() returned unexpected type {type(result)}"
        )
        assert result.shape[0] == adata.n_obs, (
            f"run_gsea() result rows {result.shape[0]} != n_obs {adata.n_obs}"
        )
        assert result.shape[1] >= 1, "run_gsea() result has 0 pathway columns"
        scores = result.iloc[:, 0].values.astype(np.float64)

    assert np.all(np.isfinite(scores)), (
        f"run_gsea() scores contain non-finite values: "
        f"{np.sum(~np.isfinite(scores))} non-finite out of {len(scores)}"
    )


# ---------------------------------------------------------------------------
# test_run_gsea_vs_decoupler
# ---------------------------------------------------------------------------
@requires_gpu
def test_run_gsea_vs_decoupler(gsm4037629_path):
    """GPU run_gsea enrichment scores Spearman ρ ≥ 0.95 vs decoupler.run_gsea.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Preprocess both (normalize + log1p).
      3. Build same synthetic net for both.
      4. singlet_gpu.enrichment.run_gsea(adata_gpu, net, seed=42).
      5. decoupler.run_gsea(adata_cpu, net, source='source', target='target',
                            times=100, min_n=5, seed=42) → result_cpu.
      6. For each pathway, extract scores from both results; compute Spearman ρ.
      7. Assert median Spearman ρ across pathways ≥ 0.95.

    Note: Both are run with the same seed; minor differences are expected due
    to GPU vs CPU floating-point ordering. ρ ≥ 0.95 is the tolerance.
    """
    dc = pytest.importorskip(
        "decoupler", reason="decoupler not installed — skip vs-reference test"
    )
    pytest.importorskip("scipy.stats", reason="scipy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _preprocess_gpu(adata_gpu)
    _preprocess_cpu(adata_cpu)

    net = _make_net(list(adata_gpu.var_names), n_sets=5, set_size=_MIN_N)

    # GPU run (inplace or returns DataFrame).
    gpu_result = singlet_gpu.enrichment.run_gsea(
        adata_gpu, net, times=_TIMES, min_n=_MIN_N, seed=_SEED
    )

    # decoupleR reference.
    try:
        dc.run_gsea(
            adata_cpu, net,
            source="source", target="target",
            times=_TIMES, min_n=_MIN_N, seed=_SEED,
        )
    except Exception as e:
        pytest.skip(f"decoupler.run_gsea failed: {e}")

    # Extract pathway scores from GPU result.
    if gpu_result is None:
        es_cols_gpu = [c for c in adata_gpu.obs.columns if c.startswith("gsea_norm_es_")]
    else:
        es_cols_gpu = list(gpu_result.columns)

    # Extract pathway scores from CPU result (decoupleR writes to obsm or obs).
    dc_key = "gsea_norm_es"
    if dc_key in adata_cpu.obsm:
        cpu_df = adata_cpu.obsm[dc_key]
        if not isinstance(cpu_df, pd.DataFrame):
            cpu_df = pd.DataFrame(cpu_df)
    else:
        # Some versions write to obs directly.
        cpu_es_cols = [c for c in adata_cpu.obs.columns if "gsea" in c.lower()]
        if not cpu_es_cols:
            pytest.skip("Cannot locate decoupler gsea scores in adata_cpu")
        cpu_df = adata_cpu.obs[cpu_es_cols]

    pathways = list(cpu_df.columns)
    if len(pathways) == 0:
        pytest.skip("decoupler returned 0 pathway scores — cannot compare")

    rhos = []
    for pw in pathways:
        # Find matching GPU column (strip 'gsea_norm_es_' prefix if present).
        gpu_col = None
        for c in es_cols_gpu:
            if pw in c or c.endswith(pw):
                gpu_col = c
                break
        if gpu_col is None:
            continue

        cpu_scores = np.asarray(cpu_df[pw], dtype=np.float64)
        if gpu_result is None:
            gpu_scores = np.asarray(adata_gpu.obs[gpu_col], dtype=np.float64)
        else:
            gpu_scores = np.asarray(gpu_result[gpu_col], dtype=np.float64)

        valid = np.isfinite(cpu_scores) & np.isfinite(gpu_scores)
        if valid.sum() < 10:
            continue
        if np.std(cpu_scores[valid]) < 1e-9 or np.std(gpu_scores[valid]) < 1e-9:
            continue
        rho = _spearman_rho(gpu_scores[valid], cpu_scores[valid])
        if np.isfinite(rho):
            rhos.append(rho)

    assert len(rhos) >= 1, (
        "Could not compute Spearman ρ for any pathway pair between GPU and decoupler. "
        "Check that pathway names match between the two results."
    )

    median_rho = float(np.median(rhos))
    assert median_rho >= _GSEA_SPEARMAN_MIN, (
        f"run_gsea median Spearman ρ {median_rho:.4f} < {_GSEA_SPEARMAN_MIN} "
        f"vs decoupler.run_gsea (computed over {len(rhos)} pathways)"
    )


# ---------------------------------------------------------------------------
# test_run_aucell_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_run_aucell_basic(gsm4037629_path):
    """run_aucell() completes and writes adata.obsm['X_aucell'].

    Procedure:
      1. Load GSM4037629, normalize + log1p.
      2. Build a synthetic 5-pathway net from var_names.
      3. Call singlet_gpu.enrichment.run_aucell(adata, net, seed=42).
      4. Assert result is either:
         (a) None (inplace) with adata.obsm['X_aucell'] shape (n_cells, n_pathways), OR
         (b) a pd.DataFrame (n_cells × n_pathways).
      5. Verify AUC values are in [0, 1].
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_gpu(adata)

    net = _make_net(list(adata.var_names), n_sets=5, set_size=_MIN_N)
    n_sets = len(net["source"].unique())

    result = singlet_gpu.enrichment.run_aucell(
        adata, net, min_n=_MIN_N, seed=_SEED
    )

    if result is None:
        # inplace path
        assert "X_aucell" in adata.obsm, (
            "run_aucell(inplace) did not write adata.obsm['X_aucell']"
        )
        auc = np.asarray(adata.obsm["X_aucell"], dtype=np.float64)
        assert auc.shape[0] == adata.n_obs, (
            f"X_aucell rows {auc.shape[0]} != n_obs {adata.n_obs}"
        )
        assert auc.shape[1] == n_sets, (
            f"X_aucell cols {auc.shape[1]} != n_sets {n_sets}"
        )
    else:
        assert isinstance(result, pd.DataFrame), (
            f"run_aucell() returned unexpected type {type(result)}"
        )
        assert result.shape == (adata.n_obs, n_sets), (
            f"run_aucell() result shape {result.shape} != ({adata.n_obs}, {n_sets})"
        )
        auc = result.values.astype(np.float64)

    assert np.all(auc >= 0.0 - 1e-9), (
        f"AUCell scores contain negative values; min={auc.min():.4e}"
    )
    assert np.all(auc <= 1.0 + 1e-9), (
        f"AUCell scores contain values > 1.0; max={auc.max():.4e}"
    )
    assert np.all(np.isfinite(auc)), (
        f"AUCell scores contain non-finite values: {np.sum(~np.isfinite(auc))}"
    )


# ---------------------------------------------------------------------------
# test_run_aucell_vs_decoupler
# ---------------------------------------------------------------------------
@requires_gpu
def test_run_aucell_vs_decoupler(gsm4037629_path):
    """GPU run_aucell AUC scores Spearman ρ ≥ 0.95 vs decoupler.run_aucell.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Preprocess both (normalize + log1p).
      3. Build same synthetic net for both.
      4. singlet_gpu.enrichment.run_aucell(adata_gpu, net, seed=42).
      5. decoupler.run_aucell(adata_cpu, net, source='source', target='target',
                              min_n=5, seed=42).
      6. For each pathway, extract AUC scores from both; compute Spearman ρ.
      7. Assert median Spearman ρ across pathways ≥ 0.95.
    """
    dc = pytest.importorskip(
        "decoupler", reason="decoupler not installed — skip vs-reference test"
    )
    pytest.importorskip("scipy.stats", reason="scipy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _preprocess_gpu(adata_gpu)
    _preprocess_cpu(adata_cpu)

    net = _make_net(list(adata_gpu.var_names), n_sets=5, set_size=_MIN_N)

    # GPU run.
    gpu_result = singlet_gpu.enrichment.run_aucell(
        adata_gpu, net, min_n=_MIN_N, seed=_SEED
    )

    # decoupleR reference.
    try:
        dc.run_aucell(
            adata_cpu, net,
            source="source", target="target",
            min_n=_MIN_N, seed=_SEED,
        )
    except Exception as e:
        pytest.skip(f"decoupler.run_aucell failed: {e}")

    # Extract CPU AUC scores.
    dc_key = "aucell_estimate"
    if dc_key in adata_cpu.obsm:
        cpu_df = adata_cpu.obsm[dc_key]
        if not isinstance(cpu_df, pd.DataFrame):
            cpu_df = pd.DataFrame(cpu_df)
    else:
        aucell_cols = [c for c in adata_cpu.obs.columns if "aucell" in c.lower()]
        if not aucell_cols:
            pytest.skip("Cannot locate decoupler AUCell scores in adata_cpu")
        cpu_df = adata_cpu.obs[aucell_cols]

    if isinstance(gpu_result, pd.DataFrame):
        gpu_df = gpu_result
    else:
        arr = adata_gpu.obsm.get("X_aucell")
        if arr is None:
            pytest.skip("run_aucell inplace wrote nothing to obsm['X_aucell']")
        gpu_df = pd.DataFrame(
            np.asarray(arr),
            columns=[f"SET_{i}" for i in range(arr.shape[1])],
        )

    pathways_cpu = list(cpu_df.columns)
    rhos = []
    for idx, pw in enumerate(pathways_cpu):
        cpu_scores = np.asarray(cpu_df[pw], dtype=np.float64)
        # Match by position if column names don't align.
        if idx < gpu_df.shape[1]:
            gpu_scores = gpu_df.iloc[:, idx].values.astype(np.float64)
        else:
            break

        valid = np.isfinite(cpu_scores) & np.isfinite(gpu_scores)
        if valid.sum() < 10:
            continue
        if np.std(cpu_scores[valid]) < 1e-9 or np.std(gpu_scores[valid]) < 1e-9:
            continue
        rho = _spearman_rho(gpu_scores[valid], cpu_scores[valid])
        if np.isfinite(rho):
            rhos.append(rho)

    assert len(rhos) >= 1, (
        "Could not compute Spearman ρ for any pathway pair. "
        "Check that both run_aucell calls completed without error."
    )

    median_rho = float(np.median(rhos))
    assert median_rho >= _AUCELL_SPEARMAN_MIN, (
        f"run_aucell median Spearman ρ {median_rho:.4f} < {_AUCELL_SPEARMAN_MIN} "
        f"vs decoupler.run_aucell (computed over {len(rhos)} pathways)"
    )
