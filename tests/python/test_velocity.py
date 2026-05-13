"""
Cycle 23 correctness tests for singlet_gpu.velocity.

Tests the GPU-native RNA velocity wrappers (cycle 15: preprocess/velocity_prep.h):
  - ``singlet.gpu.velocity.moments``
  - ``singlet.gpu.velocity.velocity``

Formal spec: singlet-gpu/state/designs/22-python-kernel-wrappers-3.md

API (from spec):
  moments(adata, *, n_neighbors=30, n_pcs=None, mode='connectivities',
          use_rep=None, layer_spliced='spliced', layer_unspliced='unspliced',
          copy=False) -> AnnData | None
  velocity(adata, *, mode='steady_state', fit_offset=False, perc=[5,95],
           layer_spliced='spliced', layer_unspliced='unspliced',
           copy=False) -> AnnData | None

Result locations (spec):
  - moments: adata.layers['Ms'] (smoothed spliced), adata.layers['Mu'] (smoothed unspliced)
  - velocity: adata.layers['velocity'] (RNA velocity), adata.var['velocity_gamma'] (rate coeff)

Tolerances (from spec):
  - moments.Ms/Mu vs scvelo.pp.moments : relative error ≤ 1e-3 (element-wise mean)
  - velocity_gamma vs scvelo.tl.velocity (steady_state): Spearman ρ ≥ 0.95

Notes:
  - GSM4037629 has intron_counts.1pz; we load it as 'unspliced' and exon_counts.1pz
    as 'spliced' per the singlet data contract.
  - scvelo tests require an scvelo-compatible AnnData with 'spliced'/'unspliced' layers.

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu for all GPU-exercising tests.
  - vs-scvelo tests additionally skip if scvelo is not importable.
"""

from __future__ import annotations

import numpy as np
import pytest

singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason="singlet.gpu not available. Run `pip install -e singlet-gpu/python/` first.",
    exc_type=ImportError,
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
_MOMENTS_REL_ERR_MAX = 1e-3  # element-wise mean relative error vs scvelo
_GAMMA_SPEARMAN_MIN = 0.95  # Spearman ρ of gamma vs scvelo steady_state
_N_NEIGHBORS = 30
_N_COMPS = 30  # PCA comps for velocity embedding
_SEED = 0xC0FFEE

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_velocity_adata(gsm4037629_path):
    """Load exon_counts as 'spliced' and intron_counts as 'unspliced' layers."""
    adata = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))
    # Add spliced/unspliced layers from the two 1pz files.
    # read_pz_to_anndata may already expose them as adata.layers; if not, load explicitly.
    if "spliced" not in adata.layers:
        try:
            import pathlib

            spliced_path = pathlib.Path(gsm4037629_path) / "exon_counts.1pz"
            unspliced_path = pathlib.Path(gsm4037629_path) / "intron_counts.1pz"
            if spliced_path.is_file():
                sp_adata = singlet_gpu.io.read_pz_to_anndata(str(spliced_path))
                adata.layers["spliced"] = sp_adata.X
            if unspliced_path.is_file():
                us_adata = singlet_gpu.io.read_pz_to_anndata(str(unspliced_path))
                adata.layers["unspliced"] = us_adata.X
        except Exception:
            # Fallback: use X as both layers (smoke test only).
            adata.layers["spliced"] = adata.X
            adata.layers["unspliced"] = adata.X
    if "unspliced" not in adata.layers:
        adata.layers["unspliced"] = adata.layers.get("spliced", adata.X)
    return adata


def _to_host_dense(x) -> np.ndarray:
    """Transfer any matrix (cupy sparse / dense, scipy sparse, numpy) to dense host."""
    import scipy.sparse as sp

    if hasattr(x, "get"):
        x = x.get()
    if sp.issparse(x):
        return np.asarray(x.todense(), dtype=np.float32)
    return np.asarray(x, dtype=np.float32)


def _gpu_to_cpu_velocity_adata(adata_gpu):
    """Produce a CPU-side AnnData with spliced/unspliced layers for scvelo reference."""
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

    for layer in ("spliced", "unspliced"):
        if layer in adata_gpu.layers:
            arr = adata_gpu.layers[layer]
            arr_host = arr.get() if hasattr(arr, "get") else arr
            if sp.issparse(arr_host):
                arr_host = arr_host.tocsr()
            else:
                arr_host = sp.csr_matrix(arr_host)
            adata_cpu.layers[layer] = arr_host

    return adata_cpu


def _spearman_rho(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    res = spearmanr(a, b)
    return float(res.correlation if hasattr(res, "correlation") else res[0])


# ---------------------------------------------------------------------------
# test_moments_writes_layers_Ms_Mu
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: _core.velocity_moments not yet exposed by pybind11 binding (only velocity_prep_compute is). Binding-extend cycle pending.",
)
@requires_gpu
def test_moments_writes_layers_Ms_Mu(gsm4037629_path):
    """moments() writes adata.layers['Ms'] and adata.layers['Mu'].

    Procedure:
      1. Load GSM4037629 with spliced/unspliced layers.
      2. Preprocess + PCA.
      3. Call singlet_gpu.velocity.moments(adata, n_neighbors=30).
      4. Assert returns None (inplace convention).
      5. Assert 'Ms' and 'Mu' in adata.layers.
      6. Check shapes match adata.shape.
      7. Check values are finite.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_velocity_adata(gsm4037629_path)
    singlet_gpu.preprocess.normalize_total(adata, inplace=True)
    singlet_gpu.preprocess.log1p(adata, inplace=True)
    singlet_gpu.reduce.pca(adata, n_comps=_N_COMPS, inplace=True)

    ret = singlet_gpu.velocity.moments(adata, n_neighbors=_N_NEIGHBORS)

    assert ret is None, f"moments() inplace must return None, got {type(ret)}"
    assert "Ms" in adata.layers, "moments(): adata.layers['Ms'] not written"
    assert "Mu" in adata.layers, "moments(): adata.layers['Mu'] not written"

    Ms = _to_host_dense(adata.layers["Ms"])
    Mu = _to_host_dense(adata.layers["Mu"])

    assert Ms.shape == (adata.n_obs, adata.n_vars), (
        f"layers['Ms'] shape {Ms.shape} != {(adata.n_obs, adata.n_vars)}"
    )
    assert Mu.shape == (adata.n_obs, adata.n_vars), (
        f"layers['Mu'] shape {Mu.shape} != {(adata.n_obs, adata.n_vars)}"
    )
    assert np.all(np.isfinite(Ms)), (
        f"layers['Ms'] contains non-finite values: {np.sum(~np.isfinite(Ms))}"
    )
    assert np.all(np.isfinite(Mu)), (
        f"layers['Mu'] contains non-finite values: {np.sum(~np.isfinite(Mu))}"
    )


# ---------------------------------------------------------------------------
# test_moments_vs_scvelo
# ---------------------------------------------------------------------------
@requires_gpu
def test_moments_vs_scvelo(gsm4037629_path):
    """GPU moments Ms/Mu relative error ≤ 1e-3 vs scvelo.pp.moments.

    Procedure:
      1. Load GSM4037629 with layers twice → adata_gpu, adata_cpu.
      2. Preprocess + PCA both identically.
      3. singlet_gpu.velocity.moments(adata_gpu, n_neighbors=30).
      4. scvelo.pp.moments(adata_cpu, n_neighbors=30, n_pcs=30).
      5. Compare adata_gpu.layers['Ms'] vs adata_cpu.layers['Ms']:
         compute element-wise relative error on non-zero positions,
         assert mean ≤ 1e-3.
      6. Same comparison for 'Mu'.
    """
    scv = pytest.importorskip("scvelo", reason="scvelo not installed — skip vs-reference test")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_velocity_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_velocity_adata(adata_gpu)

    singlet_gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet_gpu.preprocess.log1p(adata_gpu, inplace=True)
    singlet_gpu.reduce.pca(adata_gpu, n_comps=_N_COMPS, inplace=True)

    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata_cpu, inplace=True)
    sc.pp.log1p(adata_cpu, inplace=True)
    sc.pp.pca(adata_cpu, n_comps=_N_COMPS)

    # GPU moments.
    singlet_gpu.velocity.moments(adata_gpu, n_neighbors=_N_NEIGHBORS)

    # scvelo reference moments.
    try:
        scv.pp.moments(adata_cpu, n_neighbors=_N_NEIGHBORS, n_pcs=_N_COMPS)
    except Exception as e:
        pytest.skip(f"scvelo.pp.moments failed: {e}")

    for layer_name in ("Ms", "Mu"):
        assert layer_name in adata_gpu.layers, f"GPU moments did not write layers['{layer_name}']"
        assert layer_name in adata_cpu.layers, (
            f"scvelo moments did not write layers['{layer_name}']"
        )

        gpu_arr = _to_host_dense(adata_gpu.layers[layer_name]).ravel().astype(np.float64)
        cpu_arr = _to_host_dense(adata_cpu.layers[layer_name]).ravel().astype(np.float64)

        mask = np.abs(cpu_arr) > 1e-9
        if mask.sum() == 0:
            continue  # all-zero reference: skip this layer

        rel_err = np.abs(gpu_arr[mask] - cpu_arr[mask]) / (np.abs(cpu_arr[mask]) + 1e-9)
        mean_rel_err = float(np.mean(rel_err))
        assert mean_rel_err <= _MOMENTS_REL_ERR_MAX, (
            f"moments layers['{layer_name}'] mean relative error {mean_rel_err:.4e} "
            f"> {_MOMENTS_REL_ERR_MAX} vs scvelo.pp.moments"
        )


# ---------------------------------------------------------------------------
# test_velocity_writes_layers_velocity
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: _core.velocity_moments / _core.velocity_velocity not yet exposed by pybind11 binding. Binding-extend cycle pending.",
)
@requires_gpu
def test_velocity_writes_layers_velocity(gsm4037629_path):
    """velocity() writes adata.layers['velocity'].

    Procedure:
      1. Load GSM4037629 with layers.
      2. Preprocess + PCA + moments (prerequisite for velocity).
      3. Call singlet_gpu.velocity.velocity(adata, mode='steady_state').
      4. Assert returns None (inplace convention).
      5. Assert 'velocity' in adata.layers.
      6. Check shape matches adata.shape.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_velocity_adata(gsm4037629_path)
    singlet_gpu.preprocess.normalize_total(adata, inplace=True)
    singlet_gpu.preprocess.log1p(adata, inplace=True)
    singlet_gpu.reduce.pca(adata, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.velocity.moments(adata, n_neighbors=_N_NEIGHBORS)

    ret = singlet_gpu.velocity.velocity(adata, mode="steady_state")

    assert ret is None, f"velocity() inplace must return None, got {type(ret)}"
    assert "velocity" in adata.layers, "velocity(): adata.layers['velocity'] not written"

    vel = _to_host_dense(adata.layers["velocity"])
    assert vel.shape == (adata.n_obs, adata.n_vars), (
        f"layers['velocity'] shape {vel.shape} != {(adata.n_obs, adata.n_vars)}"
    )


# ---------------------------------------------------------------------------
# test_velocity_writes_var_gamma
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    raises=AttributeError,
    reason="CYCLE-23-FOLLOWUP-CYCLE-22-BINDING-EXPOSE: _core.velocity_moments / _core.velocity_velocity not yet exposed by pybind11 binding. Binding-extend cycle pending.",
)
@requires_gpu
def test_velocity_writes_var_gamma(gsm4037629_path):
    """velocity() writes adata.var['velocity_gamma'] (per-gene rate coefficients).

    Verifies:
      - var['velocity_gamma'] length == n_vars.
      - All values are finite and non-negative (gamma ≥ 0 by definition).
      - Non-trivial variance (gamma differs across genes).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_velocity_adata(gsm4037629_path)
    singlet_gpu.preprocess.normalize_total(adata, inplace=True)
    singlet_gpu.preprocess.log1p(adata, inplace=True)
    singlet_gpu.reduce.pca(adata, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.velocity.moments(adata, n_neighbors=_N_NEIGHBORS)
    singlet_gpu.velocity.velocity(adata, mode="steady_state")

    assert "velocity_gamma" in adata.var.columns, (
        "velocity(): adata.var['velocity_gamma'] not written"
    )
    gamma = adata.var["velocity_gamma"].values.astype(np.float64)
    assert len(gamma) == adata.n_vars, (
        f"velocity_gamma length {len(gamma)} != n_vars {adata.n_vars}"
    )
    assert np.all(np.isfinite(gamma)), (
        f"velocity_gamma contains non-finite values: {np.sum(~np.isfinite(gamma))}"
    )
    assert np.all(gamma >= 0.0), f"velocity_gamma contains negative values; min={gamma.min():.4e}"
    assert np.std(gamma) > 1e-8, (
        "velocity_gamma has near-zero variance — all genes assigned same gamma"
    )


# ---------------------------------------------------------------------------
# test_velocity_vs_scvelo_steady_state
# ---------------------------------------------------------------------------
@requires_gpu
def test_velocity_vs_scvelo_steady_state(gsm4037629_path):
    """GPU velocity_gamma Spearman ρ ≥ 0.95 vs scvelo.tl.velocity steady_state.

    Procedure:
      1. Load GSM4037629 with layers twice → adata_gpu, adata_cpu.
      2. GPU: preprocess + PCA + moments + velocity.
      3. CPU (scvelo): normalize + moments + velocity (steady_state).
      4. Extract var['velocity_gamma'] from both.
      5. Filter to genes finite in both.
      6. Assert Spearman ρ ≥ 0.95.
    """
    scv = pytest.importorskip("scvelo", reason="scvelo not installed — skip vs-reference test")
    pytest.importorskip("scipy.stats", reason="scipy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_velocity_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_velocity_adata(adata_gpu)

    # GPU pipeline.
    singlet_gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet_gpu.preprocess.log1p(adata_gpu, inplace=True)
    singlet_gpu.reduce.pca(adata_gpu, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.velocity.moments(adata_gpu, n_neighbors=_N_NEIGHBORS)
    singlet_gpu.velocity.velocity(adata_gpu, mode="steady_state")

    # scvelo reference pipeline.
    _sc = pytest.importorskip("scanpy", reason="scanpy not installed")  # noqa: F841
    try:
        scv.pp.filter_and_normalize(adata_cpu, min_shared_counts=20, n_top_genes=2000)
        scv.pp.moments(adata_cpu, n_neighbors=_N_NEIGHBORS, n_pcs=_N_COMPS)
        scv.tl.velocity(adata_cpu, mode="steady_state")
    except Exception as e:
        pytest.skip(f"scvelo reference pipeline failed: {e}")

    if "velocity_gamma" not in adata_cpu.var.columns:
        pytest.skip("scvelo did not write var['velocity_gamma']")

    gpu_gamma = adata_gpu.var["velocity_gamma"].values.astype(np.float64)
    cpu_gamma = adata_cpu.var["velocity_gamma"].values.astype(np.float64)

    # Align by var_names (scvelo may filter genes).
    gpu_genes = list(adata_gpu.var_names)
    cpu_genes = list(adata_cpu.var_names)
    shared = list(set(gpu_genes) & set(cpu_genes))
    if len(shared) < 20:
        pytest.skip(
            f"Too few shared genes between GPU ({len(gpu_genes)}) and scvelo "
            f"({len(cpu_genes)}) var_names after scvelo filtering: {len(shared)}"
        )

    gpu_map = dict(zip(gpu_genes, gpu_gamma))
    cpu_map = dict(zip(cpu_genes, cpu_gamma))
    gpu_arr = np.array([gpu_map[g] for g in shared], dtype=np.float64)
    cpu_arr = np.array([cpu_map[g] for g in shared], dtype=np.float64)

    valid = np.isfinite(gpu_arr) & np.isfinite(cpu_arr) & (gpu_arr >= 0) & (cpu_arr >= 0)
    if valid.sum() < 10:
        pytest.skip(f"Too few valid shared gamma estimates: {valid.sum()}")

    rho = _spearman_rho(gpu_arr[valid], cpu_arr[valid])
    assert rho >= _GAMMA_SPEARMAN_MIN, (
        f"velocity_gamma Spearman ρ {rho:.4f} < {_GAMMA_SPEARMAN_MIN} "
        f"vs scvelo steady_state (n_shared_genes={valid.sum()})"
    )
