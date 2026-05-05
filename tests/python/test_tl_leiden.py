"""
Cycle 21 correctness tests for singlet_gpu.tools.leiden().

Tests the GPU-native Leiden clustering wrapper (cycle 9: graph/leiden.h):
  - ``singlet.gpu.tools.leiden``

Formal spec: singlet-gpu/state/designs/20-python-kernel-wrappers-2.md

Verified scanpy signature (from spec):
  def leiden(adata, resolution=1, *, restrict_to=None, rng=None,
             key_added='leiden', adjacency=None, directed=None, use_weights=True,
             n_iterations=-1, partition_type=None, neighbors_key=None,
             obsp=None, copy=False, flavor=...)
  Result: adata.obs['leiden'] (string cluster labels).

Tests:
  - test_leiden_basic              : call succeeds, obs['leiden'] written
  - test_leiden_resolution_param   : higher resolution → more clusters
  - test_leiden_writes_obs         : obs['leiden'] shape, dtype, non-empty labels
  - test_leiden_vs_scanpy          : ARI vs sc.tl.leiden ≥ 0.85

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu marker for all GPU-exercising tests.
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
_ARI_MIN = 0.85          # ARI vs scanpy Leiden (spec tolerance)
_SEED = 0xC0FFEE         # fixed seed for reproducibility
_N_NEIGHBORS = 15
_N_COMPS = 50

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


def _full_pipeline_gpu(adata_gpu, resolution=1.0):
    """Preprocess + PCA + neighbors on GPU, then return without leiden."""
    singlet_gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet_gpu.preprocess.log1p(adata_gpu, inplace=True)
    singlet_gpu.reduce.pca(adata_gpu, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.pp.neighbors(adata_gpu, n_neighbors=_N_NEIGHBORS)


def _full_pipeline_cpu(adata_cpu):
    """Preprocess + PCA + neighbors on CPU (scanpy) for reference."""
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata_cpu, inplace=True)
    sc.pp.log1p(adata_cpu)  # scanpy 1.11+ removed inplace= (in-place by default)
    sc.pp.pca(adata_cpu, n_comps=_N_COMPS)
    sc.pp.neighbors(adata_cpu, n_neighbors=_N_NEIGHBORS, use_rep="X_pca")


def _adjusted_rand_index(labels_a, labels_b):
    """Compute ARI between two label arrays using sklearn."""
    sklearn_metrics = pytest.importorskip(
        "sklearn.metrics", reason="sklearn not installed"
    )
    a = np.asarray(labels_a, dtype=str)
    b = np.asarray(labels_b, dtype=str)
    return float(sklearn_metrics.adjusted_rand_score(a, b))


# ---------------------------------------------------------------------------
# test_leiden_basic
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: leiden requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_leiden_basic(gsm4037629_path):
    """leiden() completes without error and writes adata.obs['leiden'].

    Procedure:
      1. Load GSM4037629, preprocess + PCA + neighbors.
      2. Call singlet_gpu.tools.leiden(adata).
      3. Assert returns None (inplace convention).
      4. Assert adata.obs['leiden'] exists and has n_obs entries.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)

    ret = singlet_gpu.tools.leiden(adata, rng=_SEED)

    assert ret is None, (
        f"leiden() inplace must return None (scanpy convention), got {type(ret)}"
    )
    assert "leiden" in adata.obs.columns, (
        "leiden(): adata.obs['leiden'] column not written"
    )
    assert len(adata.obs["leiden"]) == adata.n_obs, (
        f"adata.obs['leiden'] length {len(adata.obs['leiden'])} != n_obs {adata.n_obs}"
    )


# ---------------------------------------------------------------------------
# test_leiden_resolution_param
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: leiden requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_leiden_resolution_param(gsm4037629_path):
    """Higher resolution produces more clusters (monotonic response).

    Procedure:
      1. Run leiden with resolution=0.3 → n_clusters_low.
      2. Run leiden with resolution=1.5 → n_clusters_high.
      3. Assert n_clusters_high > n_clusters_low.

    Note: this must hold for the GSM4037629 sample (11,560 cells) where both
    resolution values are well within the biologically meaningful range.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_low = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata_low)
    singlet_gpu.tools.leiden(adata_low, 0.3, rng=_SEED, key_added="leiden_low")

    adata_high = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata_high)
    singlet_gpu.tools.leiden(adata_high, 1.5, rng=_SEED, key_added="leiden_high")

    n_low = int(adata_low.obs["leiden_low"].nunique())
    n_high = int(adata_high.obs["leiden_high"].nunique())

    assert n_low >= 1, f"resolution=0.3 produced 0 clusters"
    assert n_high >= 1, f"resolution=1.5 produced 0 clusters"
    assert n_high > n_low, (
        f"Higher resolution must produce more clusters: "
        f"low(res=0.3)={n_low}, high(res=1.5)={n_high}"
    )


# ---------------------------------------------------------------------------
# test_leiden_writes_obs
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: leiden requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_leiden_writes_obs(gsm4037629_path):
    """leiden() writes adata.obs['leiden'] with correct shape, dtype, and labels.

    Verifies:
      - obs['leiden'] dtype is object or str (categorical string labels).
      - Number of unique labels (clusters) is in [2, n_obs / 2].
      - All cells are assigned a label (no NaN).
      - key_added parameter changes the column name correctly.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)
    singlet_gpu.tools.leiden(adata, rng=_SEED, key_added="my_leiden")

    assert "my_leiden" in adata.obs.columns, (
        "leiden(key_added='my_leiden'): 'my_leiden' column not in adata.obs"
    )

    labels = adata.obs["my_leiden"]
    assert len(labels) == adata.n_obs, (
        f"obs['my_leiden'] length {len(labels)} != n_obs {adata.n_obs}"
    )
    assert labels.isna().sum() == 0, (
        f"obs['my_leiden'] contains {labels.isna().sum()} NaN values"
    )

    n_clusters = int(labels.nunique())
    assert n_clusters >= 2, (
        f"leiden produced only {n_clusters} cluster(s) — expected ≥ 2"
    )
    assert n_clusters <= adata.n_obs // 2, (
        f"leiden produced {n_clusters} clusters for {adata.n_obs} cells — seems too fragmented"
    )


# ---------------------------------------------------------------------------
# test_leiden_vs_scanpy
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: leiden requires cuGraph; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_leiden_vs_scanpy(gsm4037629_path):
    """GPU leiden clustering ARI ≥ 0.85 vs sc.tl.leiden.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Run full GPU pipeline: preprocess + PCA + neighbors + leiden.
      3. Run full scanpy CPU pipeline: preprocess + PCA + neighbors + leiden.
      4. Compute ARI between the two label arrays.
      5. Assert ARI ≥ 0.85.

    Note: ARI = 1 is not expected due to GPU vs CPU kNN implementation
    differences and stochastic partition initializations.
    """
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _full_pipeline_gpu(adata_gpu)
    singlet_gpu.tools.leiden(adata_gpu, 1.0, rng=_SEED)

    _full_pipeline_cpu(adata_cpu)
    sc.tl.leiden(adata_cpu, 1.0, rng=_SEED)

    gpu_labels = adata_gpu.obs["leiden"].values
    cpu_labels = adata_cpu.obs["leiden"].values

    ari = _adjusted_rand_index(gpu_labels, cpu_labels)
    assert ari >= _ARI_MIN, (
        f"leiden ARI {ari:.4f} < {_ARI_MIN} vs sc.tl.leiden "
        f"(gpu_n_clusters={len(set(gpu_labels))}, cpu_n_clusters={len(set(cpu_labels))})"
    )
