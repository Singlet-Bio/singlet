"""
Cycle 21 correctness tests for singlet_gpu.tools.umap().

Tests the GPU-native UMAP embedding wrapper (cycle 10: embed/umap.h):
  - ``singlet.gpu.tools.umap``

Formal spec: singlet-gpu/state/designs/20-python-kernel-wrappers-2.md

Verified scanpy signature (from spec):
  def umap(adata, *, min_dist=0.5, spread=1.0, n_components=2, maxiter=None,
           alpha=1.0, gamma=1.0, negative_sample_rate=5, init_pos='random',
           rng=None, a=None, b=None, method='umap', key_added=None,
           neighbors_key='neighbors', copy=False)
  Note: default init_pos='random' (not 'spectral') for GPU reproducibility.
  Result: adata.obsm['X_umap'].

Tests:
  - test_umap_basic                        : call succeeds, X_umap written
  - test_umap_writes_obsm                  : X_umap shape, dtype, finite values
  - test_umap_n_components                 : n_components parameter honored
  - test_umap_vs_scanpy_trustworthiness    : trustworthiness ≥ 0.85 vs sc.tl.umap

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
_TRUST_MIN = 0.85        # trustworthiness vs scanpy UMAP (spec tolerance)
_SEED = 0xC0FFEE
_N_NEIGHBORS = 15
_N_COMPS = 50
_TRUST_K = 15            # k for trustworthiness metric

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


def _full_pipeline_gpu(adata_gpu):
    """Preprocess + PCA + neighbors (GPU). Precursor for UMAP."""
    singlet_gpu.preprocess.normalize_total(adata_gpu, inplace=True)
    singlet_gpu.preprocess.log1p(adata_gpu, inplace=True)
    singlet_gpu.reduce.pca(adata_gpu, n_comps=_N_COMPS, inplace=True)
    singlet_gpu.pp.neighbors(adata_gpu, n_neighbors=_N_NEIGHBORS)


def _full_pipeline_cpu(adata_cpu):
    """Preprocess + PCA + neighbors (scanpy CPU). Precursor for scanpy UMAP."""
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata_cpu, inplace=True)
    sc.pp.log1p(adata_cpu, inplace=True)
    sc.pp.pca(adata_cpu, n_comps=_N_COMPS)
    sc.pp.neighbors(adata_cpu, n_neighbors=_N_NEIGHBORS, use_rep="X_pca")


def _trustworthiness(X_high, X_low, k=15):
    """Compute trustworthiness of a low-dimensional embedding.

    Uses sklearn.manifold.trustworthiness if available; otherwise computes
    manually via nearest-neighbor rank comparison.
    """
    try:
        from sklearn.manifold import trustworthiness as sk_trust
        return float(sk_trust(X_high, X_low, n_neighbors=k))
    except ImportError:
        pass

    # Manual fallback: compute pairwise distances and rank-based trustworthiness.
    from sklearn.metrics import pairwise_distances  # only pairwise_distances

    n = X_high.shape[0]
    D_high = pairwise_distances(X_high)
    D_low = pairwise_distances(X_low)

    # For each point i, rank neighbours by distance in high-dim space.
    rank_high = np.argsort(np.argsort(D_high, axis=1), axis=1)
    nn_low = np.argsort(D_low, axis=1)[:, 1:k + 1]  # k NN in low-dim (exclude self)

    t_sum = 0.0
    for i in range(n):
        for j in nn_low[i]:
            r = rank_high[i, j]
            if r > k:
                t_sum += r - k

    n_factor = 2.0 / (n * k * (2 * n - 3 * k - 1))
    return float(1.0 - n_factor * t_sum)


def _to_numpy(arr):
    """Return a numpy 2D float32 array from cupy or numpy input."""
    if hasattr(arr, "get"):
        arr = arr.get()
    return np.asarray(arr, dtype=np.float32)


# ---------------------------------------------------------------------------
# test_umap_basic
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: umap_embed requires cuML; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_umap_basic(gsm4037629_path):
    """umap() completes without error and writes adata.obsm['X_umap'].

    Procedure:
      1. Load GSM4037629, preprocess + PCA + neighbors.
      2. Call singlet_gpu.tools.umap(adata, init_pos='random', rng=_SEED).
      3. Assert returns None (inplace convention).
      4. Assert adata.obsm['X_umap'] exists.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)

    ret = singlet_gpu.tools.umap(adata, init_pos="random", rng=_SEED)

    assert ret is None, (
        f"umap() inplace must return None (scanpy convention), got {type(ret)}"
    )
    assert "X_umap" in adata.obsm, (
        "umap(): adata.obsm['X_umap'] not written"
    )


# ---------------------------------------------------------------------------
# test_umap_writes_obsm
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: umap_embed requires cuML; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_umap_writes_obsm(gsm4037629_path):
    """umap() writes obsm['X_umap'] with shape (n_cells, 2), finite values.

    Verifies:
      - X_umap is 2-D with shape (n_obs, n_components=2).
      - All values are finite (no NaN/Inf).
      - Values span a non-degenerate range (std > 0 per dimension).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)
    singlet_gpu.tools.umap(adata, n_components=2, init_pos="random", rng=_SEED)

    assert "X_umap" in adata.obsm, "X_umap not written"

    X_umap = _to_numpy(adata.obsm["X_umap"])

    assert X_umap.ndim == 2, f"X_umap must be 2-D, got {X_umap.ndim}-D"
    assert X_umap.shape == (adata.n_obs, 2), (
        f"X_umap shape {X_umap.shape} != ({adata.n_obs}, 2)"
    )
    assert np.all(np.isfinite(X_umap)), "X_umap contains NaN or Inf"

    # Each UMAP dimension must have non-zero variance.
    for dim in range(2):
        std = float(np.std(X_umap[:, dim]))
        assert std > 0.01, (
            f"X_umap dimension {dim} has near-zero std={std:.4f} — degenerate embedding"
        )


# ---------------------------------------------------------------------------
# test_umap_n_components
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: umap_embed requires cuML; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_umap_n_components(gsm4037629_path):
    """umap() honors the n_components parameter (2 and 3 tested).

    Procedure:
      1. Run umap with n_components=2 → X_umap shape (n_obs, 2).
      2. Run umap with n_components=3 → X_umap shape (n_obs, 3).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    for n_comp in [2, 3]:
        adata = _load_gpu_adata(gsm4037629_path)
        _full_pipeline_gpu(adata)
        singlet_gpu.tools.umap(adata, n_components=n_comp, init_pos="random", rng=_SEED)

        X_umap = _to_numpy(adata.obsm["X_umap"])
        assert X_umap.shape == (adata.n_obs, n_comp), (
            f"n_components={n_comp}: X_umap shape {X_umap.shape} != "
            f"({adata.n_obs}, {n_comp})"
        )
        assert np.all(np.isfinite(X_umap)), (
            f"n_components={n_comp}: X_umap contains NaN or Inf"
        )


# ---------------------------------------------------------------------------
# test_umap_vs_scanpy_trustworthiness
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True, raises=RuntimeError,
    reason="INFRA-CUVS-CUGRAPH-INSTALL: umap_embed requires cuML; not installed on GPU nodes (state/blockers.md)",
)
@requires_gpu
def test_umap_vs_scanpy_trustworthiness(gsm4037629_path):
    """GPU UMAP embedding trustworthiness ≥ 0.85 on the PCA high-dim space.

    Trustworthiness measures how well local neighbourhood structure in the
    high-dimensional (PCA) space is preserved in the low-dimensional UMAP
    embedding.  We compute it for the GPU UMAP independently (not comparing
    directly against scanpy UMAP coordinates, which would be coordinate-
    dependent and rotation-sensitive).

    Procedure:
      1. Load GSM4037629, preprocess + PCA + neighbors.
      2. singlet_gpu.tools.umap(adata) → X_umap.
      3. Compute trustworthiness(X_pca[:, :_TRUST_K], X_umap, k=_TRUST_K).
      4. Assert trustworthiness ≥ 0.85.

    Note: PCA representation (X_pca) is used as the high-dim reference, not
    the raw count matrix, since UMAP operates on the kNN graph derived from PCA.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _full_pipeline_gpu(adata)
    singlet_gpu.tools.umap(adata, n_components=2, init_pos="random", rng=_SEED)

    X_pca = _to_numpy(adata.obsm["X_pca"])
    X_umap = _to_numpy(adata.obsm["X_umap"])

    # Subsample for tractability (trustworthiness is O(n²) in naïve impl).
    n_sample = min(adata.n_obs, 3_000)
    rng = np.random.default_rng(0xC0FFEE)
    idx = rng.choice(adata.n_obs, size=n_sample, replace=False)

    X_pca_sub = X_pca[idx]
    X_umap_sub = X_umap[idx]

    t = _trustworthiness(X_pca_sub, X_umap_sub, k=_TRUST_K)
    assert t >= _TRUST_MIN, (
        f"UMAP trustworthiness {t:.4f} < {_TRUST_MIN} "
        f"(computed on {n_sample} subsampled cells, k={_TRUST_K})"
    )
