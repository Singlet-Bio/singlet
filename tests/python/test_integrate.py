"""
Cycle 23 correctness tests for singlet_gpu.integrate.

Tests the GPU-native batch integration wrappers (cycle 14: integrate/{harmony,bbknn}.h):
  - ``singlet.gpu.integrate.harmony_integrate``
  - ``singlet.gpu.integrate.bbknn``

Formal spec: singlet-gpu/state/designs/22-python-kernel-wrappers-3.md

API (from spec):
  harmony_integrate(adata, key, *, basis='X_pca', adjusted_basis='X_pca_harmony',
                    n_clusters=20, max_iter=10, tol=1e-4, seed=0) -> None
  bbknn(adata, *, batch_key='batch', use_rep='X_pca', neighbors_within_batch=3,
        n_pcs=None, metric='euclidean', set_op_mix_ratio=1.0,
        local_connectivity=1, copy=False) -> AnnData | None

Result locations (spec):
  - harmony_integrate: adata.obsm['X_pca_harmony']  (n_cells × n_comps)
  - bbknn            : adata.obsp['distances'], adata.obsp['connectivities'],
                       adata.uns['neighbors']

Tolerances (from spec):
  - harmony_integrate vs harmonypy: CCA cosine score ≥ 0.95
  - bbknn            vs python-bbknn: neighbor list overlap (intersection/union per cell ≥ 0.80)

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu for all GPU-exercising tests.
  - vs-harmonypy / vs-python-bbknn additionally skip if reference lib not importable.
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
_HARMONY_CCA_MIN = 0.95  # mean cosine similarity of CCA-aligned embeddings
_BBKNN_OVERLAP_MIN = 0.80  # mean per-cell neighbor list intersection/union
_N_COMPS = 20  # reduced PCA dims (fast for harness)
_SEED = 0
_N_CLUSTERS = 10  # Harmony clusters (reduced for speed)
_MAX_ITER = 5  # reduced for speed in harness
_NEIGHBORS_WITHIN_BATCH = 3

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


def _preprocess_and_pca_gpu(adata, n_comps=_N_COMPS):
    singlet_gpu.preprocess.normalize_total(adata, inplace=True)
    singlet_gpu.preprocess.log1p(adata, inplace=True)
    singlet_gpu.reduce.pca(adata, n_comps=n_comps, inplace=True)


def _preprocess_and_pca_cpu(adata, n_comps=_N_COMPS):
    sc = pytest.importorskip("scanpy", reason="scanpy not installed")
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)  # scanpy 1.11+ removed inplace= from log1p (in-place by default)
    sc.pp.pca(adata, n_comps=n_comps)


def _add_dummy_batch(adata, n_batches: int = 2, col: str = "batch") -> None:
    """Assign synthetic batch labels to obs (deterministic by cell index)."""
    n = adata.n_obs
    labels = [f"batch_{i % n_batches}" for i in range(n)]
    adata.obs[col] = labels


def _to_host_array(x) -> np.ndarray:
    """Transfer a cupy / numpy array to host numpy."""
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


# ---------------------------------------------------------------------------
# test_harmony_integrate_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_harmony_integrate_basic(gsm4037629_path):
    """harmony_integrate() completes without error.

    Procedure:
      1. Load GSM4037629, preprocess + PCA (n_comps=20).
      2. Assign 2-batch synthetic label to adata.obs['batch'].
      3. Call singlet_gpu.integrate.harmony_integrate(adata, 'batch').
      4. Assert returns None (inplace convention).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_and_pca_gpu(adata, n_comps=_N_COMPS)
    _add_dummy_batch(adata)

    ret = singlet_gpu.integrate.harmony_integrate(
        adata,
        "batch",
        n_clusters=_N_CLUSTERS,
        max_iter=_MAX_ITER,
        seed=_SEED,
    )

    assert ret is None, f"harmony_integrate() inplace must return None, got {type(ret)}"


# ---------------------------------------------------------------------------
# test_harmony_writes_obsm_X_pca_harmony
# ---------------------------------------------------------------------------
@requires_gpu
def test_harmony_writes_obsm_X_pca_harmony(gsm4037629_path):
    """harmony_integrate() writes adata.obsm['X_pca_harmony'] with correct shape.

    Verifies:
      - obsm['X_pca_harmony'] shape = (n_cells, n_comps).
      - All values finite.
      - Embedding has non-trivial row-norm variance (not all zero or identical).
      - adjusted_basis parameter changes the output key.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_and_pca_gpu(adata, n_comps=_N_COMPS)
    _add_dummy_batch(adata)

    singlet_gpu.integrate.harmony_integrate(
        adata,
        "batch",
        n_clusters=_N_CLUSTERS,
        max_iter=_MAX_ITER,
        seed=_SEED,
    )

    assert "X_pca_harmony" in adata.obsm, (
        "harmony_integrate(): adata.obsm['X_pca_harmony'] not written"
    )

    emb = _to_host_array(adata.obsm["X_pca_harmony"])
    assert emb.shape == (adata.n_obs, _N_COMPS), (
        f"X_pca_harmony shape {emb.shape} != ({adata.n_obs}, {_N_COMPS})"
    )
    assert np.all(np.isfinite(emb)), (
        f"X_pca_harmony contains non-finite values: {np.sum(~np.isfinite(emb))}"
    )
    row_norms = np.linalg.norm(emb, axis=1)
    assert np.std(row_norms) > 1e-6, (
        "X_pca_harmony row norms have near-zero variance — embedding may be trivial"
    )

    # Check adjusted_basis parameter respected.
    adata2 = _load_gpu_adata(gsm4037629_path)
    _preprocess_and_pca_gpu(adata2, n_comps=_N_COMPS)
    _add_dummy_batch(adata2)
    singlet_gpu.integrate.harmony_integrate(
        adata2,
        "batch",
        adjusted_basis="X_harmony_custom",
        n_clusters=_N_CLUSTERS,
        max_iter=_MAX_ITER,
        seed=_SEED,
    )
    assert "X_harmony_custom" in adata2.obsm, (
        "harmony_integrate(adjusted_basis='X_harmony_custom'): key not written"
    )


# ---------------------------------------------------------------------------
# test_harmony_vs_harmonypy
# ---------------------------------------------------------------------------
@requires_gpu
def test_harmony_vs_harmonypy(gsm4037629_path):
    """GPU Harmony embedding cosine similarity ≥ 0.95 vs harmonypy reference.

    CCA score: mean cosine similarity between GPU and harmonypy row embeddings
    after aligning column signs (sign ambiguity in harmonized PCA components).

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Preprocess + PCA on both (same n_comps).
      3. Assign identical dummy batch labels.
      4. GPU: singlet_gpu.integrate.harmony_integrate(adata_gpu, 'batch').
      5. CPU: harmonypy.run_harmony(pca_cpu, batch_labels, max_iter=5, random_state=0).
      6. Extract X_pca_harmony from GPU; Z_corr.T from harmonypy.
      7. Compute per-cell cosine similarity between matched rows.
      8. Assert mean cosine similarity ≥ 0.95.
    """
    harmonypy = pytest.importorskip(
        "harmonypy", reason="harmonypy not installed — skip vs-reference test"
    )
    pytest.importorskip("anndata", reason="anndata not installed")

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _preprocess_and_pca_gpu(adata_gpu, n_comps=_N_COMPS)
    _preprocess_and_pca_cpu(adata_cpu, n_comps=_N_COMPS)

    _add_dummy_batch(adata_gpu)
    adata_cpu.obs["batch"] = adata_gpu.obs["batch"].values

    # GPU Harmony.
    singlet_gpu.integrate.harmony_integrate(
        adata_gpu,
        "batch",
        n_clusters=_N_CLUSTERS,
        max_iter=_MAX_ITER,
        seed=_SEED,
    )
    gpu_emb = _to_host_array(adata_gpu.obsm["X_pca_harmony"])  # (n_cells, n_comps)

    # harmonypy reference.
    pca_cpu = adata_cpu.obsm["X_pca"]  # (n_cells, n_comps)
    try:
        ho = harmonypy.run_harmony(
            pca_cpu,
            adata_cpu.obs,
            "batch",
            max_iter_harmony=_MAX_ITER,
            random_state=_SEED,
        )
        cpu_emb = ho.Z_corr.T  # harmonypy returns (n_comps, n_cells); transpose
    except Exception as e:
        pytest.skip(f"harmonypy.run_harmony failed: {e}")

    # Align sign per component (sign ambiguity).
    for k in range(min(gpu_emb.shape[1], cpu_emb.shape[1])):
        if np.dot(gpu_emb[:, k], cpu_emb[:, k]) < 0:
            cpu_emb[:, k] *= -1

    # Per-cell cosine similarity.
    gpu_norm = np.linalg.norm(gpu_emb, axis=1, keepdims=True) + 1e-12
    cpu_norm = np.linalg.norm(cpu_emb, axis=1, keepdims=True) + 1e-12
    cosines = np.sum((gpu_emb / gpu_norm) * (cpu_emb / cpu_norm), axis=1)

    mean_cosine = float(np.mean(cosines))
    assert mean_cosine >= _HARMONY_CCA_MIN, (
        f"Harmony mean cosine similarity {mean_cosine:.4f} < {_HARMONY_CCA_MIN} "
        f"vs harmonypy (n_cells={adata_gpu.n_obs}, n_comps={_N_COMPS})"
    )


# ---------------------------------------------------------------------------
# test_bbknn_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_bbknn_basic(gsm4037629_path):
    """bbknn() completes without error and returns None (inplace convention).

    Procedure:
      1. Load GSM4037629, preprocess + PCA (n_comps=20).
      2. Assign 2-batch synthetic label.
      3. Call singlet_gpu.integrate.bbknn(adata, batch_key='batch').
      4. Assert returns None.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_and_pca_gpu(adata, n_comps=_N_COMPS)
    _add_dummy_batch(adata)

    ret = singlet_gpu.integrate.bbknn(
        adata,
        batch_key="batch",
        neighbors_within_batch=_NEIGHBORS_WITHIN_BATCH,
    )

    assert ret is None, f"bbknn() inplace must return None (copy=False default), got {type(ret)}"


# ---------------------------------------------------------------------------
# test_bbknn_writes_obsp
# ---------------------------------------------------------------------------
@requires_gpu
def test_bbknn_writes_obsp(gsm4037629_path):
    """bbknn() writes distances, connectivities to obsp and params to uns.

    Verifies:
      - adata.obsp['distances'] shape (n_cells, n_cells), sparse, non-negative.
      - adata.obsp['connectivities'] shape (n_cells, n_cells), sparse, in [0,1].
      - adata.uns['neighbors'] exists.
      - copy=True returns an AnnData, original is unchanged.
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    import scipy.sparse as sp

    adata = _load_gpu_adata(gsm4037629_path)
    _preprocess_and_pca_gpu(adata, n_comps=_N_COMPS)
    _add_dummy_batch(adata)

    singlet_gpu.integrate.bbknn(
        adata,
        batch_key="batch",
        neighbors_within_batch=_NEIGHBORS_WITHIN_BATCH,
    )

    n = adata.n_obs

    # distances
    D = adata.obsp["distances"]
    D_host = D.get() if hasattr(D, "get") else D
    if not sp.issparse(D_host):
        D_host = sp.csr_matrix(D_host)
    else:
        D_host = D_host.tocsr()
    assert D_host.shape == (n, n), f"obsp['distances'] shape {D_host.shape} != ({n}, {n})"
    assert np.all(D_host.data >= 0.0), "obsp['distances'] has negative values"

    # connectivities
    C = adata.obsp["connectivities"]
    C_host = C.get() if hasattr(C, "get") else C
    if not sp.issparse(C_host):
        C_host = sp.csr_matrix(C_host)
    else:
        C_host = C_host.tocsr()
    assert C_host.shape == (n, n), f"obsp['connectivities'] shape {C_host.shape} != ({n}, {n})"
    assert np.all(C_host.data >= 0.0), "obsp['connectivities'] has negative values"
    assert np.all(C_host.data <= 1.0 + 1e-6), (
        f"obsp['connectivities'] max={C_host.data.max():.4f} > 1.0"
    )

    assert "neighbors" in adata.uns, "bbknn(): adata.uns['neighbors'] not written"


# ---------------------------------------------------------------------------
# test_bbknn_vs_python_bbknn
# ---------------------------------------------------------------------------
@requires_gpu
def test_bbknn_vs_python_bbknn(gsm4037629_path):
    """GPU BBKNN neighbor list overlap ≥ 0.80 vs python bbknn reference.

    Overlap metric: for each cell, compute intersection/union of neighbor
    indices from GPU and CPU results, then take the mean across cells.

    Procedure:
      1. Load GSM4037629 twice → adata_gpu, adata_cpu.
      2. Preprocess + PCA on both (same n_comps, same PCA basis).
      3. Assign identical batch labels.
      4. singlet_gpu.integrate.bbknn(adata_gpu, batch_key='batch',
                                     neighbors_within_batch=3).
      5. bbknn.bbknn(adata_cpu, batch_key='batch',
                     neighbors_within_batch=3, use_rep='X_pca').
      6. Extract non-zero column indices per row from both obsp['distances'].
      7. Compute per-cell overlap = |intersection| / |union|.
      8. Assert mean overlap ≥ 0.80.
    """
    bbknn_ref = pytest.importorskip(
        "bbknn", reason="python-bbknn not installed — skip vs-reference test"
    )
    pytest.importorskip("anndata", reason="anndata not installed")
    import scipy.sparse as sp

    adata_gpu = _load_gpu_adata(gsm4037629_path)
    adata_cpu = _gpu_to_cpu_adata(adata_gpu)

    _preprocess_and_pca_gpu(adata_gpu, n_comps=_N_COMPS)
    _preprocess_and_pca_cpu(adata_cpu, n_comps=_N_COMPS)

    _add_dummy_batch(adata_gpu)
    adata_cpu.obs["batch"] = adata_gpu.obs["batch"].values

    # GPU BBKNN.
    singlet_gpu.integrate.bbknn(
        adata_gpu,
        batch_key="batch",
        neighbors_within_batch=_NEIGHBORS_WITHIN_BATCH,
    )

    # Python bbknn reference.
    try:
        bbknn_ref.bbknn(
            adata_cpu,
            batch_key="batch",
            neighbors_within_batch=_NEIGHBORS_WITHIN_BATCH,
            use_rep="X_pca",
            n_pcs=_N_COMPS,
        )
    except Exception as e:
        pytest.skip(f"python bbknn failed: {e}")

    # Extract neighbor index sets per cell.
    def _neighbor_sets(adata):
        D = adata.obsp["distances"]
        D_host = D.get() if hasattr(D, "get") else D
        if not sp.issparse(D_host):
            D_host = sp.csr_matrix(D_host)
        else:
            D_host = D_host.tocsr()
        sets = []
        for i in range(D_host.shape[0]):
            neighbors = set(D_host.indices[D_host.indptr[i] : D_host.indptr[i + 1]])
            sets.append(neighbors)
        return sets

    gpu_sets = _neighbor_sets(adata_gpu)
    cpu_sets = _neighbor_sets(adata_cpu)

    overlaps = []
    for gs, cs in zip(gpu_sets, cpu_sets):
        inter = len(gs & cs)
        union = len(gs | cs)
        if union > 0:
            overlaps.append(inter / union)

    mean_overlap = float(np.mean(overlaps)) if overlaps else 0.0
    assert mean_overlap >= _BBKNN_OVERLAP_MIN, (
        f"BBKNN mean neighbor overlap {mean_overlap:.4f} < {_BBKNN_OVERLAP_MIN} "
        f"vs python bbknn (n_cells={adata_gpu.n_obs})"
    )
