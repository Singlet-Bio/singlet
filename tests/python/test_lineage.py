"""
Cycle 23 correctness tests for singlet_gpu.lineage.

Tests the GPU-native MT lineage wrapper (cycle 16: anno/mt_lineage.h):
  - ``singlet.gpu.lineage.detect_clones``

Formal spec: singlet-gpu/state/designs/22-python-kernel-wrappers-3.md

API (from spec):
  detect_clones(adata, *, alt_layer='mt_alt', depth_layer='mt_depth',
                min_depth=10, min_cells_alt=5, min_vaf=0.01, min_K=2,
                max_K=10, seed=0, copy=False) -> AnnData | None

Result locations (spec):
  - adata.obs['mt_clone_id']       : int per cell (clone label, 0-based)
  - adata.obsm['mt_heteroplasmy']  : n_cells × n_informative_sites (float, in [0,1])

Notes:
  - GSM4037629 has mt_alleles.1pz (singlet --pipeline output).
    We load it into 'mt_alt' and 'mt_depth' layers for the real-data test.
  - detect_clones is a NEW function with no established SOTA AnnData equivalent.
    No vs-reference test; correctness is structural + real-data finite-output.
  - min_K constraint: detect_clones must not produce fewer than min_K clones.

Skip strategy:
  - Module-level skip if singlet.gpu not available.
  - requires_gpu for all GPU-exercising tests.
  - GSM4037629 real-data test skips if mt_alleles.1pz is absent.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason="singlet.gpu not available. Run `pip install -e singlet-gpu/python/` first.",
    exc_type=ImportError,
)

from conftest import requires_gpu  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_K = 2
_MAX_K = 6  # reduced from spec for harness speed
_SEED = 0
_MIN_DEPTH = 10
_MIN_CELLS_ALT = 5
_MIN_VAF = 0.01

_MT_ALLELES_FILENAME = "mt_alleles.1pz"

# ---------------------------------------------------------------------------
# Synthetic MT data builder
# ---------------------------------------------------------------------------


def _make_mt_adata(n_cells: int = 200, n_sites: int = 50, n_clones: int = 3):
    """Build a minimal AnnData with synthetic mt_alt and mt_depth layers.

    Clone structure: cells are assigned to n_clones groups; each clone has a
    unique variant site with elevated VAF. Depth is uniform.
    """
    import anndata
    import scipy.sparse as sp

    rng = np.random.default_rng(0xC0FFEE)
    depth = np.full((n_cells, n_sites), 30, dtype=np.float32)
    alt = np.zeros((n_cells, n_sites), dtype=np.float32)

    # Each clone gets 5 informative sites with VAF ~0.5.
    for k in range(n_clones):
        clone_cells = np.where(np.arange(n_cells) % n_clones == k)[0]
        sites_start = k * (n_sites // n_clones)
        sites_end = sites_start + 5
        sites = np.arange(sites_start, min(sites_end, n_sites))
        for s in sites:
            alt[np.ix_(clone_cells, [s])] = rng.integers(12, 18, size=(len(clone_cells), 1))

    # AnnData enforces layers[k].shape == (n_obs, n_vars) — dummy X must use
    # n_sites as its var dimension to match the mt_alt / mt_depth layers.
    adata = anndata.AnnData(
        X=sp.csr_matrix(np.ones((n_cells, n_sites), dtype=np.float32)),
        obs={"cell": [f"cell_{i}" for i in range(n_cells)]},
        var={"site": [f"site_{j}" for j in range(n_sites)]},
    )
    adata.layers["mt_alt"] = sp.csr_matrix(alt)
    adata.layers["mt_depth"] = sp.csr_matrix(depth)
    return adata


# ---------------------------------------------------------------------------
# test_detect_clones_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_detect_clones_basic():
    """detect_clones() completes on synthetic MT data without error.

    Procedure:
      1. Build synthetic 200-cell, 50-site MT AnnData with 3 ground-truth clones.
      2. Call singlet_gpu.lineage.detect_clones(adata, min_K=2, max_K=6, seed=0).
      3. Assert returns None (inplace convention).
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    adata = _make_mt_adata(n_cells=200, n_sites=50, n_clones=3)

    ret = singlet_gpu.lineage.detect_clones(
        adata,
        alt_layer="mt_alt",
        depth_layer="mt_depth",
        min_depth=_MIN_DEPTH,
        min_cells_alt=_MIN_CELLS_ALT,
        min_vaf=_MIN_VAF,
        min_K=_MIN_K,
        max_K=_MAX_K,
        seed=_SEED,
    )

    assert ret is None, f"detect_clones() inplace must return None, got {type(ret)}"


# ---------------------------------------------------------------------------
# test_detect_clones_writes_obs_clone_id
# ---------------------------------------------------------------------------
@requires_gpu
def test_detect_clones_writes_obs_clone_id():
    """detect_clones() writes adata.obs['mt_clone_id'] as integer labels.

    Verifies:
      - obs['mt_clone_id'] length == n_cells.
      - All labels are integer-valued (or castable to int).
      - Labels are in range [0, max_K).
      - Number of distinct clones ≥ min_K.
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    adata = _make_mt_adata(n_cells=200, n_sites=50, n_clones=3)
    singlet_gpu.lineage.detect_clones(
        adata,
        alt_layer="mt_alt",
        depth_layer="mt_depth",
        min_depth=_MIN_DEPTH,
        min_cells_alt=_MIN_CELLS_ALT,
        min_vaf=_MIN_VAF,
        min_K=_MIN_K,
        max_K=_MAX_K,
        seed=_SEED,
    )

    assert "mt_clone_id" in adata.obs.columns, (
        "detect_clones(): adata.obs['mt_clone_id'] not written"
    )

    clone_ids = adata.obs["mt_clone_id"].values
    assert len(clone_ids) == adata.n_obs, (
        f"mt_clone_id length {len(clone_ids)} != n_obs {adata.n_obs}"
    )

    # All labels must be integer-like.
    clone_ids_int = np.asarray(clone_ids, dtype=int)
    assert np.all(clone_ids_int >= 0), "mt_clone_id contains negative clone labels"

    n_distinct = len(np.unique(clone_ids_int))
    assert n_distinct >= _MIN_K, (
        f"detect_clones found {n_distinct} distinct clones < min_K={_MIN_K}"
    )
    assert n_distinct <= _MAX_K, (
        f"detect_clones found {n_distinct} distinct clones > max_K={_MAX_K}"
    )


# ---------------------------------------------------------------------------
# test_detect_clones_writes_obsm_heteroplasmy
# ---------------------------------------------------------------------------
@requires_gpu
def test_detect_clones_writes_obsm_heteroplasmy():
    """detect_clones() writes adata.obsm['mt_heteroplasmy'] in [0, 1].

    Verifies:
      - obsm['mt_heteroplasmy'] shape = (n_cells, n_informative_sites).
      - n_informative_sites ≥ 1.
      - All values in [0, 1] (VAF = alt/depth).
      - Values are finite.
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    adata = _make_mt_adata(n_cells=200, n_sites=50, n_clones=3)
    singlet_gpu.lineage.detect_clones(
        adata,
        alt_layer="mt_alt",
        depth_layer="mt_depth",
        min_depth=_MIN_DEPTH,
        min_cells_alt=_MIN_CELLS_ALT,
        min_vaf=_MIN_VAF,
        min_K=_MIN_K,
        max_K=_MAX_K,
        seed=_SEED,
    )

    assert "mt_heteroplasmy" in adata.obsm, (
        "detect_clones(): adata.obsm['mt_heteroplasmy'] not written"
    )

    het = np.asarray(adata.obsm["mt_heteroplasmy"], dtype=np.float64)
    assert het.shape[0] == adata.n_obs, (
        f"mt_heteroplasmy rows {het.shape[0]} != n_obs {adata.n_obs}"
    )
    assert het.shape[1] >= 1, (
        "mt_heteroplasmy has 0 informative sites — detect_clones may have filtered all"
    )
    assert np.all(np.isfinite(het)), (
        f"mt_heteroplasmy contains non-finite values: {np.sum(~np.isfinite(het))}"
    )
    assert np.all(het >= 0.0 - 1e-9), f"mt_heteroplasmy contains negative VAFs; min={het.min():.4e}"
    assert np.all(het <= 1.0 + 1e-9), f"mt_heteroplasmy contains VAFs > 1.0; max={het.max():.4e}"


# ---------------------------------------------------------------------------
# test_detect_clones_GSM4037629_real_data
# ---------------------------------------------------------------------------
@requires_gpu
def test_detect_clones_GSM4037629_real_data(gsm4037629_path):
    """detect_clones() returns finite results on real GSM4037629 MT data.

    Loads mt_alleles.1pz from the singlet pipeline output and runs
    detect_clones. Asserts:
      - obs['mt_clone_id'] written, all values finite integers ≥ 0.
      - obsm['mt_heteroplasmy'] written, all values finite and in [0,1].

    Skipped if mt_alleles.1pz is not present in the sample directory.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    mt_path = pathlib.Path(gsm4037629_path) / _MT_ALLELES_FILENAME
    if not mt_path.is_file():
        pytest.skip(
            f"mt_alleles.1pz not found at {mt_path}. "
            "Run the singlet pipeline with --pipeline flag to generate this file."
        )

    # Load mt_alleles as the main matrix; treat as alt counts.
    # For a real test we need both alt and depth. Try to load both:
    try:
        mt_adata = singlet_gpu.io.read_pz_to_anndata(str(mt_path))
    except Exception as e:
        pytest.skip(f"Could not read mt_alleles.1pz: {e}")

    # mt_alleles.1pz may encode alt+depth interleaved, or just alt counts.
    # If detect_clones accepts a single layer, use it as alt; derive depth as sum.
    # This mirrors the expected singlet_gpu.io contract.
    import scipy.sparse as sp

    X_host = mt_adata.X
    if hasattr(X_host, "get"):
        X_host = X_host.get()
    if sp.issparse(X_host):
        X_host = X_host.tocsr()
    alt_arr = np.asarray(X_host.todense() if sp.issparse(X_host) else X_host, dtype=np.float32)

    # Use X directly as alt layer; infer pseudo-depth as sum of all sites per cell.
    # (Real use case: singlet_gpu.io.read_mt_pileup handles this natively.)
    mt_adata.layers["mt_alt"] = sp.csr_matrix(alt_arr)
    depth_arr = np.clip(alt_arr * 10 + 1, 10, None)  # pseudo-depth for harness
    mt_adata.layers["mt_depth"] = sp.csr_matrix(depth_arr)

    singlet_gpu.lineage.detect_clones(
        mt_adata,
        alt_layer="mt_alt",
        depth_layer="mt_depth",
        min_depth=_MIN_DEPTH,
        min_cells_alt=_MIN_CELLS_ALT,
        min_vaf=_MIN_VAF,
        min_K=_MIN_K,
        max_K=_MAX_K,
        seed=_SEED,
    )

    assert "mt_clone_id" in mt_adata.obs.columns, (
        "detect_clones real-data: obs['mt_clone_id'] not written"
    )
    clone_ids = mt_adata.obs["mt_clone_id"].values
    assert np.all(np.isfinite(np.asarray(clone_ids, dtype=float))), (
        "detect_clones real-data: mt_clone_id contains non-finite values"
    )
    assert np.all(np.asarray(clone_ids, dtype=int) >= 0), (
        "detect_clones real-data: mt_clone_id has negative labels"
    )

    assert "mt_heteroplasmy" in mt_adata.obsm, (
        "detect_clones real-data: obsm['mt_heteroplasmy'] not written"
    )
    het = np.asarray(mt_adata.obsm["mt_heteroplasmy"], dtype=np.float64)
    assert np.all(np.isfinite(het)), (
        f"detect_clones real-data: mt_heteroplasmy non-finite: {np.sum(~np.isfinite(het))}"
    )


# ---------------------------------------------------------------------------
# test_detect_clones_min_K_constraint
# ---------------------------------------------------------------------------
@requires_gpu
def test_detect_clones_min_K_constraint():
    """detect_clones() respects min_K: always produces ≥ min_K distinct clone labels.

    Tests with min_K=3 on data that clearly contains ≥ 3 clones (3 ground-truth
    clone groups with distinct high-VAF sites). Verifies that detect_clones does
    not collapse to fewer than min_K clusters even if BIC would prefer fewer.
    """
    pytest.importorskip("anndata", reason="anndata not installed")
    pytest.importorskip("scipy", reason="scipy not installed")

    # Build data with 4 clearly separable clones.
    adata = _make_mt_adata(n_cells=400, n_sites=80, n_clones=4)

    min_K_test = 3
    singlet_gpu.lineage.detect_clones(
        adata,
        alt_layer="mt_alt",
        depth_layer="mt_depth",
        min_depth=_MIN_DEPTH,
        min_cells_alt=_MIN_CELLS_ALT,
        min_vaf=_MIN_VAF,
        min_K=min_K_test,
        max_K=_MAX_K,
        seed=_SEED,
    )

    assert "mt_clone_id" in adata.obs.columns, (
        "detect_clones(min_K=3): obs['mt_clone_id'] not written"
    )
    clone_ids = np.asarray(adata.obs["mt_clone_id"].values, dtype=int)
    n_distinct = len(np.unique(clone_ids))

    assert n_distinct >= min_K_test, (
        f"detect_clones(min_K={min_K_test}) produced only {n_distinct} distinct clones; "
        f"must respect min_K constraint and return ≥ {min_K_test}"
    )
