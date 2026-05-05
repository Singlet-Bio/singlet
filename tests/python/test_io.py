"""
Cycle 19 correctness tests for singlet_gpu.io (loader.py).

Tests the high-level AnnData I/O wrappers introduced in cycle 19:
  - ``singlet.gpu.io.read_pz_to_anndata``
  - ``singlet.gpu.io.write_anndata_to_pz``

Formal spec: singlet-gpu/state/designs/19-python-kernel-wrappers-1.md

Skip strategy:
  - If ``singlet.gpu`` cannot be imported (wheel not built), the entire
    module is skipped via a module-level collect hook.
  - GPU-touching tests are individually skipped via ``requires_gpu``
    when cupy/CUDA is unavailable.

All tests use the ``gsm4037629_path`` session fixture from conftest.py.
"""
from __future__ import annotations

import pathlib
import tempfile

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level skip when the wheel hasn't been built yet.
# ---------------------------------------------------------------------------
singlet_gpu = pytest.importorskip(
    "singlet.gpu",
    reason=(
        "singlet.gpu not available. "
        "Run `pip install -e singlet-gpu/python/` first."
    ),
    exc_type=ImportError,
)

from conftest import requires_gpu  # noqa: E402 — after importorskip

# ---------------------------------------------------------------------------
# Constants for GSM4037629
# ---------------------------------------------------------------------------
_EXPECTED_GSM_ID = "GSM4037629"
_EXPECTED_GSE_ID = "GSE127918"


# ---------------------------------------------------------------------------
# test_read_pz_to_anndata_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_read_pz_to_anndata_basic(gsm4037629_path):
    """read_pz_to_anndata() returns a valid AnnData with correct shape.

    AnnData convention: .X.shape == (n_obs, n_vars) == (cells, genes).
    The on-disk .1pz CSC is stored as (genes, cells), so the loader must
    transpose.

    Checks:
      - Return type is AnnData.
      - adata.X.shape[0] (cells) < adata.X.shape[1] (genes) for this sample.
      - n_obs and n_vars are both positive.
      - adata.obs_names and adata.var_names are non-empty string arrays.
    """
    anndata = pytest.importorskip("anndata", reason="anndata not installed")

    adata = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))

    assert isinstance(adata, anndata.AnnData), (
        f"read_pz_to_anndata must return AnnData, got {type(adata)}"
    )

    n_cells, n_genes = adata.X.shape
    assert n_cells > 0, f"n_obs (cells) must be positive, got {n_cells}"
    assert n_genes > 0, f"n_vars (genes) must be positive, got {n_genes}"

    # For GSM4037629: ~11560 cells vs ~33538 genes — cells is always smaller.
    assert n_cells < n_genes, (
        f"Expected cells < genes (transposed from CSC), got shape {adata.X.shape}"
    )

    # obs_names (cell barcodes) and var_names (gene IDs) must be populated.
    assert len(adata.obs_names) == n_cells, (
        f"obs_names length {len(adata.obs_names)} != n_cells {n_cells}"
    )
    assert len(adata.var_names) == n_genes, (
        f"var_names length {len(adata.var_names)} != n_genes {n_genes}"
    )
    assert len(adata.obs_names[0]) > 0, "obs_names must contain non-empty barcodes"
    assert len(adata.var_names[0]) > 0, "var_names must contain non-empty gene IDs"


# ---------------------------------------------------------------------------
# test_anndata_metadata_in_uns
# ---------------------------------------------------------------------------
@requires_gpu
def test_anndata_metadata_in_uns(gsm4037629_path):
    """GEO metadata embedded in the .1pz TLV block must appear in adata.uns['singlify'].

    The design doc specifies that read_pz_to_anndata embeds all user_kv fields
    from the .1pz TLV into adata.uns['singlify'] as a flat dict.

    Required keys: gsm_id, gse_id.
    Non-empty string checks: protocol, organism.
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    adata = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))

    assert "singlify" in adata.uns, (
        "adata.uns must contain 'singlify' key with embedded GEO metadata"
    )
    meta = adata.uns["singlify"]

    assert meta.get("gsm_id") == _EXPECTED_GSM_ID, (
        f"uns['singlify']['gsm_id'] = {meta.get('gsm_id')!r}, "
        f"expected {_EXPECTED_GSM_ID!r}"
    )
    assert meta.get("gse_id") == _EXPECTED_GSE_ID, (
        f"uns['singlify']['gse_id'] = {meta.get('gse_id')!r}, "
        f"expected {_EXPECTED_GSE_ID!r}"
    )

    protocol = meta.get("protocol", "")
    assert isinstance(protocol, str) and len(protocol) > 0, (
        f"uns['singlify']['protocol'] must be a non-empty string, got {protocol!r}"
    )
    organism = meta.get("organism", "")
    assert isinstance(organism, str) and len(organism) > 0, (
        f"uns['singlify']['organism'] must be a non-empty string, got {organism!r}"
    )


# ---------------------------------------------------------------------------
# test_modality_param
# ---------------------------------------------------------------------------
@requires_gpu
def test_modality_param(gsm4037629_path):
    """The modality= parameter selects which .1pz file to load from the directory.

    Default ``modality='exon'`` loads ``exon_counts.1pz``.
    Explicitly passing ``modality='exon'`` must produce identical shape.

    If ``intron_counts.1pz`` exists in the directory, loading
    ``modality='intron'`` must also produce a valid AnnData with the same
    number of cells (same barcodes) but may differ in nnz.

    If the alternative modality file does not exist, the test skips gracefully
    (the loader must raise FileNotFoundError, which we catch and skip).
    """
    pytest.importorskip("anndata", reason="anndata not installed")

    # Default (exon) vs explicit exon — must be identical shape.
    adata_default = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))
    adata_explicit = singlet_gpu.io.read_pz_to_anndata(
        str(gsm4037629_path), modality="exon"
    )
    assert adata_default.X.shape == adata_explicit.shape, (
        "Default modality and explicit modality='exon' must produce identical shape"
    )

    # Intron modality — same cell count if file exists.
    intron_file = gsm4037629_path / "intron_counts.1pz"
    if not intron_file.is_file():
        pytest.skip("intron_counts.1pz not present in GSM4037629; skipping intron modality check")

    adata_intron = singlet_gpu.io.read_pz_to_anndata(
        str(gsm4037629_path), modality="intron"
    )
    # Cell barcodes (obs axis) must be the same set for both exon and intron.
    assert adata_intron.n_obs == adata_default.n_obs, (
        f"intron AnnData has {adata_intron.n_obs} cells, "
        f"expected {adata_default.n_obs} (same barcodes as exon)"
    )


# ---------------------------------------------------------------------------
# test_write_pz_roundtrip
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="write_anndata_to_pz raises NotImplementedError pending "
           "_core.from_cupy_csr_to_pz binding (CYCLE-19-FOLLOWUP-CYCLE-18-"
           "BINDING-EXPOSE)",
    strict=True,
    raises=NotImplementedError,
)
@requires_gpu
def test_write_pz_roundtrip(gsm4037629_path):
    """write_anndata_to_pz → read_pz_to_anndata round-trips data losslessly.

    Procedure:
      1. Load GSM4037629 into adata_orig via read_pz_to_anndata.
      2. Write adata_orig to a temporary directory via write_anndata_to_pz.
      3. Reload from the temp directory via read_pz_to_anndata → adata_rt.
      4. Compare shapes and non-zero values element-wise (no fp tolerance —
         the data path is integer count data, no arithmetic applied).

    The comparison uses scipy.sparse on the host copy because the test must
    not assume the runtime GPU is present (though in practice it will be —
    we still go through the GPU path since requires_gpu gates this test).
    """
    import scipy.sparse as sp

    pytest.importorskip("anndata", reason="anndata not installed")

    adata_orig = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))

    with tempfile.TemporaryDirectory() as tmpdir:
        singlet_gpu.io.write_anndata_to_pz(adata_orig, tmpdir, modality="exon")

        # Verify the output file was created.
        out_file = pathlib.Path(tmpdir) / "exon_counts.1pz"
        assert out_file.is_file(), (
            f"write_anndata_to_pz did not produce {out_file}"
        )

        adata_rt = singlet_gpu.io.read_pz_to_anndata(tmpdir, modality="exon")

    # Shape must be identical.
    assert adata_rt.X.shape == adata_orig.X.shape, (
        f"Round-trip AnnData shape {adata_rt.X.shape} != "
        f"original {adata_orig.X.shape}"
    )

    # Values: bring both to host scipy CSR and compare element-wise.
    # Use .get() if X is a cupy sparse, or .toarray() if scipy.
    def _to_scipy(x):
        if hasattr(x, "get"):
            return x.get()
        return x

    X_orig = _to_scipy(adata_orig.X)
    X_rt = _to_scipy(adata_rt.X)

    # Compare non-zeros via sorted COO for robustness (column ordering may differ).
    X_orig_coo = sp.coo_matrix(X_orig)
    X_rt_coo = sp.coo_matrix(X_rt)

    assert X_orig_coo.nnz == X_rt_coo.nnz, (
        f"Round-trip nnz {X_rt_coo.nnz} != original nnz {X_orig_coo.nnz}"
    )

    # Sort by (row, col) to make comparison order-independent.
    orig_idx = np.lexsort((X_orig_coo.col, X_orig_coo.row))
    rt_idx = np.lexsort((X_rt_coo.col, X_rt_coo.row))

    assert np.array_equal(X_orig_coo.row[orig_idx], X_rt_coo.row[rt_idx]), (
        "Round-trip: row indices differ"
    )
    assert np.array_equal(X_orig_coo.col[orig_idx], X_rt_coo.col[rt_idx]), (
        "Round-trip: col indices differ"
    )
    assert np.array_equal(
        X_orig_coo.data[orig_idx].astype(np.float32),
        X_rt_coo.data[rt_idx].astype(np.float32),
    ), "Round-trip: count values differ (should be bit-identical integer data)"

    # GEO metadata must survive the round-trip.
    assert adata_rt.uns.get("singlify", {}).get("gsm_id") == _EXPECTED_GSM_ID, (
        "uns['singlify']['gsm_id'] must survive round-trip"
    )
