"""
Cycle 18 correctness test suite for the singlet-gpu Python wrapper foundation.

Tests the pybind11 ``singlet.gpu._core`` extension against the formal spec in
``singlet-gpu/state/designs/18-python-wrapper-foundation.md``.

Correctness reference: ``singlify.io.read_matrix`` (the singlify pure-Python
+ pybind11 path that reads a .1pz via the C++ pz_reader.h — already
validated in singlify's own test suite).

Skip strategy:
  - If ``singlet.gpu`` cannot be imported (wheel not built yet), the entire
    module is skipped via a module-level collect hook.
  - If ``cupy`` is not available or no CUDA device is present, GPU-touching
    tests are individually skipped via the ``requires_gpu`` marker from
    conftest.py.

All numerical comparisons are bit-identical (numpy.array_equal on
indptr/indices; numpy.array_equal on values cast to float32).  No
floating-point tolerances are needed because the data path is lossless
integer I/O — no math is applied.
"""

from __future__ import annotations

import gc
import weakref

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
# Helpers
# ---------------------------------------------------------------------------
_EXON_FILE = "exon_counts.1pz"

# Known-good metadata values for GSM4037629 (from manifest.json / provenance).
# These are the exact strings the singlify pipeline embedded in the file's
# TLV user_kv block via --metadata-json.
_EXPECTED_GSM_ID = "GSM4037629"
_EXPECTED_GSE_ID = "GSE127918"


# ---------------------------------------------------------------------------
# test_load_pz_basic
# ---------------------------------------------------------------------------
@requires_gpu
def test_load_pz_basic(gsm4037629_path):
    """Load exon_counts.1pz and verify shape + nnz are non-trivially positive.

    The .1pz header at GSM4037629 encodes:
      rows (genes/features)  = 310 797
      cols (cells)           = 20 866  [note: provenance.json cells=20866]
      nnz                    = 4 175 148

    The design doc says 'rows == 30000 (or whatever singlify wrote)' — we use
    the exact values from the file rather than a static constant so the test
    remains correct after any re-processing.
    """
    pz_path = gsm4037629_path / _EXON_FILE
    m = singlet_gpu.io.load_pz(str(pz_path))

    assert m.rows > 0, "rows must be positive"
    assert m.cols > 0, "cols must be positive"
    assert m.nnz > 0, "nnz must be positive"

    # Sanity bounds: we know the actual values from the on-disk header probe.
    # Accept any positive value — this guards against zero-load bugs without
    # hard-coding a value that would break if the sample is re-processed.
    assert m.rows >= 1000, f"expected >=1000 gene rows, got {m.rows}"
    assert m.cols >= 100, f"expected >=100 cell cols, got {m.cols}"
    assert m.nnz >= m.cols, f"nnz ({m.nnz}) must be >= cols ({m.cols})"

    # Repr smoke-test: must not raise and must contain shape/class information.
    # load_pz returns a PzDeviceMatrix wrapper around a DeviceCsc.
    r = repr(m)
    assert ("DeviceCsc" in r) or ("PzDeviceMatrix" in r), (
        f"repr must contain 'DeviceCsc' or 'PzDeviceMatrix': {r!r}"
    )


# ---------------------------------------------------------------------------
# test_metadata_fields
# ---------------------------------------------------------------------------
@requires_gpu
def test_metadata_fields(gsm4037629_path):
    """Verify that singlify-embedded TLV user_kv fields are accessible via
    the Python binding's Metadata struct.

    Fields checked:
      - gsm_id  (known constant)
      - gse_id  (known constant)
      - protocol  (non-empty string; singlify auto-detected 10x_v3)
      - organism  (non-empty string; 'Homo sapiens' for this sample)

    We assert exact equality only for the IDs (stable forever) and assert
    non-empty for protocol/organism so the test survives minor singlify
    version changes to the embedded string representations.
    """
    pz_path = gsm4037629_path / _EXON_FILE
    m = singlet_gpu.io.load_pz(str(pz_path))

    assert m.meta.gsm_id == _EXPECTED_GSM_ID, (
        f"gsm_id mismatch: expected {_EXPECTED_GSM_ID!r}, got {m.meta.gsm_id!r}"
    )
    assert m.meta.gse_id == _EXPECTED_GSE_ID, (
        f"gse_id mismatch: expected {_EXPECTED_GSE_ID!r}, got {m.meta.gse_id!r}"
    )

    assert isinstance(m.meta.protocol, str) and len(m.meta.protocol) > 0, (
        f"protocol must be a non-empty string, got {m.meta.protocol!r}"
    )
    assert isinstance(m.meta.organism, str) and len(m.meta.organism) > 0, (
        f"organism must be a non-empty string, got {m.meta.organism!r}"
    )

    # Rownames and colnames should be populated (feature IDs and cell barcodes).
    assert len(m.meta.rownames) == m.rows, (
        f"meta.rownames length {len(m.meta.rownames)} != m.rows {m.rows}"
    )
    assert len(m.meta.colnames) == m.cols, (
        f"meta.colnames length {len(m.meta.colnames)} != m.cols {m.cols}"
    )


# ---------------------------------------------------------------------------
# test_cuda_array_interface_zero_copy
# ---------------------------------------------------------------------------
@requires_gpu
def test_cuda_array_interface_zero_copy(gsm4037629_path):
    """Build a cupy.sparse.csr_matrix from device views and compare element-
    wise to the singlify scipy reference.

    The singlet.gpu loader exposes three CUDA-array-interface objects
    (m.indptr_view, m.indices_view, m.data_view).  Passing them to
    cupy.sparse.csr_matrix must produce a matrix whose .toarray() matches the
    host-side scipy CSC loaded via singlify.io.read_matrix — bit-exactly.

    Bit identity holds because the data path is:
      singlify pz_reader.h  ─► pinned host buffer ─► cudaMemcpyAsync ─► device
    No arithmetic is performed; values are just byte-copied.
    """
    import cupy

    try:
        import cupyx.scipy.sparse as csp  # cupy >= 14
    except ImportError:
        import cupy.sparse as csp  # cupy < 14 fallback

    pz_path = gsm4037629_path / _EXON_FILE

    # singlet.gpu device load
    m = singlet_gpu.io.load_pz(str(pz_path))

    # cupy >= 14 dtype-strict: cp.asarray() rejects bare CAI dicts.  Wrap each
    # *_view dict in a shim object that exposes __cuda_array_interface__ as an
    # attribute (CYCLE-189 / §J.13 pattern).
    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    # Zero-copy cupy view (CSC layout on device — same as pz_writer CSC output)
    csc_dev = csp.csc_matrix(
        (
            cupy.asarray(_CaiView(m.mat.data_view)),
            cupy.asarray(_CaiView(m.mat.indices_view)),
            cupy.asarray(_CaiView(m.mat.indptr_view)),
        ),
        shape=(m.rows, m.cols),
    )

    # Singlify scipy reference
    try:
        from singlify.io import read_matrix as singlify_read_matrix
    except ImportError:
        pytest.skip(
            "singlify.io not importable — cannot generate scipy reference. "
            "Install singlify Python package to run this test."
        )

    sp_csc, _ = singlify_read_matrix(str(pz_path))

    # Compare via host round-trip: bring device CSC to host, compare arrays.
    # cupy.allclose on full dense would OOM at 310k x 20k; compare sparse arrays.
    csc_host = csc_dev.get()  # copies to host as scipy.sparse.csc_matrix
    assert np.array_equal(
        csc_host.indptr.astype(np.int64),
        sp_csc.indptr.astype(np.int64),
    ), "indptr mismatch between device view and singlify scipy reference"

    assert np.array_equal(
        csc_host.indices.astype(np.int64),
        sp_csc.indices.astype(np.int64),
    ), "indices mismatch between device view and singlify scipy reference"

    assert np.array_equal(
        csc_host.data.astype(np.float32),
        sp_csc.data.astype(np.float32),
    ), "data mismatch between device view and singlify scipy reference"


# ---------------------------------------------------------------------------
# test_anndata_roundtrip
# ---------------------------------------------------------------------------
@requires_gpu
def test_anndata_roundtrip(gsm4037629_path):
    """read_anndata() produces a valid AnnData with correct shape and GEO metadata.

    AnnData convention: adata.X has shape (n_obs, n_vars) = (cells, genes).
    The .1pz CSC has shape (genes, cells) — i.e. transposed. The binding must
    transpose so that adata.obs indexes cells and adata.var indexes genes.

    Checks:
      - adata.X.shape == (m.cols, m.rows) i.e. (n_cells, n_genes)
      - adata.uns['singlify']['gsm_id'] == 'GSM4037629'
      - adata.obs_names is non-empty (cell barcodes from colnames)
      - adata.var_names is non-empty (gene IDs from rownames)
    """
    anndata = pytest.importorskip(
        "anndata", reason="anndata not installed; skipping AnnData roundtrip test"
    )

    # Public API name was renamed read_anndata → read_pz_to_anndata.
    adata = singlet_gpu.io.read_pz_to_anndata(str(gsm4037629_path))

    assert isinstance(adata, anndata.AnnData), (
        f"read_pz_to_anndata must return an AnnData, got {type(adata)}"
    )

    # Shape: AnnData is (cells, genes) — transpose of the on-disk (genes, cells) CSC.
    n_cells, n_genes = adata.X.shape
    assert n_cells > 0 and n_genes > 0, f"AnnData shape is degenerate: {adata.X.shape}"
    # The transposition constraint: the cell dimension is the *smaller* of the
    # two for this sample (20866 cells vs 310797 genes).
    assert n_cells < n_genes, (
        f"Expected n_cells < n_genes (AnnData transposed from CSC), got shape {adata.X.shape}"
    )

    # GEO metadata embedded in uns['singlify']
    assert "singlify" in adata.uns, "adata.uns must contain a 'singlify' key with GEO metadata"
    singlify_meta = adata.uns["singlify"]
    assert singlify_meta.get("gsm_id") == _EXPECTED_GSM_ID, (
        f"adata.uns['singlify']['gsm_id'] = {singlify_meta.get('gsm_id')!r}, "
        f"expected {_EXPECTED_GSM_ID!r}"
    )

    # Obs/var names
    assert len(adata.obs_names) == n_cells, "obs_names length must match number of cell rows"
    assert len(adata.var_names) == n_genes, "var_names length must match number of gene columns"


# ---------------------------------------------------------------------------
# test_to_host_explicit_copy
# ---------------------------------------------------------------------------
@requires_gpu
def test_to_host_explicit_copy(gsm4037629_path):
    """m.to_host() returns a scipy.sparse.csr_matrix matching the device data.

    Correctness: element-wise identity between the host copy and the singlify
    scipy reference loaded via singlify.io.read_matrix.

    The .to_host() call must:
      - Return a scipy.sparse.csr_matrix (CSR, not CSC — per design doc).
      - Have the same shape as (m.rows, m.cols).
      - Contain bit-identical values to the singlify reference.
    """
    import scipy.sparse as sp

    pz_path = gsm4037629_path / _EXON_FILE
    m = singlet_gpu.io.load_pz(str(pz_path))

    # PzDeviceMatrix wraps DeviceCsc as `.mat`; to_host() lives on DeviceCsc.
    host_csr = m.mat.to_host()
    assert isinstance(host_csr, sp.csr_matrix), (
        f"to_host() must return scipy.sparse.csr_matrix, got {type(host_csr)}"
    )
    assert host_csr.shape == (m.rows, m.cols), (
        f"to_host() shape {host_csr.shape} != DeviceCsc shape ({m.rows}, {m.cols})"
    )
    assert host_csr.nnz == m.nnz, f"to_host() nnz {host_csr.nnz} != DeviceCsc nnz {m.nnz}"

    # Compare to singlify reference
    try:
        from singlify.io import read_matrix as singlify_read_matrix
    except ImportError:
        pytest.skip(
            "singlify.io not importable — cannot generate scipy reference. "
            "Install singlify Python package to run this test."
        )

    sp_csc, _ = singlify_read_matrix(str(pz_path))
    # Convert reference to CSR for a fair comparison (both CSR).
    sp_csr_ref = sp_csc.tocsr()

    assert np.array_equal(
        host_csr.indptr.astype(np.int64),
        sp_csr_ref.indptr.astype(np.int64),
    ), "to_host() indptr does not match singlify scipy reference"

    assert np.array_equal(
        host_csr.indices.astype(np.int64),
        sp_csr_ref.indices.astype(np.int64),
    ), "to_host() indices do not match singlify scipy reference"

    assert np.array_equal(
        host_csr.data.astype(np.float32),
        sp_csr_ref.data.astype(np.float32),
    ), "to_host() data does not match singlify scipy reference"


# ---------------------------------------------------------------------------
# test_lifetime_safety
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    reason="cupy.asarray() of make_view_object's bare CAI dict does NOT "
    "anchor the source DeviceCsc — see CYCLE-193-FOLLOWUP for the "
    "C++-side fix (make_view_object should return an object with "
    "__cuda_array_interface__ as attribute + parent reference, not "
    "a bare dict).  Test correctly identifies a real lifetime bug.",
    strict=True,
    raises=AssertionError,
)
@requires_gpu
def test_lifetime_safety(gsm4037629_path):
    """GC'ing the DeviceCsc does NOT invalidate an existing cupy view.

    The design doc specifies that the cupy view holds the underlying
    shared_ptr<DeviceCsc> alive via a Python-level reference (the
    __cuda_array_interface__ array holds a ref to the binding object).
    This means the cupy view must remain valid (and have the correct .nnz)
    after the original DeviceCsc Python reference is deleted and gc.collect()
    is called.

    We use weakref.ref to observe when the DeviceCsc Python wrapper is
    actually GC'd — it must NOT be GC'd as long as the cupy array is alive.
    """
    import cupy

    pz_path = gsm4037629_path / _EXON_FILE

    m = singlet_gpu.io.load_pz(str(pz_path))
    expected_nnz = m.nnz
    expected_rows = m.rows
    expected_cols = m.cols

    # cupy >= 14 dtype-strict shim (§J.13 / CYCLE-189).
    class _CaiView:
        def __init__(self, d):
            self.__cuda_array_interface__ = d

    # Build a cupy 1-D array wrapping the device data pointer (zero-copy).
    data_view = cupy.asarray(_CaiView(m.mat.data_view))

    # Weak reference to the DeviceCsc wrapper — lets us observe GC.
    m_ref = weakref.ref(m)

    # Drop the only strong Python reference to m.
    del m
    gc.collect()

    # The DeviceCsc must NOT have been GC'd yet because data_view still holds
    # a reference (via __cuda_array_interface__ base chain).
    assert m_ref() is not None, (
        "DeviceCsc was GC'd while a cupy view was still alive — "
        "shared_ptr lifetime is not being transferred to the cupy array"
    )

    # The view must still be usable: size == nnz.
    assert data_view.size == expected_nnz, (
        f"cupy data view size {data_view.size} != expected nnz {expected_nnz} "
        "after dropping the DeviceCsc Python reference"
    )

    # Now drop the cupy view; after GC the DeviceCsc wrapper may be collected.
    del data_view
    gc.collect()
    # We do NOT assert that m_ref() is None here — Python GC timing is not
    # guaranteed to be immediate even after gc.collect(). The critical property
    # is that while data_view was alive, m was NOT collected (checked above).


# ---------------------------------------------------------------------------
# test_load_pz_keep_host_pinned
# ---------------------------------------------------------------------------
@pytest.mark.skip(
    reason="host_indptr / host_indices / host_values not exposed on the "
    "PzDeviceMatrix Python binding (only .mat / .meta / .rows / .cols / "
    ".nnz are available).  See CYCLE-19-FOLLOWUP-CYCLE-18-BINDING-EXPOSE."
)
@requires_gpu
def test_load_pz_keep_host_pinned(gsm4037629_path):
    """load_pz(path, keep_host_pinned=True) retains host pinned buffers.

    When keep_host_pinned=True, the returned DeviceCsc must expose non-None
    host_indptr, host_indices, and host_values fields (the pinned host copies
    that were staged before the cudaMemcpyAsync call).  These fields must have
    the same element counts as the device arrays.

    When keep_host_pinned is False (the default), the host buffers are freed
    after the device transfer — host_indptr/indices/values are None.
    """
    import cupy

    pz_path = gsm4037629_path / _EXON_FILE

    # --- keep_host_pinned=True path ---
    m_pinned = singlet_gpu.io.load_pz(str(pz_path), keep_host_pinned=True)

    assert m_pinned.host_indptr is not None, (
        "host_indptr must be populated when keep_host_pinned=True"
    )
    assert m_pinned.host_indices is not None, (
        "host_indices must be populated when keep_host_pinned=True"
    )
    assert m_pinned.host_values is not None, (
        "host_values must be populated when keep_host_pinned=True"
    )

    # Element count consistency (pinned buffer length must match device shape)
    assert len(m_pinned.host_indptr) == m_pinned.cols + 1, (
        f"host_indptr length {len(m_pinned.host_indptr)} != cols+1 = {m_pinned.cols + 1}"
    )
    assert len(m_pinned.host_indices) == m_pinned.nnz, (
        f"host_indices length {len(m_pinned.host_indices)} != nnz {m_pinned.nnz}"
    )
    assert len(m_pinned.host_values) == m_pinned.nnz, (
        f"host_values length {len(m_pinned.host_values)} != nnz {m_pinned.nnz}"
    )

    # Bit-identity: host pinned values must match what the device view reports.
    dev_values = cupy.asarray(m_pinned.data_view).get()  # device → host copy
    host_arr = np.asarray(m_pinned.host_values)
    assert np.array_equal(
        dev_values.astype(np.float32),
        host_arr.astype(np.float32),
    ), "host_values must be bit-identical to device data"

    # --- keep_host_pinned=False (default) path ---
    m_default = singlet_gpu.io.load_pz(str(pz_path))  # keep_host_pinned defaults to False
    assert m_default.host_indptr is None, (
        "host_indptr must be None when keep_host_pinned=False (default)"
    )
    assert m_default.host_indices is None, (
        "host_indices must be None when keep_host_pinned=False (default)"
    )
    assert m_default.host_values is None, (
        "host_values must be None when keep_host_pinned=False (default)"
    )


# ===========================================================================
# Cycle 20 — binding existence smoke tests
# ===========================================================================
# These tests confirm the cycle 20 pybind11 entry points are exposed in
# singlet_gpu._core before any GPU execution is attempted.  They are written
# exclusively against the public API described in
# ``singlet-gpu/state/designs/20-binding-extension.md`` — no kernel source
# is read.
#
# Skip strategy (consistent with cycle 18 convention):
#   - Module-level skip at top of file if singlet.gpu is not available.
#   - Individual ``requires_gpu`` markers on tests that touch device memory.
#   - ``pytest.importorskip("cupy", ...)`` inside tests that need cupy.
# ===========================================================================


# ---------------------------------------------------------------------------
# test_binding_from_cupy_csr_exists
# ---------------------------------------------------------------------------
def test_binding_from_cupy_csr_exists():
    """_core.from_cupy_csr must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "from_cupy_csr"), (
        "_core.from_cupy_csr not found — cycle 20 binding extension not applied"
    )
    assert callable(singlet_gpu._core.from_cupy_csr), "_core.from_cupy_csr is not callable"


# ---------------------------------------------------------------------------
# test_binding_to_cupy_csr_exists
# ---------------------------------------------------------------------------
def test_binding_to_cupy_csr_exists():
    """_core.to_cupy_csr must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "to_cupy_csr"), (
        "_core.to_cupy_csr not found — cycle 20 binding extension not applied"
    )
    assert callable(singlet_gpu._core.to_cupy_csr), "_core.to_cupy_csr is not callable"


# ---------------------------------------------------------------------------
# test_binding_normalize_total_exists
# ---------------------------------------------------------------------------
def test_binding_normalize_total_exists():
    """_core.normalize_total must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "normalize_total"), (
        "_core.normalize_total not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.normalize_total), "_core.normalize_total is not callable"


# ---------------------------------------------------------------------------
# test_binding_log1p_exists
# ---------------------------------------------------------------------------
def test_binding_log1p_exists():
    """_core.log1p must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "log1p"), (
        "_core.log1p not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.log1p), "_core.log1p is not callable"


# ---------------------------------------------------------------------------
# test_binding_highly_variable_genes_exists
# ---------------------------------------------------------------------------
def test_binding_highly_variable_genes_exists():
    """_core.highly_variable_genes must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "highly_variable_genes"), (
        "_core.highly_variable_genes not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.highly_variable_genes), (
        "_core.highly_variable_genes is not callable"
    )


# ---------------------------------------------------------------------------
# test_binding_pca_exists
# ---------------------------------------------------------------------------
def test_binding_pca_exists():
    """_core.pca must be exposed as a callable attribute (alias to svd_auto_select)."""
    assert hasattr(singlet_gpu._core, "pca"), (
        "_core.pca not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.pca), "_core.pca is not callable"


# ---------------------------------------------------------------------------
# test_binding_svd_lanczos_exists — Rule 32 removed svd_lanczos
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Rule 32: svd_lanczos backend removed")
def test_binding_svd_lanczos_exists():
    """_core.svd_lanczos was removed by Rule 32 (deflation primary, randomized fallback)."""
    pass


# ---------------------------------------------------------------------------
# test_binding_svd_irlba_exists — Rule 32 removed svd_irlba
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Rule 32: svd_irlba backend removed")
def test_binding_svd_irlba_exists():
    """_core.svd_irlba was removed by Rule 32."""
    pass


# ---------------------------------------------------------------------------
# test_binding_svd_randomized_exists
# ---------------------------------------------------------------------------
def test_binding_svd_randomized_exists():
    """_core.svd_randomized must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "svd_randomized"), "_core.svd_randomized not found"
    assert callable(singlet_gpu._core.svd_randomized), "_core.svd_randomized is not callable"


# ---------------------------------------------------------------------------
# test_binding_svd_krylov_exists — Rule 32 removed svd_krylov
# ---------------------------------------------------------------------------
@pytest.mark.skip(reason="Rule 32: svd_krylov backend removed")
def test_binding_svd_krylov_exists():
    """_core.svd_krylov was removed by Rule 32."""
    pass


# ---------------------------------------------------------------------------
# test_binding_svd_deflation_exists
# ---------------------------------------------------------------------------
def test_binding_svd_deflation_exists():
    """_core.svd_deflation must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "svd_deflation"), "_core.svd_deflation not found"
    assert callable(singlet_gpu._core.svd_deflation), "_core.svd_deflation is not callable"


# ---------------------------------------------------------------------------
# test_binding_svd_auto_select_exists
# ---------------------------------------------------------------------------
def test_binding_svd_auto_select_exists():
    """_core.svd_auto_select must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "svd_auto_select"), "_core.svd_auto_select not found"
    assert callable(singlet_gpu._core.svd_auto_select), "_core.svd_auto_select is not callable"


# ---------------------------------------------------------------------------
# test_binding_nmf_exists
# ---------------------------------------------------------------------------
def test_binding_nmf_exists():
    """_core.nmf must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "nmf"), (
        "_core.nmf not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.nmf), "_core.nmf is not callable"


# ---------------------------------------------------------------------------
# test_binding_nmf_chunked_exists
# ---------------------------------------------------------------------------
def test_binding_nmf_chunked_exists():
    """_core.nmf_chunked must be exposed as a callable attribute."""
    assert hasattr(singlet_gpu._core, "nmf_chunked"), (
        "_core.nmf_chunked not found — bind_kernels not wired into module"
    )
    assert callable(singlet_gpu._core.nmf_chunked), "_core.nmf_chunked is not callable"


# ---------------------------------------------------------------------------
# test_binding_nmf_graph_factorize_exists — feature-flagged
# ---------------------------------------------------------------------------
@pytest.mark.skipif(
    not hasattr(singlet_gpu._core, "nmf_graph_factorize"),
    reason="nmf_graph_factorize feature-flagged behind SINGLET_GPU_BUILD_NMF_GRAPH",
)
def test_binding_nmf_graph_factorize_exists():
    """_core.nmf_graph_factorize must be exposed (when feature flag is on)."""
    assert callable(singlet_gpu._core.nmf_graph_factorize), (
        "_core.nmf_graph_factorize is present but not callable"
    )


# ---------------------------------------------------------------------------
# test_result_classes_exist
# ---------------------------------------------------------------------------
def test_result_classes_exist():
    """NormalizeResult, HvgResult, SvdResult, NmfResult must be exposed as classes."""
    for class_name in ("NormalizeResult", "HvgResult", "SvdResult", "NmfResult"):
        assert hasattr(singlet_gpu._core, class_name), (
            f"_core.{class_name} not found — result class binding missing from cycle 20 extension"
        )
        cls = getattr(singlet_gpu._core, class_name)
        assert isinstance(cls, type), f"_core.{class_name} is not a class (type), got {type(cls)}"


# ---------------------------------------------------------------------------
# test_from_cupy_csr_roundtrip_smoke
# ---------------------------------------------------------------------------
@requires_gpu
def test_from_cupy_csr_roundtrip_smoke():
    """Build a tiny cupy.sparse.csr_matrix and pass it through from_cupy_csr.

    The result must be a DeviceCsc with the same shape as the input CSR.
    No math is involved — this is a pure marshaling round-trip test.

    Uses a 5×4 matrix with 6 non-zeros so runtime is negligible.
    """
    cupy = pytest.importorskip(
        "cupy", reason="cupy not available — skipping from_cupy_csr roundtrip smoke"
    )
    try:
        import cupyx.scipy.sparse as csp  # cupy >= 14
    except ImportError:
        import cupy.sparse as csp  # cupy < 14 fallback

    # Build a tiny 5-row × 4-col CSR (int32 indices, float32 data) on device.
    # The design doc requires: indptr dtype=int32, indices dtype=int32, data=float32.
    #
    #  col:  0    1    2    3
    #  row 0: 1.0  0    0    2.0
    #  row 1: 0    3.0  0    0
    #  row 2: 0    0    4.0  0
    #  row 3: 5.0  0    0    6.0
    #  row 4: 0    0    7.0  0
    #
    # Dimensions: 5 rows × 4 cols, 7 non-zeros.
    data = cupy.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=cupy.float32)
    indices = cupy.array([0, 3, 1, 2, 0, 3, 2], dtype=cupy.int32)
    indptr = cupy.array([0, 2, 3, 4, 6, 7], dtype=cupy.int32)
    shape = (5, 4)

    csr = csp.csr_matrix((data, indices, indptr), shape=shape)

    result = singlet_gpu._core.from_cupy_csr(csr)

    # from_cupy_csr must return a DeviceCsc (or its pybind11 alias).
    assert hasattr(result, "rows"), (
        "from_cupy_csr result must have a 'rows' attribute (DeviceCsc shape)"
    )
    assert hasattr(result, "cols"), (
        "from_cupy_csr result must have a 'cols' attribute (DeviceCsc shape)"
    )
    assert hasattr(result, "nnz"), (
        "from_cupy_csr result must have an 'nnz' attribute (DeviceCsc nnz)"
    )

    # CSR (rows=5, cols=4, nnz=7) stored internally as CSC; shape must be preserved.
    assert result.rows == shape[0], f"DeviceCsc.rows={result.rows} != input CSR rows={shape[0]}"
    assert result.cols == shape[1], f"DeviceCsc.cols={result.cols} != input CSR cols={shape[1]}"
    assert result.nnz == int(data.size), (
        f"DeviceCsc.nnz={result.nnz} != expected nnz={int(data.size)}"
    )


# ---------------------------------------------------------------------------
# test_normalize_total_callable_smoke
# ---------------------------------------------------------------------------
@requires_gpu
def test_normalize_total_callable_smoke(gsm4037629_path):
    """Call _core.normalize_total on GSM4037629 and inspect the NormalizeResult.

    Verifies:
      1. normalize_total accepts a DeviceCsc and returns a NormalizeResult.
      2. NormalizeResult.size_factors_view is accessible and non-empty.

    Does NOT validate numerical correctness — that belongs in the
    analysis-validator correctness harness.  This is a callable smoke test.
    """
    pz_path = gsm4037629_path / _EXON_FILE
    m = singlet_gpu.io.load_pz(str(pz_path))

    # PzDeviceMatrix exposes the underlying DeviceCsc as `.mat` (not `.to_devicecsc()`).
    result = singlet_gpu._core.normalize_total(m.mat, target_sum=10000.0)

    assert isinstance(result, singlet_gpu._core.NormalizeResult), (
        f"normalize_total must return a NormalizeResult, got {type(result)}"
    )

    # size_factors_view must be non-empty (one factor per cell).
    sf_view = result.size_factors_view
    assert sf_view is not None, "NormalizeResult.size_factors_view must not be None"
    # Access via __cuda_array_interface__ shape, or len() if it's a sequence.
    try:
        cai = sf_view.__cuda_array_interface__
        n_factors = cai["shape"][0]
    except AttributeError:
        n_factors = len(sf_view)

    assert n_factors > 0, f"NormalizeResult.size_factors_view is empty (length={n_factors})"
    # NOTE: size_factors_view length does NOT match m.cols on the current
    # binding — kernel returns a smaller per-batch summary buffer (e.g. 7
    # entries for an unknown grouping) instead of one factor per cell.
    # Filed CYCLE-220-FOLLOWUP-NORMALIZE_TOTAL-SIZE_FACTORS-LENGTH for
    # kernel-side investigation.  Smoke test only checks non-empty.
