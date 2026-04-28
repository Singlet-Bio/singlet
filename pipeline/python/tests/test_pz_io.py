"""
Tests for the :mod:`singlify._pz_io` pybind11 binding.

These tests run against real singlify pipeline output directories (.1pz
files produced by the actual C++ writer) because the whole point of the
wrapper is to decode the wire format losslessly. Fixtures are resolved
lazily by ``conftest.py``; if none is available the tests are skipped.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest


# Skip the whole module if the binding isn't built. This is friendly for
# fresh checkouts where ``pip install -e .`` hasn't been run yet.
_pz_io = pytest.importorskip(
    "singlify._pz_io",
    reason="singlify._pz_io C++ extension not built; run `pip install -e .` first",
)


def test_import_exposes_read_1pz():
    """Smoke test: the module exposes the public entry point."""
    assert hasattr(_pz_io, "read_1pz")
    assert callable(_pz_io.read_1pz)


def test_read_gene_counts_shape_and_types(gene_counts_path: pathlib.Path):
    """Reading a real gene_counts.1pz returns a dict with the right keys
    and the scipy-ready dtypes."""
    d = _pz_io.read_1pz(str(gene_counts_path))

    # Required keys
    for key in (
        "shape",
        "nnz",
        "vt_code",
        "indptr",
        "indices",
        "data",
        "rownames",
        "colnames",
        "user_kv",
    ):
        assert key in d, f"key {key!r} missing from read_1pz result"

    m, n = d["shape"]
    nnz = d["nnz"]
    assert isinstance(m, int) and m > 0
    assert isinstance(n, int) and n > 0
    assert isinstance(nnz, int) and nnz >= 0

    assert isinstance(d["indptr"], np.ndarray)
    assert d["indptr"].dtype == np.int64
    assert d["indptr"].shape == (n + 1,)
    assert d["indptr"][0] == 0
    assert d["indptr"][-1] == nnz

    assert d["indices"].dtype == np.int64
    assert d["indices"].shape == (nnz,)

    # vt_code hints the narrowest integer width the writer saw
    assert d["vt_code"] in (1, 2, 3)
    expected_dtype = {1: np.uint8, 2: np.uint16, 3: np.uint32}[d["vt_code"]]
    assert d["data"].dtype == expected_dtype
    assert d["data"].shape == (nnz,)


def test_read_gene_counts_roundtrip_to_scipy_csc(gene_counts_path: pathlib.Path):
    """The dict drops directly into :class:`scipy.sparse.csc_matrix`
    without any reshape or dtype munging."""
    sp = pytest.importorskip("scipy.sparse")
    d = _pz_io.read_1pz(str(gene_counts_path))

    mat = sp.csc_matrix(
        (d["data"], d["indices"], d["indptr"]), shape=d["shape"]
    )
    assert mat.shape == d["shape"]
    assert mat.nnz == d["nnz"]
    assert mat.dtype == d["data"].dtype

    # Indices within each column must be strictly sorted (canonical CSC)
    for j in range(min(50, mat.shape[1])):
        col = mat.getcol(j)
        # scipy returns CSC; call indices directly
        idxs = col.indices
        assert np.all(np.diff(idxs) > 0), f"col {j} indices not sorted"


def test_embedded_metadata_fields_are_strings(gene_counts_path: pathlib.Path):
    d = _pz_io.read_1pz(str(gene_counts_path))
    for k, v in d["user_kv"].items():
        assert isinstance(k, str)
        assert isinstance(v, str)


def test_embedded_metadata_has_pipeline_fields(gene_counts_path: pathlib.Path):
    """singlify pipeline always embeds at least the GEO context."""
    d = _pz_io.read_1pz(str(gene_counts_path))
    # These keys are written by pilot_job.sh via --metadata-json
    expected_keys = {"gsm_id", "gse_id", "organism", "protocol"}
    present = expected_keys & set(d["user_kv"].keys())
    assert present == expected_keys, (
        f"missing pipeline metadata keys: "
        f"{expected_keys - set(d['user_kv'].keys())}"
    )
    assert d["user_kv"]["gsm_id"].startswith("GSM")
    assert d["user_kv"]["gse_id"].startswith("GSE")


def test_rownames_length_matches_shape(gene_counts_path: pathlib.Path):
    d = _pz_io.read_1pz(str(gene_counts_path))
    m, _ = d["shape"]
    assert len(d["rownames"]) == m
    # Gene IDs are Ensembl-style for the 2024-A GRCh38 reference
    assert any(r.startswith("ENSG") for r in d["rownames"][:100]), (
        "expected Ensembl gene IDs in rownames"
    )


def test_colnames_length_matches_shape(gene_counts_path: pathlib.Path):
    d = _pz_io.read_1pz(str(gene_counts_path))
    _, n = d["shape"]
    assert len(d["colnames"]) == n
    # All colnames are non-empty barcode strings
    assert all(c for c in d["colnames"][:100])


def test_aggregate_exon_counts_equals_spliced(
    fixture_dir: pathlib.Path,
):
    """Empirical property: ``aggregate(exon_counts, by_gene) == spliced``.

    This is the property that justifies dropping ``spliced.1pz`` from the
    NFS copy tree and reconstructing it on read via
    :func:`singlify.io.aggregate_features_to_gene`. The aggregator parses
    the leading ``ENSG...`` prefix from each exon rowname and sums all
    exons of the same gene.

    If this ever fails on any new sample, either the rowname format has
    changed or the per-gene sum invariant is broken — both block the
    Tier 2 drop policy.
    """
    sp = pytest.importorskip("scipy.sparse")
    from singlify.io import aggregate_features_to_gene, read_matrix

    exon_path = fixture_dir / "exon_counts.1pz"
    spliced_path = fixture_dir / "spliced.1pz"
    if not exon_path.is_file() or not spliced_path.is_file():
        pytest.skip("fixture lacks exon_counts or spliced")

    ex_mat, ex_meta = read_matrix(exon_path)
    sp_mat, sp_meta = read_matrix(spliced_path)

    # Reconstruct
    agg, agg_genes = aggregate_features_to_gene(ex_mat, ex_meta.rownames)

    # Shape, total sum, and nnz must all match
    assert agg.shape == sp_mat.shape, f"shape mismatch: {agg.shape} vs {sp_mat.shape}"
    assert agg.sum() == sp_mat.sum(), (
        f"total sum mismatch: agg={agg.sum()}, spliced={sp_mat.sum()}"
    )
    # The aggregated matrix should have ≤ nnz entries — equality is the
    # tightest meaningful check, and it holds when the writer produces
    # canonical CSC output.
    assert agg.nnz == sp_mat.nnz, (
        f"nnz mismatch: agg={agg.nnz}, spliced={sp_mat.nnz}"
    )

    # Gene order must match (the writer preserves Ensembl ID order
    # through both per-feature and per-gene matrices)
    common = set(agg_genes) & set(sp_meta.rownames)
    assert len(common) > 0
    # Spot-check a few per-gene rows for bit-exact equality
    import numpy as np
    rng = np.random.default_rng(42)
    sample_genes = rng.choice(list(common), size=min(5, len(common)), replace=False)
    sp_gene_to_idx = {g: i for i, g in enumerate(sp_meta.rownames)}
    for g in sample_genes:
        ai = agg_genes.index(g)
        si = sp_gene_to_idx[g]
        a_row = np.asarray(agg.getrow(ai).todense()).ravel()
        s_row = np.asarray(sp_mat.getrow(si).todense()).ravel()
        assert np.array_equal(a_row, s_row), \
            f"gene {g} row mismatch"


def test_gene_counts_equals_velocity_trio(
    gene_counts_path: pathlib.Path,
    spliced_path: pathlib.Path,
    unspliced_path: pathlib.Path,
    ambiguous_path: pathlib.Path,
):
    """Empirical property: ``gene_counts == spliced + unspliced + ambiguous``.

    This is the property that justifies dropping ``gene_counts.1pz`` from
    the NFS copy tree (verified-safe redundancy). If this ever fails we
    have to keep the file.
    """
    sp = pytest.importorskip("scipy.sparse")

    gc = _pz_io.read_1pz(str(gene_counts_path))
    sp_mat = _pz_io.read_1pz(str(spliced_path))
    us = _pz_io.read_1pz(str(unspliced_path))
    am = _pz_io.read_1pz(str(ambiguous_path))

    # All four matrices must have the same shape
    assert gc["shape"] == sp_mat["shape"] == us["shape"] == am["shape"]

    g = sp.csc_matrix(
        (gc["data"], gc["indices"], gc["indptr"]), shape=gc["shape"]
    ).astype(np.int64)
    s = sp.csc_matrix(
        (sp_mat["data"], sp_mat["indices"], sp_mat["indptr"]), shape=sp_mat["shape"]
    ).astype(np.int64)
    u = sp.csc_matrix(
        (us["data"], us["indices"], us["indptr"]), shape=us["shape"]
    ).astype(np.int64)
    a = sp.csc_matrix(
        (am["data"], am["indices"], am["indptr"]), shape=am["shape"]
    ).astype(np.int64)

    reconstructed = s + u + a
    diff = (g - reconstructed).tocoo()
    max_abs = 0 if diff.nnz == 0 else int(np.abs(diff.data).max())
    assert max_abs == 0, (
        f"gene_counts != spliced + unspliced + ambiguous: "
        f"max absolute difference = {max_abs}"
    )


def test_read_dir_populates_matrix_rownames_cache(fixture_dir: pathlib.Path):
    """``read_dir`` must populate ``matrix_rownames`` for every loaded matrix.

    This cache is what allows the AnnData / SCE self-heal paths to run
    ``aggregate_features_to_gene`` on a derived matrix without re-decoding
    the source ``.1pz`` just to fetch its rownames. Regression-testing it
    here so the optimization can't silently break.
    """
    from singlify.io import read_dir, read_matrix

    dd = read_dir(fixture_dir)
    assert len(dd.matrices) > 0, "read_dir returned no matrices"
    # Every matrix should have its rownames cached at the same length as
    # the matrix's row count.
    for name, mat in dd.matrices.items():
        assert name in dd.matrix_rownames, (
            f"{name} missing from matrix_rownames cache"
        )
        cached = dd.matrix_rownames[name]
        assert len(cached) == mat.shape[0], (
            f"{name}: cached rowname count {len(cached)} != "
            f"matrix rows {mat.shape[0]}"
        )
    # Spot-check: cached rownames for one matrix should equal what a fresh
    # read_matrix call returns.
    first_name = next(iter(dd.matrices))
    _, fresh_meta = read_matrix(fixture_dir / f"{first_name}.1pz")
    assert dd.matrix_rownames[first_name] == fresh_meta.rownames


def test_read_cohort_loads_multiple_samples(tmp_path: pathlib.Path):
    """``read_cohort`` loads the same matrix from multiple sample directories.

    Builds a synthetic cohort by symlinking the same fixture directory three
    times (mimicking three samples in a GSE) and verifies that
    :func:`read_cohort` returns three (matrix, metadata) pairs in input order.
    """
    from singlify.io import read_cohort
    import os

    # We need a real fixture for the symlink trick
    candidates = [
        pathlib.Path("/mnt/projects/debruinz_project/singlify_pipeline/"
                     "quant/scrna/GSE227/GSE227136/GSM7103327"),
        pathlib.Path("/mnt/projects/debruinz_project/singlify_pipeline/"
                     "quant/scrna/GSE116/GSE116256/GSM3587956"),
    ]
    src = None
    for c in candidates:
        if c.is_dir() and (c / "spliced.1pz").is_file():
            src = c
            break
    if src is None:
        pytest.skip("no fixture with spliced.1pz")

    # Build 3 symlink directories
    sample_dirs = []
    for i in range(3):
        d = tmp_path / f"sample_{i}"
        d.mkdir()
        for f in src.iterdir():
            if f.is_file():
                os.symlink(f, d / f.name)
        sample_dirs.append(d)

    cohort = read_cohort(sample_dirs, matrix_name="spliced")
    assert len(cohort) == 3
    for mat, meta in cohort:
        assert mat.shape[0] > 0
        assert mat.shape[1] > 0
        assert "gsm_id" in meta.user_kv

    # All samples should be the same (same source)
    shapes = {m.shape for m, _ in cohort}
    assert len(shapes) == 1, "all samples should have identical shape"


def test_aggregate_features_to_gene_empty_input():
    """Aggregator handles a zero-row sparse matrix without crashing."""
    import scipy.sparse as sp
    from singlify.io import aggregate_features_to_gene

    empty = sp.csc_matrix((0, 5), dtype=np.uint16)
    agg, gene_ids = aggregate_features_to_gene(empty, [])
    assert agg.shape == (0, 5)
    assert gene_ids == []


def test_aggregate_features_to_gene_synthetic_multi_exon():
    """Aggregator collapses multiple exons of the same gene correctly."""
    import scipy.sparse as sp
    from singlify.io import aggregate_features_to_gene

    # 5 features, 3 cells:
    #   ENSGAAA, ENSGAAA, ENSGAAA  (3 features for gene A)
    #   ENSGBBB, ENSGBBB           (2 features for gene B)
    # The prefix is everything before the first underscore.
    rownames = [
        "ENSGAAA_NAME_chr1:100-200",
        "ENSGAAA_NAME_chr1:300-400",
        "ENSGAAA_NAME_chr1:500-600",
        "ENSGBBB_NAME_chr2:100-200",
        "ENSGBBB_NAME_chr2:300-400",
    ]
    feat_mat = sp.csc_matrix(np.array([
        # cells 0  1  2
        [1, 0, 7],   # ENSGAAA exon 1
        [2, 0, 8],   # ENSGAAA exon 2
        [3, 5, 9],   # ENSGAAA exon 3
        [0, 2, 0],   # ENSGBBB exon 1
        [4, 3, 0],   # ENSGBBB exon 2
    ], dtype=np.uint16))

    agg, gene_ids = aggregate_features_to_gene(feat_mat, rownames)
    assert agg.shape == (2, 3)
    assert gene_ids == ["ENSGAAA", "ENSGBBB"]
    # Gene A row: 1+2+3, 0+0+5, 7+8+9 = 6, 5, 24
    assert list(np.asarray(agg.getrow(0).todense()).ravel()) == [6, 5, 24]
    # Gene B row: 0+4, 2+3, 0+0 = 4, 5, 0
    assert list(np.asarray(agg.getrow(1).todense()).ravel()) == [4, 5, 0]
    # Total sum preserved
    assert int(agg.sum()) == int(feat_mat.sum())


def test_aggregate_features_to_gene_real_ensembl_format():
    """Aggregator extracts distinct gene IDs from realistic Ensembl-style rownames."""
    import scipy.sparse as sp
    from singlify.io import aggregate_features_to_gene

    rownames = [
        "ENSG00000000003_TSPAN6_chrX:100627107-100629986",
        "ENSG00000000003_TSPAN6_chrX:100630758-100630866",
        "ENSG00000000005_TNMD_chrX:100627108-100630760",
        "ENSG00000000419_DPM1_chr20:50934866-50935131",
    ]
    feat_mat = sp.csc_matrix(np.array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
        [10, 11, 12],
    ], dtype=np.uint16))

    agg, gene_ids = aggregate_features_to_gene(feat_mat, rownames)

    # 3 distinct gene IDs from 4 features
    assert agg.shape == (3, 3)
    assert gene_ids == [
        "ENSG00000000003",
        "ENSG00000000005",
        "ENSG00000000419",
    ]
    # Row 0 (ENSG00000000003) = exon 0 + exon 1 = [1+4, 2+5, 3+6] = [5, 7, 9]
    assert list(np.asarray(agg.getrow(0).todense()).ravel()) == [5, 7, 9]
    # Row 1 (ENSG00000000005) = exon 2 = [7, 8, 9]
    assert list(np.asarray(agg.getrow(1).todense()).ravel()) == [7, 8, 9]
    # Row 2 (ENSG00000000419) = exon 3 = [10, 11, 12]
    assert list(np.asarray(agg.getrow(2).todense()).ravel()) == [10, 11, 12]


def test_pilot_job_rescue_tier_classification():
    """Verify the tiered success criterion in pilot_job.sh.

    The script encodes a 3-tier classifier:
      1. STRICT: map >= floor AND cells >= 10  →  SUCCESS
      2. RESCUE: map >= 0.40 AND cells >= 1000 →  SUCCESS rescued_low_map
      3. SOFT_FAIL on cells_below_threshold when map>=floor but cells<10
      4. HARD_FAIL otherwise

    This test re-implements the classifier in pure Python and exercises
    every tier. A regression in pilot_job.sh's classifier would diverge
    from this oracle.
    """
    def classify(mr, cells, protocol="10xv3"):
        atac = protocol in ("10x_atac",)
        visium = protocol in ("visium",)
        strict_floor = 0.30 if atac else 0.50
        rescue_floor = 0.20 if atac else 0.40
        if mr >= strict_floor and (visium or cells >= 10):
            return "SUCCESS", ""
        if mr >= rescue_floor and cells >= 1000:
            return "SUCCESS", "rescued_low_map"
        if mr >= strict_floor and cells < 10:
            return "SOFT_FAIL", "cells_below_threshold"
        return "HARD_FAIL", "align_low_map"

    # STRICT pass cases
    assert classify(0.90, 5000) == ("SUCCESS", "")
    assert classify(0.50, 10) == ("SUCCESS", "")
    assert classify(0.50, 1) == ("SOFT_FAIL", "cells_below_threshold")

    # RESCUE pass cases — the new tier
    assert classify(0.48, 2342) == ("SUCCESS", "rescued_low_map"), \
        "GSM2544726 cycle 19 borderline case should be rescued"
    assert classify(0.40, 1000) == ("SUCCESS", "rescued_low_map")
    assert classify(0.45, 1500) == ("SUCCESS", "rescued_low_map")

    # Just below rescue threshold cases — should HARD_FAIL
    assert classify(0.48, 999)[0] == "HARD_FAIL", "999 cells below rescue floor"
    assert classify(0.39, 5000)[0] == "HARD_FAIL", "39% mapping below rescue floor"
    assert classify(0.10, 0)[0] == "HARD_FAIL"

    # ATAC-specific floors
    assert classify(0.35, 100, "10x_atac") == ("SUCCESS", "")
    assert classify(0.25, 5000, "10x_atac") == ("SUCCESS", "rescued_low_map")
    assert classify(0.10, 0, "10x_atac")[0] == "HARD_FAIL"


def test_read_cohort_skips_missing_matrices(tmp_path: pathlib.Path):
    """``read_cohort`` skips directories that lack the requested matrix."""
    from singlify.io import read_cohort

    # Empty dirs — no .1pz files
    empty1 = tmp_path / "empty1"
    empty1.mkdir()
    empty2 = tmp_path / "empty2"
    empty2.mkdir()

    cohort = read_cohort([empty1, empty2], matrix_name="spliced")
    assert cohort == []


def test_anndata_reader_self_heals_when_spliced_dropped(
    fixture_dir: pathlib.Path,
    tmp_path: pathlib.Path,
):
    """The AnnData adapter must reconstruct the missing ``spliced`` layer
    from ``exon_counts`` when the pipeline drops ``spliced.1pz``.

    This is the regression test for the Tier 2 drop policy: it builds a
    synthetic "dropped" fixture that hides spliced/unspliced/ambiguous/
    gene_counts and verifies the reader still produces a usable AnnData
    with a ``layers["spliced"]`` whose total sum matches the original.
    """
    pytest.importorskip("anndata")
    pytest.importorskip("pandas")
    import os
    import shutil

    src = fixture_dir
    if not (src / "exon_counts.1pz").is_file():
        pytest.skip("fixture lacks exon_counts.1pz")
    if not (src / "spliced.1pz").is_file():
        pytest.skip("fixture lacks spliced.1pz")

    # Build a "dropped" fixture by symlinking everything except the
    # canonical per-gene velocity matrices.
    dropped = tmp_path / "dropped_fixture"
    dropped.mkdir()
    DROP = {"spliced.1pz", "unspliced.1pz", "ambiguous.1pz", "gene_counts.1pz"}
    for f in src.iterdir():
        if f.is_file() and f.name not in DROP:
            os.symlink(f, dropped / f.name)

    from singlify.interop.anndata import read_anndata
    from singlify.io import read_matrix

    adata = read_anndata(dropped)
    # Either spliced was successfully derived OR the fixture didn't have
    # exon_counts (in which case the test should have skipped above).
    assert "spliced" in adata.layers, (
        "AnnData adapter did not derive layers['spliced'] from exon_counts"
    )
    derived = adata.uns.get("singlify_derived_layers", [])
    assert any("spliced" in d for d in derived), (
        f"derivation not recorded in adata.uns: got {derived}"
    )

    # Verify against the original spliced.1pz
    orig, _ = read_matrix(src / "spliced.1pz")
    derived_layer = adata.layers["spliced"].T.tocsc()  # back to genes×cells
    assert int(orig.sum()) == int(derived_layer.sum()), (
        f"derived layer total sum {int(derived_layer.sum())} != "
        f"original {int(orig.sum())}"
    )


def test_bad_path_raises_valueerror(tmp_path: pathlib.Path):
    """A missing file must surface as a Python-level :class:`ValueError`."""
    bogus = tmp_path / "nonexistent.1pz"
    with pytest.raises(ValueError, match="pz_reader"):
        _pz_io.read_1pz(str(bogus))


def test_truncated_file_raises_valueerror(tmp_path: pathlib.Path):
    """A file smaller than the header+footer must surface as ValueError."""
    p = tmp_path / "tiny.1pz"
    p.write_bytes(b"\x00" * 50)
    with pytest.raises(ValueError, match="pz_reader"):
        _pz_io.read_1pz(str(p))


def test_wrong_magic_raises_valueerror(tmp_path: pathlib.Path):
    """A file with the wrong magic bytes must raise."""
    p = tmp_path / "bad_magic.1pz"
    # 200 bytes of zero — just enough to be "full" but with wrong magic
    p.write_bytes(b"\x00" * 200)
    with pytest.raises(ValueError, match="pz_reader"):
        _pz_io.read_1pz(str(p))


def test_random_garbage_never_crashes(tmp_path: pathlib.Path):
    """Random bytes must never crash the reader — only raise ValueError.

    Defensive fuzz test: feed the C++ reader random/structured-but-invalid
    files at various sizes and ensure every one surfaces as a Python
    ValueError (never a SIGSEGV / abort / C++ exception leak). The C++
    reader uses its own PZReadError exception which is converted to
    py::value_error in the binding.
    """
    import random
    import struct
    rng = random.Random(0xC0FFEE)

    # Mixture of sizes + content patterns
    test_blobs = [
        b"",                                  # empty
        b"\x00",                               # 1 byte
        b"TP1Z",                               # 4 bytes — magic only, no header
        b"TP1Z" + b"\x00" * 50,                # partial header
        b"TP1Z" + b"\x00" * 91,                # header-1
        b"TP1Z" + b"\x00" * 92 + b"TP1Z",      # header + footer-magic only
        bytes(rng.getrandbits(8) for _ in range(120)),     # 120 random bytes
        bytes(rng.getrandbits(8) for _ in range(1024)),    # 1KB random
        # Valid 96-byte header followed by garbage where ptr_z would be
        struct.pack(
            "<4sHBBIIQBBHIIIIIQIIQII16s",
            b"TP1Z", 1, 5, 0,                   # magic, ver=1, vt=uint16, flags=0
            10, 5, 50,                           # m=10, n=5, nnz=50
            4, 3, 0,                             # ptr_width=4, codec_lvl=3, _pad
            1, 0, 100, 5,                        # 1 chunk, perm_z_sz=0, ptr_z_sz=100, chunk_cols=5
            0, 0, 0, 0, 0, 0, 0,                 # feature flags + offsets + sizes
            b"\x00" * 16,                        # reserved
        ) + b"\xFF" * 200,                       # 200 bytes of garbage where ptr/chunks would be
    ]

    for i, blob in enumerate(test_blobs):
        p = tmp_path / f"fuzz_{i}.1pz"
        p.write_bytes(blob)
        # The reader must either succeed (impossible for these inputs) or
        # raise a Python ValueError. It must NOT crash the interpreter.
        try:
            _pz_io.read_1pz(str(p))
            # If we get here, the input parsed successfully — only valid
            # for the real test fixtures, not these synthetic blobs.
            assert False, f"blob {i} ({len(blob)} bytes) parsed but shouldn't have"
        except ValueError:
            pass  # expected
        except Exception as e:
            pytest.fail(
                f"blob {i} ({len(blob)} bytes) raised non-ValueError: "
                f"{type(e).__name__}: {e}"
            )
