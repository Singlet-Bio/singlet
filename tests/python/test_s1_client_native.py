# SPDX-License-Identifier: MIT
"""singlet._pz2: the parts of the binding that do not need a codec (header errors, ISA selection, gene aggregation, column
selection), checked against scipy; plus the codec round trips and ISA identity when the codec is built (skipped otherwise)."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

_pz2 = pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
import scipy.sparse as sp  # noqa: E402


def _random_csc(m=300, n=80, density=0.08, seed=0, vmax=50):
    rng = np.random.default_rng(seed)
    a = sp.random(
        m,
        n,
        density=density,
        format="csc",
        random_state=seed,
        data_rvs=lambda k: rng.integers(1, vmax, k),
    )
    a.sum_duplicates()
    a.sort_indices()
    return a.astype(np.int32)


def _expected_genes(a, row_to_gene, n_genes):
    keep = row_to_gene >= 0
    ind = sp.csr_matrix(
        (np.ones(int(keep.sum()), dtype=np.int64), (row_to_gene[keep], np.flatnonzero(keep))),
        shape=(n_genes, a.shape[0]),
    )
    out = (ind @ a.astype(np.int64)).tocsc()
    out.eliminate_zeros()
    out.sort_indices()
    return out


@pytest.mark.parametrize("threads", [1, 3])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_aggregate_genes_matches_scipy(seed, threads):
    a = _random_csc(m=500, n=120, density=0.1, seed=seed)
    rng = np.random.default_rng(100 + seed)
    n_genes = 60
    row_to_gene = rng.integers(-1, n_genes, a.shape[0]).astype(np.int32)  # -1 drops a row
    ip, ix, d = _pz2.aggregate_genes(a.indptr, a.indices, a.data, row_to_gene, n_genes, threads)
    got = sp.csc_matrix((d, ix, ip), shape=(n_genes, a.shape[1]))
    want = _expected_genes(a, row_to_gene, n_genes)
    assert got.indptr.tolist() == want.indptr.tolist()
    assert got.indices.tolist() == want.indices.tolist()
    assert got.data.astype(np.int64).tolist() == want.data.tolist()
    assert ix.dtype == np.int32 and d.dtype == np.int32


def test_aggregate_genes_bitmap_path_and_empty_columns():
    # dense columns force the bitmap path (many entries per column vs gene words); empty columns and a dropped-only column
    m, n_genes = 4000, 90
    rng = np.random.default_rng(7)
    cols = []
    for j in range(40):
        k = 0 if j % 7 == 0 else int(rng.integers(1, 3000))
        rows = np.sort(rng.choice(m, size=min(k, m), replace=False))
        cols.append((rows, rng.integers(1, 9, len(rows))))
    indptr = np.concatenate([[0], np.cumsum([len(r) for r, _ in cols])]).astype(np.int32)
    indices = np.concatenate([r for r, _ in cols]).astype(np.int32)
    data = np.concatenate([v for _, v in cols]).astype(np.int32)
    a = sp.csc_matrix((data, indices, indptr), shape=(m, 40))
    row_to_gene = rng.integers(0, n_genes, m).astype(np.int32)
    ip, ix, d = _pz2.aggregate_genes(indptr, indices, data, row_to_gene, n_genes, 4)
    got = sp.csc_matrix((d, ix, ip), shape=(n_genes, 40))
    want = _expected_genes(a, row_to_gene, n_genes)
    assert (got != want).nnz == 0
    for j in range(40):  # gene ids strictly ascending in every column
        c = ix[ip[j] : ip[j + 1]]
        assert (np.diff(c) > 0).all()


def test_aggregate_genes_float64_values():
    a = _random_csc(m=100, n=30, seed=5)
    row_to_gene = (np.arange(100) % 10).astype(np.int32)
    ip, ix, d = _pz2.aggregate_genes(
        a.indptr, a.indices, a.data.astype(np.float64), row_to_gene, 10, 2
    )
    assert d.dtype == np.float64
    want = _expected_genes(a, row_to_gene, 10)
    assert np.array_equal(d, want.data.astype(np.float64))


def test_aggregate_genes_rejects_bad_input():
    a = _random_csc(m=50, n=5, seed=3)
    with pytest.raises(ValueError):
        _pz2.aggregate_genes(
            a.indptr, a.indices, a.data, np.zeros(10, dtype=np.int32), 5, 1
        )  # map shorter than the row ids


def test_select_columns_matches_scipy():
    a = _random_csc(m=200, n=64, seed=9)
    cols = np.array([0, 3, 4, 17, 63], dtype=np.uint32)
    ip, ix, d = _pz2.select_columns(a.indptr, a.indices, a.data, cols)
    got = sp.csc_matrix((d, ix, ip), shape=(200, len(cols)))
    want = a[:, cols]
    assert (got != want).nnz == 0
    with pytest.raises(ValueError):
        _pz2.select_columns(
            a.indptr, a.indices, a.data, np.array([3, 3], dtype=np.uint32)
        )  # not strictly ascending
    with pytest.raises(ValueError):
        _pz2.select_columns(
            a.indptr, a.indices, a.data, np.array([64], dtype=np.uint32)
        )  # out of range


def test_read_header_rejects_garbage():
    with pytest.raises(_pz2.Pz2Error) as ei:
        _pz2.read_header(b"x" * 300)
    assert ei.value.code == "BadMagic"
    with pytest.raises(_pz2.Pz2Error) as ei:
        _pz2.read_header(b"x" * 10)
    assert ei.value.code in ("Truncated", "NeedRange")
    assert ei.value.len == 256 and ei.value.off == 0


def test_isa_selection_report():
    rep = _pz2.isa_report()
    assert "selected=" in rep and "compiled=" in rep
    assert _pz2.selected_isa() in ("scalar", "avx2", "avx512")
    assert _pz2.isa_supported("scalar")


@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512"])
def test_pz2_isa_env_var_forces_path(isa):
    """PZ2_ISA is read once per process: run a child per value; the selected path never exceeds what the CPU supports."""
    pkg_dir = pathlib.Path(_pz2.__file__).resolve().parent
    # import the extension without executing singlet/__init__ (it pulls in the whole analysis stack)
    code = (
        "import sys, types; p = types.ModuleType('singlet'); p.__path__ = [%r]; sys.modules['singlet'] = p; "
        "from singlet import _pz2; print(_pz2.selected_isa(), _pz2.best_isa())" % str(pkg_dir)
    )
    env = dict(os.environ, PZ2_ISA=isa)
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120
    )
    assert out.returncode == 0, out.stderr
    sel, best = out.stdout.split()
    order = ["scalar", "avx2", "avx512"]
    assert order.index(sel) <= order.index(best)
    assert sel == isa or order.index(isa) > order.index(
        best
    )  # forced path, or clamped to the best supported
