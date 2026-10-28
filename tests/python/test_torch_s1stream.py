# SPDX-License-Identifier: MIT
"""singlet.torch.S1Stream / S1Cells: block-shuffled minibatches from one .s1 object, locally and over HTTP ranges."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
sp = pytest.importorskip("scipy.sparse")
pytest.importorskip("anndata")
torch = pytest.importorskip("torch")

import singlet  # noqa: E402
from singlet.torch import S1Cells, S1Stream, block_batches  # noqa: E402
from test_s1_write import random_counts  # noqa: E402


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    """3,000 cells x 2,000 genes at 25% density, about 1.5M non-zeros in some 30 chunks: a block of cells is a small part of the
    bytes, next to the fixed prefix (header, chunk index, tables, names)."""
    d = tmp_path_factory.mktemp("s1stream")
    x, genes, cells = random_counts(3000, 2000, 0.25, seed=11)
    p = singlet.write_s1(x, d / "big.s1", genes=genes, barcodes=cells)
    return p, sp.csr_matrix(x), genes, cells


def _rows(x, blocks):
    return x[np.concatenate([np.arange(a, b) for a, b in blocks])].toarray()


@pytest.mark.parametrize("order", ["native", "canonical"])
def test_stream_batches_match_slices(big, order):
    p, x, genes, cells = big
    s = S1Stream(str(p), order=order, threads=2)
    assert (s.n_cells, s.n_genes) == (3000, 2000) and s.barcodes == cells
    blocks = [(2900, 2964), (64, 128), (128, 192), (1000, 1010)]
    got = s.batch(blocks).to_dense().numpy()
    want = _rows(x, blocks)
    if order == "native" and s.gene_order is not None:
        want = want[:, s.gene_order]
    assert np.array_equal(got, want)
    names = s.gene_names
    assert names == (genes if order == "canonical" or s.gene_order is None else [genes[i] for i in s.gene_order])


def test_stream_over_http_reads_only_needed_bytes(big):
    import _s1_range_server as rs

    p, x, _, _ = big
    with rs.serve(str(p.parent)) as (url, rec):
        s = S1Stream(f"{url}/{p.name}", order="canonical")
        blocks = [(1500, 1564)]
        assert np.array_equal(s.batch(blocks).to_dense().numpy(), _rows(x, blocks))
        one = s.bytes_fetched
        assert s.requests <= 3
        s.batch([(0, 3000)])
        assert one < s.bytes_fetched / 4  # one block costs a fraction of reading every cell


def test_block_batches():
    ep = block_batches(1000, batch_size=128, block_size=64, seed=1)
    cells = np.concatenate([np.arange(a, b) for bt in ep for a, b in bt])
    assert sorted(cells.tolist()) == list(range(1000))  # every cell once, last block short
    assert all(len(bt) <= 2 and bt == sorted(bt) for bt in ep)
    fixed = block_batches(1000, batch_size=128, block_size=64, seed=1, n_batches=20)
    assert len(fixed) == 20 and all(b - a == 64 for bt in fixed for a, b in bt)


def test_s1cells_epoch_and_workers(big):
    p, x, _, _ = big
    ds = S1Cells(str(p), batch_size=256, block_size=64, order="canonical", threads=1)
    seen = []
    for item in torch.utils.data.DataLoader(ds, batch_size=None, num_workers=2):
        idx = item["index"].numpy()
        assert np.array_equal(item["X"].to_dense().numpy(), x[idx].toarray())
        seen.append(idx)
    assert sorted(np.concatenate(seen).tolist()) == list(range(3000))
    first = [b[0] for b in ds.batches()]
    ds.set_epoch(1)
    assert [b[0] for b in ds.batches()] != first
