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
    assert names == (
        genes if order == "canonical" or s.gene_order is None else [genes[i] for i in s.gene_order]
    )


@pytest.mark.parametrize(
    "range_status", [206, 200]
)  # 200: the slice with Content-Range, as rclone serve s3 answers
def test_stream_over_http_reads_only_needed_bytes(big, range_status):
    import _s1_range_server as rs

    p, x, _, _ = big
    with rs.serve(str(p.parent), range_status=range_status) as (url, rec):
        s = S1Stream(f"{url}/{p.name}", order="canonical")
        blocks = [(1500, 1564)]
        assert np.array_equal(s.batch(blocks).to_dense().numpy(), _rows(x, blocks))
        one = s.bytes_fetched
        assert s.requests <= 3
        s.batch([(0, 3000)])
        assert one < s.bytes_fetched / 4  # one block costs a fraction of reading every cell


def test_stream_over_http_rejects_a_server_that_ignores_range(big):
    import _s1_range_server as rs

    p = big[0]
    with rs.serve(str(p.parent), ignore_range=True) as (url, _):
        with pytest.raises(ValueError, match="does not serve byte ranges"):
            S1Stream(f"{url}/{p.name}")


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


def test_io_threads_auto_and_checked(big):
    p = big[0]
    assert S1Stream(str(p)).io_threads == 1  # a local file: one request at a time
    assert S1Stream(str(p), io_threads=3).io_threads == 3
    with pytest.raises(ValueError):
        S1Stream(str(p), io_threads=0)


def test_stream_over_http_sends_a_batchs_requests_together(big):
    import time

    import _s1_range_server as rs

    p, x, _, _ = big
    blocks = [(i * 370, i * 370 + 64) for i in range(8)]  # 8 separate ranges, one request each
    with rs.serve(str(p.parent), delay_s=0.05) as (url, _):
        seq = S1Stream(f"{url}/{p.name}", order="canonical", io_threads=1)
        par = S1Stream(f"{url}/{p.name}", order="canonical")
        assert par.io_threads == 32  # "auto" over HTTP
        out, took = {}, {}
        for name, s in (("seq", seq), ("par", par)):
            n0, t0 = s.requests, time.perf_counter()
            out[name] = s.batch(blocks).to_dense().numpy()
            took[name] = (time.perf_counter() - t0, s.requests - n0)
        par.close()
    assert np.array_equal(out["seq"], _rows(x, blocks)) and np.array_equal(out["par"], out["seq"])
    assert took["par"][1] == took["seq"][1] >= 4  # the same requests, sent together
    assert took["par"][0] < took["seq"][0] / 2  # ~one round trip instead of eight


def test_s1cells_prefetch_gives_the_same_batches(big):
    p = big[0]
    a = list(S1Cells(str(p), batch_size=256, threads=1, prefetch=0))
    b = list(S1Cells(str(p), batch_size=256, threads=1, prefetch=3))
    assert len(a) == len(b)
    for u, v in zip(a, b):
        assert torch.equal(u["index"], v["index"])
        assert np.array_equal(u["X"].to_dense().numpy(), v["X"].to_dense().numpy())


@pytest.mark.skipif(sys.platform != "linux", reason="DataLoader forks its workers on Linux")
def test_s1cells_in_forked_workers_over_http(big):
    """The dataset opens the object in the parent (for its shape); forked workers must not share that HTTP pool."""
    import _s1_range_server as rs

    p = big[0]
    with rs.serve(str(p.parent)) as (url, _):
        ds = S1Cells(f"{url}/{p.name}", batch_size=256, threads=1, order="canonical")
        ref = {int(b["index"][0]): b["X"].to_dense().numpy() for b in ds}
        dl = torch.utils.data.DataLoader(
            ds, batch_size=None, num_workers=2, multiprocessing_context="fork"
        )
        got = {int(b["index"][0]): b["X"].to_dense().numpy() for b in dl}
    assert got.keys() == ref.keys()
    assert all(np.array_equal(got[k], ref[k]) for k in ref)


def test_read_ahead_raises_errors_and_stops_when_closed():
    from singlet.torch._s1stream import _read_ahead

    def make(i):
        if i == 3:
            raise RuntimeError("boom")
        return i

    g = _read_ahead(make, list(range(10)), 2)
    assert [next(g) for _ in range(3)] == [0, 1, 2]
    with pytest.raises(RuntimeError, match="boom"):
        next(g)
    g2 = _read_ahead(lambda i: i, list(range(1000)), 1)
    assert next(g2) == 0
    g2.close()  # returns: the producer is unblocked and stops


@pytest.mark.parametrize("order", ["native", "canonical"])
def test_decode_batch_matches_one_range_at_a_time(big, order):
    p = big[0]
    s = S1Stream(str(p), order=order, threads=3)
    ranges = [
        (2900, 2964),
        (64, 192),
        (1000, 1010),
        (5, 5),
        (2999, 3000),
    ]  # out of order, an empty range, the last cell
    bufs = [
        s._get((int(s.mem.plan(a, b)["byte_off"]), int(s.mem.plan(a, b)["byte_len"])))
        if b > a
        else b""
        for a, b in ranges
    ]
    ip, idx, val = s.mem.decode_batch(ranges, bufs, mode=order, threads=3, dtype="float32")
    one = [s.mem.decode(a, b, mode=order, threads=1, dtype="float32") for a, b in ranges if b > a]
    assert ip[-1] == len(idx) == len(val) == sum(int(d["nnz"]) for d in one)
    want = sp.vstack(
        [
            sp.csr_matrix(
                (d["data"], d["indices"], d["indptr"]), shape=(len(d["indptr"]) - 1, s.n_genes)
            )
            for d in one
        ]
    )
    got = sp.csr_matrix((val, idx, ip), shape=(len(ip) - 1, s.n_genes))
    assert (got != want).nnz == 0
    with pytest.raises(ValueError):
        s.mem.decode_batch(ranges, bufs[:-1])
    with pytest.raises(ValueError):
        s.mem.decode_batch([(0, 64)], [bufs[0][:10]])


def test_stream_keeps_no_batch_bytes_and_iter_batches_matches(big):
    p, x, _, _ = big
    s = S1Stream(str(p), order="canonical", threads=2)
    ep = block_batches(3000, batch_size=256, block_size=64, seed=4)
    head = s.mem.prefix_len + (
        1 << 18
    )  # bytes past the first read: those come from the batch's own requests
    checked = 0
    for b in ep[:6]:
        s.batch(b)
        for c0, c1 in b:
            q = s.mem.plan(c0, c1)
            if int(q["byte_off"]) > head:
                assert not s.mem.has(
                    int(q["byte_off"]), int(q["byte_len"])
                )  # nothing of a batch is kept
                checked += 1
    assert checked
    for ahead in (0, 1, 3):
        for b, t in zip(ep, s.iter_batches(ep, ahead=ahead)):
            assert np.array_equal(t.to_dense().numpy(), _rows(x, b))


def test_stream_over_http_opens_in_one_request(big):
    import _s1_range_server as rs

    p, x, _, _ = big
    with rs.serve(str(p.parent)) as (url, rec):
        s = S1Stream(f"{url}/{p.name}", order="canonical")
        assert s.requests == 1  # the prefix read holds the header, member table and matrix prefix
        blocks = [(10, 74), (2000, 2064)]
        assert np.array_equal(s.batch(blocks).to_dense().numpy(), _rows(x, blocks))
        assert s.requests <= 3  # a block inside the first read costs no request
