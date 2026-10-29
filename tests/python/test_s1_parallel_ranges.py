# SPDX-License-Identifier: MIT
"""Concurrent range requests in the atlas readers (singlet.s1.transport / client, singlet.torch.AtlasCells): ``io_threads``
resolution, the per-read budget and the per-process cap, a range answered with status 200 and its Content-Range, and, against
the test range server holding every response for a fixed delay, the same results at ``io_threads`` 1 and 8 with fewer round
trips' worth of wall time at 8.

Everything is local. The study tests are skipped when ``singlet._pz2`` or the shipped packs are missing."""

from __future__ import annotations

import copy
import json
import os
import pathlib
import sys
import threading
import time
import warnings

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import _s1_range_server as rs  # noqa: E402
from singlet.s1 import transport as T  # noqa: E402

DELAY = 0.2  # seconds the server holds every response: one "round trip"
N_CLONES = 8


@pytest.fixture()
def fresh_io(monkeypatch):
    """A clean per-process pool and cap (the module keeps them for the life of the process)."""
    monkeypatch.setattr(T, "_IO_SLOTS", None)
    monkeypatch.setattr(T, "_IO_POOL", None)
    yield
    if T._IO_POOL is not None:
        T._IO_POOL.shutdown(wait=True)


# =================================================================================================================== units
def test_io_threads_auto_is_one_locally_and_eight_over_http(tmp_path):
    local, http = T.DirTransport(str(tmp_path)), T.HttpTransport("http://127.0.0.1:9")
    assert T.io_threads("auto", local) == 1 and T.io_threads(None, local) == 1
    assert (
        T.io_threads("auto", http) == 8
        and T.io_threads(3, http) == 3
        and T.io_threads(1, local) == 1
    )
    with pytest.raises(ValueError):
        T.io_threads(0, http)


def _peak(budget, n_items, hold=0.05):
    """Run n_items calls through budget.map; return (results, the most that ran at once)."""
    lock, cur, peak = threading.Lock(), [0], [0]

    def fn(i):
        with lock:
            cur[0] += 1
            peak[0] = max(peak[0], cur[0])
        time.sleep(hold)
        with lock:
            cur[0] -= 1
        return i * i

    return budget.map(fn, range(n_items)), peak[0]


def test_budget_keeps_order_and_at_most_n_in_flight(fresh_io):
    out, peak = _peak(T.IOBudget(3), 12)
    assert out == [i * i for i in range(12)] and peak == 3
    out, peak = _peak(T.IOBudget(1), 5)
    assert out == [i * i for i in range(5)] and peak == 1


def test_budget_runs_one_at_a_time_in_the_calling_thread():
    seen = set()
    T.IOBudget(1).map(lambda _i: seen.add(threading.get_ident()), range(4))
    assert seen == {threading.get_ident()}


def test_process_cap_bounds_requests_in_flight(tmp_path, fresh_io, monkeypatch):
    (tmp_path / "o.bin").write_bytes(bytes(range(256)) * 16)
    monkeypatch.setenv("SINGLET_MAX_CONNECTIONS", "2")
    with rs.serve(str(tmp_path), delay_s=DELAY) as (url, rec):
        t = T.HttpTransport(url)
        t0 = time.perf_counter()
        bodies = T.IOBudget(8).map(
            lambda k: bytes(t.get("o.bin", (k * 16, k * 16 + 15)).body), range(8)
        )
        took = time.perf_counter() - t0
    assert bodies == [(bytes(range(256)) * 16)[k * 16 : k * 16 + 16] for k in range(8)]
    assert len(rec.log) == 8 and took >= 4 * DELAY * 0.9  # 8 requests, 2 at a time


def test_range_answered_200_with_its_content_range_is_accepted(tmp_path):
    data = os.urandom(5000)
    (tmp_path / "o.bin").write_bytes(data)
    with rs.serve(str(tmp_path), range_status=200) as (url, rec):
        t = T.HttpTransport(url)
        r = t.get("o.bin", (100, 1099), expect_total=5000)
        assert r.status == 200 and bytes(r.body) == data[100:1100] and r.total == 5000 and r.etag
        with pytest.raises(T.IntegrityError, match="expected 4999"):  # the size check still applies
            t.get("o.bin", (0, 9), expect_total=4999)
    with rs.serve(str(tmp_path), ignore_range=True) as (
        url,
        rec,
    ):  # the whole object for a range: still refused
        with pytest.raises(T.IntegrityError, match="ignored the Range"):
            T.HttpTransport(url).get("o.bin", (100, 1099), expect_total=5000)


# =================================================================================================================== study
@pytest.fixture(scope="module")
def clones(tmp_path_factory):
    """The synthetic study plus N_CLONES more samples that point at GSM9000001's object (independent reads of one object)."""
    pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
    pytest.importorskip("zstandard")
    import _s1_synth as synth

    root = tmp_path_factory.mktemp("parallel")
    try:
        s = synth.Synth(str(root)).build(labels=True)
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    mp = root / "s1" / "m" / f"{synth.GSE}.json"
    man = json.loads(mp.read_text())
    g1 = next(x for x in man["samples"] if x["gsm"] == "GSM9000001")
    gsms = []
    for k in range(N_CLONES):
        c = copy.deepcopy(g1)
        c["gsm"] = "GSM91%05d" % k
        man["samples"].append(c)
        gsms.append(c["gsm"])
    mp.write_text(json.dumps(man))
    return root, s, gsms


@pytest.fixture()
def env(tmp_path, monkeypatch):
    from singlet.s1 import client

    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("SINGLET_OFFLINE", "1")  # no study API calls for sample facts
    monkeypatch.delenv("SINGLET_DATA_BASE", raising=False)
    client.clear_memory_cache()
    yield tmp_path
    client.clear_memory_cache()


def _same(a, b):
    a, b = a.tocsr(), b.tocsr()
    return a.shape == b.shape and (a != b).nnz == 0


def _timed(root, fn):
    """fn(study) against the delaying server: (result, seconds, sample-object requests)."""
    from singlet.s1 import client

    client.clear_memory_cache()
    with rs.serve(str(root), delay_s=DELAY) as (url, rec):
        st = client.open_study("GSE900001", base=url, force=True)
        rec.log.clear()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            out = fn(st)
            took = time.perf_counter() - t0
        return out, took, len(rec.paths("/s1/s/"))


def test_study_defaults(clones, env):
    from singlet.s1 import client

    root, _s, _g = clones
    assert client.open_study("GSE900001", base=str(root)).io_threads == 1
    with rs.serve(str(root)) as (url, _rec):
        st = client.open_study("GSE900001", base=url)
        io = st.io_budget()
        assert st.io_threads == 8 and io.n == 8 and st.sample_threads(None, io, 20) == 8
        assert (
            st.sample_threads(2, io, 20) == 2 and st.io_budget(io) is io and st.io_budget(1).n == 1
        )


def test_to_anndata_same_matrix_and_fewer_round_trips_at_8(clones, env):
    root, _s, gsms = clones
    one, t1, n1 = _timed(root, lambda st: st.to_anndata(gsms, io_threads=1))
    eight, t8, n8 = _timed(root, lambda st: st.to_anndata(gsms, io_threads=8))
    assert _same(one.X, eight.X) and _same(one.layers["unspliced"], eight.layers["unspliced"])
    assert list(one.obs_names) == list(eight.obs_names) and one.n_obs == 40 * N_CLONES
    assert n1 == n8 == N_CLONES  # one request per sample either way
    assert t1 >= N_CLONES * DELAY  # one at a time
    assert t8 < t1 / 2  # the samples' requests overlap


def test_cells_two_phase_same_cells_and_fewer_round_trips_at_8(clones, env, monkeypatch):
    from singlet.s1 import client

    monkeypatch.setattr(client, "PREFIX_HINT", 4096)  # small members: plan them like large ones
    monkeypatch.setattr(client, "PARTIAL_GAP", 64)
    root, s, gsms = clones
    thr = int(np.median(s.truth["GSM9000001"]["umi_exon"]))
    q = f"n_umi_exon >= {thr}"
    one, t1, n1 = _timed(root, lambda st: st.cells(q, samples=gsms, whole_max=1, io_threads=1))
    eight, t8, n8 = _timed(root, lambda st: st.cells(q, samples=gsms, whole_max=1, io_threads=8))
    assert _same(one.X, eight.X) and list(one.obs_names) == list(eight.obs_names) and one.n_obs > 0
    assert n1 == n8 >= 3 * N_CLONES  # cells + prefixes, then chunk runs, per sample
    assert t1 >= n1 * DELAY
    assert t8 < t1 / 3


def test_atlascells_same_cells_at_1_and_8(clones, env):
    pytest.importorskip("torch")
    import pandas as pd
    from singlet.torch import AtlasCells

    root, _s, gsms = clones
    plan = pd.DataFrame([("GSE900001", g, 40) for g in gsms], columns=["gse", "gsm", "n_cells"])

    def drain(io_threads):
        def run(st):
            ds = AtlasCells(
                plan=plan,
                base=st.transport.base,
                batch_size=64,
                io_threads=io_threads,
                output="scipy",
            )
            got = {}
            for b in ds:
                for i, (g, c) in enumerate(zip(b["gsm"], b["col"].tolist())):
                    got[(g, c)] = b["X"][i].toarray().ravel()
            return got

        return run

    one, t1, n1 = _timed(root, drain(1))
    eight, t8, n8 = _timed(root, drain(8))
    assert set(one) == set(eight) and len(one) == 40 * N_CLONES
    assert all(np.array_equal(one[k], eight[k]) for k in one)
    assert n1 == n8 == N_CLONES  # (decoding dominates this one's wall time, so it is not timed)
    with pytest.raises(ValueError):
        AtlasCells(plan=plan, io_threads=0)
