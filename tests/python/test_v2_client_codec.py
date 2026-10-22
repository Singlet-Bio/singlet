# SPDX-License-Identifier: MIT
"""singlet._pz2 against the real codec: round trips in every mode, value widths, dtype conversion, gene aggregation with column
selection, ISA identity, fetch-callable behaviour (exactly the planned ranges), names and key/values, corrupt inputs, GIL release.

Skipped when ``singlet._pz2`` is not built."""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

try:  # a source checkout without an installed package
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))

_pz2 = pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
import scipy.sparse as sp  # noqa: E402


def _csc(m, n, density, seed, vmax=40, dtype=np.uint32):
    rng = np.random.default_rng(seed)
    a = sp.random(
        m,
        n,
        density=density,
        format="csc",
        random_state=seed,
        data_rvs=lambda k: rng.integers(1, vmax, k),
    ).astype(dtype)
    a.sum_duplicates()
    a.sort_indices()
    return a


def _names(prefix, k):
    return ["%s%d" % (prefix, i) for i in range(k)]


def _encode(a, **kw):
    kw.setdefault("row_names", _names("r", a.shape[0]))
    kw.setdefault("col_names", _names("c", a.shape[1]))
    return _pz2.encode(a.indptr, a.indices, a.data, a.shape[0], a.shape[1], **kw)


def _canonical(res, shape):
    m = sp.csc_matrix((res["data"], res["indices"], res["indptr"]), shape=shape)
    perm = res["row_perm"]
    if perm is not None:  # native order: map stored ids to canonical ids
        m = sp.csc_matrix((m.data, perm[m.indices], m.indptr), shape=shape)
        m.sort_indices()
    return m


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


@pytest.mark.parametrize("codec", ["rans", "generic", "auto"])
@pytest.mark.parametrize(
    "shape,density,vmax",
    [((3000, 500), 0.03, 40), ((700, 90), 0.2, 300), ((50000, 64), 0.002, 5), ((12, 5), 0.5, 3)],
)
def test_roundtrip_all_modes(codec, shape, density, vmax):
    a = _csc(*shape, density, seed=shape[0], vmax=vmax)
    blob, stats = _encode(a, codec=codec, chunk_cols=64)
    mem = _pz2.Member(blob)
    h = mem.header
    assert (h["m"], h["n"], h["nnz"]) == (shape[0], shape[1], a.nnz)
    want_digest = _pz2.digest(
        a.indptr,
        a.indices,
        a.data,
        *shape,
        value_type=h["value_type"],
        row_names=_names("r", shape[0]),
        col_names=_names("c", shape[1]),
    )
    assert h["digest"] == want_digest.hex()
    for mode in ("native", "canonical"):
        for threads in (1, 3):
            res = mem.decode(mode=mode, threads=threads)
            assert (_canonical(res, shape) != a).nnz == 0
            assert res["indices"].dtype == np.int32 and res["indptr"].dtype == np.int32
    res = _pz2.decode(blob, mode="canonical", threads=2)  # the one-shot function agrees
    assert (_canonical(res, shape) != a).nnz == 0
    for c0, c1 in [
        (0, 1),
        (3, 70),
        (63, 65),
        (0, shape[1]),
        (shape[1] - 1, shape[1]),
        (5, 5),
    ]:  # ranges that cut chunks
        if c1 > shape[1]:
            continue
        r = mem.decode(c0, c1, mode="canonical")
        sub = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(shape[0], c1 - c0))
        assert (sub != a[:, c0:c1]).nnz == 0
    with pytest.raises(ValueError):
        mem.decode(0, shape[1] + 1)


@pytest.mark.parametrize("dt,vmax", [(np.uint8, 200), (np.uint16, 60000), (np.uint32, 2**31 - 2)])
def test_value_widths_and_dtype_conversion(dt, vmax):
    a = _csc(800, 120, 0.05, 11, vmax=vmax, dtype=dt)
    blob, _ = _encode(a, codec="auto")
    mem = _pz2.Member(blob)
    r = mem.decode(mode="canonical")
    assert (
        np.dtype(r["data"].dtype).itemsize <= np.dtype(dt).itemsize
    )  # stored width: the smallest that fits
    assert np.array_equal(r["data"].astype(np.int64), a.data.astype(np.int64))
    for want in ("int32", "float64", "uint32"):
        r2 = mem.decode(mode="canonical", dtype=want)
        assert r2["data"].dtype == np.dtype(want)
        assert np.array_equal(r2["data"].astype(np.int64), a.data.astype(np.int64))


def test_float32_matrix_roundtrip():
    rng = np.random.default_rng(3)
    a = sp.random(
        400,
        60,
        density=0.1,
        format="csc",
        random_state=3,
        data_rvs=lambda k: rng.random(k).astype(np.float32) + 0.5,
    ).astype(np.float32)
    a.sort_indices()
    blob, _ = _encode(a, codec="auto")
    mem = _pz2.Member(blob)
    assert mem.header["value_type"] == "f32"
    r = mem.decode(mode="canonical")
    assert r["data"].dtype == np.float32 and np.array_equal(r["data"], a.data)


def test_empty_and_degenerate_matrices():
    for shape in [(100, 0), (0, 0), (100, 7)]:
        a = sp.csc_matrix(shape, dtype=np.uint32)
        blob, _ = _encode(a, codec="auto")
        r = _pz2.Member(blob).decode(mode="canonical")
        assert r["nnz"] == 0 and len(r["indptr"]) == shape[1] + 1 and len(r["indices"]) == 0


def test_gene_mode_matches_scipy_with_selection():
    m, n = 4000, 300
    a = _csc(m, n, 0.04, 21, vmax=30)
    blob, _ = _encode(a, codec="rans", chunk_cols=64)
    mem = _pz2.Member(blob)
    n_genes = 120
    gmap = np.random.default_rng(5).integers(-1, n_genes, m).astype(np.int32)
    cols = np.array([0, 5, 6, 70, 71, 140, 255, 299], dtype=np.uint32)
    for threads in (1, 4):
        r = mem.decode(mode="gene", gene_map=gmap, n_genes=n_genes, threads=threads, cols=cols)
        got = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(n_genes, len(cols)))
        assert (got != _expected_genes(a[:, cols], gmap, n_genes)).nnz == 0
        assert r["data"].dtype == np.int32 and r["n_rows"] == n_genes and r["n_cols"] == len(cols)
    r = mem.decode(
        60,
        200,
        mode="gene",
        gene_map=gmap,
        n_genes=n_genes,
        cols=np.array([0, 10, 139], dtype=np.uint32),
    )
    got = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(n_genes, 3))
    assert (got != _expected_genes(a[:, [60, 70, 199]], gmap, n_genes)).nnz == 0
    r = mem.decode(
        mode="gene", gene_map=gmap, n_genes=n_genes, dtype="float64"
    )  # float sums (the R path)
    assert r["data"].dtype == np.float64
    with pytest.raises(ValueError):
        mem.decode(mode="gene", n_genes=n_genes)  # gene_map missing


def test_isa_outputs_are_identical():
    a = _csc(6000, 400, 0.03, 31, vmax=100)
    blob, _ = _encode(a, codec="rans", chunk_cols=128)
    mem = _pz2.Member(blob)
    ref = mem.decode(mode="native", isa="scalar")
    for isa in ("avx2", "avx512"):
        r = mem.decode(mode="native", isa=isa)
        for k in ("indptr", "indices", "data"):
            assert np.array_equal(ref[k], r[k]), (isa, k)
    gm = (np.arange(6000) % 90).astype(np.int32)
    g0 = mem.decode(mode="gene", gene_map=gm, n_genes=90, isa="scalar")
    g1 = mem.decode(mode="gene", gene_map=gm, n_genes=90, isa="avx512", threads=4)
    assert all(np.array_equal(g0[k], g1[k]) for k in ("indptr", "indices", "data"))


def test_fetch_callable_reads_only_the_ranges_it_needs():
    a = _csc(5000, 1200, 0.02, 41, vmax=30)
    blob, _ = _encode(a, codec="rans", chunk_cols=100)
    calls = []

    def fetch(off, ln):
        calls.append((off, ln))
        return blob[off : off + ln]

    mem = _pz2.Member(fetch=fetch, prefix_hint=4096)
    n_open = len(calls)
    assert n_open >= 1
    plan = mem.plan(250, 420)
    assert plan["byte_len"] < len(blob) // 2
    r = mem.decode(250, 420, mode="canonical", threads=2)
    assert calls[n_open:] == [(plan["byte_off"], plan["byte_len"])]  # exactly the planned run, once
    sub = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(5000, 170))
    assert (sub != a[:, 250:420]).nnz == 0
    mem.decode(300, 400, mode="canonical")  # already covered: no new request
    assert len(calls) == n_open + 1


def test_missing_range_without_fetch_raises_needrange():
    a = _csc(2000, 500, 0.03, 51)
    blob, _ = _encode(a, codec="rans", chunk_cols=100)
    first = _pz2.read_header(blob)["prefix_len"]
    mem = _pz2.Member(blob[:first])
    plan = mem.plan(200, 300)
    with pytest.raises(_pz2.NeedRange) as ei:
        mem.decode(200, 300)
    assert ei.value.off == plan["byte_off"] and ei.value.len == plan["byte_len"]
    mem.add_range(plan["byte_off"], blob[plan["byte_off"] : plan["byte_off"] + plan["byte_len"]])
    assert mem.decode(200, 300, mode="canonical")["nnz"] == a[:, 200:300].nnz
    with pytest.raises(_pz2.NeedRange):
        _pz2.Member(blob[:100])  # not even the header


def test_names_kinds_and_kv():
    a = _csc(300, 40, 0.1, 61)
    m, n = a.shape
    pack_id = bytes(range(1, 9))
    blob, _ = _encode(
        a,
        row_names=_names("g", m),
        col_names={"kind": "cells", "names": _names("b", n)},
        kv={"k1": "v1", "a": "b"},
    )
    mem = _pz2.Member(blob)
    assert mem.header["col_names_kind"] == "cells" and mem.header["row_names_kind"] == "inline"
    assert mem.row_names()["names"] == _names("g", m)
    assert mem.col_names()["kind"] == "cells" and mem.col_names()["names"] is None
    assert mem.kv() == {"a": "b", "k1": "v1"}
    idx = np.sort(np.random.default_rng(1).choice(10000, n, replace=False)).astype(np.uint32)
    blob, _ = _encode(
        a,
        row_names={"kind": "pack", "pack_id": pack_id, "names": _names("g", m)},
        col_names={
            "kind": "pack_subset",
            "pack_id": pack_id,
            "pack_index": idx,
            "names": _names("b", n),
        },
    )
    mem = _pz2.Member(blob)
    h = mem.header
    assert h["row_names_kind"] == "pack" and h["row_names_id"] == pack_id.hex()
    cn = mem.col_names()
    assert (
        h["col_names_kind"] == "pack_subset"
        and cn["kind"] == "pack_subset"
        and np.array_equal(cn["pack_index"], idx)
    )


def test_corrupt_inputs_raise_instead_of_crashing():
    a = _csc(2000, 300, 0.03, 71)
    blob, _ = _encode(a, codec="rans", chunk_cols=100)
    rng = np.random.default_rng(9)
    detected = 0
    for _ in range(60):
        b = bytearray(blob)
        for _k in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= int(rng.integers(1, 256))
        try:
            _pz2.Member(bytes(b)).decode(mode="canonical")
        except (_pz2.Pz2Error, ValueError):
            detected += 1
    assert (
        detected >= 20
    )  # a flipped bit in a payload may decode to other counts; the point is that nothing crashes
    for cut in (
        10,
        255,
        300,
        len(blob) // 2,
    ):  # (the last bytes belong to sections decode does not read)
        with pytest.raises((_pz2.Pz2Error, ValueError)):
            _pz2.Member(blob[:cut]).decode(mode="canonical")


def test_threads_do_not_change_output_and_gil_is_released():
    import threading
    import time

    a = _csc(8000, 1500, 0.02, 81, vmax=60)
    blob, _ = _encode(a, codec="rans", chunk_cols=100)
    mem = _pz2.Member(blob)
    ref = mem.decode(mode="canonical", threads=1)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.0005)

    th = threading.Thread(target=ticker)
    th.start()
    t0 = time.perf_counter()
    for _ in range(20):
        r = mem.decode(mode="canonical", threads=2)
    dt = time.perf_counter() - t0
    stop.set()
    th.join()
    assert np.array_equal(r["indices"], ref["indices"]) and np.array_equal(r["data"], ref["data"])
    assert (
        len(ticks) > 0.2 * dt / 0.0005 * 0.05
    )  # the ticker thread kept running: the GIL was released while decoding


@pytest.mark.parametrize("isa", ["scalar", "avx2", "avx512"])
def test_pz2_isa_env_var_gives_identical_decodes(isa, tmp_path):
    """The whole path under each forced PZ2_ISA (a child process per value): decode a fixed member, hash the arrays. Every ISA must
    produce the same bytes; the member itself was encoded once, here."""
    import os
    import subprocess

    a = _csc(7000, 300, 0.03, 91, vmax=80)
    blob, _ = _encode(a, codec="rans", chunk_cols=64)
    f = tmp_path / "m.pz2"
    f.write_bytes(blob)
    pkg_dir = pathlib.Path(_pz2.__file__).resolve().parent
    code = (
        "import sys, types, hashlib, numpy as np; p = types.ModuleType('singlet'); p.__path__ = [%r]; sys.modules['singlet'] = p; "
        "from singlet import _pz2; m = _pz2.Member(open(%r, 'rb').read()); h = hashlib.sha256(); "
        "[h.update(np.ascontiguousarray(m.decode(mode=md, threads=2)[k]).tobytes()) for md in ('native', 'canonical') for k in ('indptr', 'indices', 'data')]; "
        "g = (np.arange(7000) %% 97).astype(np.int32); r = m.decode(mode='gene', gene_map=g, n_genes=97, threads=3); "
        "[h.update(np.ascontiguousarray(r[k]).tobytes()) for k in ('indptr', 'indices', 'data')]; "
        "print(_pz2.selected_isa(), h.hexdigest())" % (str(pkg_dir), str(f))
    )
    out = subprocess.run(
        [sys.executable, "-P", "-c", code],
        env=dict(os.environ, PZ2_ISA=isa),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr
    sel, digest = out.stdout.split()
    ref = subprocess.run(
        [sys.executable, "-P", "-c", code],
        env=dict(os.environ, PZ2_ISA="scalar"),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ref.stdout.split()[1] == digest  # identical to the scalar path whatever the CPU selected
    assert sel in ("scalar", "avx2", "avx512")
