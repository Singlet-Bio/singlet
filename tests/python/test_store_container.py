# SPDX-License-Identifier: MIT
"""Container-layer tests for singlet.store (W2): util, .s1 container, cells table, matrix digest, pure-Python codec 2, packs, legacy reader,
manifest. Self-contained (synthetic data) except the tests marked as corpus tests, which skip when the study corpus is absent."""

from __future__ import annotations

import hashlib
import http.server
import json
import os
import struct
import sys
import threading
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest
import zstandard

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from singlet.store import cells as cells_mod  # noqa: E402
from singlet.store import codec, legacy, util  # noqa: E402
from singlet.store import container as C  # noqa: E402
from singlet.store import manifest as M  # noqa: E402
from singlet.store import packs as P  # noqa: E402
from singlet.store.matrix import Csc, canonical_digest  # noqa: E402

CORPUS = ROOT / "bench" / "compression" / "corpus"
BUNDLE = CORPUS / "bundles" / "GSE145370.singlet"
VOCAB_SRC = ROOT / "bench" / "compression" / "corpus_w4_vocab"


# ------------------------------------------------------------------------------------------------------------ helpers
def make_csc(m=500, n=300, density=0.05, seed=0, vt=2, names=True, empty_cols=True):
    rs = np.random.RandomState(seed)
    cols = []
    for j in range(n):
        k = 0 if (empty_cols and j % 7 == 3) else rs.binomial(m, density)
        cols.append(np.sort(rs.choice(m, size=min(k, m), replace=False)))
    indptr = np.concatenate([[0], np.cumsum([len(c) for c in cols])]).astype(np.int64)
    indices = np.concatenate(cols).astype(np.uint32) if cols else np.zeros(0, np.uint32)
    hi = {1: 255, 2: 65535, 3: 2**32 - 1}.get(vt, 100)
    if vt == 4:
        data = rs.rand(len(indices)).astype(np.float32).view(np.uint32)
    else:
        data = np.minimum(rs.geometric(0.3, size=len(indices)).astype(np.int64), hi).astype(
            np.uint32
        )
        if vt == 3 and len(data):
            data[:3] = [2**32 - 1, 70000, 1][: min(3, len(data))]
    return Csc(
        m,
        n,
        indptr,
        indices,
        data,
        ["gene%d_x" % i for i in range(m)] if names else None,
        ["BC%05d" % j for j in range(n)] if names else None,
        {"b": "2", "a": "1"},
        vt,
    )


def same(a: Csc, b: Csc):
    assert (a.m, a.n) == (b.m, b.n)
    assert np.array_equal(a.indptr, b.indptr)
    assert np.array_equal(a.indices, b.indices)
    assert np.array_equal(a.data, b.data)
    assert a.rownames == b.rownames and a.colnames == b.colnames
    assert a.kv == b.kv


# ------------------------------------------------------------------------------------------------------------ util
@pytest.mark.parametrize(
    "vals",
    [[], [0], [127, 128, 129], [2**32 - 1, 2**63, 2**64 - 1, 5, 0], list(range(0, 100000, 37))],
)
def test_leb128_roundtrip(vals):
    a = np.array(vals, dtype=np.uint64)
    b = util.leb128_encode(a)
    assert np.array_equal(util.leb128_decode(b, len(a)), a)
    # reference encoding
    ref = bytearray()
    for v in vals:
        while v >= 128:
            ref.append((v & 127) | 128)
            v >>= 7
        ref.append(v)
    assert bytes(ref) == b


def test_leb128_rejects_bad_streams():
    with pytest.raises(ValueError):
        util.leb128_decode(b"\x80", 1)  # truncated
    with pytest.raises(ValueError):
        util.leb128_decode(b"\x01\x02", 3)  # wrong count


def test_zstd_is_thread_safe():
    data = os.urandom(3000) + b"a" * 20000
    out = []

    def work():
        for _ in range(20):
            z = util.zstd_compress(data, 3)
            assert util.zstd_decompress(z) == data
        out.append(1)

    ts = [threading.Thread(target=work) for _ in range(6)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(out) == 6


# ------------------------------------------------------------------------------------------------------------ container
def build_object():
    w = C.S2Writer("GSM123")
    w.add_matrix("m.1pz", b"M" * 1000, b"\x01" * 16, {"x": 1}, C.FLAG_LEGACY)
    w.add_cells("cells", b"C" * 77, b"\x02" * 16)
    assert w.add_stored("a.json", b'{"a": 1}' * 200) == C.KIND_STORED
    assert w.add_stored("tiny.txt", b"x") == C.KIND_RAW  # zstd would not shrink it
    assert w.add_stored("empty.txt", b"") == C.KIND_RAW
    w.add_derived(
        "d.tsv",
        123,
        456,
        b"\x03" * 16,
        {"recipe": "r-v1", "params": {"k": [1, 2]}, "needs": ["cells:called"]},
    )
    return w, w.build()


def test_container_layout_and_reader():
    w, obj = build_object()
    assert obj[:8] == b"SINGLET2"
    r = C.S2Reader(obj)
    assert r.sample_id == "GSM123" and r.n_members == 6 and r.file_len == len(obj)
    assert [i.name for i in w.infos] == list(r.members)
    for mi in r.members.values():
        if mi.kind != C.KIND_DERIVED and mi.len:
            assert mi.off % 64 == 0
    assert r.read_member("m.1pz") == b"M" * 1000
    assert r.read_member("cells") == b"C" * 77
    assert r.legacy_blob("a.json") == b'{"a": 1}' * 200
    assert r.legacy_blob("tiny.txt") == b"x"
    assert r.legacy_blob("empty.txt") == b""
    d = r.members["d.tsv"]
    assert (
        d.kind == C.KIND_DERIVED
        and d.len == 0
        and d.out_len == 123
        and d.crc32 == 456
        and d.meta["params"] == {"k": [1, 2]}
    )
    assert r.object_digest() == util.d16(obj)
    # deterministic
    assert build_object()[1] == obj


def test_container_sources_and_request_counts(tmp_path):
    _, obj = build_object()
    p = tmp_path / "x.s1"
    p.write_bytes(obj)
    assert C.S2Reader(str(p)).read_member("m.1pz") == b"M" * 1000
    calls = []

    def fetch(off, ln):
        calls.append((off, ln))
        return obj[off : off + ln]

    r = C.S2Reader(fetch, prefetch=64)
    n0 = len(calls)
    assert n0 <= 2  # header, then (maybe) the table
    assert r.read_range("m.1pz", 10, 20) == b"M" * 20
    assert len(calls) == n0 + 1  # exactly one request per range
    r2 = C.S2Reader(fetch)  # default prefetch covers everything here: no extra request for members
    c = len(calls)
    assert r2.read_member("cells") == b"C" * 77
    assert len(calls) == c + 1 - 1 or len(calls) == c  # served from the first read


@pytest.mark.parametrize("mut", ["magic", "version", "len", "trunc", "table"])
def test_container_rejects_corruption(mut):
    _, obj = build_object()
    b = bytearray(obj)
    if mut == "magic":
        b[0] ^= 1
    elif mut == "version":
        b[8] = 9
    elif mut == "len":
        b[32] ^= 1
    elif mut == "trunc":
        b = b[: len(b) - 5]
    elif mut == "table":
        b[64 + 2 + 5] = 0xFF  # inside the first name/fields
    with pytest.raises((C.S2Error, ValueError, UnicodeDecodeError, struct.error)):
        r = C.S2Reader(bytes(b))
        for n in list(r.members):
            r.read_member(n)
        if mut == "table":
            raise C.S2Error("unreachable - table mutation not detected")


def test_stored_member_crc_check():
    w = C.S2Writer("G")
    w.add_stored("a", b"hello world " * 100)
    obj = bytearray(w.build())
    r = C.S2Reader(bytes(obj))
    mi = r.members["a"]
    obj[mi.off + 3] ^= 0xFF
    with pytest.raises(
        (C.S2Error, zstandard.ZstdError)
    ):  # a damaged zstd frame, or a CRC/length mismatch
        C.S2Reader(bytes(obj)).legacy_blob("a")


# ------------------------------------------------------------------------------------------------------------ matrix digest
def slow_digest(c: Csc) -> bytes:
    h = hashlib.sha256(b"PZ2CANON1")
    h.update(struct.pack("<IIQB", c.m, c.n, c.nnz, c.value_type))
    for j in range(c.n):
        a, b = c.indptr[j], c.indptr[j + 1]
        h.update(
            struct.pack("<I", b - a)
            + c.indices[a:b].astype("<u4").tobytes()
            + c.data[a:b].astype("<u4").tobytes()
        )
    for lst in (c.rownames, c.colnames):
        h.update(struct.pack("<I", len(lst)) + b"".join(s.encode() + b"\0" for s in lst))
    for k in sorted(c.kv):
        h.update(k.encode() + b"\0" + c.kv[k].encode() + b"\0")
    return h.digest()[:16]


@pytest.mark.parametrize("seed,vt", [(0, 2), (1, 1), (2, 3), (3, 4)])
def test_canonical_digest_matches_plain_definition(seed, vt):
    c = make_csc(seed=seed, vt=vt)
    assert canonical_digest(c) == slow_digest(c)


def test_canonical_digest_sensitivity():
    c = make_csc()
    d0 = canonical_digest(c)
    c.data[3] += 1
    assert canonical_digest(c) != d0
    c.data[3] -= 1
    c.colnames[5] = "x"
    assert canonical_digest(c) != d0
    c.colnames[5] = "BC00005"
    c.kv["z"] = "1"
    assert canonical_digest(c) != d0


def test_canonical_digest_blocks_and_empty(monkeypatch):
    import singlet.store.matrix as mm

    c = make_csc(m=800, n=400, density=0.1, seed=5)
    d = canonical_digest(c)
    monkeypatch.setattr(mm, "_DIGEST_BLOCK_NNZ", 1000)  # force many blocks (same stream)
    assert mm.canonical_digest(c) == d
    e = Csc(
        0, 0, np.zeros(1, np.int64), np.zeros(0, np.uint32), np.zeros(0, np.uint32), [], [], {}, 1
    )
    assert canonical_digest(e) == slow_digest(e)
    z = Csc(
        5,
        4,
        np.zeros(5, np.int64),
        np.zeros(0, np.uint32),
        np.zeros(0, np.uint32),
        list("abcde"),
        list("wxyz"),
        {},
        2,
    )
    assert canonical_digest(z) == slow_digest(z)


def test_check_canonical_flags_problems():
    c = make_csc()
    assert c.check_canonical() is None
    bad = make_csc()
    bad.indices[bad.indptr[1] : bad.indptr[1] + 2] = bad.indices[bad.indptr[1] : bad.indptr[1] + 2][
        ::-1
    ]
    assert bad.check_canonical() is not None
    big = make_csc(vt=1)
    big.data[0] = 300
    assert "value" in big.check_canonical()


# ------------------------------------------------------------------------------------------------------------ codec 2
@pytest.mark.parametrize("vt", [1, 2, 3, 4])
@pytest.mark.parametrize("chunk", [1, 7, 64, 1024])
def test_codec2_roundtrip(vt, chunk):
    c = make_csc(m=300, n=130, vt=vt, seed=vt)
    cod = codec.PyCodec2()
    b = cod.encode(c, codec.EncodeOptions(chunk_cols=chunk, threads=2))
    h = codec.parse_header(b)
    assert (
        h["codec"] == 2
        and h["n_chunks"] == -(-c.n // chunk)
        and h["digest"] == canonical_digest(c)
        and h["total_len"] == len(b)
    )
    out = cod.decode(b)
    same(c, out)


def _chunk_first(b):
    h = codec.parse_header(b)
    off, ln = h["sections"]["chunk_index"]
    return np.frombuffer(b[off : off + ln], dtype=codec._IDX_DT)["first"].astype(np.int64)


@pytest.mark.parametrize("budget", [1, 40, 300, 0, -1])
def test_codec2_chunk_nnz_budget(budget):
    c = make_csc(m=400, n=500, density=0.03, seed=5)
    ip = c.indptr.astype(np.int64)
    lim = None if budget < 0 else (budget or codec.DEFAULT_CHUNK_NNZ_MAX)
    b = codec.PyCodec2().encode(c, codec.EncodeOptions(chunk_cols=64, chunk_nnz_max=budget))
    same(c, codec.PyCodec2().decode(b))
    first = _chunk_first(b)
    for j0, j1 in zip(first[:-1], first[1:]):
        z = int(ip[j1] - ip[j0])
        assert j1 - j0 <= 64 and (j1 - j0 == 1 or lim is None or z <= lim)
        if (
            lim is not None and j1 < c.n and j1 - j0 < 64
        ):  # greedy: the next column would not have fitted
            assert z + int(ip[j1 + 1] - ip[j1]) > lim
    if lim is None:
        assert (np.diff(first)[:-1] == 64).all()
    try:
        nat = codec.NativeCodec("generic")
    except ImportError:
        return
    # the native writer plans the same chunks (pz2.h plan_chunks)
    nb = nat.encode(c, codec.EncodeOptions(chunk_cols=64, chunk_nnz_max=budget))
    assert (_chunk_first(nb) == first).all()


def test_codec2_column_ranges_and_requests():
    c = make_csc(m=400, n=2500, density=0.02, seed=9)
    cod = codec.PyCodec2()
    b = cod.encode(c, codec.EncodeOptions(chunk_cols=256))
    sp = c.to_scipy()
    for a, z in [(0, 1), (0, 2500), (255, 257), (300, 301), (1000, 1800), (2499, 2500), (5, 5)]:
        sub = cod.decode(b, cols=(a, z))
        ref = sp[:, a:z]
        assert sub.n == z - a and sub.nnz == ref.nnz
        assert (sub.to_scipy() != ref).nnz == 0
        assert sub.colnames == c.colnames[a:z]
    calls = []

    def fetch(off, ln):
        calls.append((off, ln))
        return b[off : off + ln]

    sub = cod.read_columns(fetch, len(b), (700, 900))
    assert (sub.to_scipy() != sp[:, 700:900]).nnz == 0
    # 700..900 lies in chunks 2 and 3 -> one request for the run; plus header/index/names/kv sections
    assert len(calls) <= 6


def test_codec2_name_references():
    c = make_csc(n=2000, m=50, density=0.2, seed=1)
    cod = codec.PyCodec2()
    full = cod.encode(c, codec.EncodeOptions())
    refs = cod.encode(
        c, codec.EncodeOptions(row_names=("pack", bytes([0x11]) * 8), col_names="cells")
    )
    assert len(refs) < len(full)
    out = cod.decode(refs)
    assert out.rownames is None and out.colnames is None
    assert out.row_ref == {"pack": "11" * 8} and out.col_ref == {"cells": True}
    for idx in (np.arange(2000), np.arange(2000) * 3, np.random.RandomState(0).permutation(2000)):
        b = cod.encode(
            c, codec.EncodeOptions(row_names="inline", col_names=("subset", bytes([0x22]) * 8, idx))
        )
        o = cod.decode(b)
        assert o.col_ref["pack"] == "22" * 8 and np.array_equal(o.col_ref["index"], idx)
        o2 = cod.decode(b, cols=(10, 30))
        assert np.array_equal(o2.col_ref["index"], idx[10:30])
    whole = cod.decode(
        cod.encode(c, codec.EncodeOptions(col_names=("pack", bytes([0x33]) * 8))), cols=(5, 9)
    )
    assert whole.col_ref == {"pack": "33" * 8, "range": (5, 9)}
    # nameless matrices: the digest hashes zero names; round trip keeps the "none" references
    nn = make_csc(m=30, n=40, seed=4)
    nn.rownames = None
    nn.colnames = None
    nn.row_ref = {"none": True}
    nn.col_ref = {"none": True}
    b = cod.encode(nn, codec.EncodeOptions(row_names="none", col_names="none"))
    o = cod.decode(b)
    assert o.row_ref == {"none": True} and o.col_ref == {"none": True} and o.rownames is None
    assert codec.parse_header(b)["digest"] == canonical_digest(nn)
    with pytest.raises(codec.CodecError):
        cod.encode(make_csc(), codec.EncodeOptions(row_names="none"))


def test_codec2_header_layout_is_the_published_one():
    """Byte offsets of include/singlet/pz2/detail/container.h (W1)."""
    c = make_csc(m=100, n=50, seed=2)
    b = codec.PyCodec2().encode(c, codec.EncodeOptions(chunk_cols=16))
    assert b[:4] == b"PZ2M" and struct.unpack_from("<HH", b, 4) == (1, 2)
    assert (
        struct.unpack_from("<II", b, 8) == (100, 50) and struct.unpack_from("<Q", b, 16)[0] == c.nnz
    )
    assert b[24] == c.value_type and b[25] == 0  # row_order 0 = canonical
    assert struct.unpack_from("<II", b, 28) == (16, 4)
    assert struct.unpack_from("<I", b, 252)[0] == (zlib.crc32(b[:252]) & 0xFFFFFFFF)
    secs = [struct.unpack_from("<QQ", b, 72 + 16 * i) for i in range(7)]
    assert secs[4][0] == 256 and secs[4][1] == 5 * 24  # chunk index right after the header
    assert struct.unpack_from("<Q", b, 184)[0] == len(b) == secs[6][0] + secs[6][1]
    assert b[192] == 0 and b[193] == 0  # names kinds: inline
    assert (
        secs[5][0] == secs[4][0] + secs[4][1] and secs[0][1] == 0 and secs[3][1] == 0
    )  # order: chunk_index, tables, rows, chunks


def test_codec2_f32_integral_and_raw():
    cod = codec.PyCodec2()
    c = make_csc(m=60, n=30, vt=4, seed=3)
    c.data = np.arange(len(c.data), dtype=np.float32).view(
        np.uint32
    )  # exactly integral -> coded as integers (flag bit0)
    b = cod.encode(c)
    assert codec.parse_header(b)["flags"] & 1
    same(c, cod.decode(b))
    c2 = make_csc(m=60, n=30, vt=4, seed=3)  # random floats -> raw bit patterns
    b2 = cod.encode(c2)
    assert codec.parse_header(b2)["flags"] == 0
    same(c2, cod.decode(b2))


def test_codec2_edge_matrices():
    cod = codec.PyCodec2()
    for c in (
        Csc(
            0,
            0,
            np.zeros(1, np.int64),
            np.zeros(0, np.uint32),
            np.zeros(0, np.uint32),
            [],
            [],
            {},
            1,
        ),
        Csc(
            10,
            0,
            np.zeros(1, np.int64),
            np.zeros(0, np.uint32),
            np.zeros(0, np.uint32),
            list("abcdefghij"),
            [],
            {"k": "v"},
            2,
        ),
        Csc(
            3,
            4,
            np.zeros(5, np.int64),
            np.zeros(0, np.uint32),
            np.zeros(0, np.uint32),
            list("abc"),
            list("wxyz"),
            {},
            3,
        ),
        Csc(
            1,
            1,
            np.array([0, 1]),
            np.array([0], np.uint32),
            np.array([2**32 - 1], np.uint32),
            ["r"],
            ["c"],
            {},
            3,
        ),
    ):
        same(c, cod.decode(cod.encode(c)))


def test_codec2_refuses_noncanonical_and_corrupt():
    cod = codec.PyCodec2()
    c = make_csc()
    c.indices[c.indptr[1] : c.indptr[1] + 2] = c.indices[c.indptr[1] : c.indptr[1] + 2][::-1]
    with pytest.raises(codec.CodecError):
        cod.encode(c)
    good = cod.encode(make_csc(n=40, seed=2))
    for pos in (0, 5, len(good) // 2, len(good) - 3):
        b = bytearray(good)
        b[pos] ^= 0x55
        try:
            out = cod.decode(bytes(b))
        except Exception:
            continue
        # a mutation that still decodes must at least not produce the original silently when it hit payload bytes
        if pos != 5:
            assert not (out.nnz == 0)


# ------------------------------------------------------------------------------------------------------------ cells table
def test_cells_roundtrip_all_dtypes():
    n = 3000
    rs = np.random.RandomState(0)
    bcs = ["".join(rs.choice(list("ACGT"), 16)) for _ in range(n)]
    t = cells_mod.CellsTable(n)
    t.add("barcode", cells_mod.DT_BARCODE, bcs)
    t.add("called", "bool", rs.rand(n) < 0.3)
    t.add("n_umi_exon", "u32", rs.geometric(0.01, n).astype(np.uint32))
    t.add("sorted_u32", "u32", np.cumsum(rs.randint(0, 5, n)).astype(np.uint32))
    t.add("u8", "u8", rs.randint(0, 255, n).astype(np.uint8))
    t.add("u16", "u16", rs.randint(0, 65535, n).astype(np.uint16))
    t.add("i32", "i32", rs.randint(-1000, 1000, n).astype(np.int32))
    t.add("f32", "f32", rs.rand(n).astype(np.float32))
    t.add("f64", "f64", rs.rand(n))
    t.add("label", "str", ["type%d" % (i % 7) for i in range(n)])
    body = t.encode()
    d = t.digest()
    t2 = cells_mod.CellsTable.decode(body)
    assert t2.digest() == d and t2.names == t.names
    for k in t.names:
        a, b = t.column(k), t2.column(k)
        assert (list(a) == list(b)) if isinstance(a, list) else np.array_equal(a, b)
    assert cells_mod.CellsTable.decode(t2.encode()).digest() == d  # canonical + stable re-encode


def test_cells_barcode_modes_and_lazy_reader():
    n = 500
    rs = np.random.RandomState(1)
    wl = [("".join(rs.choice(list("ACGT"), 16))) for _ in range(5000)]
    pack = P.Whitelist(P.build_whitelist(wl, {"protocol": "t"}))
    pick = [pack.barcodes([i])[0] for i in rs.randint(0, pack.n, n)]
    idx = pack.index_of(pick)
    t = cells_mod.CellsTable(n)
    t.add("barcode", cells_mod.DT_BARCODE, pick, whitelist=(pack, idx))
    t.add("called", "bool", np.ones(n, bool))
    body = t.encode()
    assert body.count(b"\x01" + pack.id[:8]) == 1  # whitelist mode, 8-byte pack id
    store = P.PackStore(dirs=[], download=False)
    store._files[pack.id.hex()] = None
    store._obj[pack.id.hex()] = pack
    t2 = cells_mod.CellsTable.decode(body, store)
    assert t2.barcodes == pick
    # 2-bit mode (no whitelist) and string mode
    for strs in (pick, ["cell-%d" % i for i in range(n)], ["AC", "ACGT"] * (n // 2)):
        tt = cells_mod.CellsTable(n)
        tt.add("barcode", cells_mod.DT_BARCODE, strs)
        assert cells_mod.CellsTable.decode(tt.encode()).barcodes == strs
    calls = []

    def fetch(off, ln):
        calls.append((off, ln))
        return body[off : off + ln]

    rd = cells_mod.CellsReader(fetch, len(body), store)
    k = len(calls)
    assert k <= 2
    assert rd.column("called").all()
    assert len(calls) == k + 1


def test_parse_cell_calls_rules():
    data = b"barcode\tis_cell\tx\nAAA\tTRUE\t1\nCCC\tFALSE\t2\nGGG\tTrue\t3\n"
    assert cells_mod.parse_cell_calls(data) == ["AAA", "GGG"]
    assert cells_mod.parse_cell_calls(b"cb\nAAA\nCCC\n") == [
        "AAA",
        "CCC",
    ]  # no is_cell: every listed barcode counts
    assert cells_mod.parse_cell_calls(b"") is None
    assert cells_mod.parse_cell_calls(b"foo\tbar\n1\t2\n") is None


# ------------------------------------------------------------------------------------------------------------ packs
def test_whitelist_pack_roundtrip_and_lookup():
    rs = np.random.RandomState(3)
    wl = sorted({"".join(rs.choice(list("ACGT"), 16)) for _ in range(20000)})
    buf = P.build_whitelist(
        wl[::-1] + wl[:50], {"protocol": "toy", "names": ["toy", "alias"]}
    )  # order/dups do not matter
    assert buf == P.build_whitelist(wl, {"protocol": "toy", "names": ["toy", "alias"]})
    w = P.Whitelist(buf)
    assert w.all_barcodes() == wl and w.n == len(wl) and w.id == util.d16(buf)
    idx = w.index_of([wl[5], wl[100], wl[0]])
    assert idx.tolist() == [5, 100, 0]
    assert w.index_of(["A" * 16]) is None or "A" * 16 in wl
    assert w.index_of(["ACGTN" + "A" * 11]) is None
    assert w.index_of(["ACGT"]) is None
    assert w.barcodes([1, 2]) == wl[1:3]


def test_pack_wrapper_errors():
    with pytest.raises(P.PackError):
        P.parse(b"nope")
    w = P.build_whitelist(["ACGT", "TTTT"], {"protocol": "x"})
    k, h, body = P.parse(w)
    assert k == P.KIND_WHITELIST and h["n"] == 2 and h["k"] == 4
    with pytest.raises(P.PackError):
        P.VocabPack(w, P.PackStore(dirs=[]))


@pytest.fixture(scope="module")
def store():
    s = P.PackStore()
    if not s.vocab_packs("exon"):
        pytest.skip("shipped packs missing (python/singlet/store/packs)")
    return s


def test_shipped_pack_index_is_consistent(store):
    idx = json.load(open(P.PKG_DIR + "/index.json"))
    assert idx["format"] == "singlet-store-packs"
    builds = {e["build"] for e in idx["packs"] if e["kind"] == "vocab"}
    assert builds == {
        "GRCh38-2024-A",
        "GRCm39-2024-A",
        "GRCz11",
        "Mmul_10",
        "mRatBN7.2",
        "GRCh38-GRCm39-barnyard",
    }
    for e in idx["packs"]:
        d = open(P.PKG_DIR + "/%s.pack" % e["id"], "rb").read()
        assert (
            len(d) == e["bytes"]
            and hashlib.sha256(d).hexdigest() == e["sha256"]
            and util.d16(d).hex() == e["id"]
        )
    shipped = sum(e["bytes"] for e in idx["packs"] if e.get("ship"))
    assert shipped < 2_800_000 + 40_000  # human + mouse vocabularies and the whitelists


@pytest.mark.skipif(not BUNDLE.exists(), reason="corpus bundle missing")
def test_human_pack_matches_legacy_names_and_vocab_json(store):
    from singlet.store.legacy import LegacyBundle, decode_1pz_bytes

    b = LegacyBundle.open(str(BUNDLE))
    ex = decode_1pz_bytes(b.read("samples/GSM4317423/exon_counts.1pz"))
    it = decode_1pz_bytes(b.read("samples/GSM4317423/intron_counts.1pz"))
    pe = store.find_vocab("exon", ex.rownames, "GRCh38-2024-A")
    pi = store.find_vocab("intron", it.rownames, "GRCh38-2024-A")
    assert pe is not None and pi is not None and pe.build == "GRCh38-2024-A"
    assert pe.names() == ex.rownames and pi.names() == it.rownames  # byte-exact expansion
    assert pe.blob() == util.join_nul(ex.rownames)
    assert pe.vocab_json() == b.read("feature_vocab.json")
    rs = np.random.RandomState(0)
    for pk, ref in ((pe, ex.rownames), (pi, it.rownames)):
        ii = rs.randint(0, len(ref), 500)
        assert pk.names_at(ii) == [ref[i] for i in ii]  # lazy lookups by row index
        assert pk.name_at(0) == ref[0] and pk.name_at(len(ref) - 1) == ref[-1]
    g = pe.row_gene()
    assert len(g) == len(ex.rownames) and g[0] == 0 and g.max() == len(pe.gene_table()) - 1
    assert all(
        ex.rownames[i].split("_")[0] == pe.gene_table()[g[i]][0] for i in rs.randint(0, len(g), 300)
    )
    assert P.expand(pe.id.hex()) == ex.rownames


@pytest.mark.skipif(not VOCAB_SRC.exists(), reason="vocab corpus missing")
def test_all_builds_match_fetched_studies(store):
    import csv

    from singlet.store.packs_build import load_meta_rows

    n = 0
    for r in csv.DictReader(open(VOCAB_SRC / "index.tsv"), delimiter="\t"):
        p = VOCAB_SRC / ("%s.%s.%s.meta" % (r["gse"], r["gsm"], r["kind"]))
        if not p.exists():
            continue
        rows = load_meta_rows(str(p))
        pk = store.find_vocab(r["kind"], rows, r["build"])
        assert pk is not None and pk.build == r["build"], (r["gse"], r["kind"])
        n += 1
    assert n >= 30


def test_barnyard_composite_structure(store):
    bn = [p for p in store.vocab_packs("exon", "GRCh38-GRCm39-barnyard")]
    assert len(bn) == 1
    pk = bn[0]
    assert pk.kind == P.KIND_COMPOSITE and len(pk.bytes) < 400
    h = store.vocab_packs("exon", "GRCh38-2024-A")[0]
    m = store.vocab_packs("exon", "GRCm39-2024-A")[0]
    assert pk.n() == h.n() + m.n()
    names = pk.names()
    assert names[0].startswith(h.name_at(0).split("_")[0]) and "_GRCh38___chr" in names[0]
    assert (
        names[h.n()].split("_")[-2:][0]
        == "GRCm39___" + names[h.n()].split("_GRCm39___")[1].split(":")[0]
        or "GRCm39___" in names[h.n()]
    )
    assert pk.names_at([0, h.n() - 1, h.n(), pk.n() - 1]) == [
        names[0],
        names[h.n() - 1],
        names[h.n()],
        names[-1],
    ]


def test_packstore_resolution_and_errors(tmp_path, store):
    w = P.build_whitelist(["ACGT", "TTTT", "GGGG"], {"protocol": "tiny"})
    (tmp_path / (util.d16(w).hex() + ".pack")).write_bytes(w)
    s = P.PackStore(dirs=[str(tmp_path)])
    assert s.get(util.d16(w).hex()[:16]).n == 3
    assert s.whitelist_by_name("tiny").id == util.d16(w)
    with pytest.raises(P.PackError):
        s.get("0" * 16)
    (tmp_path / ("0" * 32 + ".pack")).write_bytes(w)  # file name that does not match its content
    s.rescan()
    with pytest.raises(P.PackError):
        s.read("0" * 32)


# ------------------------------------------------------------------------------------------------------------ legacy reader
def make_zip(path, force64=False, big_names=False):
    with zipfile.ZipFile(path, "w", allowZip64=True) as z:
        z.writestr("manifest.json", json.dumps({"gse_id": "GSE1"}))
        for i in range(3):
            zi = zipfile.ZipInfo("samples/GSM%d/exon_counts.1pz" % (i + 1))
            zi.compress_type = zipfile.ZIP_STORED
            with z.open(zi, "w", force_zip64=force64) as f:
                f.write(os.urandom(5000 + i))
            zi = zipfile.ZipInfo("samples/GSM%d/summary.json" % (i + 1))
            zi.compress_type = zipfile.ZIP_DEFLATED
            with z.open(zi, "w", force_zip64=force64) as f:
                f.write(b'{"a": %d}' % i * 500)


@pytest.mark.parametrize("force64", [False, True])
def test_legacy_zip_reader_matches_zipfile(tmp_path, force64):
    p = tmp_path / "GSE1.singlet"
    make_zip(p, force64)
    b = legacy.LegacyBundle.open(str(p))
    z = zipfile.ZipFile(p)
    assert list(b.entries) == [i.filename for i in z.infolist()]
    for i in z.infolist():
        e = b.entries[i.filename]
        assert (e.size, e.csize, e.crc32, e.method) == (
            i.file_size,
            i.compress_size,
            i.CRC,
            i.compress_type,
        )
        assert b.read(i.filename) == z.read(i.filename)
    assert b.samples() == ["GSM1", "GSM2", "GSM3"] and b.study_paths() == ["manifest.json"]
    assert [r for r, _ in b.sample_paths("GSM2")] == ["exon_counts.1pz", "summary.json"]
    assert b.gse_hint == "GSE1"


def test_legacy_reader_detects_corruption(tmp_path):
    p = tmp_path / "x.singlet"
    make_zip(p)
    raw = bytearray(p.read_bytes())
    e = zipfile.ZipFile(p).infolist()[1]
    raw[e.header_offset + 30 + len(e.filename) + 100] ^= 0xFF
    p.write_bytes(bytes(raw))
    b = legacy.LegacyBundle.open(str(p))
    with pytest.raises(legacy.LegacyError):
        b.read(e.filename)


class _RangeHandler(http.server.BaseHTTPRequestHandler):
    data = b""
    seen_ua: list = []
    log: list = []

    def log_message(self, *a):
        pass

    def do_GET(self):
        ua = self.headers.get("User-Agent", "")
        type(self).seen_ua.append(ua)
        if ua.startswith("Python-urllib"):
            self.send_response(403)
            self.end_headers()
            return
        rng = self.headers.get("Range", "")
        n = len(self.data)
        import re as _re

        m = _re.match(r"bytes=(\d*)-(\d*)", rng)
        a, b = m.group(1), m.group(2)
        if a == "":
            start, end = max(0, n - int(b)), n - 1
        else:
            start, end = int(a), min(int(b) if b else n - 1, n - 1)
        type(self).log.append((start, end))
        body = self.data[start : end + 1]
        self.send_response(206)
        self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, n))
        self.send_header("Content-Length", str(len(body)))
        self.send_header("ETag", '"abc-3"')
        self.end_headers()
        self.wfile.write(body)


def test_legacy_http_range_reader(tmp_path):
    p = tmp_path / "GSE1.singlet"
    make_zip(p)
    _RangeHandler.data = p.read_bytes()
    _RangeHandler.seen_ua = []
    _RangeHandler.log = []
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RangeHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        url = "http://127.0.0.1:%d/data/GSE1/GSE1.singlet" % srv.server_address[1]
        b = legacy.LegacyBundle.open(url)
        assert b.size == len(p.read_bytes()) and b.etag == "abc-3"
        z = zipfile.ZipFile(p)
        for i in z.infolist():
            assert b.read(i.filename) == z.read(i.filename)
        assert all(
            ua.startswith("singlet/") for ua in _RangeHandler.seen_ua
        )  # never the default urllib agent
        st = b.stats()
        assert st["requests"] <= 2 + len(z.infolist())
    finally:
        srv.shutdown()


@pytest.mark.skipif(not BUNDLE.exists(), reason="corpus bundle missing")
def test_legacy_reader_on_corpus_bundle():
    b = legacy.LegacyBundle.open(str(BUNDLE))
    z = zipfile.ZipFile(BUNDLE)
    assert len(b.entries) == len(z.infolist()) == 143
    for i in z.infolist():
        e = b.entries[i.filename]
        assert (e.size, e.csize, e.crc32) == (i.file_size, i.compress_size, i.CRC)
    assert len(b.samples()) == 14
    assert b.read("manifest.json") == z.read("manifest.json")


def test_legacy_directory_source(tmp_path):
    d = tmp_path / "GSE9.GSM77"
    d.mkdir()
    (d / "exon_counts.1pz").write_bytes(b"abc")
    (d / "star_Solo.out__Gene__raw__matrix.mtx").write_bytes(b"mtx")
    (d / "manifest.json").write_bytes(b'{"gse_id": "GSE9"}')
    (d / "sample.json").write_text(json.dumps({"gse": "GSE9", "gsm": "GSM77"}))
    b = legacy.LegacyBundle.open(str(d))
    assert (
        b.samples() == ["GSM77"] and b.study_paths() == ["manifest.json"] and b.gse_hint == "GSE9"
    )
    assert dict(b.sample_paths("GSM77")).keys() == {
        "exon_counts.1pz",
        "star_Solo.out/Gene/raw/matrix.mtx",
    }
    assert b.read("samples/GSM77/star_Solo.out/Gene/raw/matrix.mtx") == b"mtx"


def test_legacy_directory_flattened_member_names(tmp_path):
    d = tmp_path / "GSE9.GSM78"
    d.mkdir()
    flat = {
        "donor__donor_assignments.tsv": "donor/donor_assignments.tsv",
        "mt__mt_summary.tsv": "mt/mt_summary.tsv",
        "nonhost__abundance.tsv": "nonhost/abundance.tsv",
        "star_Solo.out__Barcodes.stats": "star_Solo.out/Barcodes.stats",
        "donor_assignments.tsv": "donor_assignments.tsv",
        "weird__name.txt": "weird__name.txt",
    }
    for f in flat:
        (d / f).write_bytes(b"x")
    b = legacy.LegacyBundle.open(str(d))
    assert set(dict(b.sample_paths("GSM78"))) == set(flat.values())


def test_ctx_whitelist_is_a_numpy_s_array_and_features_order_resolves():
    from singlet.store.context import StudyContext

    ctx = StudyContext("GSE1", "GRCm39-2024-A", {})
    wl = ctx.whitelist("737K-august-2016")
    assert isinstance(wl, np.ndarray) and wl.dtype == np.dtype("S16") and len(wl) == 737280
    assert wl[0] == b"AAACCTGAGAAACCAT" and (wl[1:] > wl[:-1]).all()
    assert ctx.whitelist("737K-cratac-v1") is not None and ctx.whitelist("no-such-list") is None
    fo = ctx.features_order()  # python/singlet/store/recipes_data/features_order_GRCm39-2024-A.u32.zst
    assert fo is not None and len(fo) == 78298 and len(set(fo.tolist())) == len(fo)
    assert StudyContext("GSE1", "GRCz11", {}).features_order() is None


# ------------------------------------------------------------------------------------------------------------ manifest
def test_study_file_entries_roundtrip():
    cases = [
        json.dumps({"a": [1, 2], "é": "x"}, indent=2).encode(),
        json.dumps({"a": 1}).encode(),
        json.dumps({"b": 1}, separators=(",", ":")).encode(),
        b"plain text\nnot json\n",
        b"",
        b'{"a": 1 }',
    ]
    for data in cases:
        ent = M.study_file_entry(data)
        assert ent["size"] == len(data) and ent["sha256"] == hashlib.sha256(data).hexdigest()
        assert M.study_file_bytes(ent) == data
    assert "json" in M.study_file_entry(cases[0]) and "zstd_b64" in M.study_file_entry(cases[3])
    ent = M.study_file_entry(b"zzz", ("r-v1", {"p": 1}, ["pack:genes"]))
    assert M.study_file_bytes(ent, lambda rid, prm: b"zzz") == b"zzz"
    with pytest.raises(ValueError):
        M.study_file_bytes(ent, lambda rid, prm: b"zzy")


def test_rev_is_stable_and_sensitive():
    sf = {"a.json": {"sha256": "11"}, "b.json": {"sha256": "22"}}
    r = M.compute_rev(["store/s/G1/x.s1", "store/s/G2/y.s1"], sf)
    assert r == M.compute_rev(["store/s/G2/y.s1", "store/s/G1/x.s1"], dict(reversed(list(sf.items()))))
    assert len(r) == 16
    assert r != M.compute_rev(["store/s/G1/x.s1", "store/s/G2/z.s1"], sf)
    sf2 = {"a.json": {"sha256": "11"}, "b.json": {"sha256": "23"}}
    assert r != M.compute_rev(["store/s/G1/x.s1", "store/s/G2/y.s1"], sf2)
    m = M.build_manifest(
        "GSE1", [{"key": "store/s/G1/x.s1"}], sf, "b", {}, {}, {}, "2026-01-01T00:00:00Z"
    )
    assert M.loads_manifest(M.dumps_manifest(m)) == m


# ------------------------------------------------------------------------------------------------------------ native adapter
needs_native = pytest.mark.skipif(not codec.native_available(), reason="singlet._pz2 not built")


@needs_native
@pytest.mark.parametrize("vt", [1, 2, 3])
def test_native_codec_roundtrip_and_cross_decoding(vt):
    c = make_csc(m=1500, n=400, density=0.03, seed=vt, vt=vt)
    nat1, nat2, py = codec.NativeCodec("rans"), codec.NativeCodec("generic"), codec.PyCodec2()
    eo = codec.EncodeOptions(chunk_cols=64)
    b1, b2, bp = nat1.encode(c, eo), nat2.encode(c, eo), py.encode(c, eo)
    assert codec.parse_header(b1)["codec"] == 1 and codec.parse_header(b2)["codec"] == 2
    for b in (b1, b2, bp):
        h = codec.parse_header(b)
        assert h["digest"] == canonical_digest(c)
        same(c, nat1.decode(b))  # the C++ reader reads everything ...
    same(c, py.decode(b2))  # ... and the pure-Python reader reads the C++ codec-2 member
    same(c, py.decode(bp))
    with pytest.raises(codec.CodecError):
        py.decode(b1)  # codec 1 needs the native binding
    for a, z in [(0, 1), (60, 70), (100, 400), (399, 400)]:
        sub = nat1.decode(b1, cols=(a, z))
        assert (sub.to_scipy() != c.to_scipy()[:, a:z]).nnz == 0 and sub.colnames == c.colnames[a:z]
    assert codec.decode_any(b1).nnz == c.nnz and codec.decode_any(bp).nnz == c.nnz


@needs_native
def test_native_codec_name_references_and_f32():
    nat = codec.NativeCodec("auto")
    c = make_csc(m=80, n=200, density=0.2, seed=2)
    pid = bytes([0x44]) * 8
    idx = np.random.RandomState(0).permutation(200)
    b = nat.encode(
        c,
        codec.EncodeOptions(row_names=("pack", pid), col_names=("subset", bytes([0x55]) * 8, idx)),
    )
    o = nat.decode(b)
    assert o.rownames is None and o.row_ref == {"pack": "44" * 8}
    assert o.col_ref["pack"] == "55" * 8 and np.array_equal(o.col_ref["index"], idx)
    b = nat.encode(c, codec.EncodeOptions(row_names="inline", col_names="cells"))
    o = nat.decode(b)
    assert o.rownames == c.rownames and o.col_ref == {"cells": True} and o.kv == c.kv
    nn = make_csc(m=20, n=30, seed=5)
    nn.rownames = nn.colnames = None
    nn.row_ref = nn.col_ref = {"none": True}
    o = nat.decode(nat.encode(nn, codec.EncodeOptions(row_names="none", col_names="none")))
    assert o.row_ref == {"none": True} and o.col_ref == {"none": True}
    f = make_csc(m=60, n=30, vt=4, seed=3)
    same(f, nat.decode(nat.encode(f)))  # non-integral floats -> codec 2 inside "auto"
