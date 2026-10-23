# SPDX-License-Identifier: MIT
"""singlet.v2.atlas: SCX1 shards, label sidecars, the index builder, queries and gather, on the synthetic v2 study of ``_v2_synth``
with deterministic fake labels (``_cidx_fixture``).

Ground truth is computed independently: counts and cell lists from the label files and cells tables by brute force, gathered
matrices from ``singlet.v2.client``'s per-sample reader for the same columns. The Worker side is tested in the website repository
against golden files written from this same fixture."""

from __future__ import annotations

import io
import json
import pathlib
import struct
import sys

import numpy as np
import pytest

try:
    import singlet  # noqa: F401
except ImportError:  # pragma: no cover
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

pytest.importorskip("singlet._pz2", reason="singlet._pz2 is not built")
pytest.importorskip("zstandard")
pytest.importorskip("pyarrow")
pytest.importorskip("anndata")

import _cidx_fixture as fx  # noqa: E402
from singlet.v2 import atlas  # noqa: E402
from singlet.v2 import client  # noqa: E402


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("cidx")
    try:
        s, res, idx = fx.build(str(root))
    except IndexError:  # pragma: no cover - packs not shipped
        pytest.skip("vocabulary packs are not available")
    return root, s, res, idx


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setenv("SINGLET_CACHE_DIR", str(tmp_path / "cache"))
    client.clear_memory_cache()
    yield


def _truth(root, s):
    """Brute-force per-cell table: (gsm, col, ct id, tissue id, sex, anc, n_umi, n_genes, ct_score, anc_prob)."""
    import pandas as pd

    rows = []
    for gsm, t in sorted(s.truth.items()):
        p = pathlib.Path(root) / "labels" / f"{gsm}.labels.parquet"
        if not p.exists():
            continue
        lab = pd.read_parquet(p)
        cols = lab["col"].to_numpy()
        umi_e = t["umi_exon"][cols].astype(np.int64)
        umi_i = np.asarray(t["intron"].sum(axis=0)).ravel()[cols] if t["intron"] is not None else np.zeros(len(cols))
        feat = np.diff(t["exon"].indptr)[cols]
        for k, c in enumerate(cols):
            rows.append(
                dict(
                    gsm=gsm,
                    col=int(c),
                    ct=lab["cell_type_cl"][k],
                    tissue=fx.TISSUE[gsm][0],
                    sex=lab["sex_chrom"][k],
                    anc=lab["ancestry_hancestro"][k],
                    n_umi=int(umi_e[k] + umi_i[k]),
                    n_genes=int(feat[k]),
                    ct_score=float(lab["cell_type_score"][k]),
                    anc_prob=float(lab["ancestry_prob"][k]),
                )
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------- SCX1
@pytest.mark.parametrize("n,gr", [(0, 8), (1, 8), (1000, 64), (70000, 65536)])
def test_scx_roundtrip(n, gr):
    rng = np.random.default_rng(n)
    cols = {
        "const": np.full(n, 7, np.uint8),
        "runs": np.repeat(np.arange(max(1, n // 50) + 1, dtype=np.uint32), 50)[:n],
        "small": rng.integers(0, 5, n).astype(np.uint16),
        "wide": rng.integers(0, 2**32, n, dtype=np.uint64).astype(np.uint32),
        "mixed": rng.integers(1000, 400000, n).astype(np.uint32),
    }
    blob = atlas.encode_shard(cols, group_rows=gr)
    sh = atlas.Shard(blob)
    assert sh.n_rows == n and sh.n_groups == -(-n // gr)
    got = sh.table()
    for k, v in cols.items():
        assert got[k].dtype == v.dtype
        np.testing.assert_array_equal(got[k], v)
    for g in range(sh.n_groups):
        lo, hi = sh.zone(g, "mixed")
        part = cols["mixed"][g * gr : (g + 1) * gr]
        assert (lo, hi) == (part.min(), part.max())
    # header-only reader with ranged fetches gives the same values
    hl = atlas.scx_header_len(blob[:64])
    sh2 = atlas.Shard(blob[:hl], fetch=lambda off, ln: blob[off : off + ln])
    if n:
        np.testing.assert_array_equal(sh2.column(0, "wide"), cols["wide"][:gr])


def test_scx_rejects_bad_input():
    with pytest.raises(atlas.ShardError):
        atlas.encode_shard({"x": np.zeros(3, np.int64)})
    with pytest.raises(atlas.ShardError):
        atlas.encode_shard({"a": np.zeros(3, np.uint8), "b": np.zeros(4, np.uint8)})
    with pytest.raises(atlas.ShardError):
        atlas.Shard(b"NOPE" + b"\0" * 60)
    blob = atlas.encode_shard({"a": np.arange(100, dtype=np.uint32)}, group_rows=64)
    with pytest.raises(atlas.ShardError):
        atlas.Shard(blob[:80])


def test_q8_and_thresholds():
    v = np.array([0, 0.5, 1, np.nan, 0.123])
    q = atlas.q8(v)
    assert q.tolist() == [0, 127, 254, 255, 31]
    # score >= 0.5 at u8 resolution keeps exactly the codes >= ceil(0.5*254)
    assert atlas.q8_threshold(0.5, upper=False) == 127
    assert atlas.q8_threshold(0.5, upper=True) == 127


# ----------------------------------------------------------------------------------------------------------------------- sidecars
def test_normalize_sidecar(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    called = np.zeros(20, bool)
    called[[1, 3, 5, 7]] = True
    t = pa.table(
        {
            "gsm": ["GSM1"] * 5,
            "col": [7, 3, 1, 5, 2],
            "cell_type_cl": ["CL:0000084", None, "CL:0000236", "CL:0000084", "CL:0000084"],
            "cell_type_label": ["T cell", None, "B cell", "T cell", "T cell"],
            "cell_type_score": np.array([0.9, np.nan, 0.5, 0.7, 0.1], np.float32),
            "sex_chrom": ["female", "female", None, "ambiguous", "male"],
            "sex_score": np.array([0.9] * 5, np.float32),
            "ancestry_hancestro": ["HANCESTRO:0005"] * 5,
            "ancestry_prob": np.array([0.8] * 5, np.float32),
            "model_version": ["m1"] * 5,
        }
    )
    p = tmp_path / "GSM1.labels.parquet"
    pq.write_table(t, p)
    data, entry = atlas.normalize_sidecar(str(p), gsm="GSM1", n_columns=20, called=called, cells_digest="ab" * 16, annot_version="a1")
    assert entry["key"] == atlas.sidecar_key("GSM1", data) and entry["key"].startswith("v2/a/GSM1/")
    assert entry["n_rows"] == 4 and entry["dropped"] == 1  # col 2 is not called
    back = atlas.read_sidecar(data)
    assert back["col"].tolist() == [1, 3, 5, 7]
    assert back["cell_type_cl"] == ["CL:0000236", None, "CL:0000084", "CL:0000084"]
    md = pq.read_table(pa.BufferReader(data)).schema.metadata
    assert md[b"gsm"] == b"GSM1" and md[b"annot_version"] == b"a1"
    # deterministic bytes -> stable content address
    assert atlas.normalize_sidecar(str(p), gsm="GSM1", n_columns=20, called=called, cells_digest="ab" * 16, annot_version="a1")[0] == data
    m = atlas.attach_annotations({"samples": [{"gsm": "GSM1"}, {"gsm": "GSM2"}]}, {"GSM1": entry}, "a1")
    assert m["annotations_default"] == "a1" and "a1" in m["samples"][0]["annotations"] and "annotations" not in m["samples"][1]
    for bad in (
        {"col": [99]},
        {"col": [1, 1]},
        {"col": [1], "cell_type_cl": ["T cell"]},
        {"col": [1], "sex_chrom": ["XY"]},
        {"col": [1], "cell_type_score": [1.5]},
        {"col": [1], "gsm": ["GSM2"]},
    ):
        bio = io.BytesIO()
        pq.write_table(pa.table(bad), bio)
        with pytest.raises(atlas.IndexError_):
            atlas.normalize_sidecar(bio.getvalue(), gsm="GSM1", n_columns=20, annot_version="a1")


# ----------------------------------------------------------------------------------------------------------------------- builder
def test_build_layout_and_totals(built):
    root, s, res, idx = built
    d = res.directory
    n_called = sum(int(t["called"].sum()) for t in s.truth.values())
    assert d["totals"]["cells"] == n_called == idx.count({})
    assert d["version"].startswith("20261101T0000Z-")
    for k in res.objects:
        assert (pathlib.Path(root) / k).exists()
        if k.startswith("v2/cidx/obj/"):
            assert k == atlas.obj_key(res.objects[k], k.rsplit(".", 1)[1])
    cur = json.loads((pathlib.Path(root) / "v2/cidx/current.json").read_text())
    assert cur["version"] == d["version"]
    assert len(d["shards"]) > 1  # shard_rows=128 in the fixture: several shards, several groups each
    # a rebuild of the same inputs is the same index (content-addressed objects, same version digest)
    _s2, res2, _ = fx.build(str(root))
    assert res2.version == res.version and sorted(res2.objects) == sorted(res.objects)


def test_dictionaries_follow_the_ontology(built):
    _root, _s, res, _idx = built
    ct = res.directory["dicts"]["cell_type"]
    ids = [e["id"] for e in ct[1:]]
    # primary-parent DFS order: T cell, then its descendants, are neighbours
    t = ids.index("CL:0000084")
    assert set(ids[t : t + 3]) == {"CL:0000084", "CL:0000624", "CL:0000625"}
    cd4 = next(e for e in ct if e["id"] == "CL:0000624")
    assert "CL:0000084" in cd4["anc"] and "CL:0000542" in cd4["anc"]
    assert res.directory["terms"]["CL:0000542"] == "lymphocyte"


def test_ordinals_are_kept_by_a_later_version(built, tmp_path):
    root, _s, res, _idx = built
    import _v2_synth as synth_mod

    res2 = atlas.build_index([synth_mod.GSE], base=str(root), previous=res.version, now="20261102T0000Z")
    a = atlas.Shard(res.objects[res.directory["samples"]["key"]]).table()
    b = atlas.Shard(res2.objects[res2.directory["samples"]["key"]]).table()
    assert dict(zip(a["gsm"].tolist(), a["so"].tolist())) == dict(zip(b["gsm"].tolist(), b["so"].tolist()))


# ----------------------------------------------------------------------------------------------------------------------- queries
QUERIES = [
    ({}, lambda t: t),
    ({"cell_type": "CL:0000084"}, lambda t: t[t.ct.isin(["CL:0000084", "CL:0000624", "CL:0000625"])]),
    ({"cell_type": "lymphocyte"}, lambda t: t[t.ct != "CL:0000576"]),
    ({"tissue": "UBERON:0002048"}, lambda t: t[t.tissue.isin(["UBERON:0002048", "UBERON:0002185"])]),
    ({"tissue": "bronchus", "sex": "ambiguous"}, lambda t: t[(t.tissue == "UBERON:0002185") & (t.sex == "ambiguous")]),
    ({"ranges": {"n_umi": [3000, None]}}, lambda t: t[t.n_umi >= 3000]),
    ({"ranges": {"n_genes": [150, 400]}}, lambda t: t[(t.n_genes >= 150) & (t.n_genes <= 400)]),
    (
        {"cell_type": "B cell", "ranges": {"cell_type_score": [0.5, None]}},
        lambda t: t[(t.ct == "CL:0000236") & (np.rint(t.ct_score * 254) >= 127)],
    ),
    ({"ancestry": "HANCESTRO:0009", "ranges": {"ancestry_prob": [None, 0.8]}}, lambda t: t[(t.anc == "HANCESTRO:0009") & (np.rint(t.anc_prob * 254) <= 203)]),
    ({"samples": ["GSM9000003"]}, lambda t: t[t.gsm == "GSM9000003"]),
    ({"studies": ["GSE900001"], "ranges": {"n_umi": [0, 2500]}}, lambda t: t[t.n_umi <= 2500]),
]


@pytest.mark.parametrize("where,expect", QUERIES)
def test_queries_match_brute_force(built, where, expect):
    root, s, _res, idx = built
    t = _truth(root, s)
    want = expect(t).sort_values(["gsm", "col"])
    assert idx.count(where) == len(want)
    got = idx.find(where, limit=None).sort_values(["gsm", "col"])
    assert got["gsm"].tolist() == want["gsm"].tolist()
    assert got["col"].tolist() == want["col"].tolist()


def test_count_by_and_postings(built):
    root, s, _res, idx = built
    t = _truth(root, s)
    by = idx.count({"organism": "Homo sapiens"}, by=["cell_type", "sex"])
    exp = t.assign(sex=t.sex.map({"female": "PATO:0000383", "male": "PATO:0000384", "ambiguous": "ambiguous"})).groupby(["ct", "sex"]).size()
    got = by.set_index(["cell_type", "sex"])["n_cells"]
    assert got.sort_index().to_dict() == exp.sort_index().to_dict()
    # the same answer with and without per-cell ranges (summary vs scan)
    a = idx.count({"cell_type": "T cell"}, by=["sample"])
    b = idx.count({"cell_type": "T cell", "ranges": {"n_umi": [0, None]}}, by=["sample"])
    assert a.sort_values("sample").reset_index(drop=True).equals(b.sort_values("sample").reset_index(drop=True))
    with pytest.raises(atlas.IndexError_):
        idx.count({}, by=["colour"])
    with pytest.raises(atlas.IndexError_):
        idx.count({"organism": "Felis catus"})
    with pytest.raises(atlas.IndexError_):
        atlas.Query.from_json({"ranges": {"n_mito": [0, 1]}})


def test_find_cells_direct_entry_point(built):
    root, _s, _res, _idx = built
    df = atlas.find_cells(cell_type="CL:0000625", min_umi=1000, base=str(root))
    assert len(df) and set(df.columns) >= {"so", "gsm", "col", "cell_type", "n_umi_exon"}
    assert (df.n_umi_exon + df.n_umi_intron >= 1000).all()
    assert atlas.count_cells(cell_type="CL:0000625", min_umi=1000, base=str(root)) == len(df)


# ----------------------------------------------------------------------------------------------------------------------- gather
def _check_against_client(root, s, a, df):
    st = client.open_study(fx_gse(), base=str(root))
    kmap, n_genes, ids, _names = client._gene_keep(st, None)
    assert list(ids) == list(a.var_names)
    for gsm in sorted(set(df.gsm)):
        cols = np.sort(df[df.gsm == gsm].col.values)
        sd = client._read_sample(st, st.samples[gsm], sel_fn=lambda rd, c=cols: c, want_cols=("barcode",), keep_map=kmap, n_genes=n_genes, threads=1)
        sub = a[a.obs.gsm == gsm]
        assert sub.obs.col.tolist() == cols.tolist()
        assert (sub.layers["spliced"] != sd.spliced.csr()).nnz == 0
        if sd.unspliced is not None:
            assert (sub.layers["unspliced"] != sd.unspliced.csr()).nnz == 0
            assert (sub.X != (sd.spliced.csr() + sd.unspliced.csr())).nnz == 0
        else:
            assert sub.layers["unspliced"].nnz == 0


def fx_gse():
    import _v2_synth as synth_mod

    return synth_mod.GSE


def test_gather_direct_matches_client(built):
    root, s, _res, idx = built
    df = idx.find({"cell_type": "CL:0000084"}, limit=None)
    a = atlas.gather(df, base=str(root), via="direct", index=idx)
    assert a.n_obs == len(df) and a.uns["singlet_gather"]["via"] == "direct"
    _check_against_client(root, s, a, df)
    # second time the prefixes come from the disk cache: fewer bytes, same matrices
    b = atlas.gather(df, base=str(root), via="direct", index=idx)
    assert b.uns["singlet_gather"]["planned_bytes"] < a.uns["singlet_gather"]["planned_bytes"]
    assert (a.X != b.X).nnz == 0


def test_gather_batches_split_samples_and_columns(built):
    root, s, _res, idx = built
    df = idx.find({}, limit=None)
    one = atlas.gather(df, base=str(root), via="direct", index=idx, modalities=["spliced"])
    many = atlas.gather(df, base=str(root), via="direct", index=idx, modalities=["spliced"], batch_cells=37)
    assert many.uns["singlet_gather"]["batches"] > 2
    assert one.obs_names.tolist() == many.obs_names.tolist()
    assert (one.layers["spliced"] != many.layers["spliced"]).nnz == 0
    assert "unspliced" not in one.layers


def test_gather_inputs_and_errors(built):
    root, _s, _res, idx = built
    df = idx.find({"samples": ["GSM9000001"]}, limit=10)
    a = atlas.gather([("GSM9000001", df.col.tolist())], base=str(root), via="direct", index=idx, modalities=["counts"])
    assert a.n_obs == 10 and "spliced" not in a.layers
    with pytest.raises(atlas.IndexError_, match="max_cells"):
        atlas.gather(df, base=str(root), via="direct", index=idx, max_cells=5)
    with pytest.raises(atlas.IndexError_, match="psi"):
        atlas.normalize_modalities(["psi"])
    with pytest.raises(atlas.IndexError_):
        atlas.gather([("GSM9000001", [10**6])], base=str(root), via="direct", index=idx)


def _stream(plan, got_bytes, names, trailer=None):
    """An SGA1 stream as the Worker writes it, from a plan and the object bytes (for the stream parser)."""
    head = json.dumps({"member_names": names, "samples": plan["samples"]}).encode()
    out = bytearray(b"SGA1" + struct.pack("<I", len(head)) + head)
    for r in plan["reads"][::-1]:  # completion order is arbitrary
        buf = got_bytes[r["key"]]
        for nm, kind, off, ln, a in r["pieces"]:
            out += atlas.FRAME.pack(r["sample"], names.index(nm), kind, 0, off, ln) + buf[a : a + ln]
    t = json.dumps(trailer or {"ok": True}).encode()
    out += atlas.FRAME.pack(0, 0, atlas.FR_END, 0, 0, len(t)) + t
    return io.BytesIO(bytes(out))


def test_gather_stream_parser(built):
    root, s, _res, idx = built
    df = idx.find({"cell_type": "CL:0000625"}, limit=None)
    want = atlas._cells_by_sample(idx, df)
    ent = [idx.sample(x) for x in sorted(want)]
    members = list(atlas.members_for(["counts"]))
    plan = atlas.plan_gather(ent, want, members)
    objs = {t["key"]: t["object"] for t in s.truth.values()}
    head, got, trailer = atlas.read_gather_stream(_stream(plan, objs, members))
    direct = atlas._fetch_direct(idx, plan)
    assert trailer["ok"] and set(got) == set(direct)
    for k in got:
        assert bytes(got[k]["prefix"]) == bytes(direct[k]["prefix"])
        assert {o: bytes(b) for o, b in got[k]["runs"].items()} == {o: bytes(b) for o, b in direct[k]["runs"].items()}
    with pytest.raises(atlas.IndexError_):
        atlas.read_gather_stream(io.BytesIO(b"SGA1\x10\x00\x00\x00{}"))
