# SPDX-License-Identifier: MIT
"""singlet.s1.atlas: SCX1 shards, label and feature sidecars, the index builder, queries, seeded sampling and gather, on two
synthetic per-sample studies with deterministic fake labels and features (``_cidx_fixture``).

Ground truth is computed independently: counts and cell lists from the label/feature files and cells tables by brute force, the
sample from the hash definition, gathered matrices from ``singlet.s1.client``'s per-sample reader for the same columns. The Worker
side is tested in the website repository against golden files written from this same fixture."""

from __future__ import annotations

import csv
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
from singlet.s1 import (
    atlas,  # noqa: E402
    client,  # noqa: E402
)


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


def _gse(gsm):
    return fx.GSE2 if gsm in fx.COPIES else "GSE900001"


def _truth(root, s):
    """Brute-force per-cell table of every labelled sample, with features as stored (decoded codes)."""
    import pandas as pd

    feats = {f.name: f for f in atlas.load_features(fx.FEATURES)}
    rows = []
    for gsm in sorted(fx.TISSUE):
        p = pathlib.Path(root) / "labels" / f"{gsm}.labels.parquet"
        if not p.exists():
            continue
        t = s.truth[fx.COPIES.get(gsm, gsm)]
        lab = pd.read_parquet(p)
        cols = lab["col"].to_numpy()
        umi_e = t["umi_exon"][cols].astype(np.int64)
        umi_i = (
            np.asarray(t["intron"].sum(axis=0)).ravel()[cols]
            if t["intron"] is not None
            else np.zeros(len(cols))
        )
        feat = np.diff(t["exon"].indptr)[cols]
        fp = pathlib.Path(root) / "features" / f"{gsm}.features.test.parquet"
        fv = pd.read_parquet(fp).set_index("col") if fp.exists() else None
        for k, c in enumerate(cols):
            r = dict(
                gsm=gsm,
                gse=_gse(gsm),
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
            for name, f in feats.items():
                x = None if fv is None else fv.loc[c, name]
                v = np.nan if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)
                r[name + "_code"] = int(f.encode([v])[0])
            props = fx.PROPS.get(gsm, {})
            r["donor"] = props.get("donor_status", "")
            r["perturb"] = props.get("perturbation", "")
            r["kind"] = props.get("sample_kind", "")
            r["disease"] = props.get("disease", "")
            rows.append(r)
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


def test_codes_thresholds_and_hash():
    q = atlas.q8([0, 0.5, 1, np.nan, 0.123])
    assert q.tolist() == [0, 127, 254, 255, 31]
    # a threshold compares against the code its own value is stored as: a score equal to the threshold always passes
    for v in np.linspace(0, 1, 101):
        assert atlas.q8([v])[0] >= atlas.q8_threshold(v)
    f = atlas.load_features([{"name": "x", "dtype": "u16", "lo": -2, "hi": 2}])[0]
    vals = np.linspace(-2, 2, 977)
    assert (f.encode(vals) >= np.array([f.bound(v, upper=False) for v in vals])).all()
    assert np.allclose(f.decode(f.encode(vals)), vals, atol=4 / 65534)
    assert f.encode([np.nan])[0] == f.na == 65535
    # the hash is pinned (the Worker computes the same values; fixture golden file)
    assert int(atlas.hash32(7, 1000, 123456)) == 1282985747
    h = atlas.hash32(0, np.arange(100000), 0)
    assert len(np.unique(h)) > 99990 and 0.45 < (h < 2**31).mean() < 0.55


def test_feature_registry_validation():
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"name": "n_umi", "version": "1"}])
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"name": "Bad-Name", "version": "1"}])
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"name": "a", "version": "1", "lo": 1, "hi": 1}])
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"name": "a", "version": "1"}, {"name": "a", "version": "2"}])
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"feature_id": "s/c", "kind": "category"}])
    with pytest.raises(atlas.IndexError_):
        atlas.load_features([{"feature_id": "s/c", "kind": "colour"}])
    specs = atlas.load_features(fx.FEATURES)
    assert [f.column for f in specs] == ["f1", "f2", "f3", "f4"]
    assert [f.id for f in specs] == [
        "test/hypoxia",
        "test/malignancy",
        "test/cycling",
        "test/phase",
    ]
    assert (
        specs[2].dtype == "u8"
        and specs[3].dtype == "u16"
        and specs[3].decode([2, 65535]).tolist() == ["G2M", None]
    )
    row = specs[0].cell_feature_row()
    assert (
        row["feature_id"] == "test/hypoxia"
        and row["storage"] == "shard"
        and json.loads(row["thresholds"]) == [0.0, 0.5]
    )
    assert atlas.load_features([{"feature_id": "a/b"}])[0].thresholds == (0.5, 0.8)


def test_estimate_cost_bounds_the_caps():
    # the worst request the caps allow stays under the per-request budget (docs/format-s1-cell-index-costs.md section 3)
    gather = atlas.estimate_cost(reads=4000, mb=1024, do_calls=2, do_writes=1)
    query = atlas.estimate_cost(reads=1100, rows=32_000_000, do_calls=2, do_writes=1)
    assert gather["usd"] < atlas.MAX_COST_USD and query["usd"] < atlas.MAX_COST_USD
    assert atlas.estimate_cost(reads=20_000)["usd"] > atlas.MAX_COST_USD


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
    data, entry = atlas.normalize_sidecar(
        str(p), gsm="GSM1", n_columns=20, called=called, cells_digest="ab" * 16, annot_version="a1"
    )
    assert entry["key"] == atlas.sidecar_key("GSM1", data) and entry["key"].startswith("s1/a/GSM1/")
    assert entry["n_rows"] == 4 and entry["dropped"] == 1
    back = atlas.read_sidecar(data)
    assert back["col"].tolist() == [1, 3, 5, 7]
    assert back["cell_type_cl"] == ["CL:0000236", None, "CL:0000084", "CL:0000084"]
    md = pq.read_table(pa.BufferReader(data)).schema.metadata
    assert md[b"gsm"] == b"GSM1" and md[b"annot_version"] == b"a1"
    assert (
        atlas.normalize_sidecar(
            str(p),
            gsm="GSM1",
            n_columns=20,
            called=called,
            cells_digest="ab" * 16,
            annot_version="a1",
        )[0]
        == data
    )
    m = atlas.attach_annotations(
        {"samples": [{"gsm": "GSM1"}, {"gsm": "GSM2"}]}, {"GSM1": entry}, "a1"
    )
    assert (
        m["annotations_default"] == "a1"
        and "a1" in m["samples"][0]["annotations"]
        and "annotations" not in m["samples"][1]
    )
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
    n_called = sum(len(fx._called(s, g)) for g in fx.TISSUE)
    assert d["totals"]["cells"] == n_called == idx.count({})
    assert d["totals"]["studies"] == 2
    assert d["version"].startswith("20261101T0000Z-")
    for k in res.objects:
        assert (pathlib.Path(root) / k).exists()
        if k.startswith("s1/cidx/obj/"):
            assert k == atlas.obj_key(res.objects[k], k.rsplit(".", 1)[1])
    cur = json.loads((pathlib.Path(root) / "s1/cidx/current.json").read_text())
    assert cur["version"] == d["version"]
    assert len(d["shards"]) > 1
    assert [f["column"] for f in d["features"]] == ["f1", "f2", "f3", "f4"]
    _s2, res2, _ = fx.build(str(root))
    assert res2.version == res.version and sorted(res2.objects) == sorted(res.objects)


def test_dictionaries_follow_the_ontology(built):
    _root, _s, res, _idx = built
    ct = res.directory["dicts"]["cell_type"]
    ids = [e["id"] for e in ct[1:]]
    t = ids.index("CL:0000084")
    assert set(ids[t : t + 3]) == {"CL:0000084", "CL:0000624", "CL:0000625"}
    cd4 = next(e for e in ct if e["id"] == "CL:0000624")
    assert "CL:0000084" in cd4["anc"] and "CL:0000542" in cd4["anc"]
    assert res.directory["terms"]["CL:0000542"] == "lymphocyte"


def test_ordinals_are_kept_by_a_later_version(built):
    root, _s, res, _idx = built
    res2 = atlas.build_index(
        ["GSE900001", fx.GSE2], base=str(root), previous=res.version, now="20261102T0000Z"
    )
    a = atlas.Shard(res.objects[res.directory["samples"]["key"]]).table()
    b = atlas.Shard(res2.objects[res2.directory["samples"]["key"]]).table()
    assert dict(zip(a["gsm"].tolist(), a["so"].tolist())) == dict(
        zip(b["gsm"].tolist(), b["so"].tolist())
    )


def test_sample_aggregates_and_d1_export(built, tmp_path):
    root, s, res, idx = built
    t = _truth(root, s)
    st = idx.samples_table()
    gsm_of = {int(so): f"GSM{g}" for so, g in zip(st["so"], st["gsm"])}
    feats = {f.name: f for f in idx.features}
    for i, so in enumerate(st["so"].tolist()):
        sub = t[t.gsm == gsm_of[so]]
        if not len(sub):
            continue
        v = np.sort(sub.n_umi.to_numpy())
        assert (
            st["umi_min"][i] == v[0]
            and st["umi_max"][i] == v[-1]
            and st["umi_q50"][i] == v[int(0.5 * (len(v) - 1))]
        )
        h = sub["hypoxia_code"].to_numpy()
        h = h[h != feats["hypoxia"].na]
        assert st["f1_n"][i] == len(h)
        assert st["f1_ge1"][i] == int((h >= feats["hypoxia"].bound(0.5, upper=False)).sum())
    files = res.d1_export(tmp_path / "d1")
    assert {pathlib.Path(f).name for f in files} == {
        "sample_cell_summary.csv",
        "cell_feature.csv",
        "cidx_sample.csv",
    }
    with open(tmp_path / "d1" / "sample_cell_summary.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert list(rows[0]) == list(atlas.SUMMARY_COLUMNS)
    ct = {(r["gsm_id"], r["item"]): int(r["n_cells"]) for r in rows if r["subject"] == "cell_type"}
    assert ct == t.groupby(["gsm", "ct"]).size().to_dict()
    sex = {(r["gsm_id"], r["item"]): int(r["n_cells"]) for r in rows if r["subject"] == "sex"}
    ts = t[t.sex != "ambiguous"]
    want_sex = (
        ts.assign(item=ts.sex.map({"female": "PATO:0000383", "male": "PATO:0000384"}))
        .groupby(["gsm", "item"])
        .size()
        .to_dict()
    )
    assert sex == want_sex
    for r in rows:
        if r["subject"] == "cell_type":
            sub = t[(t.gsm == r["gsm_id"]) & (t.ct == r["item"])]
            assert int(r["n_hi1"]) == int((atlas.q8(sub.ct_score) >= atlas.q8_threshold(0.5)).sum())
            assert r["version"] == "test-1"
    hyp = [r for r in rows if r["item"] == "test/hypoxia"]
    assert {r["gsm_id"] for r in hyp} == {g for g in set(t.gsm) if g not in fx.NO_FEATURES}
    for r in hyp:
        h = t[t.gsm == r["gsm_id"]].hypoxia_code.to_numpy()
        h = h[h != 255]
        assert int(r["n_cells"]) == len(h) and int(r["n_hi2"]) == int(
            (h >= feats["hypoxia"].bound(0.5)).sum()
        )
        assert r["version"].startswith("test-run-")
    phase = [r for r in rows if r["item"].startswith("test/phase=")]
    assert phase and all(r["n_hi1"] == "" for r in phase)
    with open(tmp_path / "d1" / "cell_feature.csv", encoding="utf-8") as fh:
        reg = list(csv.DictReader(fh))
    assert [r["feature_id"] for r in reg] == [
        "test/hypoxia",
        "test/malignancy",
        "test/cycling",
        "test/phase",
    ]
    with open(tmp_path / "d1" / "cidx_sample.csv", encoding="utf-8") as fh:
        samp = {r["gsm_id"]: r for r in csv.DictReader(fh)}
    assert (
        samp["GSM9000011"]["perturbation"] == "treated;genetic"
        and samp["GSM9000013"]["sample_kind"] == "cell_line"
    )


def test_feature_sidecar_normalize_and_manifest(built, tmp_path):
    root, s, _res, _idx = built
    specs = atlas.load_features(fx.FEATURES)
    src = pathlib.Path(root) / "features" / "GSM9000001.features.test.parquet"
    called = s.truth["GSM9000001"]["called"]
    data, entry = atlas.normalize_feature_sidecar(
        str(src), gsm="GSM9000001", specs=specs, n_columns=len(called), called=called
    )
    assert entry["key"].startswith("s1/cf/GSM9000001/test.") and entry["key"].endswith(".parquet")
    assert entry["method_version"] == "test-run-0" and entry["n_rows"] == int(called.sum())
    codes, ver = atlas.read_feature_sidecar(data, specs, np.flatnonzero(called))
    codes2, _ = atlas.read_feature_sidecar(str(src), specs, np.flatnonzero(called))
    assert ver == "test-run-0" and all((codes[k] == codes2[k]).all() for k in codes)
    m = atlas.attach_features({"samples": [{"gsm": "GSM9000001"}]}, {"GSM9000001": entry})
    assert m["samples"][0]["features"]["test"]["key"] == entry["key"]
    import pyarrow as pa
    import pyarrow.parquet as pq

    bad = tmp_path / "bad.parquet"
    pq.write_table(pa.table({"col": [0], "hypoxia": [0.1], "gene_ACTB": [1.0]}), bad)
    with pytest.raises(atlas.IndexError_, match="unregistered"):
        atlas.normalize_feature_sidecar(
            str(bad), gsm="GSM9000001", specs=specs, n_columns=len(called)
        )


def test_add_extension_rederives_the_version(built):
    _root, _s, res, _idx = built
    import copy

    r = copy.deepcopy(res)
    blob = atlas.encode_shard({"jid": np.arange(5, dtype=np.uint32)})
    key = atlas.obj_key(blob, "scx")
    v = r.add_extension("junctions", {"table_version": "x", "tot": key}, {key: blob})
    assert v != res.version and r.directory["junctions"]["tot"] == key
    assert atlas.version_key(v) in r.objects and atlas.version_key(res.version) not in r.objects
    assert json.loads(r.objects[atlas.version_key(v)])["version"] == v


# ----------------------------------------------------------------------------------------------------------------------- queries
def _code(name, v):
    return next(f for f in atlas.load_features(fx.FEATURES) if f.name == name).bound(v)


QUERIES = [
    ({}, lambda t: t),
    (
        {"cell_type": "CL:0000084"},
        lambda t: t[t.ct.isin(["CL:0000084", "CL:0000624", "CL:0000625"])],
    ),
    ({"cell_type": "lymphocyte"}, lambda t: t[t.ct != "CL:0000576"]),
    (
        {"tissue": "UBERON:0002048"},
        lambda t: t[t.tissue.isin(["UBERON:0002048", "UBERON:0002185"])],
    ),
    (
        {"tissue": "bronchus", "sex": "ambiguous"},
        lambda t: t[(t.tissue == "UBERON:0002185") & (t.sex == "ambiguous")],
    ),
    ({"ranges": {"n_umi": [3000, None]}}, lambda t: t[t.n_umi >= 3000]),
    ({"ranges": {"n_genes": [150, 400]}}, lambda t: t[(t.n_genes >= 150) & (t.n_genes <= 400)]),
    (
        {"cell_type": "B cell", "ranges": {"cell_type_score": [0.5, None]}},
        lambda t: t[(t.ct == "CL:0000236") & (atlas.q8(t.ct_score) >= 127)],
    ),
    (
        {"ancestry": "HANCESTRO:0009", "ranges": {"ancestry_prob": [None, 0.8]}},
        lambda t: t[(t.anc == "HANCESTRO:0009") & (atlas.q8(t.anc_prob) <= 203)],
    ),
    ({"samples": ["GSM9000003"]}, lambda t: t[t.gsm == "GSM9000003"]),
    (
        {"studies": [fx.GSE2], "ranges": {"n_umi": [0, 2500]}},
        lambda t: t[(t.gse == fx.GSE2) & (t.n_umi <= 2500)],
    ),
    (
        {"ranges": {"hypoxia": [0.0, None]}},
        lambda t: t[(t.hypoxia_code != 255) & (t.hypoxia_code >= _code("hypoxia", 0.0))],
    ),
    (
        {"ranges": {"malignancy": [0.25, 0.75]}},
        lambda t: t[
            (t.malignancy_code != 65535)
            & (t.malignancy_code >= _code("malignancy", 0.25))
            & (t.malignancy_code <= _code("malignancy", 0.75))
        ],
    ),
    ({"ranges": {"test/cycling": [1, 1]}}, lambda t: t[t.cycling_code == 254]),
    ({"categories": {"test/phase": ["S", "G2M"]}}, lambda t: t[t.phase_code.isin([1, 2])]),
    ({"donor_status": "diseased"}, lambda t: t[t.donor == "diseased"]),
    (
        {"donor_status": ["healthy", "unknown"], "cell_type": "B cell"},
        lambda t: t[(t.donor != "diseased") & (t.ct == "CL:0000236")],
    ),
    ({"perturbation": "genetic"}, lambda t: t[t.perturb.str.contains("genetic")]),
    (
        {"perturbation": ["treated"], "sample_kind": "organoid"},
        lambda t: t[t.perturb.str.contains("treated") & (t.kind == "organoid")],
    ),
    ({"disease": "cancer"}, lambda t: t[t.disease.isin(["MONDO:0004992", "MONDO:0005061"])]),
    ({"disease": "normal", "studies": ["GSE900001"]}, lambda t: t[t.disease == "PATO:0000461"]),
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
    exp = (
        t.assign(
            sex=t.sex.map(
                {"female": "PATO:0000383", "male": "PATO:0000384", "ambiguous": "ambiguous"}
            )
        )
        .groupby(["ct", "sex"])
        .size()
    )
    got = by.set_index(["cell_type", "sex"])["n_cells"]
    assert got.sort_index().to_dict() == exp.sort_index().to_dict()
    a = idx.count({"cell_type": "T cell"}, by=["sample"])
    b = idx.count({"cell_type": "T cell", "ranges": {"n_umi": [0, None]}}, by=["sample"])
    assert (
        a.sort_values("sample")
        .reset_index(drop=True)
        .equals(b.sort_values("sample").reset_index(drop=True))
    )
    with pytest.raises(atlas.IndexError_):
        idx.count({}, by=["colour"])
    with pytest.raises(atlas.IndexError_):
        idx.count({"organism": "Felis catus"})
    with pytest.raises(atlas.IndexError_):
        idx.count({"ranges": {"n_mito": [0, 1]}})
    with pytest.raises(atlas.IndexError_):
        atlas.Query.from_json({"colour": "red"})


# ----------------------------------------------------------------------------------------------------------------------- sampling
def test_sampling_is_the_bottom_n_of_the_hash(built):
    root, s, _res, idx = built
    t = _truth(root, s)
    where = {"cell_type": "T cell"}
    df, info = idx.sample_cells(where, n=50, seed=7, max_studies=1)
    chosen = info["studies"][0]["gse"]
    elig = t[t.ct.isin(["CL:0000084", "CL:0000624", "CL:0000625"]) & (t.gse == chosen)]
    so = idx.so_of(elig.gsm.tolist())
    h = atlas.hash32(7, so, elig.col.to_numpy())
    order = np.lexsort((elig.col.to_numpy(), so, h.astype(np.int64)))[:50]
    want = sorted(zip(so[order].tolist(), elig.col.to_numpy()[order].tolist()))
    assert list(zip(df.so.tolist(), df.col.tolist())) == want
    assert info["eligible"] == len(elig) and info["remaining"] == len(elig) - 50 and info["sampled"]
    # ranked by eligible cells; the first study wins
    counts = t[t.ct.isin(["CL:0000084", "CL:0000624", "CL:0000625"])].groupby("gse").size()
    assert chosen == counts.sort_values(ascending=False).index[0]
    # same seed, same sample; another seed, another sample
    assert idx.sample_cells(where, n=50, seed=7, max_studies=1)[0].equals(df)
    assert not idx.sample_cells(where, n=50, seed=8, max_studies=1)[0].equals(df)


def test_sampling_pages_are_disjoint_and_cover_the_eligible_set(built):
    _root, _s, _res, idx = built
    seen = []
    cur = None
    while True:
        df, info = idx.sample_cells({"tissue": "lung"}, n=60, seed=5, cursor=cur)
        seen += list(zip(df.so, df.col))
        cur = info["cursor"]
        if not cur:
            break
    assert len(seen) == len(set(seen)) == idx.count({"tissue": "lung"})


def test_sampling_caps_studies_and_ranks(built):
    _root, _s, _res, idx = built
    df, info = idx.sample_cells({}, n=10_000, max_studies=1)
    assert df.gsm.map(_gse).nunique() == 1 and [r["selected"] for r in info["studies"]] == [
        True,
        False,
    ]
    df, info = idx.sample_cells(
        {"studies": [fx.GSE2, "GSE900001"]}, n=30, max_studies=1, study_rank="given"
    )
    assert set(df.gsm.map(_gse)) == {fx.GSE2}
    with pytest.raises(atlas.IndexError_):
        idx.sample_cells({}, study_rank="given")
    with pytest.raises(atlas.IndexError_):
        idx.sample_cells({}, sampling="stratified")

    # chunk sampling takes whole chunks: the 40 cells touch far fewer chunks than a uniform sample of 40
    def chunks(frame):
        return {
            (so, int(np.searchsorted(idx._chunk_cols(so), c, side="right") - 1))
            for so, c in zip(frame.so, frame.col)
        }

    u, _ = idx.sample_cells({}, n=40, seed=3, max_studies=2)
    c, _ = idx.sample_cells({}, n=40, seed=3, max_studies=2, sampling="chunks")
    assert len(chunks(c)) < len(chunks(u))


def test_find_cells_direct_entry_point(built):
    root, _s, _res, _idx = built
    df = atlas.find_cells(cell_type="CL:0000625", min_umi=1000, base=str(root), n=20, seed=1)
    assert len(df) == 20 and set(df.columns) >= {"so", "gsm", "col", "cell_type", "n_umi_exon"}
    assert (df.n_umi_exon + df.n_umi_intron >= 1000).all()
    assert df.attrs["eligible"] == atlas.count_cells(
        cell_type="CL:0000625", min_umi=1000, base=str(root)
    )
    f = atlas.find_cells(ranges={"hypoxia": [0.5, None]}, base=str(root), n=100)
    assert "test/hypoxia" in f.columns and (f["test/hypoxia"] >= 0.5 - 1 / 254).all()
    assert set(f["test/phase"].dropna()) <= {"G1", "S", "G2M"}
    g = atlas.find_cells(
        donor_status="diseased", categories={"phase": ["G1"]}, base=str(root), n=100
    )
    assert set(g.gsm) <= {"GSM9000003", "GSM9000011"} and (g["test/phase"] == "G1").all()
    assert len(f) == min(100, f.attrs["eligible"])


# ----------------------------------------------------------------------------------------------------------------------- gather
def _check_against_client(root, a, df):
    for gsm in sorted(set(df.gsm)):
        st = client.open_study(_gse(gsm), base=str(root))
        kmap, n_genes, ids, _names = client._gene_keep(st, None)
        assert list(ids) == list(a.var_names)
        cols = np.sort(df[df.gsm == gsm].col.values)
        sd = client._read_sample(
            st,
            st.samples[gsm],
            sel_fn=lambda rd, c=cols: c,
            want_cols=("barcode",),
            keep_map=kmap,
            n_genes=n_genes,
            threads=1,
        )
        sub = a[a.obs.gsm == gsm]
        assert sub.obs.col.tolist() == cols.tolist()
        assert (sub.layers["spliced"] != sd.spliced.csr()).nnz == 0
        if sd.unspliced is not None:
            assert (sub.layers["unspliced"] != sd.unspliced.csr()).nnz == 0
            assert (sub.X != (sd.spliced.csr() + sd.unspliced.csr())).nnz == 0
        else:
            assert sub.layers["unspliced"].nnz == 0


def test_gather_direct_matches_client(built):
    root, _s, _res, idx = built
    df = idx.find({"cell_type": "CL:0000084"}, limit=None)
    a = atlas.gather(df, base=str(root), via="direct", index=idx)
    assert a.n_obs == len(df) and a.uns["singlet_gather"]["via"] == "direct"
    _check_against_client(root, a, df)
    b = atlas.gather(df, base=str(root), via="direct", index=idx)
    assert b.uns["singlet_gather"]["planned_bytes"] < a.uns["singlet_gather"]["planned_bytes"]
    assert (a.X != b.X).nnz == 0


def test_gather_direct_same_cells_at_io_threads_1_and_4(built):
    root, _s, _res, idx = built
    df = idx.find({}, limit=None)
    one = atlas.gather(df, base=str(root), via="direct", index=idx, io_threads=1)
    four = atlas.gather(df, base=str(root), via="direct", index=idx, io_threads=4)
    assert one.obs_names.tolist() == four.obs_names.tolist() and (one.X != four.X).nnz == 0
    assert one.uns["singlet_gather"]["planned_reads"] == four.uns["singlet_gather"]["planned_reads"]


def test_gather_batches_by_cells_and_studies(built):
    root, _s, _res, idx = built
    df = idx.find({}, limit=None)
    one = atlas.gather(df, base=str(root), via="direct", index=idx, modalities=["spliced"])
    many = atlas.gather(
        df, base=str(root), via="direct", index=idx, modalities=["spliced"], batch_cells=37
    )
    per_study = atlas.gather(
        df, base=str(root), via="direct", index=idx, modalities=["spliced"], batch_studies=1
    )
    assert (
        many.uns["singlet_gather"]["batches"] > 2
        and per_study.uns["singlet_gather"]["batches"] == 2
    )
    assert one.obs_names.tolist() == many.obs_names.tolist() == per_study.obs_names.tolist()
    assert (one.layers["spliced"] != many.layers["spliced"]).nnz == 0
    assert "unspliced" not in one.layers


def test_gather_inputs_and_errors(built):
    root, _s, _res, idx = built
    df = idx.find({"samples": ["GSM9000001"]}, limit=10)
    a = atlas.gather(
        [("GSM9000001", df.col.tolist())],
        base=str(root),
        via="direct",
        index=idx,
        modalities=["counts"],
    )
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
    for r in plan["reads"][::-1]:
        buf = got_bytes[r["key"]]
        for nm, kind, off, ln, a in r["pieces"]:
            out += (
                atlas.FRAME.pack(r["sample"], names.index(nm), kind, 0, off, ln) + buf[a : a + ln]
            )
    t = json.dumps(trailer or {"ok": True}).encode()
    out += atlas.FRAME.pack(0, 0, atlas.FR_END, 0, 0, len(t)) + t
    return io.BytesIO(bytes(out))


def test_gather_stream_parser(built):
    _root, s, _res, idx = built
    df = idx.find({"cell_type": "CL:0000625"}, limit=None)
    want = atlas._cells_by_sample(idx, df)
    ent = [idx.sample(x) for x in sorted(want)]
    members = list(atlas.members_for(["counts"]))
    plan = atlas.plan_gather(ent, want, members)
    objs = {t["key"]: t["object"] for t in s.truth.values()}
    _head, got, trailer = atlas.read_gather_stream(_stream(plan, objs, members))
    direct = atlas._fetch_direct(idx, plan)
    assert trailer["ok"] and set(got) == set(direct)
    for k in got:
        assert bytes(got[k]["prefix"]) == bytes(direct[k]["prefix"])
        assert {o: bytes(b) for o, b in got[k]["runs"].items()} == {
            o: bytes(b) for o, b in direct[k]["runs"].items()
        }
    with pytest.raises(atlas.IndexError_):
        atlas.read_gather_stream(io.BytesIO(b"SGA1\x10\x00\x00\x00{}"))


def test_a_sample_without_a_count_matrix_is_skipped_not_an_error():
    """Legacy samples whose bundle held only provenance/summary JSON (GSE229169, GSE83466, ...) have no cells table: the index
    leaves them out and lists them as skipped; a sample with matrices but no cells table is still an error."""

    def sample(members):
        return atlas._SampleIn(
            gsm="GSM1",
            gse="GSE1",
            key="s1/s/GSM1/x.s1",
            bytes=1,
            sha256=None,
            n_columns=0,
            members=members,
        )

    empty = atlas._load_sample(None, sample({"summary.json": {"kind": 0}}), None, None)
    assert empty.skip and not empty.error and empty.cols is None
    broken = atlas._load_sample(None, sample({"matrix_exon.1pz": {"kind": 1}}), None, None)
    assert broken.error == "no cells table" and not broken.skip


def test_organism_comes_from_a_stored_study_meta_too():
    """Most legacy study_meta.json files are not reproducible by json.dumps (trailing newline, size), so the manifest stores them
    zstd/base64 instead of inline: the index must read gsm_meta from either form, or every such sample is organism "unknown"."""
    from singlet.s1 import manifest as M

    meta = {"gsm_meta": {"GSM1": {"organism": "Homo sapiens"}}}
    stored = M.study_file_entry((json.dumps(meta, indent=2) + "\n").encode())
    inline = M.study_file_entry(json.dumps(meta, indent=2).encode())
    assert "zstd_b64" in stored and "json" in inline
    for ent in (stored, inline):
        gm = atlas._study_gsm_meta({"study_files": {"study_meta.json": ent}})
        assert gm["GSM1"]["organism"] == "Homo sapiens"
    assert atlas._study_gsm_meta({}) == {}
