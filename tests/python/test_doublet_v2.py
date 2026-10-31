# SPDX-License-Identifier: MIT
"""doublet/score_v2: array stacker vs scikit-learn, writer and registry, the R-free scorer on a planted fixture, and (when the
Clipper data is mounted) reproduction of the leave-one-study-out scores of stack.py."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
skl = pytest.importorskip("sklearn")

from singlet.features import doublet as D  # noqa: E402
from singlet.features import output as O  # noqa: E402
from singlet.features import pipeline as P  # noqa: E402
from singlet.features import registry as REG  # noqa: E402

HAVE_MODEL = D.MODEL_NPZ.exists() and D.MODEL_JSON.exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="frozen stacker not in the package data")


def _toy_stacker(n=1500, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.random((n, len(D.INPUTS)))
    y = (X[:, 2] + 0.5 * X[:, 3] + 0.3 * rng.standard_normal(n) > 0.9).astype(int)
    clf = HistGradientBoostingClassifier(**D.HGB_PARAMS).fit(X, y)
    return clf, X, D.Stacker.from_sklearn(clf, {"feature_order": list(D.INPUTS)})


def test_rank_norm_ties_and_range():
    r = D.rank_norm(np.array([3.0, 1.0, 1.0, 2.0]))
    np.testing.assert_allclose(r, [1.0, 0.375, 0.375, 0.75])


def test_array_stacker_equals_sklearn_including_missing():
    clf, X, st = _toy_stacker()
    np.testing.assert_allclose(st.predict_proba(X), clf.predict_proba(X)[:, 1], atol=1e-12, rtol=0)
    Xn = X.copy()
    Xn[::7, 3] = np.nan
    np.testing.assert_allclose(st.predict_proba(Xn), clf.predict_proba(Xn)[:, 1], atol=1e-12, rtol=0)


def test_stacker_save_load_hash(tmp_path):
    _, X, st = _toy_stacker()
    st.meta.update(sklearn_version_trained=skl.__version__)
    st.save(tmp_path / "m.npz", tmp_path / "m.json")
    back = D.Stacker.__new__(D.Stacker)
    z = np.load(tmp_path / "m.npz")
    back.__init__({k: z[k] for k in z.files}, json.loads((tmp_path / "m.json").read_text()))
    assert back.content_hash() == back.meta["content_hash"] == st.content_hash()
    np.testing.assert_array_equal(back.predict_proba(X), st.predict_proba(X))
    meta = json.loads((tmp_path / "m.json").read_text())
    meta["content_hash"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="content hash"):
        D.Stacker.load(tmp_path / "m.npz", tmp_path / "bad.json")


@needs_model
def test_packaged_stacker_loads_and_is_frozen():
    st = D.stacker()
    assert tuple(st.meta["feature_order"]) == D.INPUTS
    assert st.meta["hgb_params"] == D.HGB_PARAMS
    assert st.meta["sklearn_version_trained"]
    assert sorted(st.meta["train_datasets"]) == sorted(D.STUDY)
    rng = np.random.default_rng(1)
    p = st.predict_proba(rng.random((500, len(D.INPUTS))))
    assert p.min() >= 0 and p.max() <= 1


@needs_model
def test_registry_row_and_method_version():
    (f,) = REG.features("doublet")
    assert f.id == "doublet/score_v2" and f.kind == "score" and f.comparable == "sample"
    mv = REG.method_version("doublet")
    assert mv == f.method_version and "doublet=score_v2" in mv and D.stacker().meta["content_hash"][:8] in mv
    row = [r for r in REG.registry_rows(("doublet",))][0]
    assert row["feature_id"] == "doublet/score_v2" and row["status"] == "active"
    assert json.loads(row["organisms"]) == ["human", "mouse"]
    assert len(REG.features("hallmark")) + len(REG.features("cell_cycle")) + 1 <= 64  # cell index cap per version


@needs_model
def test_sidecar_roundtrip_and_guard(tmp_path):
    feats = REG.features("doublet")
    n = 50
    v = np.linspace(0, 1, n).astype(np.float32)
    v[3] = np.nan
    tab = O.build_table("GSM123", np.arange(n)[::-1].copy()[::-1], {"score_v2": v}, feats, REG.method_version("doublet"))
    p = O.write_features(tab, tmp_path / O.sidecar_name("GSM123", "doublet"), feats)
    t = pq.read_table(p)
    md = {k.decode(): x.decode() for k, x in t.schema.metadata.items() if not k.startswith(b"pandas")}
    assert md["feature_set"] == "doublet" and md["method_version"] == REG.method_version("doublet")
    assert t.schema.field("score_v2").type == pa.float32() and t.column_names == ["gsm", "col", "score_v2"]
    bad = O.build_table("GSM123", np.arange(n), {"score_v2": v * 2}, feats, REG.method_version("doublet"))
    with pytest.raises(O.PublicOutputError):
        O.assert_public_safe(bad, feats)


@needs_model
def test_cell_index_records_the_method_version(tmp_path):
    atlas = pytest.importorskip("singlet.s1.atlas")
    feats = REG.features("doublet")
    n = 60
    v = np.random.default_rng(3).random(n).astype(np.float32)
    tab = O.build_table("GSM123", np.arange(n), {"score_v2": v}, feats, REG.method_version("doublet"))
    p = O.write_features(tab, tmp_path / "x.parquet", feats)
    specs = atlas.load_features([f.cell_feature_row() for f in feats])
    data, entry = atlas.normalize_feature_sidecar(p, gsm="GSM123", specs=specs, n_columns=n)
    assert entry["method_version"] == REG.method_version("doublet") and entry["feature_set"] == "doublet"
    assert entry["n_rows"] == n


@needs_model
def test_score_inputs_na_rules():
    rng = np.random.default_rng(2)
    n = 400
    F = {c: rng.random(n) for c in D.INPUTS}
    F["prod"][5] = np.nan
    s = D.score_inputs(F)
    assert s.dtype == np.float32 and np.isnan(s[5]) and np.isfinite(np.delete(s, 5)).all()
    assert np.nanmin(s) >= 0 and np.nanmax(s) <= 1
    small = {c: v[: D.MIN_CELLS - 1] for c, v in F.items()}
    assert np.isnan(D.score_inputs(small)).all()


@needs_model
def test_pipeline_without_engine_score_is_na():
    class S:
        gsm, gse, species = "GSM1", "GSE1", "human"
        X = sps.csr_matrix(np.ones((300, 20), np.float32))
        gene_names = [f"G{i}" for i in range(20)]
        col = np.arange(300)

    res = P.score_sample(S(), ("doublet",))
    r = res["doublet"]
    assert np.isnan(r.values["score_v2"]).all() and "score_v2" in r.dropped
    assert r.method_version == REG.method_version("doublet")


def test_prod_tsv_reader(tmp_path):
    (tmp_path / "GSM9").mkdir()
    (tmp_path / "GSM9" / "doublet_scores.tsv").write_text(
        "barcode\tdoublet_score\tis_doublet\nAAA\t0.2\t0\nCCC\t0.9\t1\n"
    )
    p = D.find_prod_tsv(tmp_path, "GSE1", "GSM9")
    v = D.read_prod_scores(p, ["CCC", "GGG", "AAA"])
    np.testing.assert_allclose(v, [0.9, np.nan, 0.2])
    assert D.find_prod_tsv(tmp_path, "GSE1", "GSM8") is None


def test_vocab_projection_ensembl_and_symbols():
    from singlet.features import doublet_atlas as DA

    class V:
        symbols = ["CD3E", "MS4A1", "LYZ"]

        def columns(self, ids, species):
            tab = {"ENSG1": 0, "ENSG2": 1, "ENSMUSG1": 2}
            return np.array([tab.get(i, -1) for i in ids])

        def __len__(self):
            return 3

    class Pr:
        vocab = V()

    P1, n1 = DA.vocab_projection(Pr, ["ENSG1.5", "hg19_ENSG2", "ENSG9", "ENSMUSG1"])
    assert n1 == 3 and P1.shape == (4, 3) and P1.toarray()[1, 1] == 1 and P1.toarray()[2].sum() == 0
    P2, n2 = DA.vocab_projection(Pr, ["cd3e", "FOO", "Lyz"])
    assert n2 == 2 and P2.toarray()[2, 2] == 1


def _planted(n=500, g=400, frac=0.1, seed=0):
    """Counts of four cell types plus ``frac`` doublets (summed pairs of different types); returns X and the doublet flag."""
    rng = np.random.default_rng(seed)
    prof = rng.gamma(0.3, 1.0, (4, g))
    prof /= prof.sum(1, keepdims=True)
    t = rng.integers(0, 4, n)
    X = np.stack([rng.poisson(prof[k] * 1500) for k in t]).astype(np.float32)
    dbl = rng.random(n) < frac
    for i in np.flatnonzero(dbl):
        k2 = (t[i] + 1 + rng.integers(0, 3)) % 4
        X[i] = X[i] + rng.poisson(prof[k2] * 1500)
    return sps.csr_matrix(X), dbl


def test_sdfl_scores_planted_doublets():
    from sklearn.metrics import roc_auc_score

    from singlet.features import doublet_sdfl as DS

    X, dbl = _planted()
    out = DS.score(X, ("a", "c"))
    assert set(out) == {"sdfl", "sdflc"}
    for k, s in out.items():
        assert s.shape == (X.shape[0],) and s.min() >= 0 and s.max() <= 1
        assert roc_auc_score(dbl, s) > 0.6, k
    again = DS.score(X, ("c",))
    np.testing.assert_array_equal(again["sdflc"], out["sdflc"])  # one stream per variant, shared prefix


# ---------------------------------------------------------------------------------- reproduction of stack.py (needs the data)
LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
REF = os.environ.get("SINGLET_DBL_REF")  # directory with loso_ref/<name>.npy written by scripts/doublet_v2.py ref-loso


@pytest.mark.skipif(not REF or not Path(f"{LQC}/dbl/data").exists(), reason="Clipper dbl data / reference scores not mounted")
def test_loso_scores_reproduce_stack_py():
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import doublet_v2 as S

    data = S.load_training()
    names = [n for n in ("pbmc-1A-dm", "hm-6k", "pbmc-ch") if (Path(REF) / "loso_ref" / f"{n}.npy").exists()]
    assert len(names) >= 2
    for n in names:
        ref = np.load(Path(REF) / "loso_ref" / f"{n}.npy")
        got = D.loso_scores(data, n)
        np.testing.assert_allclose(got, ref, atol=1e-9, rtol=0, err_msg=n)


# ------------------------------------------------------------------------------------------------ pinned model-file fetch
def _mini_assets(tmp_path):
    import hashlib

    from singlet.features import doublet_assets as A

    src = tmp_path / "release"
    src.mkdir()
    blobs = {"centroids": b"centroid-bytes" * 100, "encoder": b"network-bytes" * 200}
    assets = {}
    for k, b in blobs.items():
        name = A.ASSETS[k].name
        (src / name).write_bytes(b)
        assets[k] = A.Asset(name, hashlib.sha256(b).hexdigest(), len(b))
    return A, src, assets


def test_fetch_by_url_checks_hash_and_caches(tmp_path):
    A, src, assets = _mini_assets(tmp_path)
    cache = tmp_path / "cache"
    base = src.as_uri()
    p = A.fetch("centroids", cache=cache, base_url=base, assets=assets)
    assert p.read_bytes() == (src / assets["centroids"].name).read_bytes() and p.parent.parent == cache
    (src / assets["centroids"].name).unlink()  # served from the cache now: no network needed
    assert A.fetch("centroids", cache=cache, base_url=base, assets=assets) == p
    assert not list(cache.rglob("*.part*"))


def test_fetch_refuses_tampered_download(tmp_path):
    A, src, assets = _mini_assets(tmp_path)
    (src / assets["encoder"].name).write_bytes(b"tampered" * 50)
    cache = tmp_path / "cache"
    with pytest.raises(A.AssetHashError):
        A.fetch("encoder", cache=cache, base_url=src.as_uri(), assets=assets)
    assert not [f for f in cache.rglob("*") if f.is_file()]  # nothing kept


def test_fetch_refetches_a_damaged_cache_entry_and_checks_overrides(tmp_path):
    A, src, assets = _mini_assets(tmp_path)
    cache = tmp_path / "cache"
    p = A.fetch("encoder", cache=cache, base_url=src.as_uri(), assets=assets)
    p.write_bytes(b"bit rot")
    again = A.fetch("encoder", cache=cache, base_url=src.as_uri(), assets=assets)
    assert again.read_bytes() == (src / assets["encoder"].name).read_bytes()
    bad = tmp_path / "local.pt"
    bad.write_bytes(b"not the network")
    with pytest.raises(A.AssetHashError):
        A.fetch("encoder", override=str(bad), assets=assets)  # an offline override is hash-checked too
    assert A.fetch("encoder", override=str(src / assets["encoder"].name), assets=assets)


def test_pins_agree_with_stacker_and_release_tag():
    from singlet.features import doublet_assets as A

    assert A.TAG == "doublet-v2-model-" + A.COMBINED_SHA256[:12]
    if HAVE_MODEL:
        m = D.stacker().meta
        assert m["atlas_encoder_sha256"] == A.ASSETS["encoder"].sha256
        assert m["centroids_sha256"] == A.ASSETS["centroids"].sha256


def test_lookup_order_store_then_cache_then_url(tmp_path, monkeypatch):
    A, src, assets = _mini_assets(tmp_path)
    store, cache = tmp_path / "store", tmp_path / "cache"
    store.mkdir()
    name = assets["encoder"].name
    ext = Path(name).suffix
    kw = dict(cache=cache, store=store, base_url=src.as_uri(), assets=assets)
    # nothing in the store or the cache: downloaded
    p1 = A.fetch("encoder", **kw)
    assert p1.parent.parent == cache
    # store entry present: wins over the cache and the URL
    (store / (assets["encoder"].sha256 + ext)).write_bytes((src / name).read_bytes())
    (src / name).unlink()
    p2 = A.fetch("encoder", **kw)
    assert p2.parent == store
    # corrupt store entry: skipped with a warning, the cache serves
    (store / (assets["encoder"].sha256 + ext)).write_bytes(b"bad")
    with pytest.warns(UserWarning, match="store"):
        assert A.fetch("encoder", **kw) == p1
    # the store directory comes from SINGLET_MODEL_STORE; an override beats everything
    monkeypatch.setenv(A.ENV_STORE, str(store))
    assert A.store_dir() == store
    own = tmp_path / "own.pt"
    own.write_bytes(p1.read_bytes())
    assert A.fetch("encoder", override=str(own), **kw) == own
