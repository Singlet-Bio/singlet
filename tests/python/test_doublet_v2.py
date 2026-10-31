# SPDX-License-Identifier: MIT
"""doublet/score_v2: array stacker vs scikit-learn, the numpy atlas encoder, the fast R-free scorer on a planted fixture, writer and
registry, the size cap and the no-torch import, and (when the Clipper data is mounted) reproduction of stack.py-style
leave-one-study-out scores and of the stored inputs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sps

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
skl = pytest.importorskip("sklearn")

from singlet.features import doublet as D  # noqa: E402
from singlet.features import doublet_atlas as DA  # noqa: E402
from singlet.features import doublet_sdfl as DS  # noqa: E402
from singlet.features import output as O  # noqa: E402
from singlet.features import pipeline as P  # noqa: E402
from singlet.features import registry as REG  # noqa: E402

DATA = Path(D.__file__).parent / "data"
HAVE_MODEL = D.MODEL_NPZ.exists() and D.MODEL_JSON.exists() and DA.ENCODER_NPZ.exists()
needs_model = pytest.mark.skipif(not HAVE_MODEL, reason="frozen model not in the package data")


def _toy_stacker(n=1500, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier

    rng = np.random.default_rng(seed)
    X = rng.random((n, len(D.INPUTS)))
    y = (X[:, 2] + 0.5 * X[:, 3] + 0.3 * rng.standard_normal(n) > 0.9).astype(int)
    clf = HistGradientBoostingClassifier(**D.HGB_PARAMS).fit(X, y)
    return clf, X, D.Stacker.from_sklearn(clf, {"feature_order": list(D.INPUTS)})


def _toy_atlas(G=60, H=16, K=8, seed=0):
    rng = np.random.default_rng(seed)
    return DA.Atlas(
        dict(
            panel_ids=np.array([f"ENSG{i:011d}" for i in range(G)]),
            panel_symbols=np.array([f"GENE{i}" for i in range(G)]),
            panel_mouse=np.array([f"ENSMUSG{i:011d}" if i % 2 == 0 else "" for i in range(G)]),
            mu=np.full(G, 0.5, np.float32),
            sd=np.full(G, 1.0, np.float32),
            W1=rng.standard_normal((G, H)).astype(np.float16),
            b1=np.zeros(H, np.float32),
            W2=rng.standard_normal((H, K + 1)).astype(np.float16),
            b2=np.zeros(K + 1, np.float32),
        )
    )


# ------------------------------------------------------------------------------------------------------------ stacker
def test_rank_norm_ties_and_range():
    np.testing.assert_allclose(D.rank_norm(np.array([3.0, 1.0, 1.0, 2.0])), [1.0, 0.375, 0.375, 0.75])


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
    back = D.Stacker.load(tmp_path / "m.npz", tmp_path / "m.json")
    assert back.content_hash() == st.content_hash()
    np.testing.assert_array_equal(back.predict_proba(X), st.predict_proba(X))
    meta = json.loads((tmp_path / "m.json").read_text())
    meta["content_hash"] = "0" * 64
    (tmp_path / "bad.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="content hash"):
        D.Stacker.load(tmp_path / "m.npz", tmp_path / "bad.json")


@needs_model
def test_model_files_are_small_and_consistent():
    total = sum(f.stat().st_size for f in DATA.glob("doublet_v2_*"))
    assert total < 1_000_000, total  # the whole model ships in the wheel
    st = D.stacker()
    assert tuple(st.meta["feature_order"]) == D.INPUTS and st.meta["hgb_params"] == D.HGB_PARAMS
    assert sorted(st.meta["train_datasets"]) == sorted(D.STUDY)
    assert st.meta["encoder_content_hash"] == D.atlas().content_hash()
    p = st.predict_proba(np.random.default_rng(1).random((500, len(D.INPUTS))))
    assert p.min() >= 0 and p.max() <= 1


def test_doublet_stage_does_not_import_torch():
    code = (
        "import sys, numpy as np, scipy.sparse as sp\n"
        "from singlet.features import doublet, doublet_atlas, doublet_sdfl, pipeline, registry\n"
        "X = sp.csr_matrix(np.random.default_rng(0).poisson(0.3, (300, 80)).astype(np.float32))\n"
        "doublet_sdfl.score(X, ('a',))\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
    )
    env = dict(os.environ, PYTHONPATH=str(Path(D.__file__).parents[2]))
    r = subprocess.run([sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stderr[-800:]


# ----------------------------------------------------------------------------------------------------- registry, writer
@needs_model
def test_registry_row_and_method_version():
    (f,) = REG.features("doublet")
    assert f.id == "doublet/score_v2" and f.kind == "score" and f.comparable == "sample"
    mv = REG.method_version("doublet")
    assert mv == f.method_version and "doublet=score_v2" in mv
    assert D.stacker().meta["content_hash"][:8] in mv and D.atlas().content_hash()[:8] in mv
    row = REG.registry_rows(("doublet",))[0]
    assert row["feature_id"] == "doublet/score_v2" and row["status"] == "active"
    assert json.loads(row["organisms"]) == ["human", "mouse"]
    assert len(REG.features("hallmark")) + len(REG.features("cell_cycle")) + 1 <= 64  # cell index cap per version


@needs_model
def test_sidecar_roundtrip_and_guard(tmp_path):
    feats = REG.features("doublet")
    n = 50
    v = np.linspace(0, 1, n).astype(np.float32)
    v[3] = np.nan
    tab = O.build_table("GSM123", np.arange(n), {"score_v2": v}, feats, REG.method_version("doublet"))
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
    _, entry = atlas.normalize_feature_sidecar(p, gsm="GSM123", specs=specs, n_columns=n)
    assert entry["method_version"] == REG.method_version("doublet") and entry["feature_set"] == "doublet" and entry["n_rows"] == n


@needs_model
def test_score_inputs_na_rules():
    rng = np.random.default_rng(2)
    n = 400
    F = {c: rng.random(n) for c in D.INPUTS}
    F["prod"][5] = np.nan
    s = D.score_inputs(F)
    assert s.dtype == np.float32 and np.isnan(s[5]) and np.isfinite(np.delete(s, 5)).all()
    assert np.nanmin(s) >= 0 and np.nanmax(s) <= 1
    assert np.isnan(D.score_inputs({c: v[: D.MIN_CELLS - 1] for c, v in F.items()})).all()


@needs_model
def test_pipeline_without_engine_score_is_na():
    class S:
        gsm, gse, species = "GSM1", "GSE1", "human"
        X = sps.csr_matrix(np.ones((300, 20), np.float32))
        gene_names = [f"G{i}" for i in range(20)]
        col = np.arange(300)

    r = P.score_sample(S(), ("doublet",))["doublet"]
    assert np.isnan(r.values["score_v2"]).all() and "score_v2" in r.dropped
    assert r.method_version == REG.method_version("doublet")


def test_prod_tsv_reader(tmp_path):
    (tmp_path / "GSM9").mkdir()
    (tmp_path / "GSM9" / "doublet_scores.tsv").write_text("barcode\tdoublet_score\tis_doublet\nAAA\t0.2\t0\nCCC\t0.9\t1\n")
    v = D.read_prod_scores(D.find_prod_tsv(tmp_path, "GSE1", "GSM9"), ["CCC", "GGG", "AAA"])
    np.testing.assert_allclose(v, [0.9, np.nan, 0.2])
    assert D.find_prod_tsv(tmp_path, "GSE1", "GSM8") is None


# ------------------------------------------------------------------------------------------------------------ the encoder
def test_atlas_gene_mapping():
    at = _toy_atlas()
    np.testing.assert_array_equal(at.columns(["ENSG00000000003.5", "hg19_ENSG00000000001", "ENSG9", "ENSMUSG00000000004"]), [3, 1, -1, 4])
    np.testing.assert_array_equal(at.columns(["gene3", "FOO", "Gene10"]), [3, -1, 10])


def test_atlas_features_are_deterministic_and_bounded():
    at = _toy_atlas()
    rng = np.random.default_rng(0)
    X = sps.csr_matrix(rng.poisson(1.0, (700, 60)).astype(np.float32))
    ids = [f"ENSG{i:011d}" for i in range(60)]
    F = at.features(X, ids, "human")
    again = at.features(X, ids, "human")
    for k in ("atlas_knn", "atlas_entropy"):
        assert F[k].shape == (700,) and np.isfinite(F[k]).all()
        np.testing.assert_array_equal(F[k], again[k])
    assert F["atlas_knn"].min() >= 0 and F["atlas_knn"].max() <= 1


def test_knn_ref_excludes_the_query_itself_and_scales_linearly():
    Q = np.array([[0.0], [10.0]], np.float32)
    ref = np.array([[0.0], [10.0], [0.1], [9.9]], np.float32)  # rows 0, 1 real (m = 2); rows 2, 3 simulated
    f = DA.Atlas.knn_ref(Q, ref, 2, 1, np.array([0, 1]))
    np.testing.assert_array_equal(f, [1.0, 1.0])  # self excluded: the nearest is the simulated neighbour
    assert DA.REF_CAP * (1 + DA.SIM_RATIO) < 10_000  # reference size is fixed, so the search is O(n)


# --------------------------------------------------------------------------------------------------------------- sdfl
def _planted(n=500, g=400, frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    prof = rng.gamma(0.3, 1.0, (4, g))
    prof /= prof.sum(1, keepdims=True)
    t = rng.integers(0, 4, n)
    X = np.stack([rng.poisson(prof[k] * 1500) for k in t]).astype(np.float32)
    dbl = rng.random(n) < frac
    for i in np.flatnonzero(dbl):
        X[i] = X[i] + rng.poisson(prof[(t[i] + 1 + rng.integers(0, 3)) % 4] * 1500)
    return sps.csr_matrix(X), dbl


def test_sdfl_scores_planted_doublets_and_is_deterministic():
    from sklearn.metrics import roc_auc_score

    X, dbl = _planted()
    out = DS.score(X, ("a", "c"))
    assert set(out) == {"sdfl", "sdflc"}
    for k, s in out.items():
        assert s.shape == (X.shape[0],) and s.min() >= 0 and s.max() <= 1
        assert roc_auc_score(dbl, s) > 0.6, k
    np.testing.assert_array_equal(DS.score(X, ("c",))["sdflc"], out["sdflc"])


def test_sdfl_scores_cells_outside_the_fitting_subsample():
    from dataclasses import replace

    from sklearn.metrics import roc_auc_score

    X, dbl = _planted(n=900)
    s = DS.score(X, ("a",), p=replace(DS.FAST, cap=300, art_min=300))["sdfl"]
    assert s.shape == (900,) and np.isfinite(s).all() and s.min() >= 0 and s.max() <= 1
    assert roc_auc_score(dbl, s) > 0.6


# ------------------------------------------------------------------------- reproduction on the real datasets (Clipper data)
LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
REF = os.environ.get("SINGLET_DBL_REF")  # OUT of scripts/doublet_v2.sbatch: pkgfeat/ (feats) and loso_ref/ (ref-loso)
have_data = bool(REF) and Path(f"{LQC}/dbl/data").exists() and (Path(REF) / "loso_ref").exists()


@pytest.mark.skipif(not have_data, reason="Clipper dbl data / reference scores not mounted")
def test_loso_scores_reproduce_stack_py_style_evaluation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import doublet_v2 as S

    data = S.load_training(Path(REF) / "pkgfeat")
    names = [n for n in ("pbmc-1A-dm", "hm-6k", "pbmc-ch") if (Path(REF) / "loso_ref" / f"{n}.npy").exists()]
    assert len(names) >= 2
    for n in names:
        np.testing.assert_allclose(D.loso_scores(data, n), np.load(Path(REF) / "loso_ref" / f"{n}.npy"), atol=1e-9, rtol=0, err_msg=n)


@needs_model
@pytest.mark.skipif(not have_data, reason="Clipper dbl data not mounted")
def test_package_reproduces_the_stored_inputs():
    from scipy.stats import spearmanr

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import doublet_v2 as S

    for n in ("pbmc-1A-dm", "hm-6k"):
        X, _, genes, species = S.load_counts(n)
        z = np.load(Path(REF) / "pkgfeat" / f"{n}.npz")
        F = D.atlas().features(X, genes, species)
        for k in ("atlas_knn", "atlas_entropy"):
            np.testing.assert_allclose(F[k], z[k], atol=1e-6, rtol=0, err_msg=f"{n} {k}")
        s = DS.score(X, ("a", "c"))
        for k in ("sdfl", "sdflc"):
            assert spearmanr(s[k], z[k])[0] > 0.999, (n, k)
