# SPDX-License-Identifier: MIT
"""singlet.doublets: the numpy combiner equals scikit-learn, the packaged model loads, the feature sidecar passes the guard, and
samples that must not be scored (genotype pools, few cells, other organisms) get no score."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.sparse as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

from singlet.doublets import score as S  # noqa: E402
from singlet.doublets.combiner import FEATURES, Combiner, basic_features, qmatrix  # noqa: E402


def test_numpy_evaluator_matches_sklearn():
    sk = pytest.importorskip("sklearn.ensemble")
    r = np.random.default_rng(0)
    Q = r.random((2000, len(FEATURES)))
    y = (Q[:, 2] + 0.5 * Q[:, 3] + 0.3 * r.standard_normal(2000) > 1.0).astype(int)
    clf = sk.HistGradientBoostingClassifier(
        max_iter=40, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
    ).fit(Q, y)
    m = Combiner.from_sklearn(clf, FEATURES, "t")
    assert np.abs(m.predict_proba(Q) - clf.predict_proba(Q)[:, 1]).max() < 1e-9


def test_packaged_combiner_scores_ranked_features():
    m = Combiner.load()
    assert m.features == FEATURES
    r = np.random.default_rng(1)
    F = {c: r.random(300) for c in FEATURES}
    s = m.score(F)
    assert s.shape == (300,) and s.dtype == np.float32 and ((s >= 0) & (s <= 1)).all()
    # a cell that ranks high on the engine score and the atlas features scores above one that ranks low on all of them
    lo = {c: np.arange(300.0) for c in FEATURES}
    assert m.score(lo)[-1] > m.score(lo)[0]


def test_basic_features_and_ranks():
    X = sps.csr_matrix(np.array([[1, 0, 3], [0, 0, 1], [5, 5, 5]]))
    f = basic_features(X)
    assert np.allclose(f["umi"], np.log1p([4, 1, 15])) and np.allclose(
        f["genes"], np.log1p([2, 1, 3])
    )
    q = qmatrix({"umi": f["umi"]}, ["umi"])
    assert q.ravel().tolist() == [2 / 3, 1 / 3, 1.0]


def test_feature_sidecar_passes_the_guard():
    from dataclasses import replace

    from singlet.features.output import assert_public_safe, build_table

    f = replace(
        S.FEATURE,
        method_version="singlet-doublets=1.0.0;combiner=x;celltype-nn=y;dbl-run=z;min_cells=100",
    )
    assert f.id == "qc/doublet_score" and f.cell_feature_row()["feature_id"] == "qc/doublet_score"
    v = np.linspace(0, 1, 5, dtype=np.float32)
    t = build_table("GSM1", np.arange(5), {"doublet_score": v}, [f], f.method_version)
    assert_public_safe(t, [f])
    rows = S.summaries(SimpleNamespace(gsm="GSM1", gse="GSE1"), S.Result(v), f.method_version)
    assert rows[0]["item"] == "qc/doublet_score" and rows[0]["n_cells"] == 5


def _sample(n, species="human", donors=None):
    return SimpleNamespace(
        X=sps.csr_matrix(np.ones((n, 5))), species=species, donor=donors, gene_ids=[], gene_names=[]
    )


@pytest.mark.parametrize(
    "sample,status",
    [
        (_sample(50), "too_few_cells"),
        (_sample(500, species=None), "other_organism"),
        (_sample(500, donors=np.array(["a", "b"] * 250)), "genotype_pool"),
    ],
)
def test_unscored_samples_get_no_score(sample, status):
    res = S.score_sample(sample, Combiner.load(), dbl_run="none", predictor=None, centroids=None)
    assert (
        res.status == status and np.isnan(res.score).all() and len(res.score) == sample.X.shape[0]
    )
