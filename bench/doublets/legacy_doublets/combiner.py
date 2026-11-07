# SPDX-License-Identifier: MIT
"""The cross-study doublet combiner: gradient boosting over within-sample rank-transformed scores.

Training (``fit``, needs scikit-learn) happens once, on real experimental labels (genotype, hashtag, MULTI-seq, species mixing)
pooled over studies, each dataset weighing the same. The fitted ensemble is exported to a ``.npz`` of flat tree arrays, and
:meth:`Combiner.predict_proba` evaluates it with numpy alone, so the batch job needs neither scikit-learn nor pickles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
from scipy.stats import rankdata

DATA = Path(__file__).with_name("data")
#: the features of ``combo_prod_atlas``, in column order
FEATURES = ("umi", "genes", "prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls")
PARAMS = dict(
    max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
)


def qmatrix(F: Dict[str, np.ndarray], cols: Sequence[str] = FEATURES) -> np.ndarray:
    """Cells x features, each column replaced by its within-sample rank in (0, 1]; NaN counts as the lowest score."""
    out = []
    for c in cols:
        v = np.asarray(F[c], np.float64)
        v = np.where(np.isfinite(v), v, np.nanmin(v) if np.isfinite(v).any() else 0.0)
        out.append(rankdata(v) / len(v))
    return np.column_stack(out)


def basic_features(X) -> Dict[str, np.ndarray]:
    """``umi`` and ``genes``: log1p of the cell's total counts and of its detected genes (cells x genes CSR)."""
    X = X.tocsr()
    return {
        "umi": np.log1p(np.asarray(X.sum(1)).ravel()),
        "genes": np.log1p(np.diff(X.indptr)),
    }


class Combiner:
    def __init__(self, arrays: Dict[str, np.ndarray], features: Sequence[str], version: str):
        self.a, self.features, self.version = arrays, tuple(features), version

    # ---------------------------------------------------------------- evaluation (numpy only)
    def predict_proba(self, Q: np.ndarray) -> np.ndarray:
        a = self.a
        n = Q.shape[0]
        raw = np.full(n, float(a["baseline"]), np.float64)
        off = a["tree_offset"]
        for t in range(len(off) - 1):
            s = slice(int(off[t]), int(off[t + 1]))
            feat, thr, left, right = a["feature"][s], a["threshold"][s], a["left"][s], a["right"][s]
            val, leaf = a["value"][s], a["is_leaf"][s]
            node = np.zeros(n, np.int64)
            while True:
                active = ~leaf[node]
                if not active.any():
                    break
                i = np.flatnonzero(active)
                nd = node[i]
                go_left = Q[i, feat[nd]] <= thr[nd]
                node[i] = np.where(go_left, left[nd], right[nd])
            raw += val[node]
        return 1.0 / (1.0 + np.exp(-raw))

    def score(self, F: Dict[str, np.ndarray]) -> np.ndarray:
        return self.predict_proba(qmatrix(F, self.features)).astype(np.float32)

    # ---------------------------------------------------------------- persistence
    def save(self, path) -> None:
        np.savez_compressed(
            path, features=np.array(self.features), version=np.array(self.version), **self.a
        )

    @classmethod
    def load(cls, path=None) -> "Combiner":
        z = np.load(path or DATA / "combiner_prod_atlas.npz", allow_pickle=False)
        arrays = {k: z[k] for k in z.files if k not in ("features", "version")}
        return cls(arrays, [str(f) for f in z["features"]], str(z["version"]))

    # ---------------------------------------------------------------- training (scikit-learn)
    @classmethod
    def fit(
        cls, Q: np.ndarray, y: np.ndarray, w: np.ndarray, features: Sequence[str], version: str
    ):
        from sklearn.ensemble import HistGradientBoostingClassifier

        clf = HistGradientBoostingClassifier(**PARAMS).fit(Q, y, sample_weight=w)
        return cls.from_sklearn(clf, features, version)

    @classmethod
    def from_sklearn(cls, clf, features: Sequence[str], version: str) -> "Combiner":
        feats, thrs, lefts, rights, vals, leaves, off = [], [], [], [], [], [], [0]
        for (p,) in clf._predictors:
            nd = p.nodes
            if nd["is_categorical"].any():
                raise ValueError("categorical splits are not exported")
            feats.append(nd["feature_idx"].astype(np.int32))
            thrs.append(nd["num_threshold"].astype(np.float64))
            lefts.append(nd["left"].astype(np.int32))
            rights.append(nd["right"].astype(np.int32))
            vals.append(nd["value"].astype(np.float64))
            leaves.append(nd["is_leaf"].astype(bool))
            off.append(off[-1] + len(nd))
        arrays = dict(
            baseline=np.asarray(float(np.ravel(clf._baseline_prediction)[0])),
            tree_offset=np.array(off, np.int64),
            feature=np.concatenate(feats),
            threshold=np.concatenate(thrs),
            left=np.concatenate(lefts),
            right=np.concatenate(rights),
            value=np.concatenate(vals),
            is_leaf=np.concatenate(leaves),
        )
        return cls(arrays, features, version)
