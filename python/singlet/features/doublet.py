# SPDX-License-Identifier: MIT
"""Feature set ``doublet``: ``doublet/score_v2``, a frozen cross-study stacker of doublet evidence (docs/cell_features.md section 8).

Per sample, nine within-sample rank-normalised inputs (``rank / n`` of each, over the sample's called cells)::

    umi           log1p total counts          genes       log1p detected genes
    prod          the engine's doublet score (``doublet_scores.tsv`` of the run directory, column doublet_score)
    atlas_knn, atlas_clf, ct_entropy, atlas_nnls     atlas-encoder features (doublet_atlas.py)
    sdfl, sdflc   artificial-doublet scorers without R (doublet_sdfl.py)

feed a ``HistGradientBoostingClassifier`` (max_iter 300, learning_rate 0.05, max_leaf_nodes 15, l2 1.0, random_state 0) trained once
on the 17 real-label datasets (genotype, hashtag, MULTI-seq and species doublets), each dataset weighted equally. The score is the
predicted probability (float32 in [0, 1], higher = more doublet-like).

The trained trees are stored as plain arrays (``data/doublet_v2_stacker.npz`` + ``.json``: sklearn version, feature order,
content hash), evaluated by :class:`Stacker` in numpy, so the scores do not depend on the installed scikit-learn; sklearn is only
needed to retrain. Single-donor samples only: genotype doublets of pooled donors stay authoritative (``donor_assignments``)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import scipy.sparse as sp

from . import __version__
from .registry import Feature

FEATURE_SET = "doublet"
SCORE_VERSION = "v2"
INPUTS = ("umi", "genes", "prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc")
HGB_PARAMS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0)
MIN_CELLS = 200  # fewer called cells: no score (k-means, PCA and the neighbour features need a population)
DATA = Path(__file__).parent / "data"
MODEL_NPZ = DATA / "doublet_v2_stacker.npz"
MODEL_JSON = DATA / "doublet_v2_stacker.json"
SPECIES = ("human", "mouse")

# the 17 real-label datasets in the order of bench/expression_sota/doublets/stack.py (training order, study = leave-out group)
STUDY = {
    "pbmc-1A-dm": "kang2018",
    "pbmc-1B-dm": "kang2018",
    "pbmc-1C-dm": "kang2018",
    "pbmc-2ctrl-dm": "kang2018",
    "pbmc-2stim-dm": "kang2018",
    "kang21": "kang2018",
    "J293t-dm": "kang2018_293t",
    "pbmc-ch": "stoeckius2018",
    "cline-ch": "stoeckius2018",
    "mkidney-ch": "mkidney_hashing",
    "HEK-HMEC-MULTI": "mcginnis2019",
    "HMEC-orig-MULTI": "mcginnis2019",
    "HMEC-rep-MULTI": "mcginnis2019",
    "pdx-MULTI": "mcginnis2019",
    "nuc-MULTI": "mcginnis2019",
    "hm-6k": "10x_hgmm",
    "hm-12k": "10x_hgmm",
}


# ------------------------------------------------------------------------------------------------------------- rank features
def rank_norm(x: np.ndarray) -> np.ndarray:
    """``rank / n`` (average ranks for ties, as scipy.stats.rankdata) of one input over one sample's cells."""
    from scipy.stats import rankdata

    x = np.asarray(x)
    return rankdata(x) / len(x)


def design(F: Dict[str, np.ndarray], cols: Sequence[str] = INPUTS) -> np.ndarray:
    return np.column_stack([rank_norm(F[c]) for c in cols])


def basic_inputs(X: sp.csr_matrix) -> Dict[str, np.ndarray]:
    X = sp.csr_matrix(X)
    return {"umi": np.log1p(np.asarray(X.sum(1)).ravel()), "genes": np.log1p(np.diff(X.indptr))}


# --------------------------------------------------------------------------------------------------------------- the stacker
class Stacker:
    """Gradient-boosted trees stored as arrays; ``predict_proba`` equals sklearn's HistGradientBoostingClassifier (binary log
    loss) on the same inputs. Per tree: ``feature_idx, num_threshold, missing_go_to_left, left, right, is_leaf, value`` for each
    node, concatenated, with ``tree_ptr`` the start of each tree; ``baseline`` is the raw prediction before the first tree."""

    def __init__(self, arrays: Dict[str, np.ndarray], meta: Dict):
        self.a = arrays
        self.meta = meta

    @classmethod
    def from_sklearn(cls, clf, meta: Dict) -> "Stacker":
        nodes = [p[0].nodes for p in clf._predictors]
        ptr = np.r_[0, np.cumsum([len(n) for n in nodes])].astype(np.int64)
        cat = lambda f, dt: np.concatenate([n[f] for n in nodes]).astype(dt)  # noqa: E731
        arrays = dict(
            tree_ptr=ptr,
            feature_idx=cat("feature_idx", np.int32),
            num_threshold=cat("num_threshold", np.float64),
            missing_go_to_left=cat("missing_go_to_left", np.uint8),
            left=cat("left", np.int32),
            right=cat("right", np.int32),
            is_leaf=cat("is_leaf", np.uint8),
            value=cat("value", np.float64),
            baseline=np.asarray(clf._baseline_prediction, np.float64).reshape(-1)[:1],
        )
        return cls(arrays, dict(meta))

    def content_hash(self) -> str:
        h = hashlib.sha256()
        for k in sorted(self.a):
            h.update(k.encode())
            h.update(np.ascontiguousarray(self.a[k]).tobytes())
        return h.hexdigest()

    def decision(self, Xr: np.ndarray) -> np.ndarray:
        a, n = self.a, len(Xr)
        raw = np.full(n, float(a["baseline"][0]))
        ptr = a["tree_ptr"]
        for t in range(len(ptr) - 1):
            s, e = int(ptr[t]), int(ptr[t + 1])
            fi, th, mg = a["feature_idx"][s:e], a["num_threshold"][s:e], a["missing_go_to_left"][s:e]
            lf, rt, leaf, val = a["left"][s:e], a["right"][s:e], a["is_leaf"][s:e], a["value"][s:e]
            node = np.zeros(n, np.int64)
            active = np.flatnonzero(~leaf[node].astype(bool))
            while len(active):
                nd = node[active]
                x = Xr[active, fi[nd]]
                go_left = np.where(np.isnan(x), mg[nd].astype(bool), x <= th[nd])
                node[active] = np.where(go_left, lf[nd], rt[nd])
                active = active[~leaf[node[active]].astype(bool)]
            raw += val[node]
        return raw

    def predict_proba(self, Xr: np.ndarray) -> np.ndarray:
        from scipy.special import expit

        return expit(self.decision(np.asarray(Xr, np.float64)))

    def save(self, npz: Path = MODEL_NPZ, js: Path = MODEL_JSON) -> None:
        np.savez_compressed(npz, **self.a)
        meta = dict(self.meta, content_hash=self.content_hash())
        Path(js).write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, npz: Path = MODEL_NPZ, js: Path = MODEL_JSON) -> "Stacker":
        meta = json.loads(Path(js).read_text(encoding="utf-8"))
        z = np.load(npz)
        s = cls({k: z[k] for k in z.files}, meta)
        if s.content_hash() != meta.get("content_hash"):
            raise ValueError(f"{npz}: content hash differs from {js}")
        if tuple(meta["feature_order"]) != INPUTS:
            raise ValueError("stacker feature order differs from doublet.INPUTS")
        return s


_STACKER: Optional[Stacker] = None


def stacker() -> Stacker:
    global _STACKER
    if _STACKER is None:
        _STACKER = Stacker.load()
    return _STACKER


# ---------------------------------------------------------------------------------------------------------------- training
def fit_hgb(
    data: Dict[str, "tuple"], train: Sequence[str], cols: Sequence[str] = INPUTS
):
    """The stack.py recipe: rank-normalised inputs of the ``train`` datasets stacked in the given order, each cell weighted
    ``1 / n_dataset`` (so every dataset counts equally) rescaled by ``n_cells / n_datasets``. ``data[name] = (inputs dict, labels)``."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr = np.vstack([design(data[m][0], cols) for m in train])
    ytr = np.concatenate([np.asarray(data[m][1]).astype(int) for m in train])
    w = np.concatenate([np.full(len(data[m][1]), 1 / len(data[m][1])) for m in train])
    return HistGradientBoostingClassifier(**HGB_PARAMS).fit(
        Xtr, ytr, sample_weight=w * len(w) / len(train)
    )


def loso_scores(
    data: Dict[str, "tuple"], name: str, cols: Sequence[str] = INPUTS, test: Optional[Dict] = None
) -> np.ndarray:
    """Leave-one-study-out score of dataset ``name``: trained on every dataset of another study that has all ``cols``; the
    held-out inputs come from ``test`` (same layout as ``data``) when given, otherwise from ``data``."""
    train = [
        m for m in STUDY if m in data and STUDY[m] != STUDY[name] and all(c in data[m][0] for c in cols)
    ]
    return fit_hgb(data, train, cols).predict_proba(design((test or data)[name][0], cols))[:, 1]


def train_frozen(data: Dict[str, "tuple"], extra_meta: Dict) -> Stacker:
    """Fit on all 17 datasets, return the array stacker (not saved)."""
    import sklearn

    train = [m for m in STUDY if m in data]
    if len(train) != len(STUDY):
        raise ValueError(f"missing datasets: {sorted(set(STUDY) - set(train))}")
    clf = fit_hgb(data, train)
    meta = dict(
        score=f"{FEATURE_SET}/score_{SCORE_VERSION}",
        feature_order=list(INPUTS),
        hgb_params=HGB_PARAMS,
        sklearn_version_trained=sklearn.__version__,
        train_datasets=train,
        n_train_cells=int(sum(len(data[m][1]) for m in train)),
        weights="each dataset counts equally (1/n per cell, rescaled by n_cells/n_datasets)",
        **extra_meta,
    )
    return Stacker.from_sklearn(clf, meta)


# --------------------------------------------------------------------------------------------------------- registry / scoring
def method_version() -> str:
    m = stacker().meta
    return (
        f"singlet-features={__version__};doublet=score_{SCORE_VERSION};stacker=hgb300/{m['content_hash'][:8]}"
        f"(sklearn {m['sklearn_version_trained']});inputs={','.join(INPUTS)};"
        f"atlas_encoder={m['atlas_encoder']}({m['atlas_encoder_sha256'][:8]});centroids={m['centroids_sha256'][:8]};"
        f"sdfl=a,c;min_cells={MIN_CELLS}"
    )


def features() -> List[Feature]:
    mv = method_version()
    return [
        Feature(
            name=f"score_{SCORE_VERSION}",
            feature_set=FEATURE_SET,
            label="Doublet score (v2)",
            description=(
                "Probability from a frozen cross-study stacker (gradient-boosted trees, trained on 17 real-label datasets) of "
                "the within-sample ranks of the engine's doublet score, depth, genes, atlas-encoder features and two "
                "artificial-doublet scorers; higher = more doublet-like. A within-sample ranking, not a calibrated doublet "
                "rate. For single-donor samples; genotype doublets of pooled donors are authoritative."
            ),
            method="stacked-hgb-rank",
            method_version=mv,
            source_ref="singlet-bio docs/cell_features.md section 8; Germain et al. 2021 (scDblFinder, method idea)",
            licence="CC0",
            organisms=SPECIES,
            thresholds=(0.5, 0.8),
            comparable="sample",
        )
    ]


def read_prod_scores(path, barcodes: Sequence[str]) -> np.ndarray:
    """The engine's ``doublet_scores.tsv`` (columns ``barcode``, ``doublet_score``, ...) as a float array aligned to
    ``barcodes`` (NaN = barcode not listed)."""
    import pandas as pd

    df = pd.read_csv(path, sep="\t", usecols=["barcode", "doublet_score"], dtype={"barcode": str})
    s = pd.Series(pd.to_numeric(df["doublet_score"], errors="coerce").to_numpy(), index=df["barcode"])
    s = s[~s.index.duplicated()]
    return s.reindex(list(barcodes)).to_numpy(np.float64)


def find_prod_tsv(prod_dir, gse: str, gsm: str) -> Optional[Path]:
    d = Path(prod_dir)
    for p in (d / gse / gsm / "doublet_scores.tsv", d / gsm / "doublet_scores.tsv", d / f"{gsm}.doublet_scores.tsv"):
        if p.exists():
            return p
    return None


def _na(n: int) -> Dict[str, np.ndarray]:
    return {f"score_{SCORE_VERSION}": np.full(n, np.nan, np.float32)}


def compute_inputs(
    X: sp.csr_matrix,
    gene_names,
    species: Optional[str],
    prod,
    *,
    device: str = "cpu",
    model_path: Optional[str] = None,
    centroids_path: Optional[str] = None,
    timings: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    from . import doublet_atlas as DA
    from . import doublet_sdfl as DS

    from . import doublet_assets as A

    meta = stacker().meta
    if meta["atlas_encoder_sha256"] != A.ASSETS["encoder"].sha256:
        raise A.AssetHashError("the stacker was trained with another encoder than the pinned asset")
    F = basic_inputs(X)
    F["prod"] = np.asarray(prod, np.float64)
    t = time.perf_counter()
    pred = DA.load_predictor(model_path)
    M = DA.load_centroids(centroids_path, meta["centroids_sha256"])
    F.update(DA.features(X, gene_names, species, pred, M, device=device))
    if timings is not None:
        timings["atlas_s"] = time.perf_counter() - t
    t = time.perf_counter()
    F.update(DS.score(X, ("a", "c")))
    if timings is not None:
        timings["sdfl_s"] = time.perf_counter() - t
    return F


def score_inputs(F: Dict[str, np.ndarray]) -> np.ndarray:
    """Stacker score from the nine raw inputs of one sample (cells with a non-finite input get NaN)."""
    n = len(F["umi"])
    out = np.full(n, np.nan, np.float32)
    ok = np.ones(n, bool)
    for c in INPUTS:
        ok &= np.isfinite(np.asarray(F[c], np.float64))
    if ok.sum() >= MIN_CELLS:
        Fo = {c: np.asarray(F[c])[ok] for c in INPUTS}
        out[ok] = stacker().predict_proba(design(Fo))
    return out


def score_sample(sample, *, device: str = "cpu", timings: Optional[dict] = None, **kw):
    """Score one sample (a SampleInput-like object with ``X``, ``gene_names``, ``gene_ids``, ``species`` and the engine's
    per-cell doublet score as ``prod_doublet_score``, aligned to the rows of ``X``). Returns ``(values, extra)`` where
    ``values = {"score_v2": float32 array}`` and ``extra`` holds the nine inputs (diagnostics, never written to a sidecar)."""
    X = sp.csr_matrix(sample.X, dtype=np.float32)
    n = X.shape[0]
    prod = getattr(sample, "prod_doublet_score", None)
    if sample.species not in SPECIES or n < MIN_CELLS or prod is None:
        return _na(n), {}
    ids = list(getattr(sample, "gene_ids", None) or []) or list(sample.gene_names)
    F = compute_inputs(X, ids, sample.species, prod, device=device, timings=timings, **kw)
    return {f"score_{SCORE_VERSION}": score_inputs(F)}, F
