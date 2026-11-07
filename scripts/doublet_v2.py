#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training, validation and cost measurement of the ``doublet/score_v2`` stacker (docs/cell_features.md section 8). Clipper jobs
(scripts/doublet_v2.sbatch); real data only: the 17 real-label datasets under $LQC/dbl/data (counts, labels) and the engine's
doublet score of each ($LQC/dbl/scores/<name>/prod.npy).

  feats      the package's inputs (sdfl, sdflc, atlas_knn, atlas_entropy) for the 17 datasets -> OUT/pkgfeat/<name>.npz + cost.json
  train      fit the frozen stacker on those inputs -> --model-dir/doublet_v2_stacker.{npz,json} (copy to python/singlet/features/data)
  loso       leave-one-study-out scores (package code) -> OUT/loso_pkg, mean AUPRC, paired CI against scDblFinder
  ref-loso   the recipe of bench/expression_sota/doublets/stack.py, unchanged, on the same inputs -> OUT/loso_ref
  score      the whole stage (inputs + stacker) on one dataset: wall / CPU seconds per stage, peak memory, sidecar parquet
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

LQC = os.environ.get("LQC", "/mnt/projects/debruinz_project/debruinz/singlet/latent_qc")
DATA, SCORES = f"{LQC}/dbl/data", f"{LQC}/dbl/scores"
PKG = ("sdfl", "sdflc", "atlas_knn", "atlas_entropy")


def load_counts(name):
    z = np.load(f"{DATA}/{name}/ds.npz")
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    return X, z["label"].astype(int), z["genes"].astype(str), str(z["species"])


def load_training(pkgdir):
    """Inputs of the 17 datasets: umi / genes from the counts, ``prod`` the engine's score, the rest from ``feats``."""
    from singlet.features import doublet as D

    data = {}
    for n in D.STUDY:
        X, y, _, _ = load_counts(n)
        F = D.basic_inputs(X)
        F["prod"] = np.load(f"{SCORES}/{n}/prod.npy")
        z = np.load(Path(pkgdir) / f"{n}.npz")
        for m in PKG:
            F[m] = z[m]
        data[n] = (F, y)
    return data


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cmd_feats(a):
    from singlet.features import doublet as D
    from singlet.features import doublet_atlas as DA
    from singlet.features import doublet_sdfl as DS

    enc = DA.Atlas.load()
    out = Path(a.out) / "pkgfeat"
    out.mkdir(parents=True, exist_ok=True)
    cost = {}
    for n in a.names or list(D.STUDY):
        X, y, genes, species = load_counts(n)
        w0, c0 = time.perf_counter(), time.process_time()
        F = enc.features(X, genes, species)
        w1, c1 = time.perf_counter(), time.process_time()
        F.update(DS.score(X, ("a", "c")))
        w2, c2 = time.perf_counter(), time.process_time()
        cost[n] = dict(n_cells=int(X.shape[0]), atlas_wall=round(w1 - w0, 2), atlas_cpu=round(c1 - c0, 2),
                       sdfl_wall=round(w2 - w1, 2), sdfl_cpu=round(c2 - c1, 2))
        np.savez(out / f"{n}.npz", **{m: np.asarray(F[m]) for m in PKG})
        print(n, cost[n], flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))


def cmd_train(a):
    from singlet.features import doublet as D
    from singlet.features import doublet_atlas as DA

    data = load_training(Path(a.out) / "pkgfeat")
    st = D.train_frozen(data, dict(
        encoder_content_hash=DA.Atlas.load().content_hash(),
        training_inputs="the 17 datasets' inputs as computed by this package (feats); engine score: prod.npy of the study",
    ))
    md = Path(a.model_dir)
    md.mkdir(parents=True, exist_ok=True)
    st.save(md / "doublet_v2_stacker.npz", md / "doublet_v2_stacker.json")
    clf = D.fit_hgb(data, list(D.STUDY))
    Xr = D.design(data["kang21"][0])
    d = np.abs(clf.predict_proba(Xr)[:, 1] - st.predict_proba(Xr)).max()
    print("saved", md, st.content_hash(), "max |sklearn - exported|:", d)
    assert d < 1e-9


def _ci(ap):
    import pandas as pd

    from singlet.features import doublet as D

    ref = pd.read_csv(f"{LQC}/dbl/eval/auprc_wide.csv", index_col=0)["scdblfinder"].reindex(list(ap))
    d = pd.Series(ap) - ref
    stu = pd.Series(D.STUDY)
    rng = np.random.default_rng(0)
    b1 = [d.iloc[rng.integers(0, len(d), len(d))].mean() for _ in range(5000)]
    g = d.groupby(stu.reindex(d.index)).mean()
    b2 = [g.iloc[rng.integers(0, len(g), len(g))].mean() for _ in range(5000)]
    q = lambda b: [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]  # noqa: E731
    return dict(vs_scdblfinder=round(float(d.mean()), 4), ci95=q(b1), ci95_study_clustered=q(b2))


def cmd_loso(a):
    from sklearn.metrics import average_precision_score

    from singlet.features import doublet as D

    data = load_training(Path(a.out) / "pkgfeat")
    out = Path(a.out) / "loso_pkg"
    out.mkdir(parents=True, exist_ok=True)
    ap = {}
    for n in a.names or list(D.STUDY):
        s = D.loso_scores(data, n)
        np.save(out / f"{n}.npy", s)
        ap[n] = float(average_precision_score(data[n][1], s))
        print("loso", n, round(ap[n], 4), flush=True)
    rep = dict(mean_auprc=round(float(np.mean(list(ap.values()))), 4), n=len(ap), per_dataset=ap)
    if len(ap) == len(D.STUDY):
        rep.update(_ci(ap))
    (out / "auprc.json").write_text(json.dumps(rep, indent=1))
    print({k: v for k, v in rep.items() if k != "per_dataset"})


def cmd_ref_loso(a):
    """stack.py of branch claude/expression-sota-31cf6d (SETS entry), verbatim recipe, on the same inputs; scores saved."""
    from scipy.stats import rankdata
    from sklearn.ensemble import HistGradientBoostingClassifier

    from singlet.features.doublet import INPUTS, STUDY

    D = load_training(Path(a.out) / "pkgfeat")
    names = list(STUDY)
    q = lambda F, cols: np.column_stack([rankdata(F[c]) / len(F[c]) for c in cols])  # noqa: E731
    out = Path(a.out) / "loso_ref"
    out.mkdir(parents=True, exist_ok=True)
    for n in a.names or names:
        tr = [m for m in names if STUDY[m] != STUDY[n]]
        Xtr = np.vstack([q(D[m][0], INPUTS) for m in tr])
        ytr = np.concatenate([D[m][1] for m in tr])
        w = np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in tr])
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
        ).fit(Xtr, ytr, sample_weight=w * len(w) / len(tr))
        np.save(out / f"{n}.npy", clf.predict_proba(q(D[n][0], INPUTS))[:, 1])
        print("ref loso", n, flush=True)


def cmd_score(a):
    from singlet.features import doublet as D
    from singlet.features import registry as REG
    from singlet.features.output import build_table, write_features

    X, y, genes, species = load_counts(a.name)
    prod = np.load(f"{SCORES}/{a.name}/prod.npy")
    species = species if species in D.SPECIES else "human"
    D.stacker(), D.atlas()  # model loading is a one-off per process, not per sample
    tim = {}
    w0, c0 = time.perf_counter(), time.process_time()
    F = D.compute_inputs(X, genes, species, prod, timings=tim)
    t = time.perf_counter()
    score = D.score_inputs(F)
    tim["stacker_s"] = time.perf_counter() - t
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    rep = dict(dataset=a.name, n_cells=int(X.shape[0]), threads=int(os.environ.get("OMP_NUM_THREADS", "0")), wall_s=round(wall, 2),
               cpu_s=round(cpu, 2), stage_wall_s={k: round(v, 2) for k, v in tim.items()}, peak_rss_mb=round(_rss_mb()),
               score_min_max=[float(np.nanmin(score)), float(np.nanmax(score))])
    feats = REG.features("doublet")
    tab = build_table("GSM9000001", np.arange(len(score)), {"score_v2": score}, feats, REG.method_version("doublet"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_features(tab, out / f"{a.name}.features.doublet.parquet", feats)
    (out / f"{a.name}.cost.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm in ("feats", "train", "loso", "ref-loso"):
        p = sub.add_parser(nm)
        p.add_argument("--out", required=True)
        p.add_argument("--names", nargs="*")
        if nm == "train":
            p.add_argument("--model-dir", required=True)
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    {"feats": cmd_feats, "train": cmd_train, "loso": cmd_loso, "ref-loso": cmd_ref_loso, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
