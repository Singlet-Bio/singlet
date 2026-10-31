#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training, validation and cost measurement of the ``doublet/score_v2`` stacker (docs/cell_features.md section 8). Clipper jobs
(scripts/doublet_v2.sbatch); real data only: the 17 real-label datasets under $LQC/dbl/data with their precomputed scores in
$LQC/dbl/scores/<name>/<method>.npy.

  train      fit the stacker on all 17 datasets and write data/doublet_v2_stacker.{npz,json} (--model-dir) with the sha256 of the
             cell type network and atlas centroid files
  feats      the atlas-encoder and sdfl inputs recomputed by the package for the 17 datasets -> OUT/pkgfeat (+ cost per dataset)
  loso       package leave-one-study-out scores -> OUT/loso_pkg/<name>.npy (--features stored | pkg | cross: train on stored,
             score the held-out dataset's package-computed inputs)
  ref-loso   the recipe of bench/expression_sota/doublets/stack.py, unchanged, scores saved -> OUT/loso_ref/<name>.npy
  score      score one dataset with the whole package (atlas encoder + sdfl + stacker) from its counts and the engine score, write
             the sidecar parquet, wall / CPU seconds / peak memory per stage, and agreement with the stored feature scores
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
STORED = ("prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc")


def load_counts(name):
    z = np.load(f"{DATA}/{name}/ds.npz")
    X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
    return X, z["label"].astype(int), z["genes"].astype(str), str(z["species"])


def load_training(kind="stored", pkgdir=None):
    """Inputs of the 17 datasets: ``stored`` = the precomputed scores of $LQC/dbl/scores; ``pkg`` = recomputed by this package
    (``feats`` command; the engine score stays the stored one)."""
    from singlet.features import doublet as D

    data = {}
    for n in D.STUDY:
        X, y, _, _ = load_counts(n)
        F = D.basic_inputs(X)
        if kind == "stored":
            for m in STORED:
                F[m] = np.load(f"{SCORES}/{n}/{m}.npy")
        else:
            z = np.load(Path(pkgdir) / f"{n}.npz")
            for m in STORED:
                F[m] = z[m]
        data[n] = (F, y)
    return data


def cmd_feats(a):
    """Package-computed inputs (atlas encoder + sdfl) for every dataset -> OUT/pkgfeat/<name>.npz and per-dataset cost."""
    from singlet.features import doublet as D

    out = Path(a.out) / "pkgfeat"
    out.mkdir(parents=True, exist_ok=True)
    cost = {}
    for n in a.names or list(D.STUDY):
        X, y, genes, species = load_counts(n)
        prod = np.load(f"{SCORES}/{n}/prod.npy")
        tim = {}
        w0, c0 = time.perf_counter(), time.process_time()
        F = D.compute_inputs(X, genes, species if species in D.SPECIES else "human", prod, device=a.device, timings=tim)
        cost[n] = dict(n_cells=int(X.shape[0]), wall_s=round(time.perf_counter() - w0, 1), cpu_s=round(time.process_time() - c0, 1),
                       **{k: round(v, 1) for k, v in tim.items()}, peak_rss_mb=round(_rss_mb()))
        np.savez(out / f"{n}.npz", **{m: np.asarray(F[m]) for m in STORED})
        print(n, cost[n], flush=True)
    (out / "cost.json").write_text(json.dumps(cost, indent=1))


def cmd_train(a):
    from singlet.features import doublet as D
    from singlet.features.doublet_atlas import ENV_CENTROIDS, ENV_MODEL, sha256_file

    data = load_training(a.features, a.pkgdir)
    meta = dict(
        atlas_encoder="celltype_nn/m1_notissue",
        atlas_encoder_sha256=sha256_file(os.environ[ENV_MODEL]),
        centroids_sha256=sha256_file(os.environ[ENV_CENTROIDS]),
        training_scores=(
            "stored scores of the 17 datasets (bench/latent_qc, bench/expression_sota): prod, atlas_*, ct_entropy, sdfl, sdflc"
            if a.features == "stored"
            else "the 17 datasets' atlas and sdfl inputs recomputed by this package (engine score: stored prod)"
        ),
    )
    st = D.train_frozen(data, meta)
    md = Path(a.model_dir)
    md.mkdir(parents=True, exist_ok=True)
    st.save(md / "doublet_v2_stacker.npz", md / "doublet_v2_stacker.json")
    print("saved", md, st.content_hash(), json.dumps(st.meta)[:300])
    # exported trees must reproduce the sklearn model they came from
    clf = D.fit_hgb(data, list(D.STUDY))
    Xr = D.design(data["kang21"][0])
    d = np.abs(clf.predict_proba(Xr)[:, 1] - st.predict_proba(Xr)).max()
    print("max |sklearn - exported| on kang21:", d)
    assert d < 1e-9


def cmd_loso(a):
    from singlet.features import doublet as D

    data = load_training("stored" if a.features != "pkg" else "pkg", a.pkgdir)
    test = load_training("pkg", a.pkgdir) if a.features == "cross" else None
    out = Path(a.out) / f"loso_{a.features}"
    out.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import average_precision_score

    ap = {}
    for n in a.names or list(D.STUDY):
        s = D.loso_scores(data, n, test=test)
        np.save(out / f"{n}.npy", s)
        ap[n] = float(average_precision_score(data[n][1], s))
        print("loso", n, round(ap[n], 4), flush=True)
    print("mean leave-one-study-out AUPRC", round(float(np.mean(list(ap.values()))), 4), "over", len(ap))
    (out / "auprc.json").write_text(json.dumps(ap, indent=1))


def cmd_ref_loso(a):
    """stack.py of branch claude/expression-sota-31cf6d (SETS["combo_pa_sdfl2"]), verbatim recipe, scores saved."""
    from scipy.stats import rankdata
    from sklearn.ensemble import HistGradientBoostingClassifier

    sys.path.insert(0, str(ROOT / "python"))
    from singlet.features.doublet import STUDY

    names = [n for n in STUDY]
    cols = ["umi", "genes", "prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc"]
    D = {}
    for n in names:
        z = np.load(f"{DATA}/{n}/ds.npz")
        X = sp.csr_matrix((z["X_data"], z["X_indices"], z["X_indptr"]), shape=tuple(z["X_shape"]))
        F = {"umi": np.log1p(np.asarray(X.sum(1)).ravel()), "genes": np.log1p(np.diff(X.indptr))}
        for m in ("prod", "atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc"):
            p = f"{SCORES}/{n}/{m}.npy"
            if os.path.exists(p):
                F[m] = np.load(p)
        D[n] = (F, z["label"].astype(int))
    q = lambda F, cols: np.column_stack([rankdata(F[c]) / len(F[c]) for c in cols])  # noqa: E731
    out = Path(a.out) / "loso_ref"
    out.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import average_precision_score

    ap = {}
    for n in a.names or names:
        tr = [m for m in names if STUDY[m] != STUDY[n] and all(c in D[m][0] for c in cols)]
        Xtr = np.vstack([q(D[m][0], cols) for m in tr])
        ytr = np.concatenate([D[m][1] for m in tr])
        w = np.concatenate([np.full(len(D[m][1]), 1 / len(D[m][1])) for m in tr])
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=0
        ).fit(Xtr, ytr, sample_weight=w * len(w) / len(tr))
        s = clf.predict_proba(q(D[n][0], cols))[:, 1]
        np.save(out / f"{n}.npy", s)
        ap[n] = float(average_precision_score(D[n][1], s))
        print("ref loso", n, round(ap[n], 4), flush=True)
    print("mean reference leave-one-study-out AUPRC", round(float(np.mean(list(ap.values()))), 4), "over", len(ap))


def _rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def cmd_score(a):
    from scipy.stats import spearmanr

    from singlet.features import doublet as D
    from singlet.features import registry as REG
    from singlet.features.output import build_table, write_features

    X, y, genes, species = load_counts(a.name)
    prod = np.load(f"{SCORES}/{a.name}/prod.npy")
    species = species if species in D.SPECIES else "human"
    tim = {}
    w0, c0 = time.perf_counter(), time.process_time()
    F = D.compute_inputs(X, genes, species, prod, device=a.device, timings=tim)
    t_stack = time.perf_counter()
    score = D.score_inputs(F)
    tim["stacker_s"] = time.perf_counter() - t_stack
    wall, cpu = time.perf_counter() - w0, time.process_time() - c0
    ch = resource.getrusage(resource.RUSAGE_CHILDREN)
    rep = dict(
        dataset=a.name,
        n_cells=int(X.shape[0]),
        device=a.device,
        threads=int(os.environ.get("OMP_NUM_THREADS", "0")),
        wall_s=round(wall, 1),
        cpu_s=round(cpu, 1),
        stage_wall_s={k: round(v, 1) for k, v in tim.items()},
        peak_rss_mb=round(max(_rss_mb(), ch.ru_maxrss / 1024.0)),
        agreement_with_stored_spearman={},
    )
    for m in ("atlas_knn", "atlas_clf", "ct_entropy", "atlas_nnls", "sdfl", "sdflc"):
        rep["agreement_with_stored_spearman"][m] = round(float(spearmanr(F[m], np.load(f"{SCORES}/{a.name}/{m}.npy"))[0]), 4)
    # sidecar through the writer and guard
    feats = REG.features("doublet")
    tab = build_table("GSM9000001", np.arange(len(score)), {"score_v2": score}, feats, REG.method_version("doublet"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_features(tab, out / f"{a.name}.features.doublet.parquet", feats)
    rep["score_min_max"] = [float(np.nanmin(score)), float(np.nanmax(score))]
    (out / f"{a.name}.cost.{a.device}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))
    # the sidecar's own score must not be quoted as accuracy: these datasets are the training data of the frozen model


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--model-dir", required=True)
    t.add_argument("--features", choices=("stored", "pkg"), default="stored")
    t.add_argument("--pkgdir")
    for nm in ("loso", "ref-loso", "feats"):
        p = sub.add_parser(nm)
        p.add_argument("--out", required=True)
        p.add_argument("--names", nargs="*")
        if nm == "loso":
            p.add_argument("--features", choices=("stored", "pkg", "cross"), default="stored")
            p.add_argument("--pkgdir")
        if nm == "feats":
            p.add_argument("--device", default="cpu")
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"train": cmd_train, "feats": cmd_feats, "loso": cmd_loso, "ref-loso": cmd_ref_loso, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
