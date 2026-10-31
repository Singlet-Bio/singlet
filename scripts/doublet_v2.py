#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Training, validation and cost measurement of the ``doublet/score_v2`` stacker (docs/cell_features.md section 8). Clipper jobs
(scripts/doublet_v2.sbatch); real data only: the 17 real-label datasets under $LQC/dbl/data with their precomputed scores in
$LQC/dbl/scores/<name>/<method>.npy.

  train      fit the stacker on all 17 datasets and write data/doublet_v2_stacker.{npz,json} (--model-dir) with the sha256 of the
             cell type network and atlas centroid files
  loso       package leave-one-study-out scores -> OUT/loso_pkg/<name>.npy
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


def load_training():
    from singlet.features import doublet as D

    data = {}
    for n in D.STUDY:
        X, y, _, _ = load_counts(n)
        F = D.basic_inputs(X)
        for m in STORED:
            F[m] = np.load(f"{SCORES}/{n}/{m}.npy")
        data[n] = (F, y)
    return data


def cmd_train(a):
    from singlet.features import doublet as D
    from singlet.features.doublet_atlas import ENV_CENTROIDS, ENV_MODEL, sha256_file

    data = load_training()
    meta = dict(
        atlas_encoder="celltype_nn/m1_notissue",
        atlas_encoder_sha256=sha256_file(os.environ[ENV_MODEL]),
        centroids_sha256=sha256_file(os.environ[ENV_CENTROIDS]),
        training_scores="bench/latent_qc + bench/expression_sota stored scores of the 17 datasets (prod, atlas_*, ct_entropy, sdfl, sdflc)",
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

    data = load_training()
    out = Path(a.out) / "loso_pkg"
    out.mkdir(parents=True, exist_ok=True)
    from sklearn.metrics import average_precision_score

    ap = {}
    for n in a.names or list(D.STUDY):
        s = D.loso_scores(data, n)
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
    for nm in ("loso", "ref-loso"):
        p = sub.add_parser(nm)
        p.add_argument("--out", required=True)
        p.add_argument("--names", nargs="*")
    s = sub.add_parser("score")
    s.add_argument("name")
    s.add_argument("--out", required=True)
    s.add_argument("--device", default="cpu")
    a = ap.parse_args()
    {"train": cmd_train, "loso": cmd_loso, "ref-loso": cmd_ref_loso, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
