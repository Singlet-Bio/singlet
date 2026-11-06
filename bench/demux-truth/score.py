#!/usr/bin/env python3
"""Score the runs of one mixture directory against truth and against each other.

usage: score.py MIXDIR   ->  MIXDIR/metrics.json
Needs numpy, scipy, scikit-learn.
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, roc_auc_score

PAIRS = [  # (name, a, b, kind)
    ("repeat_t4", "k_t4_r1", "k_t4_r2", "repeat"),
    ("repeat_t16", "k_t16_r1", "k_t16_r2", "repeat"),
    ("threads_4v8", "k_t4_r1", "k_t8_r1", "threads"),
    ("threads_4v16", "k_t4_r1", "k_t16_r1", "threads"),
    ("seed_42v7", "k_t16_r1", "k_t16_s7", "seed"),
    ("auto_repeat", "auto_t16_r1", "auto_t16_r2", "repeat"),
    ("auto_vs_given", "k_t16_r1", "auto_t16_r1", "kmode"),
]


def read_tsv(path):
    with open(path) as f:
        head = f.readline().rstrip("\n").split("\t")
        rows = [l.rstrip("\n").split("\t") for l in f]
    return {h: [r[i] if i < len(r) else "" for r in rows] for i, h in enumerate(head)}


def hungarian_map(truth, pred, pred_labels):
    """Map predicted donor labels to truth labels maximizing matches. Returns dict pred->truth."""
    tl = sorted(set(truth))
    M = np.zeros((len(tl), len(pred_labels)), dtype=np.int64)
    ti = {t: i for i, t in enumerate(tl)}
    pi = {p: i for i, p in enumerate(pred_labels)}
    for t, p in zip(truth, pred):
        if p in pi:
            M[ti[t], pi[p]] += 1
    r, c = linear_sum_assignment(-M)
    return {pred_labels[j]: tl[i] for i, j in zip(r, c)}


def load_run(d, tag):
    a = read_tsv(d / "runs" / tag / "assign.tsv")
    return np.array(a["label"]), np.array(a["prob_max"], dtype=float), np.array(a["prob_doublet"], dtype=float)


def truth_metrics(truth, lab, pmax, pdbl):
    cls = np.array(truth["class"])
    don = np.array(truth["donor_a"])
    sing = cls == "singlet"
    dbl = ~sing
    donor_labels = sorted({l for l in lab if l.startswith("donor")})
    mp = hungarian_map(list(don[sing]), list(lab[sing]), donor_labels)
    mapped = np.array([mp.get(l, "NA") for l in lab])
    correct = (mapped == don) & sing
    assigned = np.array([l.startswith("donor") for l in lab])
    unas = lab == "unassigned"
    pred_dbl = (lab == "doublet") | (pdbl > 0.5)
    tp = int((pred_dbl & dbl).sum())
    fp = int((pred_dbl & sing).sum())
    fn = int((~pred_dbl & dbl).sum())
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    score = 1.0 - pmax
    auroc = float(roc_auc_score(dbl.astype(int), score)) if dbl.any() and sing.any() else None
    # unassigned used as a doublet flag
    utp = int((unas & dbl).sum())
    ufp = int((unas & sing).sum())
    return {
        "n_cells": int(len(lab)), "n_singlets": int(sing.sum()), "n_doublets": int(dbl.sum()),
        "k_found": len(donor_labels),
        "accuracy_singlets": float(correct.sum() / sing.sum()),
        "accuracy_assigned_singlets": float(correct.sum() / max(1, (assigned & sing).sum())),
        "ari_singlets": float(adjusted_rand_score(don[sing], lab[sing])),
        "unassigned_frac": float(unas.mean()),
        "unassigned_frac_singlets": float(unas[sing].mean()),
        "unassigned_frac_doublets": float(unas[dbl].mean()) if dbl.any() else None,
        "doublet_called_n": int(pred_dbl.sum()),
        "doublet_precision": prec, "doublet_recall": rec, "doublet_f1": f1,
        "doublet_auroc_1_minus_pmax": auroc,
        "unassigned_as_doublet_precision": utp / (utp + ufp) if utp + ufp else None,
        "unassigned_as_doublet_recall": utp / dbl.sum() if dbl.any() else None,
    }


def pair_metrics(la, pa, lb, pb):
    da = sorted({l for l in la if l.startswith("donor")})
    db = sorted({l for l in lb if l.startswith("donor")})
    # map b labels onto a labels
    mp = {}
    if da and db:
        M = np.zeros((len(da), len(db)), dtype=np.int64)
        ia = {x: i for i, x in enumerate(da)}
        ib = {x: i for i, x in enumerate(db)}
        for x, y in zip(la, lb):
            if x in ia and y in ib:
                M[ia[x], ib[y]] += 1
        r, c = linear_sum_assignment(-M)
        mp = {db[j]: da[i] for i, j in zip(r, c)}
    mb = np.array([mp.get(y, y) if y.startswith("donor") else y for y in lb])
    both = np.array([x.startswith("donor") for x in la]) & np.array([y.startswith("donor") for y in lb])
    dp = np.abs(pa - pb)
    return {
        "agreement": float((la == mb).mean()),
        "ari": float(adjusted_rand_score(la, lb)),
        "k_a": len(da), "k_b": len(db),
        "dprob_mean": float(dp.mean()),
        "dprob_p99": float(np.percentile(dp, 99)),
        "dprob_max": float(dp.max()),
        "dprob_mean_both_assigned": float(dp[both].mean()) if both.any() else None,
    }


def main(d):
    d = Path(d)
    truth = read_tsv(d / "truth.tsv")
    runs = [json.loads(l) for l in open(d / "runs.jsonl")]
    out = {"mix": json.load(open(d / "mix.json")), "runs": {}, "pairs": {}}
    cache = {}
    for r in runs:
        lab, pmax, pdbl = load_run(d, r["tag"])
        cache[r["tag"]] = (lab, pmax)
        m = truth_metrics(truth, lab, pmax, pdbl)
        m.update({k: r[k] for k in ("threads", "seed", "k_arg", "wall_s", "cpu_s")})
        out["runs"][r["tag"]] = m
    for name, a, b, kind in PAIRS:
        if a in cache and b in cache:
            pm = pair_metrics(cache[a][0], cache[a][1], cache[b][0], cache[b][1])
            pm.update({"a": a, "b": b, "kind": kind})
            out["pairs"][name] = pm
    json.dump(out, open(d / "metrics.json", "w"), indent=1)
    print("scored", d, len(out["runs"]), "runs")


if __name__ == "__main__":
    main(sys.argv[1])
