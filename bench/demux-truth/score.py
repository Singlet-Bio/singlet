#!/usr/bin/env python3
"""Score the runs of one mixture directory against truth and against each other.

usage: score.py MIXDIR   ->  MIXDIR/metrics.json
MIXDIR holds truth.tsv, mix.json, runs*.jsonl (one file per tool) and runs/<tag>/assign.tsv.
Run tags are <tool>_<mode>_t<threads>_s<seed>, mode k (K given) or auto.
Needs numpy, scipy, scikit-learn.
"""
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, roc_auc_score

TAG = re.compile(r"^(?P<tool>[a-z0-9]+)_(?P<mode>k|auto)_t(?P<t>\d+)_s(?P<s>\d+)$")


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


def f1_of(pred, dbl):
    tp = int((pred & dbl).sum())
    fp = int((pred & ~dbl).sum())
    fn = int((~pred & dbl).sum())
    return (2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else None, tp, fp, fn


def truth_metrics(truth, lab, pmax, pdbl):
    cls = np.array(truth["class"])
    don = np.array(truth["donor_a"])
    sing = cls == "singlet"
    dbl = ~sing
    donor_labels = sorted({l for l in lab if l.startswith("donor")})
    mp = hungarian_map(list(don[sing]), list(lab[sing]), donor_labels) if donor_labels else {}
    mapped = np.array([mp.get(l, "NA") for l in lab])
    correct = (mapped == don) & sing
    assigned = np.array([l.startswith("donor") for l in lab])
    unas = lab == "unassigned"
    called = lab == "doublet"
    f1, tp, fp, fn = f1_of(called, dbl)
    f1_half, _, _, _ = f1_of(pdbl > 0.5, dbl)
    # doublets are scored with prob_doublet, never with 1 - prob_max
    auroc = float(roc_auc_score(dbl.astype(int), pdbl)) if dbl.any() and sing.any() and pdbl.std() > 0 else None
    return {
        "n_cells": int(len(lab)), "n_singlets": int(sing.sum()), "n_doublets": int(dbl.sum()),
        "k_found": len(donor_labels),
        "accuracy_singlets": float(correct.sum() / sing.sum()),
        "accuracy_assigned_singlets": float(correct.sum() / max(1, (assigned & sing).sum())),
        "ari_singlets": float(adjusted_rand_score(don[sing], lab[sing])),
        "unassigned_frac": float(unas.mean()),
        "unassigned_frac_singlets": float(unas[sing].mean()),
        "doublet_called_n": int(called.sum()),
        "doublet_precision": tp / (tp + fp) if tp + fp else None,
        "doublet_recall": tp / (tp + fn) if tp + fn else None,
        "doublet_f1": f1, "doublet_f1_at_0.5": f1_half,
        "doublet_auroc": auroc,
    }


def pair_metrics(la, pa, lb, pb):
    da = sorted({l for l in la if l.startswith("donor")})
    db = sorted({l for l in lb if l.startswith("donor")})
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
    dp = np.abs(pa - pb)
    return {
        "agreement": float((la == mb).mean()),
        "ari": float(adjusted_rand_score(la, lb)),
        "k_a": len(da), "k_b": len(db),
        "dprob_mean": float(dp.mean()), "dprob_p99": float(np.percentile(dp, 99)),
    }


def main(d):
    d = Path(d)
    truth = read_tsv(d / "truth.tsv")
    runs = []
    for fn in sorted(glob.glob(str(d / "runs*.jsonl"))):
        runs += [json.loads(l) for l in open(fn) if l.strip()]
    out = {"mix": json.load(open(d / "mix.json")), "runs": {}, "pairs": {}, "diag": []}
    cache = {}
    for r in runs:
        if not (d / "runs" / r["tag"] / "assign.tsv").exists():
            continue
        lab, pmax, pdbl = load_run(d, r["tag"])
        cache[r["tag"]] = (lab, pmax)
        m = truth_metrics(truth, lab, pmax, pdbl)
        m.update({k: r.get(k) for k in ("tool", "threads", "seed", "k_arg", "wall_s", "cpu_s")})
        out["runs"][r["tag"]] = m
    # pairs inside one tool: thread counts at one seed, and seeds at one thread count
    by = {}
    for tag in cache:
        mt = TAG.match(tag)
        if mt:
            by.setdefault((mt["tool"], mt["mode"]), {})[(int(mt["t"]), int(mt["s"]))] = tag
    for (tool, mode), g in by.items():
        for (t, s), tag in g.items():
            for (t2, s2), tag2 in g.items():
                if (t, s) < (t2, s2) and (t == t2 or s == s2):
                    kind = "threads" if s == s2 else "seed"
                    pm = pair_metrics(cache[tag][0], cache[tag][1], cache[tag2][0], cache[tag2][1])
                    pm.update({"a": tag, "b": tag2, "kind": kind, "tool": tool, "mode": mode})
                    out["pairs"][f"{tag}__{tag2}"] = pm
    if (d / "diag.jsonl").exists():
        out["diag"] = [json.loads(l) for l in open(d / "diag.jsonl") if l.strip()]
    json.dump(out, open(d / "metrics.json", "w"), indent=1)
    print("scored", d, len(out["runs"]), "runs")


if __name__ == "__main__":
    main(sys.argv[1])
