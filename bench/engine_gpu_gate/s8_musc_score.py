# SPDX-License-Identifier: MIT
"""Score Singlet's U calls with and without kFlank (intron rules 389 vs 397) against the MuSC long-read truth (s8_musc_truth.py).
usage: s8_musc_score.py TRUTH.tsv.gz OUT.json RULES_A=DUMP_A RULES_B=DUMP_B     (dumps: SINGLET_MOLECULE_DUMP tsv.gz; label S/U/A, D = dropped)
Molecule level: a Singlet molecule is scored when the truth has the same CB + UMI + gene. U precision / recall / F1 against Truth-LR (>= 10 intronic bases; 5 and 20 as sensitivity),
(cell, gene) level: 'gene has an unspliced molecule in this cell'. Differences B - A with a 95% bootstrap over cells (1000 resamples)."""
import json
import re
import sys

import numpy as np
import pandas as pd

truth = pd.read_csv(sys.argv[1], sep="\t")
print(truth.split.value_counts().to_dict(), file=sys.stderr)
truth["gene"] = truth.gene.str.replace(r"\..*$", "", regex=True)
tkey = truth.set_index(["CB", "UB", "gene"])
dumps = dict(a.split("=", 1) for a in sys.argv[3:])


def load(path):
    parts = []
    for ch in pd.read_csv(path, sep="\t", chunksize=4_000_000, dtype={"barcode": str, "gene_id": str, "umi": str, "label": str}):
        ch = ch[ch.label != "D"]
        ch = ch.assign(CB=ch.barcode.str.split("-").str[0], umi=ch.umi.str[:10], gene=ch.gene_id.str.replace(r"\..*$", "", regex=True))
        ch = ch[ch.CB.isin(set(truth.CB))]
        parts.append(ch[["CB", "umi", "gene", "label", "reads", "diag"]].rename(columns={"umi": "UB"}))
    d = pd.concat(parts, ignore_index=True)
    return d.join(tkey[["split", "clsLR", "clsLR5", "clsLR20"]], on=["CB", "UB", "gene"])


def prf(tp, fp, fn):
    return {"tp": int(tp), "fp": int(fp), "fn": int(fn), "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "f1": 2 * tp / max(2 * tp + fp + fn, 1)}


def stats(d, col="clsLR"):
    t = d[col].notna()
    u, tu = (d.label == "U") & t, d[col] == "U"
    return prf((u & tu).sum(), (u & ~tu).sum(), (~(d.label == "U") & tu & t).sum())


def pair_stats(d, col="clsLR"):
    t = d[d[col].notna()]
    g = t.assign(cu=(t.label == "U"), tu=(t[col] == "U")).groupby(["CB", "gene"])[["cu", "tu"]].any()
    return prf((g.cu & g.tu).sum(), (g.cu & ~g.tu).sum(), (~g.cu & g.tu).sum())


out = {"truth_molecules": int(len(truth)), "truth_cells": int(truth.CB.nunique()),
       "truth_classes": {s: truth[truth.split == s].clsLR.value_counts().to_dict() for s in ("dev", "test")}}
D = {k: load(p) for k, p in dumps.items()}
for k, d in D.items():
    out[k] = {"molecules_in_truth_cells": int(len(d)), "with_truth": int(d.clsLR.notna().sum()), "U_calls": int((d.label == "U").sum()),
              "confusion_LR": {f"{a}->{b}": int(((d.label == a) & (d.clsLR == b)).sum()) for a in "SUA" for b in "SUA"}}
    for sub in ("all", "dev", "test"):
        m = d if sub == "all" else d[d.split == sub]
        if len(m) == 0:
            continue
        out[k][sub] = {"mol": stats(m), "mol_LR5": stats(m, "clsLR5"), "mol_LR20": stats(m, "clsLR20"), "pair": pair_stats(m)}
(ka, da), (kb, db) = list(D.items())[:2]
rng = np.random.default_rng(1)
cells = np.array(sorted(truth.CB.unique()))
split_of = truth.drop_duplicates("CB").set_index("CB").split
res = {}
for sub in ("all", "test"):
    cs = cells if sub == "all" else np.array([c for c in cells if split_of[c] == sub])
    if len(cs) == 0:
        continue
    ga = {c: x for c, x in da[da.clsLR.notna()].groupby("CB")}
    gb = {c: x for c, x in db[db.clsLR.notna()].groupby("CB")}
    def cnt(g, c):
        x = g.get(c)
        if x is None:
            return np.zeros(3)
        u, tu = (x.label == "U"), (x.clsLR == "U")
        return np.array([(u & tu).sum(), (u & ~tu).sum(), (~u & tu).sum()], float)
    A = np.array([cnt(ga, c) for c in cs]); B = np.array([cnt(gb, c) for c in cs])
    f = lambda v: (2 * v[0] / max(2 * v[0] + v[1] + v[2], 1), v[0] / max(v[0] + v[1], 1), v[0] / max(v[0] + v[2], 1))   # noqa: E731
    pt = np.array(f(B.sum(0))) - np.array(f(A.sum(0)))
    bs = []
    for _ in range(1000):
        i = rng.integers(0, len(cs), len(cs))
        bs.append(np.array(f(B[i].sum(0))) - np.array(f(A[i].sum(0))))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    res[sub] = {n: {"delta": float(pt[j]), "ci95": [float(lo[j]), float(hi[j])]} for j, n in enumerate(("f1", "precision", "recall"))} | {"cells": int(len(cs))}
out[f"delta_{kb}_minus_{ka}"] = res
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
