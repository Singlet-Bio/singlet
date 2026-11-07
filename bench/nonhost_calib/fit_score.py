# SPDX-License-Identifier: MIT
"""Fit and score the calibrated infection probability of the non-host output (claims-to-specs T31) on the development libraries.

    python -P fit_score.py J OUT.json

J holds runs/<id>/nonhost/nonhost_candidates.tsv (engine, run_dev.sbatch) and kraken/<id>/{report.txt,reads.txt} (Kraken2 PlusPF
k2_pluspf_20261022, confidence 0.1, 50 M reads, kraken_pf.sbatch). Truth is taxon specific: a pair (library, taxon) is a positive when the
library is infected with the taxon (SARS-CoV-2 in ravindra dpi1 and dpi2; no EBV positive: the lymphoblastoid libraries of the corpus are not droplet data) and a negative when the
taxon is known absent (SARS-CoV-2, influenza A and Salmonella enterica in every other library; EBV is no negative anywhere: it is latent
in blood cells). Singlet's evidence for a pair is the reads and the hit-rate of the EM candidates whose names match the taxon; Kraken2's
is the clade reads of the taxon. Each tool's map to a probability is a logistic regression (ridge 1.0 on the slopes), fitted on the pairs
below: Singlet on log10 read fraction and mean hit rate, Kraken2 on log10 read fraction. Scores: class-balanced log score (mean of the mean
log probability of the truth over positives and over negatives), in sample and by leave-one-library-out.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

J, OUT = sys.argv[1:3]
POS = {"ravindra2021_dpi1": "sars2", "ravindra2021_dpi2": "sars2"}  # the two lymphoblastoid libraries (GSE230239) are 2 x 101 nt reads, not droplet libraries: no EBV positive could be run
NEG_LIBS = ["ravindra2021_mock", "pbmc1k_v3", "pbmc1k_v2", "pbmc5p_v1", "hgmm1k_v3", "neuron1k_v3", "nuclei2k_v2", "kang2018_ctrl"]
LIBS = list(POS) + NEG_LIBS
TAXA = {"sars2": (re.compile(r"NC_045512|coronavirus 2", re.I), 694009), "ebv": (re.compile(r"gammaherpesvirus 4|Epstein", re.I), 10376),
        "flua": (re.compile(r"Influenza A virus", re.I), 11320), "salm": (re.compile(r"Salmonella", re.I), 28901)}
NEGATIVE_TAXA = ["sars2", "flua", "salm"]


def singlet_evidence(lib, taxon):
    p = f"{J}/runs/{lib}/nonhost/nonhost_candidates.tsv"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, sep="\t")
    if d.empty:
        return 0.0, 0.0, 1.0
    screened = float(d.screened_reads.iloc[0])
    m = d[d.species_name.str.contains(TAXA[taxon][0])]
    reads = float(m.reads_assigned.sum())
    hit = float((m.mean_hit_rate * m.reads_assigned).sum() / reads) if reads > 0 else 0.0
    return reads, hit, screened


def kraken_evidence(lib, taxon):
    p = f"{J}/kraken/{lib}/report.txt"
    if not os.path.exists(p) or os.path.getsize(p) == 0:
        return None
    n = float(open(f"{J}/kraken/{lib}/reads.txt").read().strip())
    clade = 0.0
    for l in open(p):
        f = l.rstrip("\n").split("\t")
        if len(f) >= 6 and f[4].strip() == str(TAXA[taxon][1]):
            clade = float(f[1])
    return clade, n


rows = []
for lib in LIBS:
    for tx in TAXA:
        if tx == "ebv" and POS.get(lib) != "ebv":
            continue  # EBV is a positive in the lymphoblastoid lines and no negative anywhere
        y = 1 if POS.get(lib) == tx else 0
        if y == 0 and tx not in NEGATIVE_TAXA:
            continue
        s, k = singlet_evidence(lib, tx), kraken_evidence(lib, tx)
        if s is None or k is None:
            continue
        reads, hit, scr = s
        rows.append({"lib": lib, "taxon": tx, "y": y, "s_reads": reads, "s_hit": hit, "s_screened": scr, "k_clade": k[0], "k_reads": k[1],
                     "s_x1": np.log10((reads + 0.5) / scr), "s_x2": hit, "k_x1": np.log10((k[0] + 0.5) / k[1])})
df = pd.DataFrame(rows)
print(df[["lib", "taxon", "y", "s_reads", "s_hit", "k_clade", "s_x1", "k_x1"]].round(3).to_string(index=False))


def fit(X, y, ridge=1.0):
    X1 = np.c_[np.ones(len(X)), X]
    b = np.zeros(X1.shape[1])
    R = np.diag([0.0] + [ridge] * X.shape[1])
    for _ in range(100):
        p = 1 / (1 + np.exp(-X1 @ b))
        g = X1.T @ (p - y) + R @ b
        H = (X1 * (p * (1 - p))[:, None]).T @ X1 + R + 1e-9 * np.eye(len(b))
        step = np.linalg.solve(H, g)
        b -= step
        if np.abs(step).max() < 1e-9:
            break
    return b


def prob(b, X):
    return 1 / (1 + np.exp(-(np.c_[np.ones(len(X)), X] @ b)))


def balanced(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(0.5 * (np.log(p[y == 1]).mean() + np.log(1 - p[y == 0]).mean()))


res = {"pairs": int(len(df)), "positives": int(df.y.sum()), "negatives": int((1 - df.y).sum()), "tools": {}}
for tool, cols in (("singlet", ["s_x1", "s_x2"]), ("kraken2_pluspf", ["k_x1"])):
    X, y = df[cols].values, df.y.values
    b = fit(X, y)
    p = prob(b, X)
    lolo = np.zeros(len(df))
    for lib in df.lib.unique():
        tr = (df.lib != lib).values
        if df.y[tr].sum() == 0 or (1 - df.y[tr]).sum() == 0:
            lolo[~tr] = np.nan
            continue
        lolo[~tr] = prob(fit(X[tr], y[tr]), X[~tr])
    ok = ~np.isnan(lolo)
    res["tools"][tool] = {"features": cols, "coef": [float(v) for v in b], "log_score_in_sample": balanced(p, y),
                          "log_score_leave_one_library_out": balanced(lolo[ok], y[ok]), "n_lolo": int(ok.sum()),
                          "p_positives": [round(float(v), 4) for v in p[y == 1]], "max_p_negatives": float(p[y == 0].max())}
    df[tool + "_p"] = p
    print(tool, json.dumps(res["tools"][tool]))
df.to_csv(OUT.replace(".json", "_pairs.csv"), index=False)
json.dump(res, open(OUT, "w"), indent=1)
