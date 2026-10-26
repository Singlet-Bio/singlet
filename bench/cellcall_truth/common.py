# SPDX-License-Identifier: MIT
"""Shared code for the independent-truth cell-calling study (bench/cellcall_truth/README.md).

Truth = "this barcode holds RNA from one source other than the ambient pool": in a genotype-multiplexed pool, the barcode's SNP
alleles fit one donor (or two, a doublet) better than the ambient donor mixture; in a barnyard, its human/mouse split fits one
species better than the ambient split. Both are scored with the same likelihood-ratio machinery:

    LL(model) = sum_s alt_s log f_s + ref_s log(1 - f_s)
    cell model: f = (1 - c) * source + c * ambient, c in C_GRID (ambient contamination of a real cell)
    LLR = max over cell models - LL(ambient)

and calibrated on barcodes far from every caller's threshold: high-UMI cells thinned to m informative UMIs (positives) and
real low-UMI empty droplets pooled and thinned to m informative UMIs (negatives). Per depth m, a barcode is a cell when its LLR
exceeds the 99th percentile of the negatives, empty when it is below the 1st percentile of the positives (when the two do not
overlap, one cut at the negatives' 99th percentile), unknown in between or below the minimum depth N where both classes are recovered >= 90%.
"""

from __future__ import annotations

import gzip
import json
import os

import numpy as np
import pandas as pd
import scipy.sparse as sp

W = "/mnt/projects/debruinz_project/debruinz/singlet"
PB = f"{W}/pipeline-benchmark"
BP = f"{W}/bench-public"
STATE = f"{W}/state/cellcall_truth"

C_GRID = np.array([0.0, 0.05, 0.1, 0.2, 0.35, 0.5])
M_GRID = np.array([3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300])
UMI_BINS = [0, 100, 200, 500, 1000, 2000, 5000, np.inf]

# Per dataset: the Singlet run (stored counts, cell_calls.tsv, cell_qc_metrics.tsv), STARsolo EmptyDrops_CR run, CellBender run,
# Cell Ranger published barcodes (fetched by prep.sbatch MODE=fetch), cv2_stored 2.1 recompute.
DATASETS = {
    "kang_b2ctrl": dict(kind="genotype", run=f"{PB}/runs/kang2018_ctrl/singlet-public",
                        star=f"{PB}/runs/kang2018_ctrl/starsolo-public-cr", cb=f"{PB}/runs/kang2018_ctrl/cellbender-public"),
    "hgmm1k_v3": dict(kind="barnyard", run=f"{PB}/runs/hgmm1k_v3/singlet-public",
                      star=f"{PB}/runs/hgmm1k_v3/starsolo-public-cr", cb=f"{PB}/runs/hgmm1k_v3/cellbender-public"),
    "hgmm5k_v3": dict(kind="barnyard", run=f"{BP}/runs/hgmm5k_v3/singlet",
                      star=f"{STATE}/starsolo/hgmm5k_v3", cb=f"{STATE}/cellbender/hgmm5k_v3"),
    "hgmm1k_v2": dict(kind="barnyard", run=f"{BP}/runs/hgmm1k_v2/singlet",
                      star=f"{STATE}/starsolo/hgmm1k_v2", cb=f"{STATE}/cellbender/hgmm1k_v2"),
}


def strip(b):
    return b.split("-")[0]


def lines(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        return [ln.rstrip("\n").split("\t")[0].split(",")[0] for ln in fh if ln.strip()]


def read_mtx(path):
    """Matrix Market coordinate file -> CSR (rows x cols) int64, fast via pandas."""
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        skip = 0
        line = fh.readline()
        while line.startswith("%"):
            skip += 1
            line = fh.readline()
        m, n, _ = (int(v) for v in line.split())
    df = pd.read_csv(path, sep=r"\s+", header=None, skiprows=skip + 1, dtype=np.int64, engine="c")
    return sp.csr_matrix((df[2].to_numpy(), (df[0].to_numpy() - 1, df[1].to_numpy() - 1)), shape=(m, n))


# ------------------------------------------------------------------------------------------------- caller columns
def gather_callers(ds, min_umi=10):
    """Per-barcode table for one dataset: depth, QC and every caller's call (0/1). Universe: barcodes with umi_total >= min_umi
    in the Singlet run, plus anything any caller calls."""
    cfg = DATASETS[ds]
    cc = pd.read_csv(f"{cfg['run']}/cell_calls.tsv", sep="\t", dtype={"barcode": str})
    cc["barcode"] = cc["barcode"].map(strip)
    qc = pd.read_csv(f"{cfg['run']}/cell_qc_metrics.tsv", sep="\t", dtype={"barcode": str},
                     usecols=["barcode", "n_umi", "n_genes", "pct_mt", "intronic_pct"])
    qc["barcode"] = qc["barcode"].map(strip)
    t = cc[["barcode", "umi_total", "umi_exon", "called", "called_v2"]].rename(columns={"called": "knee", "called_v2": "singlet_run"})
    t = t.merge(qc, on="barcode", how="left")
    calls = {}
    cv2 = f"{STATE}/cv2/{ds}.tsv"
    if os.path.exists(cv2):
        c2 = pd.read_csv(cv2, sep="\t", dtype={"barcode": str})
        calls["called_v2"] = set(c2.loc[c2["called_v2"] == 1, "barcode"].map(strip))
    fb = f"{cfg['star']}/Solo.out/Gene/filtered/barcodes.tsv"
    if os.path.exists(fb):
        calls["edcr"] = set(map(strip, lines(fb)))
    cbf = f"{cfg['cb']}/cb_cell_barcodes.csv"
    if os.path.exists(cbf):
        calls["cellbender"] = set(map(strip, lines(cbf)))
    crf = f"{STATE}/cellranger/{ds}.barcodes.tsv.gz"
    if os.path.exists(crf):
        calls["cellranger"] = set(map(strip, lines(crf)))
    if ds == "kang_b2ctrl":
        a = pd.read_csv(f"{W}/modality-bench/genotype_pool/truth/GSE96583_batch2.total.tsne.df.tsv.gz", sep="\t", index_col=0)
        a = a[a["stim"] == "ctrl"]
        calls["demuxlet_table"] = set(map(strip, a.index))
    union = set().union(*calls.values()) | set(t.loc[t["singlet_run"] == 1, "barcode"])
    t = t[(t["umi_total"] >= min_umi) | t["barcode"].isin(union)].copy()
    for k, s in calls.items():
        t[k] = t["barcode"].isin(s).astype(np.int8)
    if "called_v2" not in t:
        t["called_v2"] = t["singlet_run"]
    missing = {k: len(s - set(t["barcode"])) for k, s in calls.items()}
    return t.reset_index(drop=True), missing


# ------------------------------------------------------------------------------------------------ likelihood core
def ll_matrix(AD, DP, F):
    """AD, DP: CSR barcodes x sites; F: sites x models allele probabilities -> barcodes x models log-likelihood."""
    F = np.clip(F, 1e-6, 1 - 1e-6)
    return np.asarray(AD @ np.log(F) + (DP - AD) @ np.log1p(-F))


def align(AD, DP):
    """AD with exactly DP's sparsity structure (CSR, sorted), so row slices line up."""
    DP = sp.csr_matrix(DP, dtype=np.int64)
    DP.sum_duplicates()
    DP.sort_indices()
    r, c = DP.nonzero()
    A = DP.copy()
    A.data = np.asarray(sp.csr_matrix(AD)[r, c]).ravel().astype(np.int64)
    A.data = np.minimum(A.data, DP.data)
    return A, DP


def umis_of(AD, DP, i):
    """Row i as an array of (site, is_alt) UMIs (AD aligned to DP)."""
    s, e = DP.indptr[i], DP.indptr[i + 1]
    sites, dp = DP.indices[s:e], DP.data[s:e]
    ad = AD.data[s:e]
    site = np.repeat(sites, dp)
    alt = np.concatenate([np.r_[np.ones(a, np.int8), np.zeros(d - a, np.int8)] for a, d in zip(ad, dp)]) if len(dp) else np.zeros(0, np.int8)
    return site, alt


def to_csr(umi_list, n_sites):
    rows, cols, alts = [], [], []
    for r, (s, a) in enumerate(umi_list):
        rows.append(np.full(len(s), r))
        cols.append(s)
        alts.append(a)
    rows = np.concatenate(rows) if rows else np.zeros(0, int)
    cols = np.concatenate(cols) if cols else np.zeros(0, int)
    alts = np.concatenate(alts) if alts else np.zeros(0, int)
    n = len(umi_list)
    DP = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n_sites))
    AD = sp.csr_matrix((alts.astype(float), (rows, cols)), shape=(n, n_sites))
    DP.sum_duplicates()
    AD.sum_duplicates()
    return AD, DP


def thinned_positives(umis, m, rng):
    out = []
    for s, a in umis:
        if len(s) >= m:
            k = rng.choice(len(s), m, replace=False)
            out.append((s[k], a[k]))
    return out


def pooled_negatives(umis, m, n, rng):
    """n synthetic empty droplets of exactly m informative UMIs, each the pooled UMIs of random real empty barcodes."""
    sizes = np.array([len(s) for s, _ in umis])
    idx = np.nonzero(sizes > 0)[0]
    out = []
    for _ in range(n):
        perm = rng.permutation(idx)
        k = int(np.searchsorted(np.cumsum(sizes[perm]), m) + 1)
        s = np.concatenate([umis[j][0] for j in perm[:k]])
        a = np.concatenate([umis[j][1] for j in perm[:k]])
        if len(s) < m:
            continue
        kk = rng.choice(len(s), m, replace=False)
        out.append((s[kk], a[kk]))
    return out


def calibrate(llr_fn, pos_umis, neg_umis, n_sites, seed=7, n_neg=10000, min_rate=0.90, q_neg=0.999, q_pos=0.01):
    """Per depth m: LLR distributions of thinned positives and pooled negatives, thresholds and recovery rates.
    t_cell = q_neg quantile of the negatives (false-cell rate 0.1%: the empties outnumber the cells, so 1% is too loose),
    t_empty = q_pos quantile of the positives."""
    rng = np.random.default_rng(seed)
    rows = []
    for m in M_GRID:
        P = thinned_positives(pos_umis, m, rng)
        N = pooled_negatives(neg_umis, m, n_neg, rng)
        if len(P) < 50 or len(N) < 50:
            continue
        lp = llr_fn(*to_csr(P, n_sites))
        ln = llr_fn(*to_csr(N, n_sites))
        t_cell = float(np.quantile(ln, q_neg))
        t_empty = float(np.quantile(lp, q_pos))
        if t_empty >= t_cell:  # separable: one cut at the negatives' quantile (false-cell rate <= 1 - q_neg)
            t_empty = t_cell
        rows.append(dict(m=int(m), n_pos=len(P), n_neg=len(N), t_cell=t_cell, t_empty=t_empty,
                         pos_cell=float((lp > t_cell).mean()), pos_empty=float((lp < t_empty).mean()),
                         neg_cell=float((ln > t_cell).mean()), neg_empty=float((ln < t_empty).mean()),
                         pos_llr_med=float(np.median(lp)), neg_llr_med=float(np.median(ln))))
    cal = pd.DataFrame(rows)
    ok = cal[(cal["pos_cell"] >= min_rate) & (cal["neg_empty"] >= min_rate)]
    n_min = int(ok["m"].min()) if len(ok) else None
    # use only depths at and above n_min, and require monotone ok beyond it
    return cal, n_min


def ambient_window(umi, called_any, lo=10):
    """The empty-droplet plateau: mode of the UMI-weighted log10 UMI histogram of barcodes no caller calls (>= lo UMIs), +- 2-fold.
    Weighting by UMIs puts the mode where the ambient molecules are (the plateau), not at the very-low-UMI barcodes, which in
    10x v3 data are dominated by barcode-error and swapped reads from single cells."""
    sel = (~called_any) & (umi >= lo)
    x = np.log10(umi[sel])
    h, e = np.histogram(x, bins=60, weights=umi[sel])
    k = int(np.argmax(h))
    mode = 10 ** (0.5 * (e[k] + e[k + 1]))
    return max(lo, mode / 2), mode * 2


def mixture_densities(llr_fn, bar_umis, pos_umis, neg_umis, n_sites, cap=400, n_ref=400, seed=11):
    """Per barcode: its LLR at depth d = min(depth, cap) (own UMIs thinned to cap) and the densities of that LLR under thinned
    positives (f1) and pooled negatives (f0) at exactly depth d (Gaussian KDE of n_ref reference draws each, floored).
    The ratio f1/f0 feeds a mixture estimate of the cell fraction of any group of barcodes (score.py), which works where single
    barcodes are too shallow to classify."""
    from scipy.stats import gaussian_kde
    rng = np.random.default_rng(seed)
    n = len(bar_umis)
    dep = np.array([min(len(s), cap) for s, _ in bar_umis])
    own = []
    for s, a in bar_umis:
        if len(s) > cap:
            k = rng.choice(len(s), cap, replace=False)
            s, a = s[k], a[k]
        own.append((s, a))
    llr = np.full(n, np.nan)
    f1 = np.full(n, np.nan)
    f0 = np.full(n, np.nan)
    ok = dep >= 3
    if ok.any():
        llr[ok] = llr_fn(*to_csr([own[i] for i in np.nonzero(ok)[0]], n_sites))
    psize = np.array([len(s) for s, _ in pos_umis])
    for m in np.unique(dep[ok]):
        elig = np.nonzero(psize >= m)[0]
        if len(elig) < 30:
            continue
        pick = rng.choice(elig, min(n_ref, len(elig)), replace=len(elig) < n_ref)
        P = thinned_positives([pos_umis[j] for j in pick], m, rng)
        N = pooled_negatives(neg_umis, m, n_ref, rng)
        lp = llr_fn(*to_csr(P, n_sites))
        ln = llr_fn(*to_csr(N, n_sites))
        idx = np.nonzero(ok & (dep == m))[0]
        x = llr[idx]
        for ref, out in ((lp, f1), (ln, f0)):
            sd = ref.std()
            if sd < 1e-9:
                dens = np.where(np.abs(x - ref.mean()) < 1e-6, 1.0, 0.0)
            else:
                kde = gaussian_kde(ref)
                dens = kde(x)
            floor = 0.5 / (len(ref) * max(np.ptp(np.r_[ref, x]), 1.0))  # less than one reference draw's worth of mass
            out[idx] = np.maximum(dens, floor)
    return llr, dep, f1, f0


def classify(llr, depth, cal, n_min):
    """cell / empty / unknown per barcode from its LLR and informative depth, with the calibration of the largest m <= depth."""
    out = np.full(len(llr), "unknown", dtype=object)
    if n_min is None:
        return out
    ms = cal["m"].to_numpy()
    for i, (l, d) in enumerate(zip(llr, depth)):
        if d < n_min:
            continue
        j = np.searchsorted(ms, d, side="right") - 1
        r = cal.iloc[j]
        if l > r["t_cell"]:
            out[i] = "cell"
        elif l < r["t_empty"]:
            out[i] = "empty"
    return out


def intact_state(n_umi, mito_frac, intr_frac, ref_mito, ref_intr, amb_intr, alpha=0.01):
    """Intact-cell state of each barcode from one-sided binomial tests (alpha) on its UMI counts:
    damaged       mito share significantly above the reference cells' 99th percentile, or intronic share significantly below
                  their 1st percentile;
    intact        not damaged, and intronic share significantly above the ambient pool's (nuclear RNA beyond ambient);
    undetermined  neither (too few UMIs to tell)."""
    from scipy.stats import binom
    n = np.maximum(np.nan_to_num(np.asarray(n_umi, float)).astype(int), 0)
    km = np.round(np.nan_to_num(np.asarray(mito_frac, float)) * n).astype(int)
    ki = np.round(np.nan_to_num(np.asarray(intr_frac, float)) * n).astype(int)
    q_m = float(np.quantile(ref_mito, 0.99))
    q_i = float(np.quantile(ref_intr, 0.01))
    hi_mito = binom.sf(km - 1, n, q_m) < alpha
    lo_intr = binom.cdf(ki, n, q_i) < alpha
    above_amb = binom.sf(ki - 1, n, amb_intr) < alpha
    dam = hi_mito | lo_intr
    state = np.where(dam, "damaged", np.where(above_amb, "intact", "undetermined"))
    return state, hi_mito, lo_intr, dict(mito_q99=q_m, intronic_q01=q_i, ambient_intronic=float(amb_intr),
                                         ref_intronic_median=float(np.median(ref_intr)), ref_mito_median=float(np.median(ref_mito)))


def strict_truth(truth, state):
    return np.where(truth == "cell", np.where(state == "intact", "cell", np.where(state == "damaged", "damaged", "unknown")), truth)


def dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
