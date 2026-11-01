# SPDX-License-Identifier: MIT
"""Leave-one-dataset-out choice of the hashtag caller in ``include/singlet/pileup/hto_demux.h``.

    python -P bench/adt/hto_lodo.py --zenodo DIR_WITH_BAL_data_AND_cell_line_data --out selection.json

Data: Howitt et al. 2023 (Zenodo 8304003), the public hashtag datasets with an independent per-cell truth (vireo genotype
donors per cell: donor / Doublet / Negative) that ship their hashtag counts: BAL batches 1-3 (8 TotalSeq-A hashtags, 2 captures
each) and the cell-line mix (3 MULTI-seq lipid tags, 3 captures). Metrics are the paper's: mean over hashtags of
F = tp / (tp + (fp + fn) / 2), with fp including genetic doublets and negatives called as that hashtag, and the share of genetic
doublets called singlets.

Protocol. The caller family and its grid (:data:`GRID`) are fixed here, before any held-out dataset is scored. The family is
the one in hto_demux.h: hashedDrops' fold-change rule (``confident_min``, ``doublet_nmads``, ``doublet_min``) and a per-tag
Gaussian-mixture doublet veto (``veto``: the posterior for a tag to be positive; None = no veto). For each dataset D, the
setting with the highest mean of (F minus doublets called singlet) over the *other* datasets is chosen, and that setting's score
on D is the held-out result. The setting chosen on all datasets is the C++ default (``HtoDemuxParams``). The numpy functions
below are the reference that the C++ header reproduces call for call (checked by ``bench/adt/hto_public_run.py``).

Other real hashtag sets have no independent truth for HTO calls (pbmc-ch, cline-ch and mkidney-ch of Xi & Li label doublets
from the hashtags themselves; Stoeckius 2018 awaits genotype truth), so they are not used for selection.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

NEG, DBL = -1, -2
PSEUDO = 5.0
CONFIDENT_NMADS = 3.0
GRID = {
    "confident_min": (0.0, 0.25, 0.5, 1.0, 2.0),
    "doublet_nmads": (2.0, 3.0, 4.0),
    "doublet_min": (1.0, 2.0),
    "veto": (None, 0.5, 0.8, 0.95),
}

# ------------------------------------------------------------------------------------------------------------ data


def _read_counts(path):
    d = pd.read_csv(path, index_col=0)
    d.index = [str(i) for i in d.index]
    return d


def load_howitt(z):
    """-> {dataset: ([(capture, counts DataFrame tags x cells, truth Series)], hashtag -> donor)}"""
    out = {}
    for b in (1, 2, 3):
        caps = []
        for c in (1, 2):
            h = _read_counts(os.path.join(z, "BAL_data", f"batch{b}_c{c}_hto_counts.csv"))
            dn = pd.read_csv(os.path.join(z, "BAL_data", f"batch{b}_c{c}_donors.csv"))
            caps.append((f"c{c}", h, dn.set_index("Barcode")["genetic_donor"].reindex(h.columns)))
        tags = list(caps[0][1].index)
        first = ord("A") + 8 * (b - 1)
        out[f"BAL{b}"] = (caps, {t: f"BAL {chr(first + i)}" for i, t in enumerate(tags)})
    cl = []
    for c in (1, 2, 3):
        h = _read_counts(os.path.join(z, "cell_line_data", f"lmo_counts_capture{c}.csv"))
        dn = pd.read_csv(os.path.join(z, "cell_line_data", f"lmo_donors_capture{c}.csv"))
        bc = "Barcode" if "Barcode" in dn else dn.columns[0]
        dcol = "genetic_donor" if "genetic_donor" in dn else dn.columns[-1]
        cl.append((f"c{c}", h, dn.set_index(bc)[dcol].reindex(h.columns)))
    donors = sorted({d for _, _, t in cl for d in t.dropna().unique() if d not in ("Doublet", "Negative")})
    out["CL"] = (cl, {t: d for t, d in zip(list(cl[0][1].index), donors)})
    return out


def score(calls, truth, hmap):
    """calls: tag names / 'Doublet' / 'Negative'; truth: genetic donors (Howitt et al. ``calculate_HTO_fscore``)."""
    calls = np.asarray(calls, dtype=object)
    truth = pd.Series(truth).astype(object)
    truth_tag = truth.map({v: k for k, v in hmap.items()}).to_numpy()
    per = []
    for tag in hmap:
        tp = np.sum((calls == tag) & (truth_tag == tag))
        fp = np.sum((calls == tag) & (truth_tag != tag))
        fn = np.sum((calls != tag) & (truth_tag == tag))
        per.append(tp / (tp + 0.5 * (fp + fn)) if tp + fp + fn else np.nan)
    dbl = (truth == "Doublet").to_numpy()
    return {"f": float(np.nanmean(per)), "dbl_singlet": float((dbl & np.isin(calls, list(hmap))).sum() / max(dbl.sum(), 1))}


def to_names(call, tags):
    return np.array([tags[c] if c >= 0 else ("Negative" if c == NEG else "Doublet") for c in call], dtype=object)


# --------------------------------------------------------------------------------------------------- reference caller


def _mad(x):
    m = np.median(x)
    return m, 1.4826 * np.median(np.abs(x - m))


def ambient_bimodal(X):
    """Per tag, the mean count of the lower group of a 1-D 2-means split of log1p counts (ambientProfileBimodal)."""
    out = np.zeros(X.shape[0])
    for i, row in enumerate(X):
        v = np.log1p(row)
        lo, hi = v.min(), v.max()
        if hi - lo < 1e-9:
            out[i] = row.mean()
            continue
        c = np.array([lo, hi])
        for _ in range(100):
            lab = np.abs(v[:, None] - c[None, :]).argmin(axis=1)
            new = np.array([v[lab == j].mean() if (lab == j).any() else c[j] for j in (0, 1)])
            if np.allclose(new, c):
                break
            c = new
        low = lab == int(np.argmin(c))
        out[i] = row[low].mean() if low.any() else row.mean()
    return out


def hashed_drops(X, confident_min, doublet_nmads, doublet_min):
    k, n = X.shape
    amb = ambient_bimodal(X) + 1e-9
    amb = amb / amb.sum()
    adj = X / (amb[:, None] * k)
    o = np.argsort(-adj, axis=0, kind="stable")
    srt = np.take_along_axis(adj, o, axis=0)
    amb_level = np.median(srt[2:], axis=0) if k > 2 else np.zeros(n)
    lfc_top = np.log2((srt[0] + PSEUDO) / (srt[1] + PSEUDO))
    lfc_second = np.log2((srt[1] + PSEUDO) / (amb_level + PSEUDO))
    md, mad = _mad(lfc_second)
    doublet = (lfc_second > md + doublet_nmads * mad) & (lfc_second >= doublet_min)
    mt, madt = _mad(lfc_top[~doublet]) if (~doublet).any() else (0.0, 0.0)
    confident = (~doublet) & (lfc_top >= confident_min) & (lfc_top >= mt - CONFIDENT_NMADS * madt)
    call = np.full(n, NEG, dtype=np.int64)
    call[confident] = o[0][confident]
    call[doublet] = DBL
    return call


def _mix_one(y, iters=200, tol=1e-9):
    """Posterior of the high component of a two-component Gaussian mixture of y (EM from a deterministic 2-means start)."""
    n = len(y)
    c = np.quantile(y, [0.25, 0.95])
    for _ in range(50):
        lab = np.abs(y[:, None] - c[None, :]).argmin(axis=1)
        new = np.array([y[lab == m].mean() if (lab == m).any() else c[m] for m in (0, 1)])
        if np.allclose(new, c):
            break
        c = new
    r = np.clip((lab == int(np.argmax(c))).astype(float), 0.02, 0.98)
    ll_old = -np.inf
    for _ in range(iters):
        P = []
        for w in (1 - r, r):
            w = w + 1e-9
            sw = w.sum()
            b0 = (w * y).sum() / sw
            P.append((b0, max(float((w * (y - b0) ** 2).sum() / sw), 1e-4), sw))
        dens = np.vstack([np.log(max(sw / n, 1e-9)) - 0.5 * np.log(2 * np.pi * var) - 0.5 * (y - b0) ** 2 / var
                          for b0, var, sw in P])
        m = dens.max(axis=0)
        e = np.exp(dens - m)
        ll = float((m + np.log(e.sum(axis=0))).sum())
        r = e[1] / e.sum(axis=0)
        if abs(ll - ll_old) < tol * max(1.0, abs(ll)):
            break
        ll_old = ll
    return r if P[1][0] >= P[0][0] else 1 - r


def mixture_posteriors(X, min_cells=20):
    k, n = X.shape
    q = np.zeros((k, n))
    for j in range(k):
        y = np.log1p(X[j])
        if n >= min_cells and y.std() > 0:
            q[j] = _mix_one(y)
    return q


def call_hashtags(X, params, q=None):
    """Reference of ``HtoDemux::demux`` for >= 20 cells and >= 2 tags. params: confident_min, doublet_nmads, doublet_min, veto."""
    call = hashed_drops(X, params["confident_min"], params["doublet_nmads"], params["doublet_min"])
    if params["veto"] is not None:
        q = mixture_posteriors(X) if q is None else q
        call = np.where((q >= params["veto"]).sum(axis=0) >= 2, DBL, call)
    return call


# ----------------------------------------------------------------------------------------------------------- protocol


def settings():
    keys = list(GRID)
    return [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]


def name_of(p):
    return "cm{confident_min}_nm{doublet_nmads}_dm{doublet_min}_veto{veto}".format(**p)


def run_grid(data):
    """-> DataFrame dataset x setting: f, dbl_singlet (captures called separately, scored over the whole dataset)."""
    rows = []
    for ds, (caps, hmap) in data.items():
        Xs = [(h.to_numpy(dtype=float), list(h.index), t.to_numpy(dtype=object)) for _, h, t in caps]
        qs = [mixture_posteriors(X) for X, _, _ in Xs]
        hd = {}
        for p in settings():
            key = (p["confident_min"], p["doublet_nmads"], p["doublet_min"])
            if key not in hd:
                hd[key] = [hashed_drops(X, *key) for X, _, _ in Xs]
            names, truth = [], []
            for (X, tags, tr), q, call in zip(Xs, qs, hd[key]):
                if p["veto"] is not None:
                    call = np.where((q >= p["veto"]).sum(axis=0) >= 2, DBL, call)
                names.append(to_names(call, tags))
                truth.append(tr)
            s = score(np.concatenate(names), np.concatenate(truth), hmap)
            rows.append({"dataset": ds, "setting": name_of(p), **p, **s})
    return pd.DataFrame(rows)


def choose(df, datasets):
    """Highest mean of F - (doublets called singlet) over the datasets: the two reported metrics, equally weighted."""
    d = df[df.dataset.isin(datasets)].assign(j=lambda x: x.f - x.dbl_singlet)
    return d.groupby("setting").j.mean().idxmax()


def _params(r):
    return {"confident_min": float(r.confident_min), "doublet_nmads": float(r.doublet_nmads),
            "doublet_min": float(r.doublet_min), "veto": None if pd.isna(r.veto) else float(r.veto)}


def lodo(df):
    ds = sorted(df.dataset.unique())
    held = []
    for test in ds:
        m = choose(df, [x for x in ds if x != test])
        r = df[(df.dataset == test) & (df.setting == m)].iloc[0]
        held.append({"held_out": test, "chosen_on_others": m, "params": _params(r), "f": float(r.f),
                     "dbl_singlet": float(r.dbl_singlet)})
    final = choose(df, ds)
    r = df[df.setting == final].iloc[0]
    return {"criterion": "max mean of (F - doublets called singlet) over the training datasets",
            "grid": {k: list(v) for k, v in GRID.items()}, "held_out": held, "shipped": final,
            "shipped_params": _params(r),
            "shipped_in_sample": df[df.setting == final][["dataset", "f", "dbl_singlet"]].to_dict("records")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zenodo", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = run_grid(load_howitt(a.zenodo))
    res = lodo(df)
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump({**res, "all_settings": df.drop(columns=["setting"]).to_dict("records")}, fh, indent=1,
                      default=lambda o: None)


if __name__ == "__main__":
    sys.exit(main())
