"""Genotype ambient truth (Kang 2018 pooled PBMC lanes) and calibration metrics of harness estimates.

Truth as in bench/latent_qc/ambient_kang/kang_ambient.py (PR #43): in a donor's confident singlet, reads with an allele the donor
lacks (at sites where the donor's pooled singlets are homozygous) came from other donors through the ambient pool:
    rho_c = (X_c - e N_c) / Q_c,  Q_c = sum_s n_s q_s,  q_s = pool alt (or ref) frequency at site s
pooled over groups of cells: (sum X - e sum N) / sum Q. Per-cell truth is noisy, so calibration is read on groups.

usage:
  score_kang.py truth RUN_DIR DUMP_DIR TRUTH_TSV LANE     LANE = ctrl (batch 2, stim == ctrl) or A / B / C (batch 1)
      -> DUMP_DIR/truth_cells.csv.gz (per confident singlet: barcode, celltype, X, N, Q) and truth_meta.json (error rate).
         Derived counts only; stays on Clipper (never committed: it is computed from private genotype data).
  score_kang.py score DUMP_DIR OUT_JSONL NAME=PREFIX [NAME=PREFIX ...]
      PREFIX.rho.f64 in the order of DUMP_DIR/cells.txt (or PREFIX.cells.txt if present) -> one JSON line per estimate.
"""
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io

RARE = ("Dendritic cells", "NK cells", "FCGR3A+ Monocytes")


def mtx(path):
    with gzip.open(path, "rb") as f:
        return scipy.io.mmread(f).tocsc()


def lines(path):
    with gzip.open(path, "rt") as f:
        return [l.rstrip("\n").split("\t")[0] for l in f]


def truth(run, dump, truth_tsv, lane):
    AD, DP = mtx(f"{run}/private/snp_ad.mtx.gz"), mtx(f"{run}/private/snp_dp.mtx.gz")
    bcs = np.array(lines(f"{run}/private/snp_ad_barcodes.tsv.gz"))
    A = pd.read_csv(f"{run}/donor/donor_assignments.tsv", sep="\t").set_index("cell").reindex(bcs)
    t = pd.read_csv(truth_tsv, sep="\t", index_col=0)
    if lane == "ctrl":
        t = t[t["stim"] == "ctrl"]
        ct_col = "cell"
    else:
        t = t[t["batch"] == lane]
        ct_col = "cell.type"
    t.index = [i.split("-")[0] for i in t.index]
    t = t[~t.index.duplicated(keep=False)]
    tm = t.reindex([b.split("-")[0] for b in bcs])
    conf = (A.donor_id.astype(str).str.startswith("donor") & (A.prob_max >= 0.99) & (A.cell_status == "cell")).values
    singlet = conf & (tm.multiplets.values == "singlet")
    amb = (A.cell_status == "ambient").values
    donor = A.donor_id.astype(str).values
    ADr, DPr = AD.tocsr(), DP.tocsr()
    a_amb = np.asarray(ADr[:, amb].sum(1)).ravel()
    d_amb = np.asarray(DPr[:, amb].sum(1)).ravel()
    f = np.where(d_amb >= 10, a_amb / np.maximum(d_amb, 1), np.nan)
    donors = sorted(set(donor[singlet]))
    G = np.full((AD.shape[0], len(donors)), np.nan)
    for j, dn in enumerate(donors):
        m = singlet & (donor == dn)
        a = np.asarray(ADr[:, m].sum(1)).ravel()
        dd = np.asarray(DPr[:, m].sum(1)).ravel()
        af = np.where(dd >= 8, a / np.maximum(dd, 1), np.nan)
        G[:, j] = np.where(af <= 0.1, 0.0, np.where(af >= 0.9, 1.0, np.nan))
    cells = np.flatnonzero(singlet)
    dj = {dn: j for j, dn in enumerate(donors)}
    X = np.zeros(len(cells)); N = np.zeros(len(cells)); Q = np.zeros(len(cells))
    Xe = Ne = 0.0
    for k, c in enumerate(cells):
        s0, s1 = DP.indptr[c], DP.indptr[c + 1]
        sites = DP.indices[s0:s1]
        n = DP.data[s0:s1].astype(float)
        a = np.zeros(len(sites))
        a0, a1 = AD.indptr[c], AD.indptr[c + 1]
        a[np.searchsorted(sites, AD.indices[a0:a1])] = AD.data[a0:a1]
        g = G[sites, dj[donor[c]]]
        fs = f[sites]
        ok = np.isfinite(g) & np.isfinite(fs)
        x = np.where(g == 0, a, n - a)
        q = np.where(g == 0, fs, 1 - fs)
        inf = ok & (q >= 0.05)
        err = ok & (q < 0.01)
        Xe += x[err].sum(); Ne += n[err].sum()
        X[k], N[k], Q[k] = x[inf].sum(), n[inf].sum(), (n * q)[inf].sum()
    e = Xe / max(Ne, 1)
    D = pd.DataFrame(dict(barcode=bcs[cells], celltype=tm[ct_col].values[cells], X=X, N=N, Q=Q))
    D.to_csv(f"{dump}/truth_cells.csv.gz", index=False)
    meta = dict(lane=lane, error_rate=e, singlets=int(len(D)), donors=len(donors), ambient_barcodes=int(amb.sum()),
                pooled_truth=float((X.sum() - e * N.sum()) / Q.sum()))
    json.dump(meta, open(f"{dump}/truth_meta.json", "w"), indent=1)
    print(json.dumps(meta))


def metrics(D, e, est):
    """D: truth cells; est: estimate aligned to D (NaN = cell not in the estimate's cell set)."""
    m0 = np.isfinite(est)
    X, N, Q = D.X.values, D.N.values, D.Q.values

    def pooled(m):
        return float((X[m].sum() - e * N[m].sum()) / Q[m].sum())

    r = dict(cells=int(m0.sum()))
    t_all = pooled(m0)
    r["truth"] = t_all
    r["est_mean"] = float(est[m0].mean())
    r["ratio"] = r["est_mean"] / t_all
    ct = D.celltype.values
    per = {}
    for c in sorted(set(ct[m0].astype(str))):
        m = m0 & (ct == c)
        if m.sum() >= 50:
            per[c] = dict(cells=int(m.sum()), truth=pooled(m), est=float(est[m].mean()), ratio=float(est[m].mean() / pooled(m)))
    r["celltype"] = per
    mr = m0 & np.isin(ct, RARE)
    r["rare_ratio"] = float(est[mr].mean() / pooled(mr)) if mr.sum() else float("nan")
    rows = [k for k in per if k not in RARE and k != "Megakaryocytes"]
    mc = m0 & np.isin(ct, rows)
    r["common_ratio"] = float(est[mc].mean() / pooled(mc)) if mc.sum() else float("nan")
    # worst cell-type ratio (log scale) over groups of >= 100 cells
    big = [v["ratio"] for v in per.values() if v["cells"] >= 100]
    r["max_abs_log2_celltype_ratio"] = float(np.max(np.abs(np.log2(big)))) if big else float("nan")
    idx = np.flatnonzero(m0)
    q = pd.qcut(pd.Series(est[idx]).rank(method="first"), 5, labels=False).values
    xs, ys = [], []
    for b in range(5):
        mm = np.zeros(len(D), bool)
        mm[idx[q == b]] = True
        xs.append(float(est[mm].mean())); ys.append(pooled(mm))
    r["quintile_est"] = xs
    r["quintile_truth"] = ys
    r["quintile_slope"] = float(np.polyfit(xs, ys, 1)[0])
    r["est_spread_q5_q1"] = xs[4] - xs[0]
    r["truth_spread_q5_q1"] = ys[4] - ys[0]
    return r


def score(dump, out_jsonl, specs):
    D = pd.read_csv(f"{dump}/truth_cells.csv.gz")
    e = json.load(open(f"{dump}/truth_meta.json"))["error_rate"]
    default_cells = pd.read_csv(f"{dump}/cells.txt", header=None)[0].astype(str).values
    with open(out_jsonl, "a") as fo:
        for s in specs:
            name, prefix = s.split("=", 1)
            if not os.path.exists(prefix + ".rho.f64"):
                print("missing", prefix, flush=True)
                continue
            rho = np.fromfile(prefix + ".rho.f64", np.float64)
            cf = prefix + ".cells.txt"
            cb = pd.read_csv(cf, header=None)[0].astype(str).values if os.path.exists(cf) else default_cells
            est = pd.Series(rho, index=cb).reindex(D.barcode.astype(str)).values
            r = metrics(D, e, est)
            r["name"] = name
            r["dump"] = dump
            js = prefix + ".json"
            if os.path.exists(js):
                h = json.load(open(js))
                r["seconds"] = h.get("seconds")
                r["mean_rho_all_cells"] = h.get("mean_rho")
            fo.write(json.dumps(r) + "\n")
            print(f"{name}: ratio {r['ratio']:.3f} slope {r['quintile_slope']:.3f} rare {r['rare_ratio']:.3f} "
                  f"common {r['common_ratio']:.3f} spread est {r['est_spread_q5_q1']:.4f} truth {r['truth_spread_q5_q1']:.4f}",
                  flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "truth":
        truth(*sys.argv[2:6])
    else:
        score(sys.argv[2], sys.argv[3], sys.argv[4:])
