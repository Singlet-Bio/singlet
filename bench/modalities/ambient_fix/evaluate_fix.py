"""Score the new gene-level ambient estimator/correction (harness.cpp output) on hgmm1k_v3 with the same truth and metrics as
bench/modalities/ambient/evaluate.py (estimate vs cross-species truth, correction table, marker retention).
Usage: python -I evaluate_fix.py TAG [TAG ...]   (TAG = directory under $A/fix_out). Run in a Slurm job."""
import json
import os
import sys

import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, "/mnt/projects/debruinz_project/debruinz/singlet/jobs/modalities/ambient")
from common import A, WORK, read_mtx_dir  # noqa: E402

R = f"{A}/results_fix"
os.makedirs(R, exist_ok=True)
meta = pd.read_csv(f"{WORK}/cells.csv")
genes = pd.read_csv(f"{WORK}/genes.csv")
alpha_m = json.load(open(f"{WORK}/prep.json"))["alpha_m"]
gsp = (genes.species == "mouse").values
raw_cells = scipy.io.mmread(f"{WORK}/cells.mtx").tocsc()
hum = (meta.species == "human").values
mou = (meta.species == "mouse").values
pure = hum | mou
alpha_other = np.where(hum, alpha_m, np.where(mou, 1 - alpha_m, np.nan))
own_share = 1 - alpha_other
tr = meta.rho_truth.values
depth_bins = pd.qcut(meta.N, 3, labels=["low", "mid", "high"]).astype(str).values
nc = len(meta)

rawfull, gid, _, _ = read_mtx_dir(f"{WORK}/raw")
tot = np.asarray(rawfull.sum(0)).ravel()
pool = np.where((tot >= 1) & (tot <= 100))[0]
pool = pool[np.argsort(-tot[pool])[:50000]]
amb_counts = np.asarray(rawfull[:, pool].sum(1)).ravel().astype(float)
amb_frac = (amb_counts + 0.5) / (amb_counts + 0.5).sum()
del rawfull


def species_umis(X, mouse_rows):
    Xr = X.tocsr()
    return np.asarray(Xr[mouse_rows].sum(0)).ravel(), np.asarray(Xr[~mouse_rows].sum(0)).ravel()


def score(before, after, label):
    Mb, Hb = species_umis(before, gsp)
    Ma, Ha = species_umis(after, gsp)
    oth_b = np.where(hum, Mb, np.where(mou, Hb, np.nan)); oth_a = np.where(hum, Ma, np.where(mou, Ha, np.nan))
    own_b = np.where(hum, Hb, np.where(mou, Mb, np.nan)); own_a = np.where(hum, Ha, np.where(mou, Ma, np.nan))
    m = pure
    o = dict(tool=label)
    o["cross_species_umis_before"] = float(oth_b[m].sum())
    o["cross_species_umis_after"] = float(oth_a[m].sum())
    o["cross_species_residual_pct"] = 100 * o["cross_species_umis_after"] / max(o["cross_species_umis_before"], 1)
    removed_own = float((own_b - own_a)[m].sum())
    o["own_species_removed_pct"] = 100 * removed_own / float(own_b[m].sum())
    exp_own = float((oth_b / alpha_other * own_share)[m].sum())
    o["own_removed_over_expected_own_ambient"] = removed_own / exp_own
    o["total_removed_pct_of_umis"] = 100 * float((before.sum() - after.sum()) / before.sum())
    implied = 1 - np.asarray(after.sum(0)).ravel() / np.maximum(np.asarray(before.sum(0)).ravel(), 1)
    o["implied_rho_mean"] = float(implied[m].mean())
    return o, implied


def est_row(name, v):
    m = pure & np.isfinite(v)
    d = dict(estimator=name, mean_est=float(np.mean(v[m])), mean_truth=float(np.mean(tr[m])))
    d["bias_ratio"] = d["mean_est"] / d["mean_truth"]
    d["spearman"] = float(spearmanr(v[m], tr[m])[0]) if np.std(v[m]) > 0 else np.nan
    d["pearson"] = float(pearsonr(v[m], tr[m])[0]) if np.std(v[m]) > 0 else np.nan
    for b in ("low", "mid", "high"):
        mb = m & (depth_bins == b)
        d[f"est_{b}"] = float(np.mean(v[mb])); d[f"truth_{b}"] = float(np.mean(tr[mb]))
    return d


cm = np.asarray(raw_cells.mean(1)).ravel()
tops = {}
for sname, mask_cells, mask_genes in (("human", hum, ~gsp), ("mouse", mou, gsp)):
    cand = np.where(mask_genes & (cm >= 0.3))[0]
    cell_frac = np.asarray(raw_cells[:, mask_cells].sum(1)).ravel() / float(raw_cells[:, mask_cells].sum())
    tops[sname] = cand[np.argsort(-(cell_frac[cand] / amb_frac[cand]))[:100]]

est_rows, corr_rows, mk_rows, extra = [], [], [], {}
for tag in sys.argv[1:]:
    d = f"{A}/fix_out/{tag}"
    rho = np.fromfile(f"{d}/rho.f64", dtype=np.float64)
    ip = np.fromfile(f"{d}/corr.indptr", dtype=np.int64)
    ix = np.fromfile(f"{d}/corr.indices", dtype=np.int32)
    dv = np.fromfile(f"{d}/corr.data", dtype=np.uint32)
    C = sp.csc_matrix((dv.astype(np.float64), ix, ip), shape=(raw_cells.shape[0], nc))
    o, imp = score(raw_cells, C, f"Singlet gene-level EM [{tag}]")
    corr_rows.append(o)
    est_rows.append(est_row(f"Singlet gene-level EM rho [{tag}]", rho))
    est_rows.append(est_row(f"Singlet gene-level EM implied [{tag}]", imp))
    r = {"tool": f"Singlet gene-level EM [{tag}]"}
    for sname, mask in (("human", hum), ("mouse", mou)):
        top = tops[sname]
        r[f"{sname}_marker_retention_pct"] = 100 * float(C[top][:, mask].sum() / raw_cells[top][:, mask].sum())
    mk_rows.append(r)
    # integer / bound sanity on the real data
    Cc = C.tocsr(); Rr = raw_cells.tocsr()
    diff = (Rr - Cc)
    extra[tag] = dict(max_excess_over_observed=float((Cc - Rr).max()), min_removed=float(diff.min()),
                      integer=bool(np.all(dv == np.round(dv))), run=json.load(open(f"{d}/run.json")),
                      clusters=int(len(np.unique(np.fromfile(f"{d}/cluster.u32", dtype=np.uint32)))))

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
est = pd.DataFrame(est_rows); corr = pd.DataFrame(corr_rows); mk = pd.DataFrame(mk_rows)
base_e = f"{A}/results/estimates.csv"; base_c = f"{A}/results/correction.csv"; base_m = f"{A}/results/markers.csv"
tabs = []
for new, base, name in ((est, base_e, "estimates"), (corr, base_c, "correction"), (mk, base_m, "markers")):
    old = pd.read_csv(base) if os.path.exists(base) else pd.DataFrame()
    both = pd.concat([old, new], ignore_index=True)
    both.to_csv(f"{R}/{name}_vs_baselines.csv", index=False)
    new.to_csv(f"{R}/{name}_fix.csv", index=False)
    print(f"\n## {name} (baselines + new)")
    print(both.round(4).to_string(index=False))
json.dump(extra, open(f"{R}/summary_fix.json", "w"), indent=1, default=float)
print(json.dumps(extra, indent=1, default=float))
