"""Score ambient estimates / corrections and doublet calls against the barnyard truth.
Writes results/{summary.json,correction.csv,estimates.csv,markers.csv,doublets.csv}. Run in a Slurm job (python -I)."""
import gzip
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import *  # noqa: E402,F403

R = f"{A}/results"
os.makedirs(R, exist_ok=True)
meta = pd.read_csv(f"{WORK}/cells.csv")
genes = pd.read_csv(f"{WORK}/genes.csv")
prep = json.load(open(f"{WORK}/prep.json"))
alpha_m = prep["alpha_m"]
gsp = (genes.species == "mouse").values
raw_cells = scipy.io.mmread(f"{WORK}/cells.mtx").tocsc()  # genes x cells
bc = meta.barcode.values
hum = (meta.species == "human").values
mou = (meta.species == "mouse").values
pure = hum | mou
alpha_other = np.where(hum, alpha_m, np.where(mou, 1 - alpha_m, np.nan))  # other species' share of the ambient pool
own_share = 1 - alpha_other

# ambient profile (Singlet's rule: 1..100 UMI barcodes, top 50k, pseudocount 0.5) for marker-gene selection
rawfull, gid, _, rbc = read_mtx_dir(f"{WORK}/raw")
tot = np.asarray(rawfull.sum(0)).ravel()
pool = np.where((tot >= 1) & (tot <= 100))[0]
pool = pool[np.argsort(-tot[pool])[:50000]]
amb_counts = np.asarray(rawfull[:, pool].sum(1)).ravel().astype(float)
amb_frac = (amb_counts + 0.5) / (amb_counts + 0.5).sum()
del rawfull

# ---------------- corrected matrices (genes x cells, cells.csv order) ----------------
corr = {}
for name in ["soupx_leiden", "soupx_species", "decontx_default", "decontx_background"]:
    p = f"{R}/{name}_corrected.mtx"
    if os.path.exists(p):
        corr[name] = scipy.io.mmread(p).tocsc()
cbp = f"{A}/cellbender/cb.h5"
if os.path.exists(cbp):
    with h5py.File(cbp, "r") as f:
        g = f["matrix"] if "matrix" in f else f
        cbbc = np.array([b.decode().replace("-1", "") for b in g["barcodes"][:]])
        X = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
    ix = pd.Series(np.arange(len(cbbc)), index=cbbc).reindex(bc)
    if ix.notna().all():
        corr["cellbender"] = X[:, ix.astype(int).values].tocsc()
    else:
        print("CellBender: %d of %d cells missing from its output" % (ix.isna().sum(), len(bc)))


def singlet_exon_eval():
    """Singlet's correct_counts() (ambient_correction.h) is never invoked by the CLI; apply its formula to the exon matrix
    with the profile and per-cell rho the binary wrote:  corrected = max(0, obs - round(rho * N * fraction_g))."""
    d = SINGLET
    with gzip.open(f"{d}/exon_counts.mtx.gz", "rt") as f:
        E = scipy.io.mmread(f).tocsc()
    feats = pd.read_csv(f"{d}/exon_counts_features.tsv.gz", sep="\t", header=None)[0].astype(str).values
    ebc = pd.read_csv(f"{d}/exon_counts_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).str.replace("-1", "", regex=False).values
    prof = pd.read_csv(f"{d}/ambient_profile.tsv", sep="\t")
    assert (prof.feature.values == feats).all()
    frac = prof.ambient_fraction.values
    rho = pd.read_csv(f"{d}/ambient_contamination.tsv", sep="\t").set_index("barcode").rho
    ci = pd.Series(np.arange(len(ebc)), index=ebc).reindex(bc).astype(int).values
    E = E[:, ci]
    rh = rho.reindex(bc).values
    Nc = np.asarray(E.sum(0)).ravel()
    Ec = E.tocoo()
    s = np.round(rh[Ec.col] * Nc[Ec.col] * frac[Ec.row])
    newd = np.maximum(Ec.data - s, 0)
    C = sp.csc_matrix((newd, (Ec.row, Ec.col)), shape=E.shape)
    C.eliminate_zeros()
    esp = pd.Series(feats).str.startswith("ENSMUSG").values
    return dict(E=E, C=C, esp=esp)


sg = singlet_exon_eval()


def species_umis(X, mouse_rows):
    Xr = X.tocsr()
    return np.asarray(Xr[mouse_rows].sum(0)).ravel(), np.asarray(Xr[~mouse_rows].sum(0)).ravel()  # mouse, human


def score(before, after, mouse_rows, label):
    Mb, Hb = species_umis(before, mouse_rows)
    Ma, Ha = species_umis(after, mouse_rows)
    oth_b = np.where(hum, Mb, np.where(mou, Hb, np.nan)); oth_a = np.where(hum, Ma, np.where(mou, Ha, np.nan))
    own_b = np.where(hum, Hb, np.where(mou, Mb, np.nan)); own_a = np.where(hum, Ha, np.where(mou, Ma, np.nan))
    m = pure
    o = dict(tool=label)
    o["cross_species_umis_before"] = float(oth_b[m].sum())
    o["cross_species_umis_after"] = float(oth_a[m].sum())
    o["cross_species_residual_pct"] = 100 * o["cross_species_umis_after"] / max(o["cross_species_umis_before"], 1)
    removed_own = float((own_b - own_a)[m].sum())
    o["own_species_removed_pct"] = 100 * removed_own / float(own_b[m].sum())
    # ambient molecules in each pure cell, inferred from its other-species UMIs: other / other_share; own part = * own_share
    exp_own = float((oth_b / alpha_other * own_share)[m].sum())
    o["own_removed_over_expected_own_ambient"] = removed_own / exp_own
    o["total_removed_pct_of_umis"] = 100 * float((before[:, :].sum() - after.sum()) / before.sum())
    implied = 1 - np.asarray(after.sum(0)).ravel() / np.maximum(np.asarray(before.sum(0)).ravel(), 1)
    o["implied_rho_mean"] = float(implied[m].mean())
    return o, implied


rows, implied_all = [], {}
o, imp = score(sg["E"], sg["C"], sg["esp"], "Singlet correct_counts formula (exon matrix)")
rows.append(o)
implied_all["Singlet"] = imp
labels = {"soupx_leiden": "SoupX (leiden clusters)", "soupx_species": "SoupX (species clusters)",
          "decontx_default": "DecontX (default)", "decontx_background": "DecontX (empty-droplet background)",
          "cellbender": "CellBender 0.3.2"}
for name, label in labels.items():
    if name in corr:
        o, imp = score(raw_cells, corr[name], gsp, label)
        rows.append(o)
        implied_all[label] = imp
corr_tab = pd.DataFrame(rows)

# marker retention: per species, the 100 genes with the highest (cell mean fraction / ambient fraction) among genes with mean >= 0.3 UMI
cm = np.asarray(raw_cells.mean(1)).ravel()
mk_rows = []
tops = {}
for sname, mask_cells, mask_genes in (("human", hum, ~gsp), ("mouse", mou, gsp)):
    cand = np.where(mask_genes & (cm >= 0.3))[0]
    cell_frac = np.asarray(raw_cells[:, mask_cells].sum(1)).ravel() / float(raw_cells[:, mask_cells].sum())
    ratio = cell_frac[cand] / amb_frac[cand]
    tops[sname] = cand[np.argsort(-ratio)[:100]]
for name, X in corr.items():
    r = {"tool": labels[name]}
    for sname, mask in (("human", hum), ("mouse", mou)):
        top = tops[sname]
        r[f"{sname}_marker_retention_pct"] = 100 * float(X[top][:, mask].sum() / raw_cells[top][:, mask].sum())
    mk_rows.append(r)
r = {"tool": "Singlet correct_counts formula (exon matrix)"}
mk_rows.append(r)  # exon-level features differ; retention reported from the totals above only

# ---------------- per-cell contamination estimates vs truth ----------------
est = {"Singlet rho (ambient_contamination.tsv)": meta.rho_singlet.values}
for v in ("default", "background"):
    p = f"{R}/decontx_{v}_rho.csv"
    if os.path.exists(p):
        est[f"DecontX {v} (reported)"] = pd.read_csv(p).set_index("barcode").rho.reindex(bc).values
for k, v in implied_all.items():
    est[f"{k} (implied by removed UMIs)"] = v
tr = meta.rho_truth.values
depth_bins = pd.qcut(meta.N, 3, labels=["low", "mid", "high"]).astype(str).values
est_rows = []
for k, v in est.items():
    m = pure & np.isfinite(v)
    if m.sum() < 10:
        continue
    d = dict(estimator=k, mean_est=float(np.mean(v[m])), mean_truth=float(np.mean(tr[m])))
    d["bias_ratio"] = d["mean_est"] / d["mean_truth"]
    d["spearman"] = float(spearmanr(v[m], tr[m])[0]) if np.std(v[m]) > 0 else np.nan
    d["pearson"] = float(pearsonr(v[m], tr[m])[0]) if np.std(v[m]) > 0 else np.nan
    for b in ("low", "mid", "high"):
        mb = m & (depth_bins == b)
        d[f"est_{b}"] = float(np.mean(v[mb])); d[f"truth_{b}"] = float(np.mean(tr[mb]))
    est_rows.append(d)
est_tab = pd.DataFrame(est_rows)
rj = f"{R}/r_tools.json"
rtools = json.load(open(rj)) if os.path.exists(rj) else {}
glob_truth = float((meta.rho_truth * meta.N)[pure].sum() / meta.N[pure].sum())

# ---------------- doublets ----------------
truth_def = {}
for cut in (0.1, 0.2, 0.3):
    truth_def[f"own_minor>={cut}"] = (meta.minor_frac >= cut).values
    truth_def[f"CR_minor>={cut}"] = (meta.cr_minor_frac.fillna(meta.minor_frac) >= cut).values
dbl_calls = {"Singlet barnyard (classify_barnyard)": (meta.singlet_barnyard == "doublet").values,
             "Singlet detect_doublets (is_doublet)": meta.singlet_dbl.astype(str).str.lower().eq("true").values}
dbl_scores = {"Singlet detect_doublets (is_doublet)": meta.singlet_dbl_score.values,
              "Singlet barnyard (classify_barnyard)": meta.minor_frac.values}
sd = f"{R}/scdblfinder.csv"
if os.path.exists(sd):
    t = pd.read_csv(sd).set_index("barcode").reindex(bc)
    dbl_calls["scDblFinder"] = (t.call == "doublet").values; dbl_scores["scDblFinder"] = t.score.values
for er in ("0.03", "0.08"):
    p = f"{R}/scrublet_er{er}.csv"
    if os.path.exists(p):
        t = pd.read_csv(p).set_index("barcode").reindex(bc)
        dbl_calls[f"Scrublet (rate {er})"] = t.call.astype(bool).values; dbl_scores[f"Scrublet (rate {er})"] = t.score.values
dbl_rows = []
for tname, tv in truth_def.items():
    for k, c in dbl_calls.items():
        tp = int((c & tv).sum()); fp = int((c & ~tv).sum()); fn = int((~c & tv).sum())
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        d = dict(truth=tname, n_truth=int(tv.sum()), tool=k, n_called=int(c.sum()), tp=tp, fp=fp, fn=fn,
                 precision=pr, recall=rc, f1=2 * pr * rc / max(pr + rc, 1e-9))
        sc = dbl_scores.get(k)
        if sc is not None and 0 < tv.sum() < len(tv) and np.isfinite(sc).all():
            d["auroc"] = float(roc_auc_score(tv, sc)); d["auprc"] = float(average_precision_score(tv, sc))
        dbl_rows.append(d)
dbl_tab = pd.DataFrame(dbl_rows)

by = pd.read_csv(f"{SINGLET}/barnyard_classification.tsv", sep="\t")
summary = dict(alpha_m=alpha_m, alphas=prep["alphas"], n_cells=int(len(meta)), n_pure=int(pure.sum()),
               species_counts=meta.species.value_counts().to_dict(), global_truth_rho_umi_weighted=glob_truth,
               singlet_json=json.load(open(f"{SINGLET}/ambient_contamination.json")), r_tools=rtools,
               barnyard_all_barcodes=by.species_call.value_counts().to_dict(),
               barnyard_called_cells=meta.singlet_barnyard.value_counts().to_dict())
json.dump(summary, open(f"{R}/summary.json", "w"), indent=1, default=float)
corr_tab.to_csv(f"{R}/correction.csv", index=False); est_tab.to_csv(f"{R}/estimates.csv", index=False)
dbl_tab.to_csv(f"{R}/doublets.csv", index=False); pd.DataFrame(mk_rows).to_csv(f"{R}/markers.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
for n, t in [("correction", corr_tab), ("estimates", est_tab), ("markers", pd.DataFrame(mk_rows)), ("doublets", dbl_tab)]:
    print("\n##", n)
    print(t.round(4).to_string(index=False))
print(json.dumps(summary, indent=1, default=float))
