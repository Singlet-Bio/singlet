"""Doublet benchmark. Runs scrublet on Singlet's filtered matrix (same cells), then scores Singlet and scrublet
against the genotype truth (Kang multiplets=='doublet' vs 'singlet'). Usage: doublets.py TRUTH SINGLET_DIR OUT.json [vireo_donor_ids.tsv]"""
import sys, json, time, os, glob, numpy as np, pandas as pd, scipy.io, scipy.sparse as sp, gzip
import scrublet as scr
from common import *
truth_f, sdir, out = sys.argv[1:4]; vireo = sys.argv[4] if len(sys.argv) > 4 else None
T = load_truth(truth_f)
mdir = os.path.join(sdir, "filtered_feature_bc_matrix")
def op(p): return gzip.open(p, "rt") if p.endswith(".gz") else open(p)
mf = [p for p in glob.glob(mdir + "/matrix.mtx*")][0]; bf = [p for p in glob.glob(mdir + "/barcodes.tsv*")][0]
X = scipy.io.mmread(mf).T.tocsr()            # all barcodes x genes
allb = [l.strip().split("-")[0] for l in op(bf)]
cc = pd.read_csv(os.path.join(sdir, "cell_calls.tsv"), sep="	"); called = set(cc.loc[cc.is_cell.astype(str).str.upper() == "TRUE", "barcode"].str.split("-").str[0])
keep = np.array([i for i, b in enumerate(allb) if b in called]); X = X[keep]; bcs = [allb[i] for i in keep]
print("matrix", X.shape, flush=True)
t0 = time.time()
rate = float((T.multiplets == "doublet").mean())
sc = scr.Scrublet(X, expected_doublet_rate=rate, random_state=0)
s_scr, p_scr = sc.scrub_doublets(min_counts=2, min_cells=3, n_prin_comps=30, verbose=False)
t_scr = time.time() - t0
df = pd.DataFrame({"scrublet": s_scr, "scrublet_call": p_scr}, index=bcs)
S = pd.read_csv(os.path.join(sdir, "doublet_scores.tsv"), sep="\t")
S["bc"] = S.barcode.str.split("-").str[0]; S = S.drop_duplicates("bc").set_index("bc")
df["singlet"] = S["doublet_score"]; df["singlet_call"] = S["is_doublet"].astype(str).eq("True")
df["umis"] = S["total_umis"]
if vireo:
    V = pd.read_csv(vireo, sep="\t"); V["bc"] = V.cell.str.split("-").str[0]; df["vireo_geno"] = V.drop_duplicates("bc").set_index("bc")["prob_doublet"]
df = df.join(T[["multiplets", "ind", "cell"]], how="inner")
d = df[df.multiplets.isin(["singlet", "doublet"])].copy()
y = (d.multiplets == "doublet").values
rate_d = float(y.mean())
res = {"cells_in_matrix": len(bcs), "cells_with_truth_label": len(d), "truth_doublets": int(y.sum()), "truth_doublet_rate": rate_d,
       "scrublet_seconds": t_scr}
for name in ["singlet", "scrublet", "vireo_geno", "umis"]:
    if name not in d or d[name].isna().all(): continue
    s = d[name].fillna(d[name].min()).values
    auc, ap = auc_ap(y, s); rec, prec = recall_at_rate(y, s, rate_d)
    r = dict(auroc=auc, auprc=ap, recall_at_truth_rate=rec, precision_at_truth_rate=prec)
    if name + "_call" in d:
        c = d[name + "_call"].values.astype(bool)
        r.update(native_calls=int(c.sum()), native_rate=float(c.mean()), native_recall=float(c[y].mean()), native_precision=float(y[c].mean()) if c.sum() else None)
    res[name] = r
# UMI-only baseline above shows how much of the signal is library size (heterotypic doublets carry more UMIs)
json.dump(res, open(out, "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
d.to_csv(out.replace(".json", "_scores.csv.gz"))
