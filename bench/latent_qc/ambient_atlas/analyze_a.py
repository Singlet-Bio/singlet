"""Atlas ambient profiles: composition, markers (overall / per tissue), structure (kNN / silhouette / variance explained).

usage: analyze_a.py REC_DIR SAMPLES_TSV OUT_DIR [nuclei_intron_threshold]
"""
import json
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from aa_load import HB3, load_records

rec_dir, meta_tsv, out = sys.argv[1], sys.argv[2], sys.argv[3]
NUC_THR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.35
import os
os.makedirs(out, exist_ok=True)
S, M, genes, sym, gcl = load_records(rec_dir, meta_tsv, org=os.environ.get("AA_ORG", "human"))
S["nuclei"] = np.where(S.intron_frac_called > NUC_THR, "nuclei", "cells")
print("samples", len(S), "core genes", len(genes), "pool_ok", int(S.pool_ok.sum()), flush=True)

# ---- composition ------------------------------------------------------------------------------------------------
def share(mat, mask):
    return mat[:, mask].sum(1) / np.maximum(mat.sum(1), 1)

isHB3 = np.isin(sym, list(HB3))
for tag, key in (("amb", "pool1"), ("cell", "called")):
    S[f"{tag}_mt"] = share(M[key], gcl == "MT")
    S[f"{tag}_hb"] = share(M[key], isHB3)
    S[f"{tag}_malat1"] = share(M[key], gcl == "MALAT1")
    S[f"{tag}_rp"] = share(M[key], gcl == "RP")
    S[f"{tag}_ig"] = share(M[key], gcl == "IG")
S.drop(columns=[], inplace=True)
S.to_csv(f"{out}/sample_summary.csv", index=False)
ok = S.pool_ok.values
Sok = S[ok].reset_index(drop=True)
Mok = {k: v[ok] for k, v in M.items()}
n = len(Sok)

# ---- fractions and per-sample enrichment ------------------------------------------------------------------------
A = Mok["pool1"].astype(np.float64)
C = Mok["called"].astype(np.float64)
Na, Nc = A.sum(1, keepdims=True), C.sum(1, keepdims=True)
fa, fc = A / Na, C / Nc
enr = np.log2(((A + 0.5) / Na) / ((C + 0.5) / Nc))
evalu = np.maximum(A, Na * fc) >= 5
enr = np.where(evalu, enr, np.nan)


def marker_table(idx, label):
    e = enr[idx]
    ne = evalu[idx].sum(0)
    with np.errstate(all="ignore"):
        med = np.nanmedian(e, axis=0)
        frac2 = np.nansum(e > 1.0, axis=0) / np.maximum(ne, 1)
    ma, mc = fa[idx].mean(0), fc[idx].mean(0)
    df = pd.DataFrame(dict(gene=genes, symbol=sym, cls=gcl, tissue=label, n_samples=len(idx), n_eval=ne, median_log2enr=med,
                           frac_gt2x=frac2, pooled_log2enr=np.log2((ma + 1e-7) / (mc + 1e-7)),
                           mean_amb_cpm=ma * 1e6, mean_cell_cpm=mc * 1e6))
    return df


allidx = np.arange(n)
mo = marker_table(allidx, "all")
cand = mo[(mo.n_eval >= max(30, 0.3 * n)) & (mo.mean_amb_cpm >= 50)].sort_values("median_log2enr", ascending=False)
cand.head(3000).to_csv(f"{out}/markers_overall.csv", index=False)
print("TOP overall ambient-enriched (median log2 amb/cell):")
print(cand.head(40)[["symbol", "cls", "n_eval", "median_log2enr", "frac_gt2x", "mean_amb_cpm", "mean_cell_cpm"]].to_string(index=False))
print("class summary of top 200:", cand.head(200).cls.replace("", "other").value_counts().to_dict())

# tissue-level
rows = []
oth_all = fa.mean(0)
for t, g in Sok.groupby("tissue_coarse"):
    idx = g.index.values
    if len(idx) < 4 or g.gse.nunique() < 2:
        continue
    df = marker_table(idx, t)
    other = np.setdiff1d(allidx, idx)
    df["tissue_excess_log2"] = np.log2((fa[idx].mean(0) + 1e-7) / (fa[other].mean(0) + 1e-7))
    df = df[(df.n_eval >= max(4, 0.6 * len(idx))) & (df.mean_amb_cpm >= 100)]
    top1 = df.sort_values("median_log2enr", ascending=False).head(25)
    top2 = df.sort_values("tissue_excess_log2", ascending=False).head(25)
    rows.append(pd.concat([top1.assign(rank_by="median_log2enr"), top2.assign(rank_by="tissue_excess")]))
mt = pd.concat(rows)
mt.to_csv(f"{out}/markers_by_tissue.csv", index=False)
for t in mt.tissue.unique():
    x = mt[(mt.tissue == t) & (mt.rank_by == "tissue_excess")].head(8)
    y = mt[(mt.tissue == t) & (mt.rank_by == "median_log2enr")].head(8)
    print(f"[{t}] n={int(x.n_samples.iloc[0])} excess: " + ", ".join(f"{a}({b:.1f})" for a, b in zip(x.symbol, x.tissue_excess_log2)))
    print(f"      enriched: " + ", ".join(f"{a}({b:.1f})" for a, b in zip(y.symbol, y.median_log2enr)))

# ---- structure --------------------------------------------------------------------------------------------------
cpm = lambda f: np.log2(f * 1e6 + 5.0)
XA, XC = cpm(fa), cpm(fc)
gsel = (fa.mean(0) * 1e6 > 5) | (fc.mean(0) * 1e6 > 5)
print("genes for structure:", int(gsel.sum()))
XA, XC = XA[:, gsel], XC[:, gsel]
labels = {"tissue": Sok.tissue_coarse.values, "chemistry": Sok.chem.values, "nuclei": Sok.nuclei.values}
studies = Sok.gse.values


def loso_knn(Z, lab, k=5):
    Zn = Z / np.linalg.norm(Z, axis=1, keepdims=True)
    sim = Zn @ Zn.T
    hit, tot = 0, 0
    for i in range(len(Z)):
        m = studies != studies[i]
        order = np.argsort(-sim[i][m])[:k]
        votes = pd.Series(lab[m][order]).value_counts()
        hit += votes.index[0] == lab[i]
        tot += 1
    return hit / tot


def pcs(X, k=30):
    Xc = X - X.mean(0)
    return PCA(n_components=min(k, len(X) - 1), random_state=0).fit_transform(Xc)


rng = np.random.default_rng(0)
res = []
Z = {"ambient": pcs(XA), "cells": pcs(XC)}
for pn, Zp in Z.items():
    for ln, lab in labels.items():
        acc = loso_knn(Zp, lab)
        shuf = np.mean([loso_knn(Zp, rng.permutation(lab)) for _ in range(5)])
        vc = pd.Series(lab).value_counts(normalize=True)
        keep = pd.Series(lab).map(pd.Series(lab).value_counts()) >= 2
        sil = silhouette_score(Zp[keep.values], lab[keep.values], metric="cosine") if keep.sum() > 3 and len(set(lab[keep.values])) > 1 else np.nan
        res.append(dict(profile=pn, label=ln, knn5_loso_acc=acc, shuffled_acc=shuf, majority_frac=vc.iloc[0], silhouette_cosine=sil))
struct = pd.DataFrame(res)
struct.to_csv(f"{out}/structure_knn.csv", index=False)
print(struct.to_string(index=False))
np.save(f"{out}/pc_ambient.npy", Z["ambient"][:, :2]); np.save(f"{out}/pc_cells.npy", Z["cells"][:, :2])
Sok[["gsm", "gse", "tissue_coarse", "chem", "nuclei"]].to_csv(f"{out}/pc_labels.csv", index=False)

# eta^2 (adjusted) of tissue / study per gene, in the log-CPM space
def adj_eta(X, lab):
    lv = pd.factorize(lab)[0]
    k = lv.max() + 1
    gm = X.mean(0)
    sst = ((X - gm) ** 2).sum(0)
    ssb = np.zeros(X.shape[1])
    for j in range(k):
        m = lv == j
        ssb += m.sum() * (X[m].mean(0) - gm) ** 2
    ssw = sst - ssb
    return 1 - (ssw / (len(X) - k)) / (sst / (len(X) - 1)), k

for pn, X in (("ambient", XA), ("cells", XC)):
    for ln, lab in (("tissue", Sok.tissue_coarse.values), ("study", studies)):
        e, k = adj_eta(X, lab)
        print(f"adjusted eta2 {pn} by {ln} (k={k}): median over genes {np.median(e):.3f}")
        struct.loc[len(struct)] = dict(profile=pn, label=f"adj_eta2_{ln}", knn5_loso_acc=np.median(e))
struct.to_csv(f"{out}/structure_knn.csv", index=False)

# variance of an ambient profile explained by own cells vs tissue atlas mean of OTHER studies (per sample, across genes)
r2 = []
for i in range(n):
    oth = (studies != studies[i])
    same = oth & (Sok.tissue_coarse.values == Sok.tissue_coarse.values[i])
    y = XA[i]
    xc = XC[i]
    xt = XA[same].mean(0) if same.sum() >= 1 else XA[oth].mean(0)
    xg = XA[oth].mean(0)

    def r2of(cols):
        D = np.column_stack([np.ones(len(y))] + cols)
        beta, *_ = np.linalg.lstsq(D, y, rcond=None)
        return 1 - ((y - D @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    r2.append(dict(gsm=Sok.gsm[i], tissue=Sok.tissue_coarse[i], n_same_tissue_other_studies=int(same.sum()), r2_cells=r2of([xc]),
                   r2_tissue=r2of([xt]), r2_global=r2of([xg]), r2_cells_tissue=r2of([xc, xt]), r2_cells_global=r2of([xc, xg])))
R2 = pd.DataFrame(r2)
R2.to_csv(f"{out}/r2_by_sample.csv", index=False)
print(R2.drop(columns=["gsm", "tissue"]).describe().loc[["mean", "25%", "50%", "75%"]].round(3).to_string())
print(R2.groupby("tissue")[["r2_cells", "r2_tissue", "r2_cells_tissue"]].median().round(3).to_string())
json.dump({"n_samples": n, "n_core_genes": int(len(genes)), "nuclei_threshold": NUC_THR}, open(f"{out}/analyze_a.json", "w"))
