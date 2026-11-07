"""Singlet Visium vs Space Ranger published outputs.

    python compare_visium.py --singlet RUN_DIR --tenx-dir DIR --prefix NAME --out result.json

Space Ranger 1.1 counts exonic reads only, so Singlet's exon_counts (spliced + ambiguous molecules) is compared.
Genes are matched by Ensembl id (version dropped). Tissue: Space Ranger's image-based in_tissue vs Singlet's
count-based call (visium_tissue.h).
"""
import argparse, gzip, io, json, os, re, tarfile
import h5py, numpy as np, pandas as pd, scipy.io, scipy.sparse as sp
from scipy.stats import spearmanr, pearsonr

ap = argparse.ArgumentParser()
ap.add_argument("--singlet", required=True); ap.add_argument("--tenx-dir", required=True)
ap.add_argument("--prefix", required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
T = lambda x: f"{a.tenx_dir}/{a.prefix}_{x}"
R = {"dataset": a.prefix, "reference_tool": "Space Ranger 1.1.0"}

def read_h5(p):
    with h5py.File(p, "r") as f:
        m = f["matrix"]
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=m["shape"][:])
        bc = [b.decode().replace("-1", "") for b in m["barcodes"][:]]
        ids = [i.decode().split(".")[0] for i in m["features"]["id"][:]]
    return X, bc, ids

Xt, bt, gt = read_h5(T("raw_feature_bc_matrix.h5"))
with tarfile.open(T("spatial.tar.gz")) as tf:
    mem = [m for m in tf.getmembers() if "tissue_positions" in m.name][0]
    txt = tf.extractfile(mem).read().decode()
pos = pd.read_csv(io.StringIO(txt), header=0 if txt.startswith("barcode") else None)
pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
pos["barcode"] = pos.barcode.str.replace("-1", "", regex=False)
pos = pos.set_index("barcode")

# Singlet
S = a.singlet
Xs = scipy.io.mmread(f"{S}/exon_counts.mtx.gz").tocsc()
bs = [l.strip().replace("-1", "") for l in gzip.open(f"{S}/exon_counts_barcodes.tsv.gz", "rt")]
fs = [l.strip().split("\t")[0] for l in gzip.open(f"{S}/exon_counts_features.tsv.gz", "rt")]
gs = [re.match(r"(ENS[A-Z]*G\d+)", f).group(1) if re.match(r"ENS[A-Z]*G\d+", f) else f for f in fs]
spt = pd.read_csv(f"{S}/spatial_positions.tsv", sep="\t", dtype={"spot": str}).set_index("barcode")
spt.index = spt.index.str.replace("-1", "", regex=False)
summ = json.load(open(f"{S}/spatial_summary.json"))
R["singlet_summary"] = summ

# Aggregate Singlet columns onto spots (columns that map to the same spot are summed)
col_spot = spt.spot.reindex(bs)
ok = col_spot.notna().to_numpy()
spots = sorted(set(col_spot[ok]))
sidx = {b: i for i, b in enumerate(spots)}
A = sp.csr_matrix((np.ones(ok.sum()), (np.flatnonzero(ok), [sidx[b] for b in col_spot[ok]])), shape=(len(bs), len(spots)))
Xs_spot = (Xs @ A).tocsc()  # genes x spots
R["columns"] = {"singlet_columns": len(bs), "columns_on_slide": int(ok.sum()), "spots_with_columns": len(spots),
                "columns_per_spot_max": int(np.bincount([sidx[b] for b in col_spot[ok]]).max())}

# Common genes and spots
gi_t = {g: i for i, g in enumerate(gt)}
common_g = [g for g in dict.fromkeys(gs) if g in gi_t]
rs = [gs.index(g) for g in common_g]; rt_ = [gi_t[g] for g in common_g]
bi_t = {b: i for i, b in enumerate(bt)}
tissue_sr = set(pos.index[pos.in_tissue == 1])
common_s = [b for b in spots if b in bi_t]
cs = [sidx[b] for b in common_s]; ct = [bi_t[b] for b in common_s]
Ms = Xs_spot[rs][:, cs].tocsc(); Mt = Xt[rt_][:, ct].tocsc()
umi_s = np.asarray(Ms.sum(0)).ravel(); umi_t = np.asarray(Mt.sum(0)).ravel()
gen_s = np.asarray((Ms > 0).sum(0)).ravel(); gen_t = np.asarray((Mt > 0).sum(0)).ravel()
in_sr = np.array([b in tissue_sr for b in common_s])

def corr(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    return {"n": int(m.sum()), "spearman": float(spearmanr(x[m], y[m])[0]),
            "pearson_log": float(pearsonr(np.log1p(x[m]), np.log1p(y[m]))[0]),
            "median_ratio_singlet_over_spaceranger": float(np.median((x[m] + 1) / (y[m] + 1)))}

R["genes_common"] = len(common_g)
R["per_spot_umis_tissue"] = corr(umi_s[in_sr], umi_t[in_sr])
R["per_spot_genes_tissue"] = corr(gen_s[in_sr].astype(float), gen_t[in_sr].astype(float))
R["per_spot_umis_all"] = corr(umi_s, umi_t)
# per-spot expression profile correlation (log1p CPM over common genes), tissue spots
def lcpm(M):
    M = M.astype(float).tocsc(); tot = np.asarray(M.sum(0)).ravel(); tot[tot == 0] = 1
    return np.log1p(M.multiply(1e4 / tot).toarray())
Ls, Lt = lcpm(Ms[:, np.flatnonzero(in_sr)]), lcpm(Mt[:, np.flatnonzero(in_sr)])
pr = [pearsonr(Ls[:, j], Lt[:, j])[0] for j in range(Ls.shape[1]) if Ls[:, j].any() and Lt[:, j].any()]
R["per_spot_profile_pearson_tissue"] = {"median": float(np.median(pr)), "p10": float(np.percentile(pr, 10)), "n": len(pr)}
gt_s = np.asarray(Ms[:, in_sr].sum(1)).ravel(); gt_t = np.asarray(Mt[:, in_sr].sum(1)).ravel()
m = (gt_s + gt_t) > 0
R["per_gene_totals_tissue"] = corr(gt_s[m].astype(float), gt_t[m].astype(float))

# Tissue agreement on all 4,992 slide spots
sp_sing = spt.dropna(subset=["spot"]).groupby("spot").in_tissue.max()
allspots = pos.index
y_sr = pos.in_tissue.reindex(allspots).fillna(0).astype(int).to_numpy()
y_s = sp_sing.reindex(allspots).fillna(0).astype(int).to_numpy()
tp = int(((y_sr == 1) & (y_s == 1)).sum()); fp = int(((y_sr == 0) & (y_s == 1)).sum())
fn = int(((y_sr == 1) & (y_s == 0)).sum()); tn = int(((y_sr == 0) & (y_s == 0)).sum())
n = tp + fp + fn + tn
po = (tp + tn) / n; pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / n / n
R["tissue"] = {"spots": n, "spaceranger_in_tissue": tp + fn, "singlet_in_tissue": tp + fp, "both": tp,
               "agreement": po, "jaccard": tp / max(1, tp + fp + fn), "cohen_kappa": (po - pe) / (1 - pe),
               "recall_of_spaceranger": tp / max(1, tp + fn), "precision_vs_spaceranger": tp / max(1, tp + fp)}
# Baseline for context: the best single global UMI threshold (oracle, chosen with the Space Ranger labels)
u_all = pd.Series(dict(zip(common_s, umi_s))).reindex(allspots).fillna(0).to_numpy()
best = max(((np.mean((u_all >= th) == (y_sr == 1)), th) for th in np.unique(np.quantile(u_all, np.linspace(0, 1, 201)))))
R["tissue"]["oracle_umi_threshold_agreement"] = float(best[0])

def rt(p):
    if not os.path.exists(p):
        return None
    txt = open(p).read()
    w = re.search(r"Elapsed \(wall clock\) time.*: (.*)", txt).group(1)
    parts = [float(x) for x in w.split(":")]
    sec = parts[-1] + 60 * parts[-2] + (3600 * parts[-3] if len(parts) == 3 else 0)
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", txt).group(1))
    cpu = float(re.search(r"User time \(seconds\): ([\d.]+)", txt).group(1)) + \
        float(re.search(r"System time \(seconds\): ([\d.]+)", txt).group(1))
    return {"wall_s": sec, "max_rss_gb": rss / 1048576, "cpu_h": cpu / 3600}
R["runtime"] = {"singlet_process": rt(f"{S}/time.txt"), "singlet_encode": rt(f"{S}/time_encode.txt")}
json.dump(R, open(a.out, "w"), indent=1, default=float)
print(json.dumps(R, indent=1, default=float))
