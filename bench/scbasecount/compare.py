"""Compare one sample across Singlet (legacy .singlet bundle), scBaseCount (2026-01-12 h5ad) and the authors' deposit.

Usage (inside a Clipper job):  python -P compare.py GSE GSM SRX ORGANISM_DIR OUT
  needs EDROP = a checkout with bench/empty_droplets/fetch.py and singlet._pz built in place.

Per sample it writes OUT/<GSM>.json (metrics) and OUT/<GSM>.cells.parquet (one row per barcode called by anyone:
Singlet exon/intron totals, scBaseCount Gene / GeneFull_Ex50pAS / Velocyto totals, author totals, call flags).

Definitions (what is compared with what):
  * Singlet exon            ~ STARsolo Gene (exonic) ~ Cell Ranger (exonic, include-introns off)
  * Singlet exon + intron   ~ STARsolo GeneFull_Ex50pAS (exon priority, introns counted) ~ Cell Ranger 7+ default
  * Singlet intron / total  ~ Velocyto unspliced / (spliced + unspliced + ambiguous)
Genes are matched by Ensembl gene ID without version.
"""
import gzip
import io
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.request

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

GSE, GSM, SRX, ORG, OUT = sys.argv[1:6]
EDROP = os.environ["EDROP"]
WORK = os.environ.get("WORK", "/mnt/projects/debruinz_project/debruinz/singlet")
CACHE = os.path.join(WORK, "data/scbasecount/samples")
os.makedirs(OUT, exist_ok=True)
os.makedirs(CACHE, exist_ok=True)
UA = {"User-Agent": "singlet-bench-scbasecount/1.0"}
GCS = "https://storage.googleapis.com/arc-institute-virtual-cell-atlas/scbasecount/2026-01-12/h5ad"
GEO = "https://ftp.ncbi.nlm.nih.gov/geo"
BC_RE = re.compile(r"[ACGT]{16}|[ACGT]{14}|[ACGT]{12}")
res = {"gse": GSE, "gsm": GSM, "srx": SRX, "organism_dir": ORG}


def get(url, tries=5):
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=600) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def download(url, path):
    if not os.path.exists(path):
        tmp = path + ".part"
        with open(tmp, "wb") as fh:
            fh.write(get(url))
        os.replace(tmp, path)
    return path


def norm_bc(s):
    m = BC_RE.search(str(s))
    return m.group(0) if m else None


def strip_ver(g):
    return str(g).split(".")[0]


# ---------------------------------------------------------------- Singlet
snpz = os.path.join(CACHE, GSE, f"{GSM}.npz")
if not os.path.exists(snpz):
    subprocess.run([sys.executable, "-P", os.path.join(EDROP, "bench/empty_droplets/fetch.py"), GSE, "--gsm", GSM,
                    "--out", CACHE], check=True)
z = np.load(snpz, allow_pickle=False)
smeta = json.loads(str(z["meta"]))
s_bc = np.array([norm_bc(b) for b in z["barcodes"]])
s_genes = np.array([strip_ver(g) for g in z["genes"]])
shape = (len(s_genes), len(s_bc))
s_ex = sp.csc_matrix((z["exon_data"], z["exon_indices"], z["exon_indptr"]), shape=shape)
s_in = sp.csc_matrix((z["intron_data"], z["intron_indices"], z["intron_indptr"]), shape=shape)
s_called = z["called"].astype(bool)
ps = smeta.get("pileup_stats", {}) or {}
summ = smeta.get("summary", {}) or {}
ws, eh = ps.get("wrong_strand"), ps.get("exon_hits")
res["singlet"] = {
    "n_barcodes": int(len(s_bc)), "n_called": int(s_called.sum()),
    "protocol": summ.get("protocol") or (smeta.get("gsm_meta") or {}).get("protocol"),
    "reference": summ.get("reference") or summ.get("genome"),
    "wrong_strand_frac": (ws / (ws + eh)) if (ws is not None and eh) else None,
    "reverse_strand": ps.get("reverse_strand"),
    "call_method": (smeta.get("cell_calls") or {}).get("call_method"),
    "total_reads": summ.get("total_reads"), "n_input_reads": summ.get("n_input_reads"),
    "input_reads_capped": summ.get("input_reads_capped"), "sequencing_saturation": summ.get("sequencing_saturation"),
    "median_umi_per_cell": summ.get("median_umi_per_cell"), "reference_build": summ.get("reference_build"),
    "singlet_version": summ.get("singlet_version"), "protocol_name": summ.get("protocol_name"),
}

# ---------------------------------------------------------------- scBaseCount
def load_sbc(feature):
    os.makedirs(os.path.join(CACHE, "sbc", feature), exist_ok=True)
    path = download(f"{GCS}/{feature}/{ORG}/{SRX}.h5ad", os.path.join(CACHE, "sbc", feature, f"{SRX}.h5ad"))
    with h5py.File(path, "r") as f:
        def mat(grp):
            g = f[grp]
            if isinstance(g, h5py.Dataset):
                return sp.csr_matrix(g[()])
            enc = g.attrs.get("encoding-type", "csr_matrix")
            shp = tuple(g.attrs["shape"])
            M = (sp.csr_matrix if "csr" in str(enc) else sp.csc_matrix)((g["data"][()], g["indices"][()], g["indptr"][()]), shape=shp)
            return M.tocsr()

        def idx(grp):
            g = f[grp]
            key = g.attrs.get("_index", "_index")
            v = g[key][()]
            return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in v])

        obs, var = idx("obs"), idx("var")
        varcols = {k: f["var"][k] for k in f["var"].keys()}
        gene_ids = None
        for k in ("gene_ids", "gene_id", "ensembl_id", "gene_ids-0"):
            if k in varcols and isinstance(varcols[k], h5py.Dataset):
                gene_ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in varcols[k][()]])
        if gene_ids is None:
            gene_ids = var
        X = mat("X")
        layers = {k: mat(f"layers/{k}") for k in (f["layers"].keys() if "layers" in f else [])}
        obs_cols = list(f["obs"].keys())
        var_cols = list(f["var"].keys())
    return {"obs": obs, "genes": np.array([strip_ver(g) for g in gene_ids]), "X": X, "layers": layers,
            "obs_cols": obs_cols, "var_cols": var_cols, "bytes": os.path.getsize(path)}


sbc = {}
for feat in ("GeneFull_Ex50pAS", "Gene", "Velocyto"):
    try:
        sbc[feat] = load_sbc(feat)
    except Exception as e:  # noqa: BLE001
        res.setdefault("errors", []).append(f"sbc {feat}: {type(e).__name__}: {e}")
if "GeneFull_Ex50pAS" not in sbc:
    json.dump(res, open(os.path.join(OUT, f"{GSM}.json"), "w"), indent=1, default=str)
    sys.exit(0)
b = sbc["GeneFull_Ex50pAS"]
b_bc = np.array([norm_bc(x) for x in b["obs"]])
res["sbc"] = {"n_cells": int(len(b_bc)), "layers": {k: sorted(v["layers"].keys()) for k, v in sbc.items()},
              "obs_cols": b["obs_cols"], "var_cols": b["var_cols"], "obs_example": b["obs"][:3].tolist(),
              "h5ad_bytes": {k: v["bytes"] for k, v in sbc.items()}, "n_genes": int(len(b["genes"]))}

# ---------------------------------------------------------------- authors (Cell Ranger filtered matrix in GEO)
def geo_dir(acc, kind):
    return f"{GEO}/{kind}/{acc[:-3]}nnn/{acc}/suppl/"


def listing(url):
    try:
        return sorted(set(re.findall(r'href="([^"?/][^"]*)"', get(url).decode("utf-8", "replace"))))
    except Exception:  # noqa: BLE001
        return []


def read_10x_h5(data):
    with h5py.File(io.BytesIO(data), "r") as f:
        g = f["matrix"] if "matrix" in f else f[list(f.keys())[0]]
        M = sp.csc_matrix((g["data"][()], g["indices"][()], g["indptr"][()]), shape=tuple(g["shape"][()]))
        bcs = [x.decode() for x in g["barcodes"][()]]
        feat = g["features"] if "features" in g else g
        ids = [x.decode() for x in (feat["id"][()] if "id" in feat else feat["genes"][()])]
        ftype = [x.decode() for x in feat["feature_type"][()]] if "feature_type" in feat else ["Gene Expression"] * len(ids)
    keep = np.array([t == "Gene Expression" for t in ftype])
    return M[keep], bcs, [strip_ver(i) for i, k in zip(ids, keep) if k]


def read_mtx_triplet(files):
    from scipy.io import mmread
    M = sp.csc_matrix(mmread(io.BytesIO(files["matrix"])))
    bcs = files["barcodes"].decode().split()
    feats = [l.split("\t") for l in files["features"].decode().splitlines() if l.strip()]
    ids = [strip_ver(f[0]) for f in feats]
    if len(feats[0]) >= 3:
        keep = np.array([f[2] == "Gene Expression" for f in feats])
        M, ids = M[keep], [i for i, k in zip(ids, keep) if k]
    if M.shape[0] != len(ids) and M.shape[1] == len(ids):
        M = M.T.tocsc()
    return M, bcs, ids


def author_matrix():
    files = listing(geo_dir(GSM, "samples"))
    h5 = [f for f in files if f.endswith(".h5") and "raw" not in f.lower()]
    for f in h5:
        try:
            return (*read_10x_h5(get(geo_dir(GSM, "samples") + f)), f)
        except Exception:  # noqa: BLE001
            pass
    trip = {}
    for f in files:
        lf = f.lower()
        if "raw" in lf:
            continue
        for key in ("matrix", "barcodes", "features", "genes"):
            if re.search(key + r"\.(mtx|tsv)(\.gz)?$", lf):
                trip["features" if key == "genes" else key] = f
    if {"matrix", "barcodes", "features"} <= set(trip):
        d = {k: get(geo_dir(GSM, "samples") + v) for k, v in trip.items()}
        d = {k: gzip.decompress(v) if trip[k].endswith(".gz") else v for k, v in d.items()}
        return (*read_mtx_triplet(d), trip["matrix"])
    for f in [f for f in files if re.search(r"\.(tar\.gz|tgz|tar)$", f, re.I)]:
        try:
            data = get(geo_dir(GSM, "samples") + f)
            d = {}
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
                for m in tf.getmembers():
                    n = m.name.lower()
                    if "raw" in n or not m.isfile():
                        continue
                    if n.endswith(".h5"):
                        return (*read_10x_h5(tf.extractfile(m).read()), f + ":" + m.name)
                    for key in ("matrix", "barcodes", "features", "genes"):
                        if re.search(key + r"\.(mtx|tsv)(\.gz)?$", n):
                            raw = tf.extractfile(m).read()
                            d["features" if key == "genes" else key] = gzip.decompress(raw) if n.endswith(".gz") else raw
            if {"matrix", "barcodes", "features"} <= set(d):
                return (*read_mtx_triplet(d), f)
        except Exception:  # noqa: BLE001
            continue
    return None


try:
    am = author_matrix()
except Exception as e:  # noqa: BLE001
    am = None
    res.setdefault("errors", []).append(f"author: {type(e).__name__}: {e}")
a_bc = a_tot = a_M = a_genes = None
if am is not None:
    a_M, a_bcs_raw, a_genes, a_src = am
    a_bc = np.array([norm_bc(x) for x in a_bcs_raw])
    a_tot = np.asarray(a_M.sum(0)).ravel()
    res["author"] = {"source": a_src, "n_cells": int(len(a_bc)), "n_genes": int(a_M.shape[0])}
    if len(a_bc) > 150000:
        res["author"]["note"] = "looks like a raw (unfiltered) matrix; ignored for cell calls"
        a_bc = a_tot = a_M = None

# ---------------------------------------------------------------- per-barcode table
s_idx = {bc: i for i, bc in enumerate(s_bc) if bc}
s_ex_tot = np.asarray(s_ex.sum(0)).ravel()
s_in_tot = np.asarray(s_in.sum(0)).ravel()
s_ex_ng = np.diff(s_ex.indptr)
s_all = (s_ex + s_in).tocsc()
s_all_ng = np.diff(s_all.indptr)

union = set(s_bc[s_called]) | set(b_bc)
if a_bc is not None:
    union |= set(a_bc)
union.discard(None)
union = sorted(union)
T = pd.DataFrame({"barcode": union})
si = T.barcode.map(s_idx)
T["singlet_in_matrix"] = si.notna()
sii = si.fillna(-1).astype(int).values
ok = sii >= 0
for name, arr in (("singlet_exon", s_ex_tot), ("singlet_intron", s_in_tot), ("singlet_genes", s_all_ng),
                  ("singlet_exon_genes", s_ex_ng)):
    v = np.full(len(T), np.nan)
    v[ok] = arr[sii[ok]]
    T[name] = v
T["singlet_called"] = False
T.loc[ok, "singlet_called"] = s_called[sii[ok]]
b_idx = {bc: i for i, bc in enumerate(b_bc)}
T["sbc_called"] = T.barcode.isin(b_idx)


def put_tot(prefix, M, bcs):
    ix = {bc: i for i, bc in enumerate(bcs)}
    tot = np.asarray(M.sum(1)).ravel()
    ng = np.diff(M.tocsr().indptr)
    j = T.barcode.map(ix)
    for nm, arr in ((prefix, tot), (prefix + "_genes", ng)):
        v = np.full(len(T), np.nan)
        okk = j.notna().values
        v[okk] = arr[j[okk].astype(int).values]
        T[nm] = v


put_tot("sbc_genefull_ex50", b["X"], b_bc)
if "Gene" in sbc:
    put_tot("sbc_gene", sbc["Gene"]["X"], [norm_bc(x) for x in sbc["Gene"]["obs"]])
    for lk in sbc["Gene"]["layers"]:
        if "EM" in lk:
            put_tot("sbc_gene_em", sbc["Gene"]["layers"][lk], [norm_bc(x) for x in sbc["Gene"]["obs"]])
if "Velocyto" in sbc:
    v = sbc["Velocyto"]
    vb = [norm_bc(x) for x in v["obs"]]
    for lk in ("spliced", "unspliced", "ambiguous"):
        if lk in v["layers"]:
            put_tot("sbc_" + lk, v["layers"][lk], vb)
if a_bc is not None:
    T["author_called"] = T.barcode.isin(set(a_bc))
    put_tot("author", a_M.T.tocsr(), a_bc)
T.to_parquet(os.path.join(OUT, f"{GSM}.cells.parquet"))

# ---------------------------------------------------------------- metrics


def jacc(a, b2):
    a, b2 = set(a), set(b2)
    return len(a & b2) / max(1, len(a | b2))


calls = {"singlet": set(T.barcode[T.singlet_called]), "sbc": set(T.barcode[T.sbc_called])}
if a_bc is not None:
    calls["author"] = set(T.barcode[T.author_called])
res["calls"] = {k: len(v) for k, v in calls.items()}
res["calls"]["singlet&sbc"] = len(calls["singlet"] & calls["sbc"])
res["calls"]["jaccard_singlet_sbc"] = jacc(calls["singlet"], calls["sbc"])
res["calls"]["sbc_not_singlet_in_singlet_matrix"] = int((T.sbc_called & ~T.singlet_called & T.singlet_in_matrix).sum())
if "author" in calls:
    res["calls"]["jaccard_singlet_author"] = jacc(calls["singlet"], calls["author"])
    res["calls"]["jaccard_sbc_author"] = jacc(calls["sbc"], calls["author"])
    res["calls"]["singlet_in_author"] = len(calls["singlet"] & calls["author"]) / max(1, len(calls["singlet"]))
    res["calls"]["sbc_in_author"] = len(calls["sbc"] & calls["author"]) / max(1, len(calls["sbc"]))
    res["calls"]["author_recovered_by_singlet"] = len(calls["singlet"] & calls["author"]) / max(1, len(calls["author"]))
    res["calls"]["author_recovered_by_sbc"] = len(calls["sbc"] & calls["author"]) / max(1, len(calls["author"]))

shared = T[T.singlet_in_matrix & T.sbc_called].copy()
res["n_shared_cells"] = int(len(shared))


def ratio_stats(x, y):
    m = (x > 0) & (y > 0)
    if m.sum() < 20:
        return None
    r = (x[m] / y[m])
    return {"median_ratio": float(np.median(r)), "p10": float(np.quantile(r, 0.1)), "p90": float(np.quantile(r, 0.9)),
            "log_pearson": float(pearsonr(np.log1p(x[m]), np.log1p(y[m]))[0]), "n": int(m.sum())}


S_tot = (shared.singlet_exon + shared.singlet_intron).values
res["totals"] = {
    "singlet_total_vs_sbc_genefull_ex50": ratio_stats(S_tot, shared.sbc_genefull_ex50.values),
    "singlet_exon_vs_sbc_gene": ratio_stats(shared.singlet_exon.values, shared.get("sbc_gene", pd.Series(np.nan, index=shared.index)).values),
    "median_umi": {"singlet_total": float(np.median(S_tot)) if len(S_tot) else None,
                   "sbc_genefull_ex50": float(shared.sbc_genefull_ex50.median()) if len(shared) else None},
}
if "sbc_unspliced" in shared:
    tot_v = shared[["sbc_spliced", "sbc_unspliced", "sbc_ambiguous"]].sum(1)
    res["unspliced"] = {"singlet_median": float(np.nanmedian(shared.singlet_intron / S_tot.clip(1))),
                        "sbc_median": float(np.nanmedian(shared.sbc_unspliced / tot_v.clip(1))),
                        "sbc_ambiguous_median": float(np.nanmedian(shared.sbc_ambiguous / tot_v.clip(1)))}
if a_bc is not None:
    sa = T[T.singlet_in_matrix & T.author_called & T.sbc_called]
    res["totals"]["singlet_exon_vs_author"] = ratio_stats(sa.singlet_exon.values, sa.author.values)
    res["totals"]["singlet_total_vs_author"] = ratio_stats((sa.singlet_exon + sa.singlet_intron).values, sa.author.values)
    if "sbc_gene" in sa:
        res["totals"]["sbc_gene_vs_author"] = ratio_stats(sa.sbc_gene.values, sa.author.values)
    res["totals"]["sbc_genefull_ex50_vs_author"] = ratio_stats(sa.sbc_genefull_ex50.values, sa.author.values)

# gene-level: per-cell gene-vector correlation and pseudobulk, on shared cells and common genes
common = sorted(set(s_genes) & set(b["genes"]))
res["genes"] = {"singlet": int(len(set(s_genes))), "sbc": int(len(set(b["genes"]))), "common": len(common)}
if len(shared) >= 20 and len(common) > 1000:
    sg_ix = pd.Series(np.arange(len(s_genes)), index=s_genes)
    sg_ix = sg_ix[~sg_ix.index.duplicated()]
    bg_ix = pd.Series(np.arange(len(b["genes"])), index=b["genes"])
    bg_ix = bg_ix[~bg_ix.index.duplicated()]
    rows_s = sg_ix[common].values
    rows_b = bg_ix[common].values
    cells = shared.sample(min(2000, len(shared)), random_state=0) if len(shared) > 2000 else shared
    cs = cells.barcode.map(s_idx).astype(int).values
    cb = cells.barcode.map(b_idx).astype(int).values
    S = s_all[:, cs][rows_s, :].toarray().astype(np.float32)          # genes x cells
    Sx = s_ex[:, cs][rows_s, :].toarray().astype(np.float32)
    B = b["X"][cb, :][:, rows_b].T.toarray().astype(np.float32)
    rr = [pearsonr(np.log1p(S[:, k]), np.log1p(B[:, k]))[0] for k in range(S.shape[1])]
    # depth-matched: thin scBaseCount's counts per cell to Singlet's per-cell total, so the read cap is taken out
    rng = np.random.default_rng(0)
    p_keep = np.clip(S.sum(0) / np.maximum(B.sum(0), 1), 0, 1)
    Bt = rng.binomial(B.astype(np.int64), p_keep[None, :]).astype(np.float32)
    rr_t = [pearsonr(np.log1p(S[:, k]), np.log1p(Bt[:, k]))[0] for k in range(S.shape[1])]
    # noise floor: two independent thinnings of scBaseCount itself to Singlet's depth
    Bt2 = rng.binomial(B.astype(np.int64), p_keep[None, :]).astype(np.float32)
    rr_self = [pearsonr(np.log1p(Bt2[:, k]), np.log1p(Bt[:, k]))[0] for k in range(S.shape[1])]
    res["gene_level"] = {"median_cell_corr_total_vs_ex50": float(np.nanmedian(rr)),
                         "median_cell_corr_total_vs_ex50_depth_matched": float(np.nanmedian(rr_t)),
                         "median_cell_corr_sbc_self_at_singlet_depth": float(np.nanmedian(rr_self)),
                         "pseudobulk_spearman_total_vs_ex50": float(spearmanr(S.sum(1), B.sum(1))[0]),
                         "n_cells": int(S.shape[1])}
    if "Gene" in sbc:
        g = sbc["Gene"]
        gb = pd.Series(np.arange(len(g["genes"])), index=g["genes"])
        gb = gb[~gb.index.duplicated()]
        gob = {bc: i for i, bc in enumerate(norm_bc(x) for x in g["obs"])}
        cg = cells.barcode.map(gob)
        okg = cg.notna().values
        G = g["X"][cg[okg].astype(int).values, :][:, gb[common].values].T.toarray().astype(np.float32)
        rr2 = [pearsonr(np.log1p(Sx[:, k]), np.log1p(G[:, j]))[0] for j, k in enumerate(np.where(okg)[0])]
        res["gene_level"]["median_cell_corr_exon_vs_gene"] = float(np.nanmedian(rr2))
        res["gene_level"]["pseudobulk_spearman_exon_vs_gene"] = float(spearmanr(Sx[:, okg].sum(1), G.sum(1))[0])
    # genes with the largest pseudobulk disagreement (symbols from the Singlet bundle)
    sym = dict(zip([strip_ver(x) for x in z["genes"]], z["symbols"]))
    pbS, pbB = S.sum(1), B.sum(1)
    lr = np.log2((pbS + 1) / (pbB + 1))
    top = np.argsort(-np.abs(lr) * np.log1p(pbS + pbB))[:15]
    res["gene_level"]["top_discordant"] = [{"gene": common[i], "symbol": str(sym.get(common[i], "")),
                                            "singlet": float(pbS[i]), "sbc": float(pbB[i]), "log2": float(lr[i])} for i in top]
    if a_M is not None:
        ag = pd.Series(np.arange(len(a_genes)), index=a_genes)
        ag = ag[~ag.index.duplicated()]
        common_a = [x for x in common if x in ag.index]
        aidx = {bc: i for i, bc in enumerate(a_bc)}
        ca = cells.barcode.map(aidx)
        oka = ca.notna().values
        if oka.sum() >= 20 and len(common_a) > 1000:
            A = a_M[ag[common_a].values, :][:, ca[oka].astype(int).values].toarray().astype(np.float32)
            posS = pd.Series(range(len(common)), index=common)[common_a].values
            Sx_a = Sx[posS][:, oka]
            B_a = B[posS][:, oka]
            res["gene_level"]["median_cell_corr_singlet_exon_vs_author"] = float(np.nanmedian(
                [pearsonr(np.log1p(Sx_a[:, k]), np.log1p(A[:, k]))[0] for k in range(A.shape[1])]))
            res["gene_level"]["median_cell_corr_sbc_ex50_vs_author"] = float(np.nanmedian(
                [pearsonr(np.log1p(B_a[:, k]), np.log1p(A[:, k]))[0] for k in range(A.shape[1])]))
            pa = np.clip(Sx_a.sum(0) / np.maximum(A.sum(0), 1), 0, 1)
            At = rng.binomial(A.astype(np.int64), pa[None, :]).astype(np.float32)
            res["gene_level"]["median_cell_corr_singlet_exon_vs_author_depth_matched"] = float(np.nanmedian(
                [pearsonr(np.log1p(Sx_a[:, k]), np.log1p(At[:, k]))[0] for k in range(A.shape[1])]))
            res["gene_level"]["pseudobulk_spearman_singlet_exon_vs_author"] = float(spearmanr(Sx_a.sum(1), A.sum(1))[0])
            res["gene_level"]["pseudobulk_spearman_sbc_ex50_vs_author"] = float(spearmanr(B_a.sum(1), A.sum(1))[0])

json.dump(res, open(os.path.join(OUT, f"{GSM}.json"), "w"), indent=1, default=str)
print(json.dumps({k: res.get(k) for k in ("calls", "totals", "gene_level", "unspliced", "errors")}, indent=1, default=str))
