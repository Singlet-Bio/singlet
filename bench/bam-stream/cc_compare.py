# SPDX-License-Identifier: MIT
"""Cell calls of two runs of one sample (stream vs sorted): overlap, per-group quality, high-quality bars, downstream.

usage: cc_compare.py RUN_A RUN_B OUT_JSON [--kang TRUTH.tsv.gz] [--celltypist MODEL.pkl]
RUN_* are phase1 run directories (out/ inside, mtx output format). A = stream, B = sorted.
Groups: both (called by A and B), a_only, b_only. A barcode's metrics come from the run that called it (B for "both").
"""
import argparse, gzip, json, os, re
import numpy as np, pandas as pd, scipy.io, scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("run_a"); ap.add_argument("run_b"); ap.add_argument("out")
ap.add_argument("--kang"); ap.add_argument("--celltypist")
args = ap.parse_args()


def bare(idx):
    return pd.Index(pd.Series(idx, dtype=str).str.replace("-1$", "", regex=True))


def load(run):
    o = os.path.join(run, "out")
    cc = pd.read_csv(f"{o}/cell_calls.tsv", sep="\t", dtype={"barcode": str}).set_index("barcode")
    qc = pd.read_csv(f"{o}/cell_qc_metrics.tsv", sep="\t", dtype={"barcode": str}).set_index("barcode")
    cc.index, qc.index = bare(cc.index), bare(qc.index)
    # gene x barcode = exon + intron molecules (exon feature = GENEID_NAME_coords)
    mats, genes = [], None
    for kind in ("exon", "intron"):
        p = f"{o}/{kind}_counts.mtx.gz"
        if not os.path.exists(p):
            continue
        m = scipy.io.mmread(p).tocsr()
        f = pd.read_csv(f"{o}/{kind}_counts_features.tsv.gz", sep="\t", header=None)[0].astype(str)
        g = f.str.split("_").str[0].values
        bcs = pd.read_csv(f"{o}/{kind}_counts_barcodes.tsv.gz", sep="\t", header=None)[0].astype(str).values
        codes, uniq = pd.factorize(g)
        agg = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
        mats.append(pd.Series(range(len(uniq)), index=uniq)); mats.append((agg @ m).tocsc()); mats.append(bcs)
    # align exon and intron gene axes
    gidx = mats[0].index.union(mats[3].index) if len(mats) > 3 else mats[0].index
    total = None
    for i in range(0, len(mats), 3):
        pos = gidx.get_indexer(mats[i].index)
        m = mats[i + 1]
        r = sp.csr_matrix((np.ones(len(pos)), (pos, np.arange(len(pos)))), shape=(len(gidx), len(pos))) @ m
        total = r if total is None else total + r
    bcs = mats[2]
    amb = pd.read_csv(f"{o}/ambient_profile.tsv", sep="\t")
    amb["gene"] = amb["feature"].astype(str).str.split("_").str[0]
    amb = amb.groupby("gene")["ambient_fraction"].sum().reindex(gidx).fillna(0).values
    donors = None
    if os.path.exists(f"{o}/donor_assignments.tsv"):
        donors = pd.read_csv(f"{o}/donor_assignments.tsv", sep="\t", dtype={"cell": str}).set_index("cell")
        donors.index = bare(donors.index)
    log = open(os.path.join(run, "run.log")).read()
    knee = re.search(r"knee umi_min=(\d+)", log)
    return dict(cc=cc, qc=qc, X=total.tocsc(), genes=gidx, bcs=bare(bcs), amb=amb, donors=donors,
                knee=int(knee.group(1)) if knee else None)


A, B = load(args.run_a), load(args.run_b)
def called(R):
    # the final call is called_v2 (knee + retained); is_cell / called are the knee-only subset
    col = "called_v2" if "called_v2" in R["cc"] else "is_cell"
    return set(R["cc"].index[R["cc"][col].astype(str).str.upper().isin(["TRUE", "1"])])
ca, cb = called(A), called(B)
groups = {"both": sorted(ca & cb), "a_only": sorted(ca - cb), "b_only": sorted(cb - ca)}
src = {"both": B, "a_only": A, "b_only": B}


def cos_ambient(R, bcs):
    idx = R["bcs"].get_indexer(bcs)
    idx = idx[idx >= 0]
    X = R["X"][:, idx].toarray().astype(float)
    a = R["amb"] / max(R["amb"].sum(), 1e-12)
    num = a @ X
    den = np.linalg.norm(X, axis=0) * np.linalg.norm(a)
    return num / np.where(den > 0, den, 1)


def q(v, ps=(5, 25, 50, 75, 95)):
    v = pd.to_numeric(pd.Series(v), errors="coerce").dropna()
    if not len(v):
        return None
    return {f"p{p}": round(float(np.percentile(v, p)), 4) for p in ps} | {"n": int(len(v)), "mean": round(float(v.mean()), 4)}


res = {"run_a": args.run_a, "run_b": args.run_b, "n_called": {"a": len(ca), "b": len(cb)},
       "groups": {k: len(v) for k, v in groups.items()}, "knee_umi_min": {"a": A["knee"], "b": B["knee"]}}

for R, key in ((A, "a"), (B, "b")):
    qc = R["qc"]
    cells = qc.index.intersection(list(called(R)))
    res.setdefault("frac_reads_in_cells", {})[key] = round(float(qc.loc[cells, "n_reads"].sum() / max(qc["n_reads"].sum(), 1)), 5)

# empty-droplet reference: uncalled barcodes with 50-100 UMIs in B
emp = B["cc"].index[(~B["cc"].index.isin(list(cb))) & B["cc"]["umi_total"].between(50, 100)][:2000]
res["ambient_cosine_empty_reference"] = q(cos_ambient(B, list(emp)))

per = {}
for g, bcs in groups.items():
    R = src[g]
    qc = R["qc"].reindex(bcs); cc = R["cc"].reindex(bcs)
    d = {"umi": q(qc["n_umi"]), "genes": q(qc["n_genes"]), "pct_mt": q(qc["pct_mt"]),
         "doublet_score": q(qc["doublet_score"]), "doublet_frac": round(float(pd.to_numeric(qc["is_doublet"], errors="coerce").fillna(0).mean()), 4) if len(bcs) else None,
         "fdr": q(cc["fdr"]), "fdr_na_frac": round(float(cc["fdr"].isna().mean()), 4) if len(bcs) else None,
         "call_method": cc["call_method"].value_counts().to_dict() if "call_method" in cc else None,
         "ambient_cosine": q(cos_ambient(R, bcs)) if len(bcs) else None}
    other = B if R is A else A    # the same barcode in the run that did not call it
    if g != "both":
        oq = other["qc"].reindex(bcs)
        d["umi_in_other_run"] = q(oq["n_umi"])
    if R["donors"] is not None and len(bcs):
        dn = R["donors"].reindex(bcs)
        d["singlet_demux_prob_max"] = q(dn["prob_max"])
        d["singlet_demux_status"] = dn["cell_status"].value_counts().to_dict() if "cell_status" in dn else None
    per[g] = d
res["per_group"] = per

# high-quality bars: is every barcode above the bar called by both runs?
both_umi = B["qc"].reindex(groups["both"])["n_umi"]
bars = {"knee_umi_min_b": B["knee"], "umi_1000": 1000, "p10_of_joint_cells": int(np.percentile(both_umi, 10))}
hq = {}
for name, bar in bars.items():
    if bar is None:
        continue
    above = set(A["qc"].index[A["qc"]["n_umi"] >= bar]) | set(B["qc"].index[B["qc"]["n_umi"] >= bar])
    hq[name] = {"bar_umi": int(bar), "n_above_in_either": len(above), "called_by_both": len(above & ca & cb),
                "called_a_only": len(above & (ca - cb)), "called_b_only": len(above & (cb - ca)), "called_by_neither": len(above - ca - cb)}
res["high_quality_bars"] = hq

# labels: Kang truth (demuxlet donor, multiplet status, published cell type) or celltypist
labels = None
if args.kang:
    t = pd.read_csv(args.kang, sep="\t")
    t.index = t.index.astype(str).str.replace("-1$", "", regex=True)
    t = t[~t.index.duplicated()]
    labels = t["cell"].astype(str)
    kt = {}
    for g, bcs in groups.items():
        tb = t.reindex(bcs)
        kt[g] = {"in_truth": int(tb["ind"].notna().sum()), "truth_singlet": int((tb["multiplets"] == "singlet").sum()),
                 "truth_doublet": int((tb["multiplets"] == "doublet").sum()), "not_in_truth": int(tb["ind"].isna().sum())}
    res["kang_truth"] = kt
elif args.celltypist and os.path.exists(args.celltypist):
    import anndata as ad, celltypist
    union = sorted(ca | cb)
    Ra = [b for b in union if b in cb]; Rb = [b for b in union if b not in cb]
    cols = []
    for R, bcs in ((B, Ra), (A, Rb)):
        cols.append(R["X"][:, R["bcs"].get_indexer(bcs)])
    X = sp.hstack(cols).T.tocsr().astype(np.float32)
    gs = B["genes"]
    names = pd.read_csv(os.path.join(args.run_b, "out", "exon_counts_features.tsv.gz"), sep="\t", header=None)[0].astype(str)
    sym = dict(zip(names.str.split("_").str[0], names.str.split("_").str[1]))
    adata = ad.AnnData(X=X, obs=pd.DataFrame(index=Ra + Rb), var=pd.DataFrame(index=[sym.get(x, x) for x in gs]))
    adata.var_names_make_unique()
    import scanpy as sc
    sc.pp.normalize_total(adata, target_sum=1e4); sc.pp.log1p(adata)
    pr = celltypist.annotate(adata, model=args.celltypist)
    labels = pr.predicted_labels["predicted_labels"].astype(str)
    conf = pr.probability_matrix.max(axis=1)
    res["celltype_confidence"] = {g: q(conf.reindex(bcs)) for g, bcs in groups.items()}

if labels is not None:
    res["celltype_by_group"] = {g: labels.reindex(bcs).fillna("unlabelled").value_counts().head(12).to_dict() for g, bcs in groups.items()}
    # downstream on high-quality cells (>= knee of the sorted run, called by that run): per-type counts and pseudobulk
    bar = B["knee"] or 1000
    down = {}
    for R, key, cset in ((A, "a", ca), (B, "b", cb)):
        hqc = [b for b in cset if R["qc"]["n_umi"].get(b, 0) >= bar]
        lab = labels.reindex(hqc).fillna("unlabelled")
        down[key] = {"n_hq": len(hqc), "per_type": lab.value_counts().to_dict(), "_lab": lab, "_R": R}
    pb = {}
    for t in sorted(set(down["a"]["per_type"]) & set(down["b"]["per_type"])):
        vs = []
        for key in ("a", "b"):
            R = down[key]["_R"]; lab = down[key]["_lab"]
            bcs = list(lab.index[lab == t])
            v = np.asarray(R["X"][:, R["bcs"].get_indexer(bcs)].sum(axis=1)).ravel()
            vs.append(pd.Series(v, index=R["genes"]))
        a, b = vs[0].align(vs[1], fill_value=0)
        la, lb = np.log1p(a / max(a.sum(), 1) * 1e6), np.log1p(b / max(b.sum(), 1) * 1e6)
        pb[t] = {"n_a": int((down["a"]["_lab"] == t).sum()), "n_b": int((down["b"]["_lab"] == t).sum()),
                 "pearson_log_cpm": round(float(np.corrcoef(la, lb)[0, 1]), 6)}
    res["downstream_hq"] = {"bar_umi": int(bar), "n_hq": {k: down[k]["n_hq"] for k in down},
                            "per_type": {k: down[k]["per_type"] for k in down}, "pseudobulk": pb}

json.dump(res, open(args.out, "w"), indent=1, default=str)
print(json.dumps({k: res[k] for k in ("n_called", "groups", "high_quality_bars")}, indent=1))
