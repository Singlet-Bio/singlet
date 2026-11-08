"""Score Singlet, STARsolo and alevin-fry (USA) against REAL ground truth. No simulated reads anywhere.

Truths (each is a physical fact about the sample, not the output of one of the aligners being scored):
  barnyard   hgmm1k_v3   HEK293T + NIH3T3 mixed 1:1. A cell is human or mouse; a UMI of the other species inside a pure
                         cell is a mapping error or ambient RNA. Ambient RNA is the same for every tool on the same
                         reads, so differences between tools are differences in cross-species assignment.
  chrY       hgmm1k_v3   HEK293T is female-derived (no Y chromosome): every UMI on a chrY gene (outside the PARs) in a
                         human cell is a false positive, mostly X/Y gametologs (ZFX/ZFY, DDX3X/DDX3Y, KDM5C/KDM5D ...).
  cite       pbmc1k_cite_v3  CITE-seq: 13 protein markers measured on the same cells by antibody-derived tags.
                         RNA of the matching gene should rank cells like the protein does (orthogonal assay).
Cells scored: the barcodes of the published Cell Ranger filtered matrix (same set for every tool).

  python score.py --runs RUNS_DIR --data DATA_DIR --gtf GENES.GTF --out OUT_DIR [--layout evp|legacy]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import loaders as L  # noqa: E402

RNG = np.random.default_rng(20261025)
# Protein -> gene map and pair rules are those of the published gex-counts metric `adt_rna_spearman` (bench/public/capabilities/gex_counts.py
# at origin/bench/public-counts 772eb0cd): RNA = log1p(UMI per 10k, the tool's own total), protein = CLR over all antibodies, mean Spearman
# across cells over the pairs every tool can score; no isoform-ambiguous pairs (CD45RA / CD45RO vs total PTPRC are not paired).
C_ADT_GENE = {"cd3": "CD3E", "cd4": "CD4", "cd8a": "CD8A", "cd14": "CD14", "cd16": "FCGR3A", "cd56": "NCAM1", "cd19": "CD19",
              "cd25": "IL2RA", "pd-1": "PDCD1", "cd127": "IL7R", "tigit": "TIGIT", "cd15": "FUT4"}


def tool_set(runs, ds, layout):
    """name -> zero-argument loader. Names are the public tool names used in the contract."""
    b = os.path.join(runs, ds)
    if layout == "evp":
        sdir, odir, adir = (os.path.join(b, "singlet"), os.path.join(b, "starsolo"), os.path.join(b, "alevinfry"))
    else:  # pipeline-benchmark layout
        sdir, odir, adir = (os.path.join(b, "singlet-campaign-t8"), os.path.join(b, "starsolo"), os.path.join(b, "alevinfry"))
    t = {
        "singlet": lambda: L.load_singlet(sdir),
        "STARsolo": lambda: L.load_starsolo(odir, "GeneFull"),
        "alevin-fry": lambda: L.load_alevinfry(adir, "af_quant", "usa"),
    }
    if os.path.isdir(os.path.join(adir, "af", "af_quant_em", "alevin")):
        t["alevin-fry-em"] = lambda: L.load_alevinfry(adir, "af_quant_em", "usa")
    return t


def cr_path(data, ds):
    for f in os.listdir(os.path.join(data, ds)):
        if f.endswith(".h5") and "filtered" in f:
            return os.path.join(data, ds, f)
    raise FileNotFoundError(ds)


def align(m: L.Mat, bcs):
    """Rows of m for the barcodes in `bcs` (zero rows for barcodes the tool never saw), columns unchanged."""
    idx = {b: i for i, b in enumerate(m.barcodes)}
    present = np.array([b in idx for b in bcs])
    rows = [idx.get(b, 0) for b in bcs]
    X = m.X[rows]
    X = sp.diags(present.astype(float)) @ X
    return sp.csr_matrix(X), present


def boot_ratio(num, den, n=2000):
    """Bootstrap 95% CI over cells of sum(num)/sum(den)."""
    k = len(num)
    idx = RNG.integers(0, k, size=(n, k))
    r = num[idx].sum(1) / np.maximum(den[idx].sum(1), 1)
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def barnyard(ds, tools, cr, chrom, out):
    """Cross-species UMI fraction inside pure cells, per tool."""
    human_cr = np.array([g.startswith("ENSG") for g in cr.genes])
    tot = np.asarray(cr.X.sum(1)).ravel()
    fh = np.asarray(cr.X[:, human_cr].sum(1)).ravel() / np.maximum(tot, 1)
    ok = tot >= 500
    pure_h, pure_m = ok & (fh >= 0.9), ok & (fh <= 0.1)
    bcs = cr.barcodes
    res = {"n_cells_cr": int(len(bcs)), "n_pure_human": int(pure_h.sum()), "n_pure_mouse": int(pure_m.sum()),
           "n_mixed": int((ok & ~pure_h & ~pure_m).sum()), "tools": {}}
    rows = {"barcode": bcs, "cr_frac_human": fh, "cr_total": tot}
    top = {}
    all_tools = dict(tools); all_tools["cellranger-published"] = lambda: cr
    for name, load in all_tools.items():
        m = load()
        X, present = align(m, bcs)
        ish = np.array([g.startswith("ENSG") for g in m.genes])
        h = np.asarray(X[:, ish].sum(1)).ravel(); mo = np.asarray(X[:, ~ish].sum(1)).ravel()
        tt = h + mo
        wrong = np.where(pure_h, mo, np.where(pure_m, h, 0.0))
        sel = pure_h | pure_m
        rows[name + "_total"] = tt; rows[name + "_wrong"] = wrong
        frac = float(wrong[sel].sum() / max(tt[sel].sum(), 1))
        res["tools"][name] = {
            "cells_present": int(present.sum()),
            "umi_total_pure": float(tt[sel].sum()),
            "wrong_species_umi": float(wrong[sel].sum()),
            "wrong_species_fraction": frac,
            "ci": boot_ratio(wrong[sel], tt[sel]),
            "median_cell_wrong_fraction": float(np.median(wrong[sel] / np.maximum(tt[sel], 1))),
            "median_umi_per_cell": float(np.median(tt[sel])),
            "frac_cells_ge_1pct_wrong": float(np.mean((wrong[sel] / np.maximum(tt[sel], 1)) >= 0.01)),
        }
        # which genes carry the wrong-species UMIs
        Xh = X[pure_m][:, ish].sum(0).A1 if pure_m.any() else np.zeros(ish.sum())
        Xm = X[pure_h][:, ~ish].sum(0).A1 if pure_h.any() else np.zeros((~ish).sum())
        gh, gm = m.genes[ish], m.genes[~ish]
        order = np.argsort(-Xh)[:15]; orderm = np.argsort(-Xm)[:15]
        top[name] = {"human_genes_in_mouse_cells": [(chrom.get(gh[i], ("", gh[i], ""))[1] or gh[i], float(Xh[i])) for i in order],
                     "mouse_genes_in_human_cells": [(chrom.get(gm[i], ("", gm[i], ""))[1] or gm[i], float(Xm[i])) for i in orderm]}
        # reads of the right species per cell vs CR (agreement)
        res["tools"][name]["pearson_total_vs_cr_pure"] = float(np.corrcoef(tt[sel], tot[sel])[0, 1])
    res["top_wrong_genes"] = top
    write_tsv(os.path.join(out, f"{ds}_barnyard_cells.tsv"), rows)
    return res


def chry(ds, tools, cr, chrom, out, cell_mask):
    """UMIs on non-PAR chrY genes in cells that cannot carry a Y chromosome."""
    ycols = {g for g, (c, n, t) in chrom.items() if c.endswith("chrY") and g.startswith("ENSG") and "_PAR_Y" not in g}
    res = {"n_chrY_genes": len(ycols), "n_cells": int(cell_mask.sum()), "tools": {}}
    all_tools = dict(tools); all_tools["cellranger-published"] = lambda: cr
    per_gene = {}
    for name, load in all_tools.items():
        m = load(); X, present = align(m, cr.barcodes)
        X = X[cell_mask]
        isy = np.array([g in ycols for g in m.genes])
        isH = np.array([g.startswith("ENSG") for g in m.genes])
        y = np.asarray(X[:, isy].sum(1)).ravel(); tot = np.asarray(X[:, isH].sum(1)).ravel()
        res["tools"][name] = {"chrY_umi": float(y.sum()), "human_umi": float(tot.sum()),
                              "chrY_per_million": float(1e6 * y.sum() / max(tot.sum(), 1)),
                              "ci_per_million": [1e6 * v for v in boot_ratio(y, tot)],
                              "cells_with_chrY": int((y > 0).sum()), "frac_cells_with_chrY": float((y > 0).mean())}
        gs = X[:, isy].sum(0).A1
        per_gene[name] = {chrom[g][1] or g: float(v) for g, v in zip(m.genes[isy], gs) if v > 0}
    res["per_gene_umi"] = per_gene
    return res


def clr(a):
    a = np.log1p(a)
    return a - a.mean(1, keepdims=True)


def spearman(x, y):
    if np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def cite(ds, tools, cr, cr_all, out):
    """Spearman between RNA (log1p CP10k of the tool's counts) and protein (CLR) across cells, per marker pair."""
    ab = L.load_cr_h5(cr_all, "Antibody Capture")
    P = ab.X.toarray()
    pnames = list(ab.genes)
    Pclr = clr(P)
    res = {"n_cells": int(len(cr.barcodes)), "pairs": [], "tools": {}}
    all_tools = dict(tools); all_tools["cellranger-published"] = lambda: cr
    chosen = []
    for pj, prot in enumerate(pnames):
        g = C_ADT_GENE.get(prot.split("_")[0].lower())
        if g:
            chosen.append((prot, g, pj))
    per_tool = {}
    for name, load in all_tools.items():
        m = load(); X, present = align(m, cr.barcodes)
        lib = np.asarray(X.sum(1)).ravel(); lib[lib == 0] = 1
        nm = {}
        gi = {g: i for i, g in enumerate(m.names)} if m.names is not None else {}
        # symbol lookup via the CR h5 (all tools share Ensembl ids)
        sym2id = {n: i for n, i in zip(cr.names, cr.genes)}
        id2col = {g: i for i, g in enumerate(m.genes)}
        vals = {}
        for prot, gene, pj in chosen:
            gid = sym2id.get(gene)
            if gid is None or gid not in id2col:
                continue
            r = np.log1p(1e4 * X[:, id2col[gid]].toarray().ravel() / lib)
            vals[(prot, gene)] = (r, Pclr[:, pj], X[:, id2col[gid]].toarray().ravel())
        per_tool[name] = vals
    # use only pairs expressed (>=3% cells with UMI) in the Cell Ranger reference and present in every tool
    keys = [k for k in per_tool["cellranger-published"] if all(k in per_tool[t] for t in per_tool)]
    # keep only pairs that every tool can score (a tool with zero counts for a gene has no correlation)
    keys = [k for k in keys if all(not np.isnan(spearman(per_tool[t][k][0], per_tool[t][k][1])) for t in per_tool)]
    res["pairs"] = [list(k) for k in keys]
    n = len(cr.barcodes)
    for name in per_tool:
        rho = {f"{k[0]}~{k[1]}": spearman(per_tool[name][k][0], per_tool[name][k][1]) for k in keys}
        meanrho = float(np.nanmean(list(rho.values())))
        bs = []
        for _ in range(300):
            ix = RNG.integers(0, n, n)
            bs.append(np.nanmean([spearman(per_tool[name][k][0][ix], per_tool[name][k][1][ix]) for k in keys]))
        res["tools"][name] = {"mean_spearman": meanrho, "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "per_pair": rho,
                              "pair_umi_total": {f"{k[0]}~{k[1]}": float(per_tool[name][k][2].sum()) for k in keys}}
    # paired bootstrap of the difference to each tool (same resampled cells)
    names = list(per_tool)
    diffs = {a: {} for a in names}
    for a in names:
        for b in names:
            if a == b:
                continue
            d = []
            for _ in range(300):
                ix = RNG.integers(0, n, n)
                d.append(np.nanmean([spearman(per_tool[a][k][0][ix], per_tool[a][k][1][ix]) - spearman(per_tool[b][k][0][ix], per_tool[b][k][1][ix]) for k in keys]))
            diffs[a][b] = [float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]
    res["paired_diff_mean_ci"] = diffs
    return res


def write_tsv(path, cols):
    keys = list(cols)
    with open(path, "w") as f:
        f.write("\t".join(keys) + "\n")
        for i in range(len(cols[keys[0]])):
            f.write("\t".join(str(cols[k][i]) if not isinstance(cols[k][i], float) else f"{cols[k][i]:.6g}" for k in keys) + "\n")


def _parse_time(p):
    wall = cpu = rss = 0.0
    for ln in open(p):
        ln = ln.strip()
        if ln.startswith("Elapsed (wall clock)"):
            t = [float(x) for x in ln.split("): ")[1].split(":")]
            wall = t[-1] + 60 * t[-2] + (3600 * t[-3] if len(t) == 3 else 0)
        elif ln.startswith("User time") or ln.startswith("System time"):
            cpu += float(ln.split(": ")[1])
        elif ln.startswith("Maximum resident"):
            rss = float(ln.split(": ")[1]) / 1048576
    return wall, cpu / 3600, rss


def timings(runs, ds):
    """Per tool: wall_s, cpu_h, max_rss_gb. Singlet is reported with and without its .1fq encode step (encode_* keys)."""
    out = {}
    for tool in ("alevinfry", "starsolo", "singlet"):
        p = os.path.join(runs, ds, tool, "time.txt")
        if not os.path.exists(p):
            continue
        w, c, r = _parse_time(p)
        d = {"wall_s": w, "cpu_h": c, "max_rss_gb": r}
        if tool == "singlet":
            pe = os.path.join(runs, ds, tool, "encode.time.txt")
            if os.path.exists(pe):
                we, ce, re_ = _parse_time(pe)
                d.update({"encode_wall_s": we, "encode_cpu_h": ce, "encode_rss_gb": re_,
                          "wall_s_with_encode": w + we, "cpu_h_with_encode": c + ce, "max_rss_gb": max(r, re_)})
        if tool == "alevinfry":
            pq = os.path.join(runs, ds, tool, "time_quant_em.txt")
            if os.path.exists(pq):
                wq, cq, _ = _parse_time(pq)
                d.update({"em_quant_wall_s": wq, "em_quant_cpu_h": cq})
        out[tool] = d
    return out


def read_accounting(runs, ds):
    """Where each tool puts the reads: STAR genome alignment and feature classes vs piscem's mapping to the splici index."""
    b = os.path.join(runs, ds); out = {}
    fin = os.path.join(b, "starsolo", "Log.final.out")
    if os.path.exists(fin):
        d = {}
        for ln in open(fin):
            if "|" in ln:
                k, v = ln.split("|", 1); d[k.strip()] = v.strip()
        n = int(d["Number of input reads"])
        out["star"] = {"input_reads": n,
                       "uniquely_mapped_fraction": int(d["Uniquely mapped reads number"]) / n,
                       "multi_mapped_fraction": int(d["Number of reads mapped to multiple loci"]) / n,
                       "too_many_loci_fraction": int(d["Number of reads mapped to too many loci"]) / n,
                       "unmapped_fraction": 1 - (int(d["Uniquely mapped reads number"]) + int(d["Number of reads mapped to multiple loci"]) + int(d["Number of reads mapped to too many loci"])) / n}
    fs = os.path.join(b, "starsolo", "Solo.out", "GeneFull", "Features.stats")
    if os.path.exists(fs):
        d = {ln.split()[0]: int(ln.split()[1]) for ln in open(fs) if len(ln.split()) == 2}
        tot = d.get("yesWLmatch", 0) + d.get("noNoFeature", 0)  # reads whose barcode matched and that aligned
        out["star_genefull_features"] = {k: v for k, v in d.items()}
    mi = os.path.join(b, "alevinfry", "af", "af_map", "map_info.json")
    if os.path.exists(mi):
        j = json.load(open(mi))
        out["piscem"] = {"num_reads": j["num_reads"], "num_mapped": j["num_mapped"], "mapped_fraction": j["num_mapped"] / j["num_reads"]}
    qc = os.path.join(b, "singlet", "cell_qc_metrics.tsv")
    if os.path.exists(qc):
        import csv
        rows = list(csv.DictReader(open(qc), delimiter="	"))
        cols = [c for c in rows[0] if any(w in c for w in ("exonic", "intronic", "antisense", "mito", "intergenic", "multimapped", "unmapped"))] if rows else []
        out["singlet_cell_qc_sums"] = {c: sum(float(r[c]) for r in rows if r[c] not in ("", "NA")) for c in cols}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--gtf", required=True); ap.add_argument("--gtf-mouse", default=None)
    ap.add_argument("--out", required=True); ap.add_argument("--layout", default="evp")
    ap.add_argument("--datasets", default="hgmm1k_v3,pbmc1k_cite_v3,pbmc1k_v3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    chrom = L.gene_chrom(a.gtf)
    result = {}
    for ds in a.datasets.split(","):
        h5 = cr_path(a.data, ds)
        cr = L.load_cr_h5(h5)
        tools = tool_set(a.runs, ds, a.layout)
        r = {"timing": timings(a.runs, ds), "timing_t4": timings(a.runs, ds + "_t4") if os.path.isdir(os.path.join(a.runs, ds + "_t4")) else {}, "read_accounting": read_accounting(a.runs, ds)}
        if ds.startswith("hgmm"):
            r["barnyard"] = barnyard(ds, tools, cr, chrom, a.out)
            tot = np.asarray(cr.X.sum(1)).ravel()
            fh = np.asarray(cr.X[:, [g.startswith("ENSG") for g in cr.genes]].sum(1)).ravel() / np.maximum(tot, 1)
            r["chrY"] = chry(ds, tools, cr, chrom, a.out, (tot >= 500) & (fh >= 0.9))
        # chrY is only a truth in the barnyard (HEK293T, female-derived); the PBMC donors are male (chrY in >90% of cells)
        if "cite" in ds:
            r["cite"] = cite(ds, tools, cr, h5, a.out)
        result[ds] = r
        print(ds, "done", flush=True)
    with open(os.path.join(a.out, "scores.json"), "w") as f:
        json.dump(result, f, indent=1)


if __name__ == "__main__":
    main()
