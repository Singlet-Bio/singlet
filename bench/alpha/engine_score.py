"""Score the published atlas engine at campaign settings (runs/<ds>/singlet-alpha30m) against the pipeline benchmark's
truth sets, reusing its scorers (bench/pipeline/analysis/{loaders,compare}.py from fix/molecule-labels).

    python -P engine_score.py PB ANALYSIS_DIR DATASET OUT_JSON [--barnyard] [--cite]
      env EDROP = checkout with singlet._pz built (reads the campaign's .1pz output)

Truth / references (all from the same FASTQs at full depth, same reference):
  * STARsolo 2.7.11b, Cell Ranger 4 emulation, EmptyDrops_CR (`starsolo`), Gene / GeneFull / Velocyto
  * the Cell Ranger matrix 10x published with the dataset (`cellranger_published`)
  * barnyard (hgmm1k_v3): species of each cell from its human / mouse UMI share
  * CITE-seq (pbmc1k_cite_v3): clusters from CLR-normalised ADT counts of the published Cell Ranger h5
Also scored for context: the atlas engine at full depth (`singlet-baseline-t4`, b1335a8) and the fixed engine.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

PB, ANA, DS, OUTJ = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
sys.path.insert(0, ANA)
import compare as C  # noqa: E402
import loaders  # noqa: E402

_spec = importlib.util.spec_from_file_location("edfetch", os.path.join(os.environ["EDROP"], "bench/empty_droplets/fetch.py"))
F = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(F)
runs = PB / "runs" / DS
R: dict = {"dataset": DS, "engine": "alpha (published atlas)", "engine_run": "singlet-alpha30m",
           "dataset_scope": "public 10x demo dataset re-run with the campaign binary and settings (not an atlas sample)"}


def load_1pz_run(out: Path, gene_ids: list[str]) -> ad.AnnData | None:
    if not (out / "exon_counts.1pz").exists():
        return None
    pz = F.load_pz()
    gene_of = loaders._feature_gene_resolver(gene_ids)
    parts = {}
    for k in ("exon", "intron"):
        p = out / f"{k}_counts.1pz"
        if p.exists():
            r = pz.read_1pz(str(p))
            m = sp.csc_matrix((r["data"], r["indices"], r["indptr"]), shape=(r["m"], r["n"]))
            parts[k] = (list(r["rownames"]), m, list(r["colnames"]))
    allg = sorted(set(gene_ids) | {gene_of(f) for (rn, _, _) in parts.values() for f in rn})
    gidx = {g: i for i, g in enumerate(allg)}
    bcs = parts["exon"][2]
    full = {}
    for k, (rn, m, cn) in parts.items():
        assert cn == bcs
        rows = np.array([gidx[gene_of(f)] for f in rn])
        agg = sp.csr_matrix((np.ones(len(rows)), (rows, np.arange(len(rows)))), shape=(len(allg), len(rows)))
        full[k] = sp.csr_matrix(agg @ m)
    exon, intron = full["exon"], full.get("intron", sp.csr_matrix(full["exon"].shape))
    a = ad.AnnData(X=sp.csr_matrix((exon + intron).T), obs=pd.DataFrame(index=[loaders._bare(b) for b in bcs]),
                   var=pd.DataFrame(index=[loaders._gid(g) for g in allg]))
    a.layers["exon"] = sp.csr_matrix(exon.T)
    a.layers["spliced"] = sp.csr_matrix(exon.T)
    a.layers["unspliced"] = sp.csr_matrix(intron.T)
    a.var_names_make_unique()
    a.obs["is_cell"] = False
    cc = out / "cell_calls.tsv"
    if cc.exists():
        _, cset, methods = F.parse_cell_calls(cc.read_text())
        a.obs["is_cell"] = a.obs_names.isin({loaders._bare(b) for b in cset})
        R["alpha_call_method"] = methods
    if (out / "pileup_stats.json").exists():
        a.uns["pileup_stats"] = json.loads((out / "pileup_stats.json").read_text())
    return a


solo_feats = runs / "starsolo" / "Solo.out" / "Gene" / "raw" / "features.tsv"
gene_ids = pd.read_csv(solo_feats, sep="\t", header=None)[0].astype(str).tolist()
tools = {}
a30 = load_1pz_run(runs / "singlet-alpha30m", gene_ids)
if a30 is None:
    R["error"] = "no singlet-alpha30m output"
    json.dump(R, open(OUTJ, "w"), indent=1)
    sys.exit(0)
tools["alpha30m"] = a30
for t in ("singlet-baseline-t4", "singlet-fixed"):
    x = loaders.load_singlet(runs / t)
    if x is not None:
        tools[t] = x
tools.update(loaders.load_starsolo(runs / "starsolo"))
crh5 = next(iter((PB / "data" / DS).glob("cellranger_*.h5")), None)
adt = None
if crh5 is not None:
    cr, adt = loaders.load_cellranger_h5(crh5)
    if cr is not None:
        tools["cellranger_published"] = cr
R["tools"] = sorted(tools)
ps = a30.uns.get("pileup_stats", {})
ws, eh = ps.get("wrong_strand"), ps.get("exon_hits")
R["alpha_wrong_strand_frac"] = (ws / (ws + eh)) if (ws is not None and eh) else None
R["alpha_reverse_strand"] = ps.get("reverse_strand")
for line in open(runs / "singlet-baseline-t4" / "run.log", errors="replace"):
    if ".1fq: " in line and " reads" in line:
        R["reads_available"] = int(line.split(".1fq: ")[1].split()[0])
        break
R["reads_used"] = 30000000
R["reads_used_over_available"] = R["reads_used"] / R["reads_available"] if R.get("reads_available") else None
R["resources"] = {t: loaders.load_time(runs / t / "time.txt") for t in ("singlet-alpha30m", "singlet-baseline-t4", "starsolo")}
R["resources"]["singlet-alpha30m"]["encode"] = loaders.load_time(runs / "singlet-alpha30m" / "encode.time.txt")

# ---- cell calls vs truth sets
called = {t: set(a.obs_names[a.obs["is_cell"]]) for t, a in tools.items() if "is_cell" in a.obs and a.obs["is_cell"].any()}
R["cell_calls"] = {t: len(s) for t, s in called.items()}
calls = {}
for t in ("alpha30m", "singlet-baseline-t4", "singlet-fixed"):
    for ref in ("cellranger_published", "starsolo_GeneFull"):
        if t in called and ref in called:
            s, r = called[t], called[ref]
            calls[f"{t}|{ref}"] = {"recovery": len(s & r) / max(1, len(r)), "precision": len(s & r) / max(1, len(s)),
                                   "jaccard": len(s & r) / max(1, len(s | r)), "n_tool": len(s), "n_ref": len(r)}
R["calls"] = calls

# ---- counts on cells called by alpha30m and STARsolo
ref_cells = called.get("starsolo_GeneFull", set()) & called["alpha30m"]
for t, a in tools.items():
    ref_cells &= set(a.obs_names)
shared = sorted(ref_cells)
R["shared_cells"] = len(shared)
conc = {}
pairs = [("alpha30m", None, "starsolo_GeneFull", None, "exon+intron vs GeneFull"),
         ("alpha30m", "exon", "starsolo_Gene", None, "exon vs Gene"),
         ("alpha30m", "exon", "cellranger_published", None, "exon vs published Cell Ranger"),
         ("alpha30m", "unspliced", "starsolo_Velocyto", "unspliced", "intron vs velocyto unspliced"),
         ("singlet-baseline-t4", None, "starsolo_GeneFull", None, "atlas engine full depth, exon+intron vs GeneFull"),
         ("singlet-baseline-t4", "exon", "starsolo_Gene", None, "atlas engine full depth, exon vs Gene")]
for ta, la, tb, lb, what in pairs:
    if ta in tools and tb in tools and shared:
        try:
            conc[f"{ta}[{la or 'X'}] vs {tb}[{lb or 'X'}]"] = {"what": what, **C.gene_concordance(tools[ta], tools[tb], shared, la, lb)}
        except Exception as e:  # noqa: BLE001
            conc[f"{ta} vs {tb}"] = {"error": repr(e)}
R["concordance"] = conc


# ---- depth-matched per-cell correlation: STARsolo Gene thinned to alpha30m's exon total per cell
def depth_matched(a, la, b, lb, cells, n=1500, seed=0):
    rng = np.random.default_rng(seed)
    cells = list(cells)
    if len(cells) > n:
        cells = list(rng.choice(cells, n, replace=False))
    genes = sorted(set(a.var_names) & set(b.var_names))
    A = C.subset(a, cells, genes)[:, genes]
    B = C.subset(b, cells, genes)[:, genes]
    X = (A.X if la is None else A.layers[la]).toarray().astype(np.int64)
    Y = (B.X if lb is None else B.layers[lb]).toarray().astype(np.int64)
    tgt = np.minimum(X.sum(1), Y.sum(1))
    th = lambda M: rng.binomial(M, np.clip(tgt / np.maximum(M.sum(1), 1), 0, 1)[:, None])  # noqa: E731
    Xt, Yt, Yt2 = th(X), th(Y), th(Y)

    def med(P, Q):
        r = [np.corrcoef(np.log1p(p), np.log1p(q))[0, 1] for p, q in zip(P, Q) if p.std() > 0 and q.std() > 0]
        return float(np.median(r)) if r else None
    return {"cell_corr_depth_matched": med(Xt, Yt), "noise_floor": med(Yt2, Yt), "cell_corr_raw": med(X, Y),
            "n_cells": len(cells), "median_depth": float(np.median(tgt))}


dm = {}
for ta, la, tb, lb in (("alpha30m", "exon", "starsolo_Gene", None), ("alpha30m", None, "starsolo_GeneFull", None),
                       ("alpha30m", "exon", "cellranger_published", None)):
    if ta in tools and tb in tools and len(shared) >= 50:
        dm[f"{ta}[{la or 'X'}] vs {tb}"] = depth_matched(tools[ta], la, tools[tb], lb, shared)
R["depth_matched"] = dm

# ---- barnyard
if "--barnyard" in sys.argv:
    R["barnyard"] = {}
    for t, a in tools.items():
        cs = sorted(called[t]) if t in called else shared
        if cs and t != "starsolo_Velocyto":
            R["barnyard"][t] = C.barnyard(a, cs)[0]

# ---- CITE-seq: RNA clusters vs ADT clusters on shared cells
if "--cite" in sys.argv and adt is not None and len(shared) >= 200:
    from sklearn.metrics import adjusted_rand_score as ari
    truth = C.adt_clusters(adt, shared)
    labs = {}
    for t in ("alpha30m", "singlet-baseline-t4", "starsolo_GeneFull", "cellranger_published"):
        if t in tools:
            labs[t] = C.cluster(tools[t], shared)
    R["ari_vs_adt"] = {t: float(ari(v, truth)) for t, v in labs.items()}
    R["adt_clusters"] = int(len(set(truth)))

json.dump(R, open(OUTJ, "w"), indent=1, default=float)
print(json.dumps({k: R.get(k) for k in ("cell_calls", "calls", "depth_matched", "barnyard", "ari_vs_adt", "alpha_wrong_strand_frac",
                                         "reads_used_over_available")}, indent=1, default=float))
