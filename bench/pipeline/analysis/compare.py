"""Compare all tools on one dataset; write results/<dataset>/{metrics.json,*.csv,*.png}.

    python -I compare.py <PB> <dataset_id> [--barnyard] [--cite]

Comparisons (all on the same reference, same FASTQs):
  * double count: Singlet baseline (atlas engine) vs fixed engine, same barcodes
  * per-cell totals and per-gene pseudobulk vs STARsolo GeneFull/Gene/Velocyto, alevin-fry USA,
    and the 10x-published Cell Ranger matrix (older reference, exonic only for v3.0 data)
  * spliced / unspliced / ambiguous split
  * cell calls (Jaccard of called barcodes)
  * clustering agreement (ARI) and marker overlap on shared cells
  * barnyard: species purity, doublet rate, cross-species (ambient) fraction
  * CITE-seq: ARI of RNA clusters against ADT-derived clusters (protein ground truth)
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import loaders  # noqa: E402

warnings.filterwarnings("ignore")


def totals(a, layer=None):
    m = a.X if layer is None else a.layers[layer]
    return np.asarray(m.sum(axis=1)).ravel()


def subset(a, cells, genes=None):
    b = a[a.obs_names.isin(cells)]
    b = b[[c for c in cells if c in set(b.obs_names)]]
    if genes is not None:
        b = b[:, b.var_names.isin(genes)]
    return b


def pearson_log(x, y):
    lx, ly = np.log1p(x), np.log1p(y)
    return float(np.corrcoef(lx, ly)[0, 1]) if len(x) > 2 else float("nan")


def spearman(x, y):
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).correlation)


def gene_concordance(a, b, cells, layer_a=None, layer_b=None):
    """Per-cell totals and per-gene pseudobulk of a vs b on shared cells and shared genes."""
    genes = sorted(set(a.var_names) & set(b.var_names))
    A, B = subset(a, cells, genes), subset(b, cells, genes)
    A, B = A[:, genes], B[:, genes]
    ma = A.X if layer_a is None else A.layers[layer_a]
    mb = B.X if layer_b is None else B.layers[layer_b]
    ca, cb = np.asarray(ma.sum(1)).ravel(), np.asarray(mb.sum(1)).ravel()
    ga, gb = np.asarray(ma.sum(0)).ravel(), np.asarray(mb.sum(0)).ravel()
    expressed = (ga + gb) > 0
    # per-cell gene-vector correlation (median over cells)
    ma_c, mb_c = sp.csr_matrix(ma), sp.csr_matrix(mb)
    rs = []
    for i in range(0, ma_c.shape[0], max(1, ma_c.shape[0] // 300)):
        xa = np.log1p(ma_c[i].toarray().ravel())
        xb = np.log1p(mb_c[i].toarray().ravel())
        if xa.std() > 0 and xb.std() > 0:
            rs.append(np.corrcoef(xa, xb)[0, 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(cb > 0, ca / cb, np.nan)
    return {
        "n_cells": int(len(ca)),
        "n_genes": int(expressed.sum()),
        "cell_total_ratio_median": float(np.nanmedian(ratio)),
        "cell_total_ratio_iqr": [float(np.nanpercentile(ratio, 25)), float(np.nanpercentile(ratio, 75))],
        "cell_total_pearson_log": pearson_log(ca, cb),
        "gene_pseudobulk_spearman": spearman(ga[expressed], gb[expressed]),
        "gene_pseudobulk_pearson_log": pearson_log(ga[expressed], gb[expressed]),
        "per_cell_gene_pearson_log_median": float(np.median(rs)) if rs else float("nan"),
        "total_ratio": float(ca.sum() / max(cb.sum(), 1)),
        "genes_detected_ratio_median": float(np.median(
            (np.asarray((ma > 0).sum(1)).ravel() + 1) / (np.asarray((mb > 0).sum(1)).ravel() + 1))),
    }


def suf_split(a):
    tot = a.X.sum()
    out = {"total": float(tot)}
    for k in ("spliced", "unspliced", "ambiguous"):
        if k in a.layers:
            out[k] = float(a.layers[k].sum() / tot) if tot else float("nan")
    return out


def cluster(a, cells, res=1.0, seed=0):
    import scanpy as sc

    b = subset(a, cells).copy()
    b.X = sp.csr_matrix(b.X, dtype=np.float32)
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    sc.pp.highly_variable_genes(b, n_top_genes=2000, flavor="seurat")
    b = b[:, b.var.highly_variable].copy()
    sc.pp.scale(b, max_value=10)
    sc.tl.pca(b, n_comps=30, random_state=seed)
    sc.pp.neighbors(b, n_neighbors=15, random_state=seed)
    sc.tl.leiden(b, resolution=res, random_state=seed, flavor="igraph", n_iterations=2)
    return b.obs["leiden"].reindex(cells).astype(str).values


def markers(a, cells, labels, n=50):
    import scanpy as sc

    b = subset(a, cells).copy()
    b.X = sp.csr_matrix(b.X, dtype=np.float32)
    sc.pp.normalize_total(b, target_sum=1e4)
    sc.pp.log1p(b)
    b.obs["lab"] = pd.Categorical(labels)
    keep = b.obs["lab"].value_counts()
    keep = keep[keep >= 10].index
    b = b[b.obs["lab"].isin(keep)].copy()
    sc.tl.rank_genes_groups(b, "lab", method="wilcoxon")
    names = pd.DataFrame(b.uns["rank_genes_groups"]["names"])
    return {c: set(names[c].head(n)) for c in names.columns}


def adt_clusters(adt, cells):
    """Protein ground truth: CLR-normalised ADT -> PCA -> Leiden (independent of RNA quantifier)."""
    import scanpy as sc

    b = subset(adt, cells).copy()
    X = b.X.toarray().astype(float)
    X = np.log1p(X) - np.log1p(X).mean(axis=1, keepdims=True)  # CLR per cell
    b.X = X
    sc.pp.scale(b)
    sc.tl.pca(b, n_comps=min(10, X.shape[1] - 1), random_state=0)
    sc.pp.neighbors(b, n_neighbors=15, random_state=0)
    sc.tl.leiden(b, resolution=0.6, random_state=0, flavor="igraph", n_iterations=2)
    return b.obs["leiden"].reindex(cells).astype(str).values


def species_of(var_names):
    v = pd.Index(var_names)
    mouse = v.str.contains("ENSMUSG") | v.str.startswith("GRCm39") | v.str.startswith("mm10")
    human = ~mouse
    return human.values, mouse.values


def barnyard(a, cells):
    b = subset(a, cells)
    h, m = species_of(b.var_names)
    hu = np.asarray(b.X[:, h].sum(1)).ravel()
    mu = np.asarray(b.X[:, m].sum(1)).ravel()
    frac_h = hu / np.maximum(hu + mu, 1)
    call = np.where(frac_h >= 0.9, "human", np.where(frac_h <= 0.1, "mouse", "doublet"))
    cross = np.where(call == "human", 1 - frac_h, np.where(call == "mouse", frac_h, np.nan))
    return {
        "n_cells": int(len(call)),
        "human": int((call == "human").sum()),
        "mouse": int((call == "mouse").sum()),
        "doublet": int((call == "doublet").sum()),
        "doublet_rate": float((call == "doublet").mean()),
        "cross_species_frac_median": float(np.nanmedian(cross)),
        "cross_species_frac_mean": float(np.nanmean(cross)),
    }, pd.Series(call, index=b.obs_names)


def main():
    import faulthandler

    faulthandler.dump_traceback_later(900, repeat=True)  # where is a slow run spending time
    pb, ds = Path(sys.argv[1]), sys.argv[2]
    is_barnyard = "--barnyard" in sys.argv
    is_cite = "--cite" in sys.argv
    runs = pb / "runs" / ds
    out = pb / "results" / ds
    out.mkdir(parents=True, exist_ok=True)
    M: dict = {"dataset": ds}

    tools = {}
    for t in ("singlet-baseline", "singlet-fixed", "singlet-baseline-t4", "singlet-fixed-t4"):
        a = loaders.load_singlet(runs / t)
        if a is not None:
            tools[t] = a
    tools.update(loaders.load_starsolo(runs / "starsolo"))
    af = loaders.load_alevinfry(runs / "alevinfry")
    if af is not None:
        tools["alevinfry"] = af
    crh5 = next(iter((pb / "data" / ds).glob("cellranger_*.h5")), None)
    adt = None
    if crh5 is not None:
        cr, adt = loaders.load_cellranger_h5(crh5)
        if cr is not None:
            tools["cellranger_published"] = cr
    M["tools"] = sorted(tools)

    # runtime / memory
    M["resources"] = {}
    for t in ("singlet-baseline", "singlet-fixed", "singlet-baseline-t4", "singlet-fixed-t4", "starsolo",
              "alevinfry"):
        r = loaders.load_time(runs / t / "time.txt")
        enc = loaders.load_time(runs / t / "encode.time.txt")
        if enc:
            r["encode_wall_s"] = enc.get("wall_s")
            r["encode_max_rss_gb"] = enc.get("max_rss_gb")
        M["resources"][t] = r

    # ── 1. double count: atlas engine vs fixed engine on identical barcodes ──
    def double_count(old_t, new_t):
        old, new = tools[old_t], tools[new_t]
        common = sorted(set(old.obs_names) & set(new.obs_names))
        cells = [c for c in common if c in set(new.obs_names[new.obs["is_cell"]])] or common
        o, n = subset(old, cells), subset(new, cells)
        genes = sorted(set(o.var_names) & set(n.var_names))
        o, n = o[:, genes], n[:, genes]
        to, tn = totals(o), totals(n)
        over = (to - tn) / np.maximum(tn, 1)
        go, gn = np.asarray(o.X.sum(0)).ravel(), np.asarray(n.X.sum(0)).ravel()
        gover = (go - gn) / np.maximum(gn, 1)
        expressed = gn >= 50
        ps = new.uns.get("pileup_stats", {})
        mol = sum(ps.get(k, 0) for k in ("molecules_spliced", "molecules_unspliced", "molecules_ambiguous"))
        xw = loaders.cross_worker_exon_hits(runs / old_t / "run.log")
        res = {
            "n_cells": len(cells),
            "molecules_total": mol,
            "molecules_legacy_double": ps.get("molecules_legacy_double"),
            "legacy_double_frac": ps.get("molecules_legacy_double", 0) / mol if mol else None,
            "cross_gene_intron_reads": ps.get("cross_gene_intron_reads"),
            "old_cross_worker_exon_molecules": xw,
            "old_cross_worker_frac": xw / mol if (xw is not None and mol) else None,
            "cell_overcount_median": float(np.median(over)),
            "cell_overcount_p10_p90": [float(np.percentile(over, 10)), float(np.percentile(over, 90))],
            "total_overcount": float(to.sum() / tn.sum() - 1),
            "gene_overcount_median_expressed": float(np.median(gover[expressed])),
            "gene_overcount_p99_expressed": float(np.percentile(gover[expressed], 99)),
            "exon_old_vs_new_ratio": float(totals(o, "exon").sum() / max(totals(n, "exon").sum(), 1)),
            "intron_old_vs_new_ratio": float(totals(o, "unspliced").sum() / max(totals(n, "unspliced").sum(), 1)),
            "called_cells_old": int(old.obs["is_cell"].sum()),
            "called_cells_new": int(new.obs["is_cell"].sum()),
        }
        return res, cells, to, tn, over, genes, go, gn, gover, expressed

    # "double_count" = the atlas configuration (4 pileup workers, as the campaign ran);
    # "_16w" = 16 workers. BAM re-runs use run() (no workers) and only have the plain pair.
    for tag, (ot, nt) in {"double_count": ("singlet-baseline-t4", "singlet-fixed-t4"),
                          "double_count_16w": ("singlet-baseline", "singlet-fixed")}.items():
        if {ot, nt} <= set(tools):
            res, cells, to, tn, over, genes, go, gn, gover, expressed = double_count(ot, nt)
            M[tag] = res
            sfx = "" if tag == "double_count" else "_16w"
            pd.DataFrame({"barcode": cells, "old_total": to, "new_total": tn, "overcount": over}).to_csv(
                out / f"double_count_cells{sfx}.csv", index=False)
            top = np.argsort(-gover * expressed)[:50]
            pd.DataFrame({"gene": np.array(genes)[top], "old": go[top], "new": gn[top],
                          "overcount": gover[top]}).to_csv(out / f"double_count_top_genes{sfx}.csv", index=False)
    if "double_count" not in M and "double_count_16w" in M:
        M["double_count"] = M.pop("double_count_16w")
        for f in ("double_count_cells", "double_count_top_genes"):
            (out / f"{f}_16w.csv").replace(out / f"{f}.csv")

    # Worker-count (in)variance: the same engine with 4 vs 16 pileup workers, same cells/genes
    inv = {}
    for eng in ("singlet-baseline", "singlet-fixed"):
        if {eng, eng + "-t4"} <= set(tools):
            a, b = tools[eng], tools[eng + "-t4"]
            cs = sorted(set(a.obs_names) & set(b.obs_names))
            gs = sorted(set(a.var_names) & set(b.var_names))
            xa, xb = subset(a, cs, gs)[:, gs].X, subset(b, cs, gs)[:, gs].X
            d = abs(xa - xb)
            inv[eng] = {"total_16w_over_4w": float(xa.sum() / max(xb.sum(), 1)),
                        "entries_differing": int((d > 0).sum()), "abs_diff_total": float(d.sum())}
    if inv:
        M["worker_invariance"] = inv

    # ── 2. cell calls ──
    called = {t: set(a.obs_names[a.obs["is_cell"]]) for t, a in tools.items()
              if "is_cell" in a.obs and a.obs["is_cell"].any()}
    # STARsolo's default --soloCellFilter CellRanger2.2 3000 0.99 10 (the knee filter behind the
    # atlas bundles' `called` flag): cells = UMI >= (99th pct of the top 3000 barcodes) / 10
    def knee_cr22(a, n_exp=3000, q=0.99, ratio=10):
        t = totals(a)
        top = np.sort(t)[::-1][:n_exp]
        thr = np.percentile(top, 100 * q) / ratio if len(top) else np.inf
        return set(a.obs_names[t >= max(thr, 1)])

    for t in ("starsolo_GeneFull", "singlet-fixed", "singlet-baseline-t4"):
        if t in tools:
            called[f"{t}__knee_cr22"] = knee_cr22(tools[t])
    M["cell_calls"] = {t: len(s) for t, s in called.items()}
    jac = {}
    for t1 in called:
        for t2 in called:
            if t1 < t2:
                s1, s2 = called[t1], called[t2]
                jac[f"{t1}|{t2}"] = {"jaccard": len(s1 & s2) / max(len(s1 | s2), 1),
                                     "only_first": len(s1 - s2), "only_second": len(s2 - s1)}
    M["cell_call_overlap"] = jac

    # reference cell set for count concordance: cells called by STARsolo AND Singlet-fixed
    ref_cells = set(called.get("starsolo_GeneFull") or called.get("starsolo_Gene") or set())
    if "singlet-fixed" in called:
        ref_cells = ref_cells & called["singlet-fixed"] if ref_cells else called["singlet-fixed"]
    for t, a in tools.items():
        ref_cells &= set(a.obs_names)
    shared = sorted(ref_cells)
    M["shared_cells"] = len(shared)

    # ── 3. count concordance (vs STARsolo GeneFull = each molecule once, exon+intron) ──
    conc = {}
    pairs = [
        ("singlet-fixed", None, "starsolo_GeneFull", None, "total (exon+intron)"),
        ("singlet-baseline-t4", None, "starsolo_GeneFull", None, "total (exon+intron), atlas config"),
        ("singlet-baseline", None, "starsolo_GeneFull", None, "total (exon+intron), 16 workers"),
        ("alevinfry", None, "starsolo_GeneFull", None, "total (S+U+A)"),
        ("singlet-fixed", "exon", "starsolo_Gene", None, "exonic (S+A)"),
        ("singlet-baseline-t4", "exon", "starsolo_Gene", None, "exonic, atlas config"),
        ("cellranger_published", None, "starsolo_Gene", None, "exonic (CR published)"),
        ("singlet-fixed", "exon", "cellranger_published", None, "exonic vs CR"),
        ("singlet-fixed", "spliced", "starsolo_Velocyto", "spliced", "spliced"),
        ("singlet-fixed", "unspliced", "starsolo_Velocyto", "unspliced", "unspliced"),
        ("singlet-fixed", "ambiguous", "starsolo_Velocyto", "ambiguous", "ambiguous"),
        ("singlet-fixed", "spliced", "alevinfry", "spliced", "spliced"),
        ("singlet-fixed", "unspliced", "alevinfry", "unspliced", "unspliced"),
        ("singlet-fixed", None, "alevinfry", None, "total"),
    ]
    for ta, la, tb, lb, what in pairs:
        if ta in tools and tb in tools and shared:
            try:
                conc[f"{ta}[{la or 'X'}] vs {tb}[{lb or 'X'}]"] = {"what": what, **gene_concordance(
                    tools[ta], tools[tb], shared, la, lb)}
            except Exception as e:  # noqa: BLE001
                conc[f"{ta} vs {tb}"] = {"error": repr(e)}
    M["concordance"] = conc
    M["suf_split"] = {t: suf_split(subset(a, shared)) for t, a in tools.items()
                      if "spliced" in a.layers and shared}

    # per-cell totals table for figures
    tab = pd.DataFrame(index=shared)
    for t, a in tools.items():
        s = subset(a, shared)
        tab[t] = pd.Series(totals(s), index=s.obs_names)
        if "exon" in a.layers:
            tab[t + "_exon"] = pd.Series(totals(s, "exon"), index=s.obs_names)
    tab.to_csv(out / "cell_totals.csv")

    # ── 4. clustering agreement + markers (on shared cells) ──
    if len(shared) >= 200:
        from sklearn.metrics import adjusted_rand_score as ari

        labs = {}
        for t in ("starsolo_GeneFull", "starsolo_Gene", "singlet-fixed", "singlet-baseline-t4", "alevinfry",
                  "cellranger_published"):
            if t in tools:
                try:
                    labs[t] = cluster(tools[t], shared)
                except Exception as e:  # noqa: BLE001
                    M.setdefault("errors", {})[f"cluster {t}"] = repr(e)
        M["n_clusters"] = {t: int(len(set(v))) for t, v in labs.items()}
        M["ari"] = {f"{a}|{b}": float(ari(labs[a], labs[b])) for a in labs for b in labs if a < b}
        ref = "starsolo_GeneFull" if "starsolo_GeneFull" in labs else next(iter(labs), None)
        if ref:
            mk = {t: markers(tools[t], shared, labs[ref]) for t in labs}
            M["marker_jaccard_vs_" + ref] = {
                t: float(np.mean([len(mk[t][c] & mk[ref][c]) / len(mk[t][c] | mk[ref][c]) for c in mk[ref]]))
                for t in mk if t != ref}
        if is_cite and adt is not None:
            truth = adt_clusters(adt, shared)
            M["ari_vs_adt"] = {t: float(ari(v, truth)) for t, v in labs.items()}
            M["adt_clusters"] = int(len(set(truth)))
        pd.DataFrame(labs, index=shared).to_csv(out / "clusters.csv")

    # ── 5. barnyard ──
    if is_barnyard:
        M["barnyard"] = {}
        calls = {}
        for t, a in tools.items():
            cs = sorted(set(a.obs_names[a.obs["is_cell"]])) if a.obs["is_cell"].any() else shared
            if cs:
                M["barnyard"][t], calls[t] = barnyard(a, cs)
        pd.DataFrame(calls).to_csv(out / "barnyard_calls.csv")

    (out / "metrics.json").write_text(json.dumps(M, indent=1, default=float))
    print(json.dumps({k: M[k] for k in M if k in ("double_count", "cell_calls", "shared_cells", "ari",
                                                  "ari_vs_adt", "barnyard")}, indent=1, default=float))


if __name__ == "__main__":
    main()
