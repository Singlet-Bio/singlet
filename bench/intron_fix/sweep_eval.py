"""Score every intron-rule variant of a Singlet run on the public benchmark's real-data checks.

    python sweep_eval.py BENCH_CAPS_DIR PB TAG OUT.json [datasets...]

BENCH_CAPS_DIR = bench/public/capabilities of a checkout of origin/bench/public-counts (its extract and metric code is used
unchanged). For runs/<ds>/<TAG>/intron_sweep/{main,rules_<r>}.tsv.gz (gene-level exon / intron / ambiguous per barcode,
written by SINGLET_INTRON_SWEEP), and for the public Singlet run itself (variant "public"), it computes:
  spliced-unspliced  unspliced and spliced+ambiguous agreement with STARsolo Velocyto and alevin-fry USA (pass/fail as published)
  gex-counts         exonic and exon+intron agreement; CITE protein-vs-RNA Spearman (pbmc1k_cite_v3)
  barnyard           cross-species UMI fraction in pure cells (hgmm1k_v3; cells and purity from the published Cell Ranger matrix)
"""
from __future__ import annotations

import gzip
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

CAPS = Path(sys.argv[1])
sys.path.insert(0, str(CAPS))
sys.path.insert(0, str(CAPS / "extract"))
import _familycmp as F  # noqa: E402
import _pubcommon as C  # noqa: E402
import counts as K  # noqa: E402
import runio as R  # noqa: E402
import gex_counts as GX  # noqa: E402
import spliced_unspliced as SU  # noqa: E402

PB, TAG, OUT = Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
DATASETS = sys.argv[5:] or ["pbmc1k_v3", "nuclei2k_v2", "hgmm1k_v3", "pbmc1k_cite_v3", "pbmc1k_v2", "neuron1k_v3", "pbmc5p_v1"]


def layers_from_tsv(p: Path, cells: list[str], universe: list[str]) -> dict[str, R.GeneMat]:
    df = pd.read_csv(p, sep="\t", dtype={"barcode": str, "gene_id": str})
    df["bc"] = df["barcode"].map(R.bare)
    df["g"] = df["gene_id"].map(R.canon_gene)
    cpos = {c: i for i, c in enumerate(cells)}
    gpos = {g: i for i, g in enumerate(universe)}
    df = df[df["bc"].isin(cpos) & df["g"].isin(gpos)]
    r = df["bc"].map(cpos).to_numpy()
    c = df["g"].map(gpos).to_numpy()
    present = np.zeros(len(cells), bool)
    present[np.unique(r)] = True
    out = {}
    for k, col in (("E", "exon"), ("U", "intron"), ("A", "ambiguous")):
        X = sp.csr_matrix((df[col].to_numpy(float), (r, c)), shape=(len(cells), len(universe)))
        X.eliminate_zeros()
        out[k] = R.GeneMat(universe, X, cells, present)
    return out


def layers_public(ctx, cells, universe):
    s = R.load_singlet(ctx.singlet, cells)
    return {k: s[k].pad(universe) for k in ("E", "U", "A")}


def metric_rows(metrics):
    rows = []
    for m in metrics:
        comp = [(c["tool"], c["value"]) for c in m.get("competitors", [])]
        rows.append({"id": m["id"], "singlet": m["singlet"]["value"], "competitors": comp, "pass": m["pass"],
                     "notes": m.get("notes", "")[:300]})
    return rows


def barnyard(ctx, layers_by_name):
    h5 = next(iter(sorted(ctx.data.glob("cellranger_*filtered*.h5"))))
    import h5py
    with h5py.File(h5, "r") as f:
        g = f["matrix"]
        M = sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]), shape=tuple(g["shape"][:]))
        bcs = [R.bare(b.decode()) for b in g["barcodes"][:]]
        ids = [x.decode() for x in g["features"]["id"][:]]
        ft = np.array([x.decode() for x in g["features"]["feature_type"][:]])
    keep = ft == "Gene Expression"
    M = M[keep]
    ids = [i for i, k in zip(ids, keep) if k]
    ish = np.array(["ENSG" in i for i in ids])
    tot = np.asarray(M.sum(0)).ravel()
    fh = np.asarray(M[ish].sum(0)).ravel() / np.maximum(tot, 1)
    ok = tot >= 500
    ph, pm = ok & (fh >= 0.9), ok & (fh <= 0.1)
    res = {"n_pure_human": int(ph.sum()), "n_pure_mouse": int(pm.sum())}
    for name, (gm, cells) in layers_by_name.items():
        pos = {c: i for i, c in enumerate(cells)}
        rows = [pos.get(b, -1) for b in bcs]
        X = gm.X
        hum = np.array([g.startswith("ENSG") for g in gm.genes])
        H = np.asarray(X[:, hum].sum(1)).ravel()
        Mo = np.asarray(X[:, ~hum].sum(1)).ravel()
        h = np.array([H[r] if r >= 0 else 0 for r in rows]); mo = np.array([Mo[r] if r >= 0 else 0 for r in rows])
        wrong = np.where(ph, mo, np.where(pm, h, 0.0))
        sel = ph | pm
        res[name] = float(wrong[sel].sum() / max((h + mo)[sel].sum(), 1))
    return res


def main():
    result = {}
    for ds in DATASETS:
        ctx = K.Ctx(PB, ds)
        sweep = ctx.runs / TAG / "intron_sweep"
        if not sweep.is_dir():
            print(ds, "no sweep dir", flush=True)
            continue
        cells = R.starsolo_cells(ctx.star_cr)
        universe = sorted({R.canon_gene(g) for g in R.singlet_known_genes(ctx.singlet)})
        vel = {k: R.load_starsolo(ctx.star_cr, "Velocyto", cells, layer=k) for k in ("spliced", "unspliced", "ambiguous")}
        af = R.load_alevinfry(ctx.af, cells)
        gene = R.load_starsolo(ctx.star_cr, "Gene", cells)
        genefull = R.load_starsolo(ctx.star_cr, "GeneFull", cells)
        af_total = af["S"].add(af["U"]).add(af["A"])
        variants = {"public": None}
        for f in sorted(sweep.glob("*.tsv.gz")):
            variants[f.name[: -len(".tsv.gz")]] = f
        dsres = {}
        for vname, f in variants.items():
            L = layers_public(ctx, cells, universe) if f is None else layers_from_tsv(f, cells, universe)
            E, U, A = L["E"], L["U"], L["A"]
            S = R.GeneMat(universe, E.X - A.X, cells, E.present)
            S.X.data = np.maximum(S.X.data, 0); S.X.eliminate_zeros()
            fams = {"unspliced": {"singlet_U": U, "starsolo_U": vel["unspliced"], "alevinfry_U": af["U"]},
                    "spliced_ambiguous": {"singlet_SA": S.add(A), "starsolo_SA": vel["spliced"].add(vel["ambiguous"]),
                                          "alevinfry_SA": af["S"].add(af["A"])},
                    "exonic": {"singlet_exon": E, "starsolo_gene": gene},
                    "total": {"singlet_total": E.add(U), "starsolo_genefull": genefull, "alevinfry_total": af_total}}
            cols, gt = {}, []
            for fam, tools in fams.items():
                c, g, _ = K.family_tables(fam, tools, cells)
                cols.update(c); gt.append(g)
            cdf = pd.DataFrame(cols, index=pd.Index(cells, name="barcode"))
            gdf = pd.concat(gt, axis=1)
            stats = []
            for spec in SU.SPECS + GX.SPECS:
                st = F.collect(ds, spec, cdf, gdf, C.DATASET_ORDER.index(ds) if ds in C.DATASET_ORDER else 0)
                if st is not None:
                    stats.append(st)
            metrics = C.stamp(F.build_metrics(stats))
            if ds == "pbmc1k_cite_v3":
                with tempfile.TemporaryDirectory() as td:
                    K.extract_adt_pairs(ctx, Path(td), cells, {"singlet_exon": E, "singlet_total": E.add(U), "starsolo_gene": gene,
                                                               "starsolo_genefull": genefull, "alevinfry_total": af_total})
                    metrics += C.stamp(GX.adt_metrics(Path(td)))
            entry = {"metrics": metric_rows(metrics),
                     "totals": {k: float(L[k].X.sum()) for k in ("E", "U", "A")}}
            if ds.startswith("hgmm"):
                by = {"singlet_U": (U, cells), "singlet_E": (E, cells), "singlet_EU": (E.add(U), cells)}
                if vname == "public":
                    by.update({"starsolo_velo_U": (vel["unspliced"], cells), "starsolo_gene": (gene, cells),
                               "starsolo_genefull": (genefull, cells), "alevinfry_U": (af["U"], cells),
                               "alevinfry_SA": (af["S"].add(af["A"]), cells), "alevinfry_USA": (af_total, cells)})
                entry["barnyard"] = barnyard(ctx, by)
            dsres[vname] = entry
            npass = sum(m["pass"] == "pass" for m in entry["metrics"])
            print(ds, vname, f"pass {npass}/{len(entry['metrics'])}", entry.get("barnyard", ""), flush=True)
        result[ds] = dsres
        OUT.write_text(json.dumps(result, indent=1))
    OUT.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
