#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark cell type methods on Singlet's own counts against author labels (Cell Ontology ids curated by CELLxGENE).

Truth comes from ``scripts/annotate_truth_cxg.py`` (CELLxGENE cells matched to Singlet called cells by barcode): one
``<truth>/<GSE>/<GSM>.truth.parquet`` per sample and ``<truth>/manifest.json``. Every sample is annotated from its Singlet
counts (``X`` = exon + intron at gene level, the matrix the atlas serves), and only the cells that have a truth label are scored.

Methods
  guide              cell guide, gating with kNN smoothing (singlet.annotate.celltype, method='guide')
  selftrain          guide + per-sample logistic regression on its confident calls (the default)
  selftrain_pure     self-training without the back-off to the guide (min_prob = 0): the variant benchmarked in singlet-agents
  guide_exon         guide on exon-only counts (what most author pipelines counted)
  guide_all_cells    guide with no minimum-genes filter (to choose that filter)
  marker_cluster     singlet.annotate_cell_types (existing tool): marker overlap per Leiden cluster, markers = the guide's
                     depth <= 3 nodes (each with its own and its ancestors' positive markers)
  knn_transfer       singlet.transfer_labels (existing tool): kNN in a reference PCA; reference = truth-labelled cells of
                     the OTHER studies of the same species (leave-study-out), at most --ref-cap cells, 2000 shared HVGs

Reported per sample and per study: accuracy at guide depth 1-4, exact/coarse/finer/wrong/unassigned fractions, calibration of
the score, accuracy by genes detected, seconds and seconds per million cells (wall clock x threads used).

  python scripts/annotate_bench.py --truth <dir> --bundles <dir> --out results.json [--methods guide,selftrain] [--threads 4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

ALL = [
    "guide",
    "selftrain",
    "selftrain_pure",
    "guide_exon",
    "guide_all_cells",
    "marker_cluster",
    "knn_transfer",
]
GENE_BINS = [(0, 100), (100, 200), (200, 500), (500, 1000), (1000, 10**9)]


def _load(bundle: Path, gsm: str):
    """(SampleInput, spliced CSR aligned to it)."""
    import scipy.sparse as sps
    from singlet.annotate import io as IO
    from singlet.bundle import SingletBundle

    s = IO.read_legacy(bundle, gsm)
    with SingletBundle(bundle) as b:
        a = b.raw_counts(gsm)
    assert [str(x) for x in a.obs_names] == s.barcodes
    spl = a.layers.get("spliced") if hasattr(a, "layers") else None
    return s, (sps.csr_matrix(spl) if spl is not None else None)


def _marker_dict(g, depth=3):
    out = {}
    for j in range(1, len(g.nodes)):
        if g.depth[j] > depth or g.n_markers_found[j] == 0:
            continue
        genes = set()
        for a in g.path(j):
            genes |= set((g.nodes[a].get("markers") or {}).get("pos") or {})
        out[g.ids[j]] = sorted(genes)
    return out


def m_marker_cluster(s, g):
    import anndata as ad
    import pandas as pd
    from singlet._annotate_cell_types import annotate_cell_types
    from singlet.annotate import celltype as CT

    X = CT.lognorm(s.X)
    _, idx = CT.neighbours(s.X, k=15)
    cl = CT.overcluster(idx, 1.0)
    a = ad.AnnData(
        X=X,
        obs=pd.DataFrame(
            {"leiden": cl.astype(str) if cl is not None else np.zeros(X.shape[0]).astype(str)}
        ),
        var=pd.DataFrame(index=pd.Index(s.gene_names).astype(str)),
    )
    a.var_names_make_unique()
    md = _marker_dict(g)
    if s.species == "mouse":
        from singlet.annotate.guide import human_to_mouse

        md = {k: [m for h in v for m in human_to_mouse(h)[:1]] for k, v in md.items()}
    annotate_cell_types(a, md, groupby="leiden", min_score=0.1)
    node = np.array([g.index.get(x, 0) for x in a.obs["cell_type"]], np.int32)
    return node, np.asarray(a.obs["cell_type_score"], np.float32)


class Reference:
    """Leave-study-out reference: truth-labelled cells of the other studies of one species (log1p CP10k)."""

    def __init__(self, samples, cap, seed=0):
        self.samples, self.cap, self.seed = samples, cap, seed

    def build(self, exclude_gse, species, query_genes):
        import anndata as ad
        import pandas as pd
        import scipy.sparse as sps
        from singlet.annotate import celltype as CT
        from sklearn.decomposition import TruncatedSVD

        parts, labs = [], []
        pool = [x for x in self.samples if x["gse"] != exclude_gse and x["species"] == species]
        if not pool:
            return None
        per = max(1, self.cap // len(pool))
        rng = np.random.default_rng(self.seed)
        common = None
        for x in pool:
            gs = {g: i for i, g in enumerate(x["genes"])}
            common = set(gs) if common is None else common & set(gs)
        common = sorted(common & set(query_genes))
        for x in pool:
            gs = {g: i for i, g in enumerate(x["genes"])}
            cols = np.array([gs[g] for g in common])
            take = rng.permutation(len(x["truth_rows"]))[:per]
            rows = x["truth_rows"][take]
            parts.append(CT.lognorm(x["X"][rows])[:, cols])
            labs.append(x["truth_cl"][take])
        R = sps.vstack(parts).tocsr()
        mu = np.asarray(R.mean(0)).ravel()
        var = np.asarray(R.multiply(R).mean(0)).ravel() - mu**2
        hv = np.sort(np.argsort(-var)[:2000])
        Rd = R[:, hv].toarray() - mu[hv]
        svd = TruncatedSVD(30, random_state=0).fit(Rd)
        ref = ad.AnnData(
            X=sps.csr_matrix(R[:, hv]),
            obs=pd.DataFrame({"cl": np.concatenate(labs)}),
            var=pd.DataFrame(index=pd.Index(np.array(common)[hv])),
        )
        ref.obsm["X_pca"] = svd.transform(Rd)
        ref.varm["PCs"] = svd.components_.T
        ref.uns["pca"] = {"mean": mu[hv]}
        return ref


def m_knn_transfer(s, g, ref):
    import anndata as ad
    import pandas as pd
    from singlet._transfer_labels import transfer_labels
    from singlet.annotate import celltype as CT
    from singlet.annotate.evaluate import TruthMapper

    gi = {x: i for i, x in enumerate(s.gene_names)}
    cols = np.array([gi[x] for x in ref.var_names])
    q = ad.AnnData(X=CT.lognorm(s.X)[:, cols], var=pd.DataFrame(index=ref.var_names.copy()))
    transfer_labels(q, ref, "cl", n_neighbors=15)
    tm = TruthMapper(g)
    node = np.array([tm(c) for c in q.obs["cl_transferred"].astype(str)], np.int32)
    return node, np.asarray(q.obs["cl_confidence"], np.float32)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default=",".join(ALL))
    ap.add_argument(
        "--studies",
        default="",
        help="comma-separated GSE ids (default: every kept study of the manifest)",
    )
    ap.add_argument(
        "--max-cells", type=int, default=30000, help="skip samples with more called cells"
    )
    ap.add_argument("--ref-cap", type=int, default=30000)
    ap.add_argument(
        "--threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)),
    )
    a = ap.parse_args(argv)

    import pandas as pd
    from singlet.annotate import celltype as CT
    from singlet.annotate import evaluate as EV
    from singlet.annotate.guide import Guide

    methods = [m for m in a.methods.split(",") if m]
    man = json.loads((a.truth / "manifest.json").read_text())
    studies = man if isinstance(man, list) else man.get("studies", list(man.values()))
    studies = [x for x in studies if x.get("kept", True)]
    if a.studies:
        want = set(a.studies.split(","))
        studies = [x for x in studies if x["gse"] in want]

    # ---- load every sample with truth once
    samples = []
    for st in studies:
        gse = st["gse"]
        bundle = a.bundles / f"{gse}.singlet"
        if not bundle.exists():
            bundle = Path(st.get("bundle_path", bundle))
        for tf in sorted((a.truth / gse).glob("*.truth.parquet")):
            gsm = tf.name.split(".")[0]
            t = pd.read_parquet(tf)
            try:
                s, spl = _load(bundle, gsm)
            except Exception as e:  # noqa: BLE001
                print(f"{gse}/{gsm}: cannot load ({e})", flush=True)
                continue
            if len(s.col) > a.max_cells or s.species is None:
                print(f"{gse}/{gsm}: skipped ({len(s.col)} cells, species {s.species})", flush=True)
                continue
            pos = {b: i for i, b in enumerate(s.barcodes)}
            keep = t["barcode"].astype(str).map(pos)
            ok = keep.notna().to_numpy()
            rows = keep[ok].astype(int).to_numpy()
            samples.append(
                dict(
                    gse=gse,
                    gsm=gsm,
                    s=s,
                    spl=spl,
                    species=s.species,
                    genes=s.gene_names,
                    X=s.X,
                    truth_rows=rows,
                    truth_cl=t["cl_id"].astype(str).to_numpy()[ok],
                    tissue=st.get("tissue", ""),
                )
            )
            print(
                f"{gse}/{gsm}: {len(s.col)} cells, {len(rows)} with truth, tissue text {s.tissue!r}",
                flush=True,
            )
    ref = Reference(samples, a.ref_cap)

    results = []
    for x in samples:
        s = x["s"]
        g = Guide(
            s.gene_names, species=s.species
        )  # full guide: predictions of every method are scored on it
        rec = dict(
            gse=x["gse"],
            gsm=x["gsm"],
            species=s.species,
            tissue=s.tissue,
            n_cells=len(s.col),
            n_truth=len(x["truth_rows"]),
            methods={},
        )
        ngenes = np.diff(s.X.indptr)[x["truth_rows"]]
        for m in methods:
            t0 = time.time()
            try:
                if m in ("guide", "selftrain", "selftrain_pure", "guide_all_cells", "guide_exon"):
                    X = x["spl"] if m == "guide_exon" else s.X
                    if X is None:
                        continue
                    kw = dict(
                        species=s.species,
                        tissue=s.tissue,
                        disease=s.disease,
                        method="guide" if m.startswith("guide") else "selftrain",
                    )
                    if m == "selftrain_pure":
                        kw["min_prob"] = 0.0
                    if m == "guide_all_cells":
                        kw["min_genes"] = 0
                    r = CT.annotate_cell_types(X, s.gene_names, **kw)
                    remap = np.array([g.index.get(i, 0) for i in r.guide.ids], np.int32)
                    node, score = remap[r.node], r.score
                elif m == "marker_cluster":
                    gt = Guide(s.gene_names, species=s.species, tissue=s.tissue, disease=s.disease)
                    node_t, score = m_marker_cluster(s, gt)
                    remap = np.array([g.index.get(i, 0) for i in gt.ids], np.int32)
                    node = remap[node_t]
                elif m == "knn_transfer":
                    rf = ref.build(x["gse"], s.species, s.gene_names)
                    if rf is None:
                        continue
                    t0 = (
                        time.time()
                    )  # the reference build is shared work, not charged to the sample
                    node, score = m_knn_transfer(s, g, rf)
                else:
                    raise ValueError(m)
                secs = time.time() - t0
                tr = x["truth_rows"]
                rep = EV.report(g, x["truth_cl"], node[tr], score[tr])
                acc_by_genes = {}
                if m in ("guide_all_cells", "selftrain", "guide"):
                    tm = EV.TruthMapper(g)
                    tn = np.array([tm(c) for c in x["truth_cl"]])
                    cat = EV.classify(g, tn, node[tr])
                    for lo, hi in GENE_BINS:
                        sel = (ngenes >= lo) & (ngenes < hi) & (cat != "untestable")
                        if sel.sum() >= 20:
                            d3 = g.ancestor_at_depth(tn[sel], 3) == g.ancestor_at_depth(
                                node[tr][sel], 3
                            )
                            acc_by_genes[f"{lo}-{hi}"] = dict(
                                n=int(sel.sum()),
                                consistent=float(
                                    np.isin(cat[sel], ["exact", "coarse", "finer"]).mean()
                                ),
                                d3_match=float(d3.mean()),
                                unassigned=float((cat[sel] == "unassigned").mean()),
                            )
                rep.update(
                    seconds=secs,
                    s_per_million=secs * a.threads / max(len(s.col), 1) * 1e6,
                    threads=a.threads,
                    coverage=float((node > 0).mean()),
                    acc_by_genes=acc_by_genes,
                )
                rec["methods"][m] = rep
                bd = " ".join(f"d{d}:{v['accuracy']:.2f}" for d, v in rep["by_depth"].items())
                print(
                    f"{x['gse']}/{x['gsm']} {m:16s} {bd} wrong {rep['fractions']['wrong']:.2f} unassigned {rep['fractions']['unassigned']:.2f} "
                    f"{secs:.1f}s",
                    flush=True,
                )
            except Exception as e:  # noqa: BLE001
                rec["methods"][m] = dict(error=f"{type(e).__name__}: {e}")
                print(f"{x['gse']}/{x['gsm']} {m}: ERROR {e}", flush=True)
        results.append(rec)
        a.out.write_text(
            json.dumps(
                dict(results=results, methods=methods, threads=a.threads), indent=1, default=float
            )
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
