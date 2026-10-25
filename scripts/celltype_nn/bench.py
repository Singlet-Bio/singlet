#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Benchmark the cell type network, each ensemble member alone and the members' vote on Singlet's own counts.

Same truth and samples as ``scripts/annotate_bench.py`` (author labels from CELLxGENE transferred by barcode,
``scripts/annotate_truth_cxg.py``; 16 studies none of which is in the training corpus). Every method is scored with
``singlet.celllabels.evaluate`` on the full guide (depth 1-4 accuracy, exact / coarse / finer / wrong), plus:

* consistency in Cell Ontology terms directly (the call is the truth term, an ancestor or a descendant of it);
* calibration of the network: the call's rolled-up probability against "the truth lies in the call's subtree" (calls finer than
  the truth cannot be checked and are left out), and the argmax probability the same way; ECE over 10 equal-width bins and the
  reliability table;
* seconds per million cells for the network (prediction only, and with reading the counts).

Methods: ``nn`` (network on X = exon + intron, as served), ``nn_exon`` (on the exon-only layer: robustness to intronic reads),
``nn_argmax`` (argmax term instead of the 0.5 roll-up call), ``guide``, ``celltypist``, ``singler``, ``vote`` (consensus of the
three members).

  python scripts/celltype_nn/bench.py --model $D/models/m1/model.pt --data $D --truth <truth dir> --bundles <dir> --out bench.json
"""

from __future__ import annotations

import argparse
import re
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "python"))
sys.path.insert(0, str(HERE.parent))

ALL = ["nn", "nn_argmax", "nn_exon", "nn_no_tissue", "guide", "celltypist", "singler", "vote"]
BINS = np.linspace(0, 1, 11)


def calibration(prob: np.ndarray, correct: np.ndarray) -> dict:
    """ECE and reliability over equal-width bins (cells with ``correct`` NaN are left out)."""
    m = ~np.isnan(correct)
    p, c = prob[m], correct[m]
    rows, ece = [], 0.0
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if s.sum():
            rows.append(dict(lo=float(lo), hi=float(hi), n=int(s.sum()), mean_prob=float(p[s].mean()), accuracy=float(c[s].mean())))
            ece += s.sum() / max(len(p), 1) * abs(p[s].mean() - c[s].mean())
    return dict(n=int(len(p)), ece=float(ece), bins=rows)


def subtree_correct(onto, pred, truth) -> np.ndarray:
    """1 when the truth term is the predicted term or below it, 0 on another branch, NaN when the prediction is finer."""
    out = np.full(len(pred), np.nan)
    for i, (p, t) in enumerate(zip(pred, truth)):
        if not isinstance(p, str):
            continue
        if p in onto.ancestors(t):
            out[i] = 1.0
        elif t in onto.ancestors(p):
            out[i] = np.nan
        else:
            out[i] = 0.0
    return out


def cl_consistency(onto, pred, truth) -> dict:
    from singlet.celltype_nn.ontology import is_consistent

    cats = [is_consistent(onto, p, t) if isinstance(p, str) else "unassigned" for p, t in zip(pred, truth)]
    v, n = np.unique(cats, return_counts=True)
    f = dict(zip(v.tolist(), (n / len(cats)).round(4).tolist()))
    f["consistent"] = round(f.get("exact", 0) + f.get("coarse", 0) + f.get("finer", 0), 4)
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--data", type=Path, required=True, help="corpus dir: SingleR references, CellTypist crosswalk")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--bundles", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--methods", default=",".join(ALL))
    ap.add_argument("--celltypist-models", type=Path, default=None)
    ap.add_argument("--studies", default="")
    ap.add_argument("--max-cells", type=int, default=30000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    ap.add_argument("--save-probs", type=Path, default=None, help="write per-cell NN outputs for the scored cells here")
    a = ap.parse_args(argv)

    import pandas as pd
    from singlet.bundle import SingletBundle
    from singlet.celllabels import celltype as CT
    from singlet.celllabels import evaluate as EV
    from singlet.celllabels import io as IO
    from singlet.celllabels.guide import Guide, ontology
    from singlet.celltype_nn import CellTypePredictor
    from singlet.celltype_nn.ensemble import pool_celltypist, run_celltypist, vote
    from ensemble_run import load_singler

    methods = [m for m in re.split(r"[,+]", a.methods) if m]  # "+" too: sbatch --export splits values on commas
    onto = ontology()
    pred = CellTypePredictor.load(a.model)
    xws = json.loads((a.data / "celltypist_crosswalk.json").read_text()) if (a.data / "celltypist_crosswalk.json").exists() else {}
    refs = {}
    man = json.loads((a.truth / "manifest.json").read_text())
    studies = [x for x in man["studies"] if x.get("kept", True)]
    if a.studies:
        studies = [x for x in studies if x["gse"] in set(a.studies.split(","))]
    results, cells_out = [], []
    for st in studies:
        gse = st["gse"]
        bundle = a.bundles / f"{gse}.singlet"
        for tf in sorted((a.truth / gse).glob("*.truth.parquet")):
            gsm = tf.name.split(".")[0]
            t_read = time.time()
            try:
                s = IO.read_legacy(bundle, gsm)
                with SingletBundle(bundle) as b:
                    A = b.raw_counts(gsm)
                gene_ids = [str(x) for x in A.var_names]
                assert [str(x) for x in A.obs_names] == s.barcodes
                spl = A.layers["spliced"] if "spliced" in A.layers else None
            except Exception as e:  # noqa: BLE001
                print(f"{gse}/{gsm}: cannot load ({e})", flush=True)
                continue
            t_read = time.time() - t_read
            if len(s.col) > a.max_cells or s.species is None:
                print(f"{gse}/{gsm}: skipped ({len(s.col)} cells, species {s.species})", flush=True)
                continue
            t = pd.read_parquet(tf)
            pos = {bc: i for i, bc in enumerate(s.barcodes)}
            keep = t["barcode"].astype(str).map(pos)
            ok = keep.notna().to_numpy()
            tr = keep[ok].astype(int).to_numpy()
            truth = t["cl_id"].astype(str).to_numpy()[ok]
            g = Guide(s.gene_names, species=s.species)
            tm = EV.TruthMapper(g)
            rec = dict(gse=gse, gsm=gsm, species=s.species, tissue=s.tissue, cxg_tissue=st.get("tissue", ""),
                       n_cells=len(s.col), n_truth=len(tr), methods={})
            calls, scores, extra = {}, {}, {}
            if any(m.startswith("nn") for m in methods):
                t0 = time.time()
                P = pred.predict(s.X, gene_ids, s.species, tissue=s.tissue, threads=a.threads)
                secs = time.time() - t0
                summ = pred.summarise(P)
                calls["nn"], scores["nn"] = summ["call"], summ["call_prob"]
                calls["nn_argmax"], scores["nn_argmax"] = summ["argmax"], summ["argmax_prob"]
                extra["nn"] = dict(seconds=secs, s_per_million=secs * a.threads / len(s.col) * 1e6,
                                   s_per_million_with_read=(secs + t_read) * a.threads / len(s.col) * 1e6)
                if a.save_probs is not None:
                    cells_out.append(pd.DataFrame(dict(gse=gse, gsm=gsm, truth=truth, call=summ["call"][tr], call_prob=summ["call_prob"][tr],
                                                       argmax=summ["argmax"][tr], argmax_prob=summ["argmax_prob"][tr],
                                                       top1=summ["topk"][tr, 0], top2=summ["topk"][tr, 1], top3=summ["topk"][tr, 2],
                                                       p1=summ["topk_prob"][tr, 0], p2=summ["topk_prob"][tr, 1], p3=summ["topk_prob"][tr, 2])))
                if "nn_no_tissue" in methods and pred.net.n_context:
                    Pn = pred.predict(s.X, gene_ids, s.species, tissue=None, threads=a.threads)
                    sn = pred.summarise(Pn)
                    calls["nn_no_tissue"], scores["nn_no_tissue"] = sn["call"], sn["call_prob"]
                if "nn_exon" in methods and spl is not None:
                    Pe = pred.predict(spl, gene_ids, s.species, tissue=s.tissue, threads=a.threads)
                    se = pred.summarise(Pe)
                    calls["nn_exon"], scores["nn_exon"] = se["call"], se["call_prob"]
            if "guide" in methods or "vote" in methods:
                t0 = time.time()
                r = CT.annotate_cell_types(s.X, s.gene_names, species=s.species, tissue=s.tissue, disease=s.disease, method="guide")
                calls["guide"] = r.cl
                scores["guide"] = np.nan_to_num(r.path_score, nan=0.0)
                extra["guide"] = dict(seconds=time.time() - t0)
            if "celltypist" in methods or "vote" in methods:
                t0 = time.time()
                raw = run_celltypist(s.X, s.gene_names, s.species, s.tissue, a.celltypist_models)
                pooled = pool_celltypist(raw, xws, onto)
                calls["celltypist"], scores["celltypist"] = pooled.cl, pooled.score
                extra["celltypist"] = dict(seconds=time.time() - t0, models=sorted(raw))
            if "singler" in methods or "vote" in methods:
                t0 = time.time()
                if s.species not in refs:
                    refs[s.species] = load_singler(a.data, s.species)
                ref = refs[s.species]
                if ref is not None:
                    from singlet.celllabels.celltype import lognorm

                    c = ref.score(lognorm(pred.vocab.project(s.X, gene_ids, s.species)))
                    calls["singler"], scores["singler"] = c.cl, c.score
                extra["singler"] = dict(seconds=time.time() - t0)
            if "vote" in methods:
                calls["vote"] = vote([calls[k] for k in ("guide", "celltypist", "singler") if k in calls], onto)
                scores["vote"] = np.ones(len(s.col), np.float32)
            for m in methods:
                if m not in calls:
                    continue
                cl = np.asarray(calls[m], dtype=object)
                node = np.array([tm(c) if isinstance(c, str) else 0 for c in cl], np.int32)
                rep = EV.report(g, truth, node[tr], np.asarray(scores[m], np.float32)[tr])
                rep["cl"] = cl_consistency(onto, cl[tr], truth)
                rep["coverage"] = float(np.mean([isinstance(c, str) for c in cl]))
                if m.startswith("nn"):
                    rep["calibration_subtree"] = calibration(np.asarray(scores[m], np.float64)[tr], subtree_correct(onto, cl[tr], truth))
                rep.update(extra.get(m, {}))
                rec["methods"][m] = rep
                bd = " ".join(f"d{d}:{v['accuracy']:.2f}" for d, v in rep["by_depth"].items())
                print(f"{gse}/{gsm} {m:11s} {bd} wrong {rep['fractions']['wrong']:.2f} cl-consistent {rep['cl']['consistent']:.2f}",
                      flush=True)
            results.append(rec)
            a.out.write_text(json.dumps(dict(results=results, methods=methods, model=str(a.model), threads=a.threads), indent=1, default=float))
    if a.save_probs is not None and cells_out:
        pd.concat(cells_out, ignore_index=True).to_parquet(a.save_probs, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
