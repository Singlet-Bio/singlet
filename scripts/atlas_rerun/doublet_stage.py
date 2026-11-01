#!/usr/bin/env python3
"""Doublet stage: after the engine, replace the sample's doublet scores with the cross-study stacker of `singlet.features.doublet`
(the doublet feature set; docs/cell_features.md section 8).

Reads the engine's out dir (called cells, exon + intron count matrices, the engine's `doublet_scores.tsv`, summary.json species),
scores the called cells and rewrites the files so that

  doublet_scores.tsv          the stacker's score (columns barcode, total_umis, doublet_score) when it could be computed,
  doublet_scores_engine.tsv   the engine's own score, always (a withheld, private-sidecar-only file: policy.omitted_layer),
  doublet_stage.json          status, method and version, wall / CPU seconds and peak RSS of this step.

When the stacker cannot score the sample (fewer than 200 called cells, an organism other than human / mouse, an error), there is
no `doublet_scores.tsv`: the layer never mixes two methods. Never raises: a failed step is recorded and the sample goes on.

  doublet_stage.py --out OUT
"""
import argparse
import json
import os
try:
    import resource
except ImportError:   # Windows: no timings
    resource = None
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "python"))

ENGINE = "doublet_scores_engine.tsv"


def _read_mtx(path):
    import scipy.io
    import scipy.sparse as sp
    return sp.csc_matrix(scipy.io.mmread(path))


def _lines(path):
    import gzip
    with gzip.open(path, "rt") as f:
        return [x.rstrip("\n") for x in f]


def _gene_ids(features):
    """Gene id of a feature row: the first underscore field ('ENSG..._NAME_chr:a-b')."""
    return [f.split("\t")[0].split("_")[0] for f in features]


def called_counts(out_dir):
    """-> (X called cells x genes float32 CSR [exon + intron], gene ids, called barcodes)."""
    import numpy as np
    import scipy.sparse as sp
    from singlet.s1.cells import parse_cell_calls

    called = parse_cell_calls(open(os.path.join(out_dir, "cell_calls.tsv"), "rb").read())
    if not called:
        raise ValueError("no called cells")
    parts, genes = [], None
    for kind in ("exon", "intron"):
        mp = os.path.join(out_dir, kind + "_counts.mtx.gz")
        if not os.path.exists(mp):
            continue
        M = _read_mtx(mp)                                    # features x barcodes
        bcs = _lines(os.path.join(out_dir, kind + "_counts_barcodes.tsv.gz"))
        gid = _gene_ids(_lines(os.path.join(out_dir, kind + "_counts_features.tsv.gz")))
        if genes is None:
            genes = sorted(set(gid))
        gi = {g: i for i, g in enumerate(genes)}
        rows = np.fromiter((gi.get(g, -1) for g in gid), np.int64, count=len(gid))
        pos = {b: i for i, b in enumerate(bcs)}
        src = np.fromiter((pos.get(b, -1) for b in called), np.int64, count=len(called))
        if (src < 0).any():
            raise ValueError("%d called barcodes are not columns of %s_counts" % (int((src < 0).sum()), kind))
        ok = rows >= 0
        P = sp.csr_matrix((np.ones(int(ok.sum()), np.float32), (np.flatnonzero(ok), rows[ok])), shape=(len(gid), len(genes)))
        parts.append((M[:, src].T.tocsr() @ P).tocsr())
    if not parts:
        raise ValueError("no count matrices")
    X = parts[0]
    for p in parts[1:]:
        X = (X + p).tocsr()
    return X.astype(np.float32), genes, list(called)


def run(out_dir, log=print):
    import numpy as np
    import pandas as pd

    t0, ru0 = time.time(), resource.getrusage(resource.RUSAGE_SELF) if resource else None
    rec = {"status": "failed", "reason": ""}
    eng = os.path.join(out_dir, "doublet_scores.tsv")
    try:
        if not os.path.exists(eng):
            raise ValueError("no engine doublet_scores.tsv (the stacker takes it as an input)")
        os.replace(eng, os.path.join(out_dir, ENGINE))      # the engine's score is private from here on, whatever happens next
        from singlet.features import doublet as D

        rec.update(method=D.method_version(), score_version=D.SCORE_VERSION)
        summ = json.load(open(os.path.join(out_dir, "summary.json")))
        species = {"human": "human", "mouse": "mouse"}.get(str(summ.get("species") or "").lower())
        X, genes, called = called_counts(out_dir)
        prod = D.read_prod_scores(os.path.join(out_dir, ENGINE), called)
        sample = types.SimpleNamespace(X=X, gene_names=genes, gene_ids=genes, species=species, prod_doublet_score=prod)
        timings = {}
        vals, _extra = D.score_sample(sample, timings=timings)
        score = vals["score_" + D.SCORE_VERSION]
        rec["n_called"] = len(called)
        if not np.isfinite(score).any():
            rec.update(status="skipped", reason="no score: %s" % (
                "organism %r" % summ.get("species") if species is None else "fewer than %d called cells or non-finite inputs" % D.MIN_CELLS))
        else:
            df = pd.DataFrame({"barcode": called, "total_umis": np.asarray(X.sum(axis=1)).ravel().astype(np.int64),
                               "doublet_score": score})
            df.to_csv(eng, sep="\t", index=False, float_format="%.6g", na_rep="NA")
            rec.update(status="scored", n_scored=int(np.isfinite(score).sum()), timings={k: round(v, 2) for k, v in timings.items()})
    except Exception as e:  # noqa: BLE001  never fails the sample; the layer is simply absent
        rec["reason"] = ("%s: %s" % (type(e).__name__, e))[:300]
    ru1 = resource.getrusage(resource.RUSAGE_SELF) if resource else None
    rec.update(wall_s=round(time.time() - t0, 1),
               cpu_s=round((ru1.ru_utime + ru1.ru_stime) - (ru0.ru_utime + ru0.ru_stime), 1) if ru1 else None,
               peak_rss_gb=round(ru1.ru_maxrss / 1e6, 2) if ru1 else None)   # ru_maxrss is KiB on Linux
    with open(os.path.join(out_dir, "doublet_stage.json"), "w") as f:
        json.dump(rec, f, indent=1)
    log("[doublet-stage] %s" % json.dumps(rec)[:400])
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv).out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
