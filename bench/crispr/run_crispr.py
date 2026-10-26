# SPDX-License-Identifier: MIT
"""Count guide UMIs of one CRISPR guide-capture library and assign guides to cells with every caller.

    python -P run_crispr.py --fastq-dir DIR --guides feature_reference.csv | --guide-table guides.tsv
        --whitelist WL --cells barcodes.tsv[.gz] --out OUT [--translation translation.txt.gz] [--threads 8]

``--cells``: the GEX columns (one barcode per line, Cell Ranger ``barcodes.tsv`` or a cells table); the first
``--called`` of them (or those listed in ``--called-file``) are the called cells. Writes ``OUT/counts.npz`` (guides x
columns, ambient profile), ``OUT/assign_<caller>.csv`` per caller and ``OUT/run.json``.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import resource
import sys
import time

import numpy as np


def read_list(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return [ln.split()[0].split(",")[0] for ln in f if ln.strip() and not ln.startswith(("barcode", "#"))]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir", required=True)
    ap.add_argument("--r1-glob", default="*_R1_*.fastq.gz")
    ap.add_argument("--guides", help="Cell Ranger feature reference CSV")
    ap.add_argument("--guide-table", help="TSV with id, sequence, target_gene (+ name, kind)")
    ap.add_argument("--pattern", default=None, help="override the feature-reference pattern, e.g. '(BC)GTTTAAGAGC'")
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--translation", default=None)
    ap.add_argument("--cells", required=True)
    ap.add_argument("--called-file", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-reads", type=int, default=None)
    ap.add_argument("--umi-len", type=int, default=None)
    ap.add_argument("--shards", type=int, default=0)
    ap.add_argument("--tmp", default=None)
    ap.add_argument("--reuse-counts", action="store_true", help="assign again from OUT/counts.npz (no counting)")
    a = ap.parse_args(argv)

    from singlet.adt import count as AC
    from singlet.adt import seq as S
    from singlet.crispr import assign as AS
    from singlet.crispr import count as CC
    from singlet.crispr import guides as G

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    r1 = sorted(glob.glob(os.path.join(a.fastq_dir, "**", a.r1_glob), recursive=True))
    r2 = [p.replace("_R1_", "_R2_") for p in r1]
    if not r1:
        sys.exit("no FASTQs")
    if a.guides:
        lib = G.from_feature_reference(a.guides)
    else:
        with open(a.guide_table) as f:
            lib = G.from_table(csv.DictReader(f, delimiter="\t"), source=os.path.basename(a.guide_table))
    import dnaio

    sample = []
    with dnaio.open(r1[0], file2=r2[0]) as f:
        for i, (x, y) in enumerate(f):
            sample.append((x.sequence.encode(), y.sequence.encode()))
            if i >= 200_000:
                break
    r1_len = len(sample[0][0])
    umi_len = a.umi_len or (12 if r1_len >= 28 else 10)
    L = max(len(s) for s in lib.sequences)
    pattern = a.pattern or lib.pattern
    det = G.detect_geometry([y for _, y in sample[:50_000]], L, lib.sequences)
    geom = det["geometry"]
    if pattern:
        pg = G.parse_pattern(pattern, L)
        # the pattern's anchor / offset, the strand from the reads
        geom = G.Geometry(L, pg.anchor, pg.offset, geom.reverse if geom.anchor == pg.anchor else False)
    print(f"[run_crispr] {len(lib)} guides; geometry {geom.describe()}; detected {det['candidates'][:3]}", flush=True)
    wl = AC.load_barcode_list(a.whitelist)
    wl_codes, bad = S.encode_strings(wl, 16)
    wl_codes = np.sort(wl_codes[bad == 0])
    cols = read_list(a.cells)
    called = set(read_list(a.called_file)) if a.called_file else set(cols)
    gex_codes, _ = S.encode_strings([c.split("-")[0] for c in cols], 16)
    trans, trep = None, {"use": False}
    if a.translation:
        cb = S.encode_strings([x[:16] for x, _ in sample], 16)[0]
        trans, trep = AC.best_translation(cb, np.sort(gex_codes), a.translation)
    print(f"[run_crispr] translation {trep}", flush=True)
    t1 = time.time()
    npz = os.path.join(a.out, "counts.npz")
    prev = os.path.join(a.out, "run.json")
    if a.reuse_counts and os.path.exists(npz) and os.path.exists(prev):
        # the callers changed, not the counting: reuse the counts (and their read statistics and timing)
        M, feats, amb, outside, meta = AC.load_npz(npz)
        lc = AC.LibraryCounts(feats, M, amb, outside, meta["stats"], meta.get("unmatched", []))
        t_count = json.load(open(prev))["timings_s"]["count_s"]
    else:
        lc = CC.count_library(r1, r2, library=lib, geometry=geom, cb_len=16, umi_len=umi_len, whitelist_codes=wl_codes,
                              gex_barcodes=cols, translation=trans, max_reads=a.max_reads, shards=a.shards, tmpdir=a.tmp)
        t_count = time.time() - t1
        AC.save_npz(npz, lc, {"geometry": geom.describe(), "translation": trep})
    cmask = np.array([c in called for c in cols])
    Mc = lc.matrix[:, np.flatnonzero(cmask)]
    p = AS.ambient_profile(lc.ambient)
    # ambient load of a droplet: the uncalled GEX columns when there are many (empty droplets), else the barcodes
    # outside the columns (libraries whose columns are only the published cells)
    unc = np.flatnonzero(~cmask)
    if len(unc) >= 1000:
        b_empty = AS.ambient_load(np.asarray(lc.matrix[:, unc].sum(axis=0)).ravel())
        b_src = f"0.95 quantile of {len(unc)} uncalled columns"
    else:
        b_empty = float(np.median(lc.outside_totals)) if len(lc.outside_totals) else 0.0
        b_src = f"median of {len(lc.outside_totals)} barcodes outside the columns with >= 1 molecule"
    print(f"[run_crispr] ambient load per droplet {b_empty:.2f} ({b_src})", flush=True)
    timings = {"count_s": t_count}
    res = {}
    for name, fn in (("singlet", lambda: AS.call_ambient(Mc, p, b_empty)), ("cellranger_gmm", lambda: AS.call_gmm(Mc)),
                     ("poisson_gaussian", lambda: AS.call_poisson_gaussian(Mc))):
        t2 = time.time()
        C = fn()
        timings[f"assign_{name}_s"] = time.time() - t2
        lab = AS.summarize(C, lib.ids, lib.target_gene)
        lab.insert(0, "barcode", [c for c, m in zip(cols, cmask) if m])
        lab["umis"] = np.asarray(Mc.sum(axis=0)).ravel()
        lab.to_csv(os.path.join(a.out, f"assign_{name}.csv"), index=False)
        res[name] = lab["class"].value_counts().to_dict()
    run = {"guides": len(lib), "geometry": geom.describe(), "detected": det["candidates"], "translation": trep,
           "ambient_load": b_empty, "ambient_load_source": b_src,
           "stats": lc.stats, "called_cells": int(cmask.sum()), "classes": res, "timings_s": timings,
           "wall_s": time.time() - t0, "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6,
           "unmatched_top": lc.unmatched[:20]}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    print(json.dumps({k: run[k] for k in ("classes", "timings_s", "wall_s")}, default=str), flush=True)


if __name__ == "__main__":
    main()
