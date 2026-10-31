# SPDX-License-Identifier: MIT
"""Run the Singlet V(D)J assembler on one library's FASTQs.

    python -P run_vdj.py --fastq-dir DIR [--r1-glob '*_R1_*.fastq.gz'] --whitelist 737K-august-2016.txt
        --species human --out OUT [--threads 16] [--max-reads N]

Writes ``OUT/contigs.csv`` (all contigs, kept or not), ``OUT/cells.csv`` (per-barcode chains, pairing, clonotypes)
and ``OUT/run.json`` (read statistics, timings, peak memory).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import resource
import sys
import time

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir", required=True)
    ap.add_argument("--r1-glob", default="*_R1_*.fastq.gz")
    ap.add_argument("--whitelist", required=True)
    ap.add_argument("--species", default="human")
    ap.add_argument("--ref", default=None, help="reference TSV (default: the packaged one for --species)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--max-reads", type=int, default=None)
    ap.add_argument("--umi-len", type=int, default=None)
    ap.add_argument("--param", action="append", default=[], help="assembler parameter override, e.g. umi_read_frac=0.02")
    a = ap.parse_args(argv)

    from singlet.adt import count as AC
    from singlet.adt import seq as S
    from singlet.vdj import assemble as A
    from singlet.vdj import clonotype as CL

    os.makedirs(a.out, exist_ok=True)
    r1 = sorted(glob.glob(os.path.join(a.fastq_dir, "**", a.r1_glob), recursive=True))
    r2 = [p.replace("_R1_", "_R2_") for p in r1]
    if not r1 or not all(os.path.exists(p) for p in r2):
        sys.exit(f"no R1/R2 pairs under {a.fastq_dir}")
    print(f"[run_vdj] {len(r1)} file pairs", flush=True)
    import dnaio

    with dnaio.open(r1[0]) as f:
        r1_len = len(next(iter(f)).sequence)
    umi_len = a.umi_len or (12 if r1_len in (28, 29, 30, 31) else 10)
    t0 = time.time()
    wl = AC.load_barcode_list(a.whitelist)
    wl_codes, bad = S.encode_strings(wl, 16)
    wl_codes = np.sort(wl_codes[bad == 0])
    ref = A.Reference.load(a.ref or a.species)
    params = A.Params(threads=a.threads)
    for kv in a.param:
        k, v = kv.split("=", 1)
        t = type(getattr(params, k))
        setattr(params, k, v.lower() in ("1", "true", "yes") if t is bool else t(v))
    res = A.run_library(r1, r2, ref=ref, cb_len=16, umi_len=umi_len, whitelist_codes=wl_codes,
                        params=params, max_reads=a.max_reads,
                        log=lambda *x: print(*x, flush=True))
    res.contigs.to_csv(os.path.join(a.out, "contigs.csv"), index=False)
    cells = CL.assign_clonotypes(CL.cell_table(res.contigs, tie_reads=params.tie_reads))
    cells.to_csv(os.path.join(a.out, "cells.csv"), index=False)
    run = {"fastqs": [os.path.basename(p) for p in r1], "r1_len": r1_len, "umi_len": umi_len, "stats": res.stats,
           "params": vars(params),
           "timings_s": res.timings, "wall_s": time.time() - t0, "threads": a.threads,
           "peak_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6}
    with open(os.path.join(a.out, "run.json"), "w") as f:
        json.dump(run, f, indent=1, default=str)
    print(json.dumps({k: run[k] for k in ("wall_s", "peak_rss_gb")}), res.stats.get("contigs_kept"), flush=True)


if __name__ == "__main__":
    main()
