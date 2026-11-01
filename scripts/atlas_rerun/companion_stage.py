#!/usr/bin/env python3
"""Companion stage: after the engine, turn the staged companion libraries of one GEX sample into members of its `.s1`.

Inputs: the engine's `out/` (exon_counts.1pz / cell_calls.tsv give the GEX columns and the called cells) and the sample's claim dir
(manifest.json + companions/<type>/<lib_gsm>/<run>_R{1,2}.fastq.gz, from fetch_worker --companions). Output, under out/companions/:
`<type>.s2` (the package's sidecar object; migrate attaches its members to the sample object) and `<type>.features.parquet`, plus
`status.json`: one record per library with status ok | failed | missing, a reason, wall time and core-seconds per M reads.
A companion that fails or is not supported yet never fails the GEX sample: it is recorded and the modality is simply absent.

  companion_stage.py --out OUT --claim CLAIM --whitelist-dir DIR [--threads N]
"""
import argparse
import json
import os
import resource
import sys
import time

SPECIES = {"Homo sapiens": "human", "Mus musculus": "mouse"}
WHITELIST_BY_R1 = {26: "737K-august-2016.txt", 28: "3M-february-2018.txt"}   # 10x R1 = 16 nt barcode + 10 / 12 nt UMI


def gex_inputs(out_dir):
    """-> (columns, called): barcodes of the sample's GEX columns in col order, and the called-cell barcodes."""
    from singlet.s1.cells import parse_cell_calls
    from singlet.s1.legacy import decode_1pz_bytes

    cols = None
    for name in ("exon_counts.1pz", "intron_counts.1pz"):
        p = os.path.join(out_dir, name)
        if os.path.exists(p):
            c = decode_1pz_bytes(open(p, "rb").read())
            if c.n > 0 and c.colnames:
                cols = list(c.colnames)
                break
    if cols is None:
        raise ValueError("no GEX columns (exon/intron counts missing or empty)")
    called = parse_cell_calls(open(os.path.join(out_dir, "cell_calls.tsv"), "rb").read())
    if called is None:
        raise ValueError("no usable cell_calls.tsv")
    return cols, list(called)


def r1_length(path):
    import gzip

    with gzip.open(path, "rt") as f:
        f.readline()
        return len(f.readline().rstrip("\n"))


def whitelist_codes(whitelist_dir, r1_len):
    import numpy as np
    from singlet.adt import count as AC
    from singlet.adt import seq as S

    name = WHITELIST_BY_R1.get(r1_len)
    if name is None:
        raise ValueError("no barcode whitelist for R1 length %d" % r1_len)
    codes, bad = S.encode_strings(AC.load_barcode_list(os.path.join(whitelist_dir, name)), 16)
    return np.sort(codes[bad == 0]), r1_len - 16


def library_files(claim, lib_type, recs):
    r1, r2, names = [], [], []
    for r in recs:
        d = os.path.join(claim, "companions", lib_type, r["lib_gsm"])
        for run in r["runs"]:
            r1.append(os.path.join(d, run + "_R1.fastq.gz"))
            r2.append(os.path.join(d, run + "_R2.fastq.gz"))
        names.append(r["lib_gsm"])
    return r1, r2, names


def run_vdj(out_dir, claim, recs, cols, called, wl_dir, species, threads):
    import numpy as np
    from singlet.vdj import assemble as A
    from singlet.vdj import library as VL
    from singlet.vdj import members as ST

    r1, r2, names = library_files(claim, "vdj", recs)
    codes, umi_len = whitelist_codes(wl_dir, r1_length(r1[0]))
    res = VL.process_library(r1, r2, gex_columns=cols, gex_called=called, whitelist_codes=codes, species=species,
                             umi_len=umi_len, params=A.Params(threads=threads), libraries=[os.path.basename(p) for p in r1],
                             log=lambda *a: None)
    d = os.path.join(out_dir, "companions")
    os.makedirs(d, exist_ok=True)
    qc = res["qc"]
    if res["members"] is None:
        return {"status": "failed", "reason": "library QC verdict fail: %s" % ",".join(map(str, qc.get("failed", [])))[:200], "qc_verdict": qc.get("verdict")}
    with open(os.path.join(d, "vdj.s2"), "wb") as f:
        f.write(ST.build_sidecar("companion", res["members"]))
    if res["labels"] is not None:
        res["labels"].to_parquet(os.path.join(d, "vdj.features.parquet"), index=False)
    return {"status": "ok", "qc_verdict": qc.get("verdict"), "vdj_cells": qc.get("vdj_cells"), "pairing_rate": qc.get("pairing_rate")}


RUNNERS = {"vdj": run_vdj}


def run_stage(out_dir, claim, wl_dir, threads=4, log=print):
    man = json.load(open(os.path.join(claim, "manifest.json")))
    by_type = {}
    for r in man.get("companions", []):
        by_type.setdefault(r["lib_type"], []).append(r)
    species = SPECIES.get(man.get("organism") or "")
    status = []
    cols = called = None
    for lib_type, recs in sorted(by_type.items()):
        base = {"lib_type": lib_type, "libs": [r["lib_gsm"] for r in recs]}
        bad = [r for r in recs if r.get("status") != "ok"]
        recs = [r for r in recs if r.get("status") == "ok"]
        if bad:
            base["fetch_failed"] = [{"lib_gsm": r["lib_gsm"], "reason": r.get("reason")} for r in bad]
        if not recs:
            status.append(dict(base, status="failed", reason="fetch failed"))
            continue
        runner = RUNNERS.get(lib_type)
        if runner is None:
            status.append(dict(base, status="missing", reason="companion stage for %s not implemented yet" % lib_type))
            continue
        if species is None:
            status.append(dict(base, status="missing", reason="no V(D)J reference for organism %r" % man.get("organism")))
            continue
        t0, ru0 = time.time(), (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
        try:
            if cols is None:
                cols, called = gex_inputs(out_dir)
            rec = runner(out_dir, claim, recs, cols, called, wl_dir, species, threads)
        except Exception as e:  # noqa: BLE001  a companion never fails the GEX sample
            rec = {"status": "failed", "reason": ("%s: %s" % (type(e).__name__, e))[:300]}
        ru1 = (resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN))
        cpu = sum((b.ru_utime + b.ru_stime) - (a.ru_utime + a.ru_stime) for a, b in zip(ru0, ru1))
        reads = sum(r.get("reads", 0) for r in recs)
        rec.update(base, wall_s=round(time.time() - t0, 1), cpu_s=round(cpu, 1), reads=reads,
                   core_s_per_Mreads=round(cpu / (reads / 1e6), 2) if reads else None)
        status.append(rec)
        log("[companion-stage] %s %s" % (lib_type, json.dumps(rec, default=str)[:400]))
    os.makedirs(os.path.join(out_dir, "companions"), exist_ok=True)
    with open(os.path.join(out_dir, "companions", "status.json"), "w") as f:
        json.dump(status, f, indent=1, default=str)
    return status


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--claim", required=True)
    ap.add_argument("--whitelist-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(argv)
    run_stage(a.out, a.claim, a.whitelist_dir, a.threads)
    return 0


if __name__ == "__main__":
    sys.exit(main())
