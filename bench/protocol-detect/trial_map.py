"""Trial mapping: the referee between close protocol candidates.

Runs the released campaign engine (CPU binary of the build) on a head subsample of about 1 M spots under each candidate engine
tag (`singlet download --protocol TAG`, then a plain run without the pipeline extras) and picks by the share of input reads that
become counted molecules in called cells, then by genes per cell and the knee.  The winner must beat the runner-up by
``WIN_RATIO``; otherwise the result is ``undecided`` and the caller keeps the detector's order.

    python trial_map.py --build B --seg-bc BC.fq --seg-cdna CDNA.fq --tags 10x-3p-v2,10x-5p-v2 --genome GRCh38-2024-A
                        --work DIR [--threads 4] [--json OUT]

Run it inside a Slurm job (CPU partition, about 40 GB for the human STAR index).  It writes nothing outside --work.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
import time

REF = os.environ.get("SINGLET_REF_BASE", "/mnt/projects/debruinz_project/cellarium/reference")
# The winner's usable share must be at least this multiple of the runner-up's.  A wrong 3'/5' or version call loses most valid
# barcodes or most gene-assigned reads (measured on the development half: see docs/PROTOCOL_DETECTION_2026-11.md).
WIN_RATIO = 1.25


def _solo_summary(out: str) -> dict:
    """STARsolo Summary.csv of the Gene (or GeneFull) feature, as {key: float}, if the engine left one."""
    for pat in ("**/Solo.out/Gene/Summary.csv", "**/Solo.out/GeneFull/Summary.csv", "**/Summary.csv"):
        for p in glob.glob(os.path.join(out, pat), recursive=True):
            d = {}
            for row in csv.reader(open(p)):
                if len(row) >= 2:
                    try:
                        d[row[0].strip()] = float(row[1])
                    except ValueError:
                        pass
            if d:
                return d
    return {}


def _knee(out: str) -> dict:
    """Knee shape from the per-barcode UMI totals of the raw matrix, when a Solo raw barcodes/matrix pair is present: the ratio
    of the median UMI of the top 500 barcodes to the median of barcodes 2000-5000 (a real knee gives a large ratio)."""
    for m in glob.glob(os.path.join(out, "**/Solo.out/Gene/raw/matrix.mtx"), recursive=True):
        tot = {}
        with open(m) as f:
            for line in f:
                if line.startswith("%"):
                    continue
                break
            for line in f:
                p = line.split()
                if len(p) >= 3:
                    tot[p[1]] = tot.get(p[1], 0) + float(p[2])
        v = sorted(tot.values(), reverse=True)
        if len(v) < 600:
            return {"barcodes": len(v)}
        top = v[min(250, len(v) - 1)]
        low = v[min(3500, len(v) - 1)]
        return {"barcodes": len(v), "umi_rank250": top, "umi_rank3500": low, "knee_ratio": round(top / max(low, 1.0), 2)}
    return {}


def run_tag(build: str, bc: str, cdna: str, tag: str, genome: str, work: str, threads: int) -> dict:
    eng = os.path.join(build, "bin", "singlet-cpu")
    d = os.path.join(work, tag)
    os.makedirs(d, exist_ok=True)
    fq = os.path.join(d, "trial.1fq")
    t0 = time.time()
    r = subprocess.run([eng, "download", "--r1", bc, "--r2", cdna, "--protocol", tag, "-o", fq], capture_output=True, text=True)
    res = {"tag": tag, "download_exit": r.returncode, "download_tail": (r.stderr or "")[-400:]}
    if r.returncode != 0 or not os.path.isfile(fq):
        res["usable"] = 0.0
        return res
    out = os.path.join(d, "out")
    gdir = os.path.join(REF, genome, "star_2.7.11b")
    gtf = os.path.join(REF, genome, "genes", "genes.gtf")
    cmd = [eng, fq, "--genome-dir", gdir, "--exons", gtf, "--out-prefix", out + "/", "--threads", str(threads),
           "--genome-shared", "off", "--ref-base", REF, "--output-format", "mtx", "--no-hla", "--no-editing"]
    r = subprocess.run(cmd, capture_output=True, text=True, env=dict(os.environ, SINGLET_ENGINE="cpu"))
    res.update({"run_exit": r.returncode, "run_tail": (r.stderr or "")[-600:], "seconds": round(time.time() - t0, 1)})
    summ = {}
    sp = os.path.join(out, "summary.json")
    if os.path.isfile(sp):
        try:
            summ = json.load(open(sp))
        except ValueError:
            pass
    solo = _solo_summary(out)
    res["summary"] = {k: summ.get(k) for k in ("protocol_name", "n_input_reads", "uniquely_mapped_pct", "n_cells_called",
                                               "median_umi_per_cell", "median_genes_per_cell", "strand_auto_flipped")}
    res["solo"] = solo
    res["knee"] = _knee(out)
    valid = solo.get("Reads With Valid Barcodes")
    gene = solo.get("Reads Mapped to Gene: Unique Gene") or solo.get("Reads Mapped to GeneFull: Unique GeneFull")
    incell = solo.get("Fraction of Unique Reads in Cells")
    if valid is not None and gene is not None and incell is not None:
        res["usable"] = round(valid * gene * incell, 4)
        res["usable_basis"] = "valid barcode x unique gene x in cells (STARsolo Summary.csv)"
    else:   # fall back on the engine summary
        um = (summ.get("uniquely_mapped_pct") or 0) / (100.0 if (summ.get("uniquely_mapped_pct") or 0) > 1 else 1.0)
        res["usable"] = round(um * min(1.0, (summ.get("n_cells_called") or 0) / 500.0), 4)
        res["usable_basis"] = "unique mapping x min(1, cells/500) (engine summary)"
    res["genes_per_cell"] = summ.get("median_genes_per_cell") or solo.get("Median Gene per Cell") or 0
    return res


def referee(results: list) -> dict:
    ok = sorted(results, key=lambda r: (-(r.get("usable") or 0), -(r.get("genes_per_cell") or 0)))
    if not ok or (ok[0].get("usable") or 0) <= 0:
        return {"chosen_tag": "", "verdict": "undecided", "why": "no candidate produced counted reads"}
    a = ok[0]
    b = ok[1] if len(ok) > 1 else None
    if b is None:
        return {"chosen_tag": a["tag"], "verdict": "single", "why": f"only candidate: usable {a['usable']}"}
    ratio = (a["usable"] or 0) / max(b.get("usable") or 0, 1e-6)
    if ratio >= WIN_RATIO:
        return {"chosen_tag": a["tag"], "verdict": "clear", "ratio": round(ratio, 2),
                "why": f"{a['tag']} usable {a['usable']} vs {b['tag']} {b.get('usable')} (x{ratio:.2f}); genes/cell "
                       f"{a.get('genes_per_cell')} vs {b.get('genes_per_cell')}"}
    # Tied usable shares: the engine sets the strand itself (forcing 10x-3p-vN or 10x-5p-vN gives the same counts), so when every
    # run names the same engine protocol and it is a candidate, that call settles 3' against 5'.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # python -I leaves the script folder off the path
    import diagnose
    calls = {diagnose.norm_engine_protocol((r.get("summary") or {}).get("protocol_name") or "") for r in results if r.get("usable")}
    tags = {r["tag"] for r in results}
    if len(calls) == 1 and next(iter(calls)) in tags:
        c = next(iter(calls))
        return {"chosen_tag": c, "verdict": "engine_call", "ratio": round(ratio, 2),
                "why": f"usable shares tie (x{ratio:.2f}); every trial run reports engine protocol {c}"}
    return {"chosen_tag": "", "verdict": "undecided", "ratio": round(ratio, 2),
            "why": f"{a['tag']} usable {a['usable']} vs {b['tag']} {b.get('usable')} (x{ratio:.2f} < {WIN_RATIO})"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--seg-bc", required=True)
    ap.add_argument("--seg-cdna", required=True)
    ap.add_argument("--tags", required=True)
    ap.add_argument("--genome", default="GRCh38-2024-A")
    ap.add_argument("--work", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--json", default="")
    ap.add_argument("--id", default="")
    a = ap.parse_args(argv)
    results = [run_tag(a.build, a.seg_bc, a.seg_cdna, t, a.genome, a.work, a.threads) for t in a.tags.split(",") if t]
    out = {"schema": "trial-map/1", "id": a.id, "tags": a.tags.split(","), "results": results, "referee": referee(results)}
    txt = json.dumps(out, indent=1, default=str)
    if a.json:
        open(a.json, "w").write(txt)
    print(txt)


if __name__ == "__main__":
    sys.exit(main())
