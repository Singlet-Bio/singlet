# SPDX-License-Identifier: MIT
"""Development libraries only: the raw reads behind reference chains Singlet does not report at all.

    python -P diag_reads.py --fastq-dir DIR --ref filtered_contig_annotations.csv --contigs singlet/contigs.csv
        --run singlet/run.json --chains TCR [--n 400] [--out x.json]

For up to ``n`` reference top contigs (cell, chain) of reference-paired cells for which Singlet has no contig of that chain
in that barcode (whatever its filters), the read pairs whose R1 barcode is the cell's (exact) and whose insert (R2 reverse
complement, and the R1 insert) shares a 15-mer with the reference junction: per UMI its read count, compared with
Singlet's per-library molecule read threshold (``umi_read_threshold`` of run.json). Classes per chain: ``no_reads``,
``all_umis_below_threshold``, ``some_umi_above_threshold``.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "vdj"))
import compare_vdj as CV  # noqa: E402

_C = str.maketrans("ACGTN", "TGCAN")
PAIRS = [("TRA", "TRB"), ("TRG", "TRD"), ("IGH", "IGK"), ("IGH", "IGL")]


def rc(s):
    return s.translate(_C)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--contigs", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--chains", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    keep = CV.TCR if a.chains == "TCR" else CV.BCR
    run = json.load(open(a.run))
    thr = run["stats"]["umi_read_threshold"]
    umi_len = run["umi_len"]
    ref = CV.load_cellranger(a.ref)
    ref = ref[ref.chain.isin(keep)]
    rt = ref.sort_values("umis", ascending=False, kind="mergesort").drop_duplicates(["barcode", "chain"])
    rtop = {(b, c): (n, u) for b, c, n, u in zip(rt.barcode, rt.chain, rt.cdr3_nt, rt.umis)}
    allc = pd.read_csv(a.contigs)
    have = set(zip(allc.barcode, allc.chain))
    miss = []
    for bc in sorted({b for b, _ in rtop}):
        for x, y in PAIRS:
            if (bc, x) in rtop and (bc, y) in rtop:
                miss += [(bc, c, rtop[(bc, c)][0], rtop[(bc, c)][1]) for c in (x, y) if (bc, c) not in have]
                break
    tot = len(miss)
    miss = miss[: a.n]
    want = defaultdict(list)
    for b, c, n, u in miss:
        want[b[:16]].append(n)
    kms = {b: [(n, {n[i:i + 15] for i in range(len(n) - 14)}) for n in ns] for b, ns in want.items()}
    umis = defaultdict(Counter)
    r1s = sorted(glob.glob(os.path.join(a.fastq_dir, "**", "*_R1_*.fastq.gz"), recursive=True))
    cbu = 16 + umi_len
    for p1 in r1s:
        p2 = p1.replace("_R1_", "_R2_")
        with gzip.open(p1, "rt") as f1, gzip.open(p2, "rt") as f2:
            for i, (x, y) in enumerate(zip(f1, f2)):
                if i % 4 != 1:
                    continue
                b = x[:16]
                if b not in kms:
                    continue
                ins = (rc(y.strip()), x.strip()[cbu:])
                for n, ks in kms[b]:
                    if any(s[j:j + 15] in ks for s in ins for j in range(0, len(s) - 14, 2)):
                        umis[(b, n)][x[16:cbu]] += 1
    rows = []
    for b, c, n, u in miss:
        cnt = umis.get((b[:16], n), Counter())
        mx = max(cnt.values()) if cnt else 0
        k = "no_reads" if not cnt else ("all_umis_below_threshold" if mx < thr else "some_umi_above_threshold")
        rows.append({"chain": c, "class": k, "ref_umis": u, "umis_seen": len(cnt), "reads": sum(cnt.values()), "max_reads_per_umi": mx,
                     "umis_above_thr": sum(v >= thr for v in cnt.values())})
    d = pd.DataFrame(rows)
    out = {"missing_chains_total": tot, "examined": len(d), "umi_read_threshold": thr,
           "classes": d["class"].value_counts().to_dict() if len(d) else {},
           "by_class_median": d.groupby("class")[["ref_umis", "umis_seen", "reads", "max_reads_per_umi", "umis_above_thr"]].median().to_dict("index") if len(d) else {},
           "max_reads_per_umi_quantiles": d.max_reads_per_umi.quantile([.1, .25, .5, .75, .9]).to_dict() if len(d) else {}}
    s = json.dumps(out, indent=1, default=str)
    print(s)
    if a.out:
        open(a.out, "w").write(s)


if __name__ == "__main__":
    main()
