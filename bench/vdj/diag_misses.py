# SPDX-License-Identifier: MIT
"""Why Cell Ranger contigs are missed: the reads of the missed cells that hold (part of) the junction.

    python -P diag_misses.py FASTQ_DIR filtered_contig_annotations.csv contigs.csv [--n 300]

For up to ``n`` missed (barcode, chain, junction): the read pairs of that barcode (exact R1 barcode) whose R2 or its
reverse complement shares a 15-mer with the junction, their UMIs, and whether a read spans the whole junction
(+ 10 nt on each side). Prints a summary and a few examples.
"""
import argparse
import glob
import gzip
import os
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import compare_vdj as CV  # noqa: E402

_C = str.maketrans("ACGTN", "TGCAN")


def rc(s):
    return s.translate(_C)[::-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fastq_dir")
    ap.add_argument("ref")
    ap.add_argument("contigs")
    ap.add_argument("--n", type=int, default=300)
    a = ap.parse_args()
    ref = CV.load_cellranger(a.ref)
    m = CV.load_singlet(a.contigs)
    K = set(zip(m.barcode, m.chain, m.cdr3_nt))
    miss = [(b, c, n) for b, c, n in zip(ref.barcode, ref.chain, ref.cdr3_nt) if (b, c, n) not in K][: a.n]
    want = defaultdict(list)
    for b, c, n in miss:
        want[b[:16]].append(n)
    kms = {}
    for b, ns in want.items():
        kms[b] = [(n, {n[i:i + 15] for i in range(len(n) - 14)}) for n in ns]
    hits = defaultdict(lambda: {"reads": 0, "umis": set(), "span": 0, "lens": []})
    r1s = sorted(glob.glob(os.path.join(a.fastq_dir, "**", "*_R1_*.fastq.gz"), recursive=True))
    for p1 in r1s:
        p2 = p1.replace("_R1_", "_R2_")
        with gzip.open(p1, "rt") as f1, gzip.open(p2, "rt") as f2:
            for i, (x, y) in enumerate(zip(f1, f2)):
                if i % 4 != 1:
                    continue
                b = x[:16]
                if b not in kms:
                    continue
                y = y.strip()
                s = rc(y)
                for n, ks in kms[b]:
                    if any(s[j:j + 15] in ks for j in range(0, len(s) - 14, 3)):
                        h = hits[(b, n)]
                        h["reads"] += 1
                        h["umis"].add(x[16:26])
                        h["lens"].append(len(s))
                        k = s.find(n)
                        if k >= 10 and k + len(n) + 10 <= len(s):
                            h["span"] += 1
    rows = []
    for b, c, n in miss:
        h = hits.get((b[:16], n), {"reads": 0, "umis": set(), "span": 0, "lens": []})
        rows.append({"barcode": b, "chain": c, "len": len(n), "reads": h["reads"], "umis": len(h["umis"]), "spanning": h["span"]})
    d = pd.DataFrame(rows)
    print(d.describe().T)
    print("no junction reads:", (d.reads == 0).mean(), " no spanning read:", (d.spanning == 0).mean(),
          " 1 UMI:", (d.umis == 1).mean())
    print(d.sort_values("reads").head(10).to_string())
    print(d.sort_values("reads").tail(10).to_string())


if __name__ == "__main__":
    main()
