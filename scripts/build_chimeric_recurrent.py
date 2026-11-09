# SPDX-License-Identifier: MIT
"""Recurrent chimeric gene pairs from samples that carry no fusions (non-malignant tissue), for the chimeric table's `filter` column.

A gene pair is recurrent when its molecules (canonical-motif rows, all barcodes, read-through and same-gene rows left out as the
evaluation does) reach --min-umis in at least --min-samples of the given samples. Pairs are unordered and written once.

    python scripts/build_chimeric_recurrent.py --min-umis 2 --min-samples 2 --out data/fusion/chimeric_recurrent_GRCh38.tsv \
        pbmc5p_v1=OUT1/chimeric_junctions.tsv.gz pbmc1k_v3=OUT2/chimeric_junctions.tsv.gz ...
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip


def pairs(path, min_dist):
    tot = collections.Counter()
    with gzip.open(path, "rt") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            a, b = r["donor_gene"], r["acceptor_gene"]
            if a == "." or b == "." or a == b or int(r["junction_type"]) < 1:
                continue
            if r["donor_chr"] == r["acceptor_chr"] and abs(int(r["donor_pos"]) - int(r["acceptor_pos"])) < min_dist:
                continue
            tot[tuple(sorted((a, b)))] += int(r["n_umis"])
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("samples", nargs="+", help="name=chimeric_junctions.tsv.gz")
    ap.add_argument("--min-umis", type=int, default=2)
    ap.add_argument("--min-samples", type=int, default=2)
    ap.add_argument("--min-dist", type=int, default=1_000_000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    seen = collections.defaultdict(list)
    names = []
    for s in a.samples:
        name, path = s.split("=", 1)
        names.append(name)
        for p, n in pairs(path, a.min_dist).items():
            if n >= a.min_umis:
                seen[p].append(name)
    rec = sorted(p for p, v in seen.items() if len(v) >= a.min_samples)
    with open(a.out, "w") as o:
        o.write(f"# recurrent chimeric gene pairs: >= {a.min_umis} molecules in >= {a.min_samples} of {len(names)} samples without fusions: "
                + ",".join(names) + "\n")
        o.write("gene_a\tgene_b\tsamples\n")
        for p in rec:
            o.write(f"{p[0]}\t{p[1]}\t{','.join(seen[p])}\n")
    print(len(rec), "recurrent pairs")


if __name__ == "__main__":
    main()
