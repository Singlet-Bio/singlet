# SPDX-License-Identifier: MIT
"""Guide-barcode table for Norman 2019 (GSE133344) gem group 1, learnt on half of the cells.

    python -P prep_norman.py DATA_DIR [--gemgroup 1] [--reads 4000000]

Perturb-seq identifies each guide pair by an 18-nt guide barcode (GBC) in the vector transcript, read in R2 before
``CCTAGAGGGCCCG``. GEO gives every cell's identity (``GSE133344_filtered_cell_identities.csv.gz``) but no GBC sequences.
The barcode -> identity table is learnt from the *training* half of the published cells (barcode hash even): each
training cell's dominant GBC votes for the cell's identity, and a GBC takes the identity most of its cells have. All
callers are then scored on the *test* half only (``test_cells.txt``), so the table never saw the cells it is judged on.

Writes ``guides.tsv`` (id, sequence, target_gene, kind), ``reference_gg<N>.csv`` (barcode, guide_ids, target, split)
and ``test_cells.txt``.
"""
import argparse
import glob
import gzip
import hashlib
from collections import Counter, defaultdict

import pandas as pd

ANCHOR = "CCTAGAGGGCCCG"
L = 18


def targets(identity):
    """'ARID1A_NegCtrl0__ARID1A_NegCtrl0' -> ['ARID1A']; 'CEBPE_RUNX1T1__...' -> ['CEBPE', 'RUNX1T1']."""
    first = identity.split("__")[0]
    return sorted({g for g in first.split("_") if g and not g.startswith("NegCtrl")})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("--gemgroup", type=int, default=1)
    ap.add_argument("--reads", type=int, default=6_000_000)
    a = ap.parse_args()
    ident = pd.read_csv(f"{a.data}/GSE133344_filtered_cell_identities.csv.gz")
    ident = ident[ident["gemgroup"] == a.gemgroup]
    ident["barcode"] = ident["cell_barcode"].str.split("-").str[0] + "-1"
    ident["split"] = ["train" if int(hashlib.sha1(b.encode()).hexdigest(), 16) % 2 == 0 else "test" for b in ident["barcode"]]
    want = dict(zip(ident["barcode"].str[:16], ident["split"]))
    # dominant GBC per published cell from the reads (exact cell barcodes are enough to learn names)
    per_cell = defaultdict(Counter)
    n = 0
    r1s = sorted(glob.glob(f"{a.data}/fastqs/*_R1_*.fastq.gz"))
    for r1 in r1s:
        r2 = r1.replace("_R1_", "_R2_")
        with gzip.open(r1, "rt") as f1, gzip.open(r2, "rt") as f2:
            for i, (x, y) in enumerate(zip(f1, f2)):
                if i % 4 != 1:
                    continue
                cb = x[:16]
                if cb in want:
                    j = y.find(ANCHOR)
                    if j >= L:
                        per_cell[cb][y[j - L:j]] += 1
                n += 1
                if n >= a.reads:
                    break
    votes = defaultdict(Counter)
    for b, idt, sp in zip(ident["barcode"].str[:16], ident["guide_identity"], ident["split"]):
        if sp == "train" and per_cell.get(b):
            gbc, _ = per_cell[b].most_common(1)[0]
            votes[gbc][idt] += 1
    rows = []
    for gbc, c in votes.items():
        idt, k = c.most_common(1)[0]
        if k >= 3 and k >= 0.8 * sum(c.values()) and "N" not in gbc:
            t = targets(idt)
            rows.append({"id": f"{idt}:{gbc[:8]}", "sequence": gbc, "target_gene": "+".join(t) if t else "non-targeting",
                         "kind": "non-targeting" if not t else "targeting", "identity": idt, "train_cells": k})
    g = pd.DataFrame(rows).sort_values("id")
    g.to_csv(f"{a.data}/guides.tsv", sep="\t", index=False)
    ident["target"] = ["+".join(targets(x)) or "non-targeting" for x in ident["guide_identity"]]
    ident.rename(columns={"guide_identity": "guide_ids"})[["barcode", "guide_ids", "target", "split"]].to_csv(
        f"{a.data}/reference_gg{a.gemgroup}.csv", index=False)
    ident.loc[ident["split"] == "test", "barcode"].to_csv(f"{a.data}/test_cells.txt", index=False, header=False)
    print(f"{len(ident)} published cells ({(ident.split == 'test').sum()} test); {len(g)} GBCs named for "
          f"{g.identity.nunique()} of {ident.guide_ids.nunique()} identities; read pairs scanned {n:,}")


if __name__ == "__main__":
    main()
