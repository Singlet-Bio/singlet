"""Are the junction tables of two runs the same once rows are matched by junction identity instead of row number?

    python determinism_sj.py DIR_A DIR_B

Compares sj_counts (feature x cell, junction name from sj_counts_features.tsv.gz, cell barcode from sj_counts_barcodes.tsv.gz) and
splice_psi, and prints the small non-timing diffs of rrna_report.json and read_stats.tsv. Needs numpy and pandas.
"""
import gzip
import io
import sys
from pathlib import Path

import pandas as pd


def load(d, stem):
    d = Path(d)
    feats = [ln.split("\t")[0] for ln in gzip.open(d / f"{stem}_features.tsv.gz", "rt").read().split("\n") if ln]
    bcs = [ln.split("\t")[0] for ln in gzip.open(d / f"{stem}_barcodes.tsv.gz", "rt").read().split("\n") if ln]
    txt = gzip.open(d / f"{stem}.mtx.gz", "rt").read().split("\n")
    body = [ln for ln in txt if ln and not ln.startswith("%")][1:]
    m = pd.read_csv(io.StringIO("\n".join(body)), sep=r"\s+", header=None, names=["r", "c", "v"])
    m["feature"] = pd.Series(feats).reindex(m.r - 1).to_numpy()
    m["cell"] = pd.Series(bcs).reindex(m.c - 1).to_numpy()
    return m, feats, bcs


def main():
    A, B = sys.argv[1:3]
    for stem in ("sj_counts", "splice_psi"):
        a, fa, ba = load(A, stem)
        b, fb, bb = load(B, stem)
        print(f"== {stem}: features A {len(fa):,} B {len(fb):,}; same set {set(fa) == set(fb)}; same order {fa == fb}; barcodes same order {ba == bb}")
        x = a.merge(b, on=["feature", "cell"], how="outer", suffixes=("_a", "_b"), indicator=True)
        both = x[x._merge == "both"]
        print(f"   entries A {len(a):,} B {len(b):,}; only A {int((x._merge == 'left_only').sum()):,}; only B {int((x._merge == 'right_only').sum()):,}; "
              f"value differs {int((both.v_a != both.v_b).sum()):,}; sum A {a.v.sum():.4f} B {b.v.sum():.4f}")
    for f in ("rrna_report.json", "read_stats.tsv"):
        la = (Path(A) / f).read_text().splitlines()
        lb = (Path(B) / f).read_text().splitlines()
        print(f"== {f}")
        for x, y in zip(la, lb):
            if x != y:
                print("   A:", x[:200], "\n   B:", y[:200])


if __name__ == "__main__":
    main()
