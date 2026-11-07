"""Compare two Singlet output directories file by file and say exactly what differs.

    python compare_dirs.py DIR_A DIR_B

For every file present in both (logs and timing files skipped): identical content (gzip-decoded) or not; for Matrix Market
files the number of (row, column) entries that differ in value, are only in A or only in B, and the total count (UMIs) in each;
for other tables the number of rows that differ. Output is plain text, one block per differing file. Needs numpy and pandas.
"""
import gzip
import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SKIP = ("run.log", "Log.", "star_Log", "time", "mem.json", "node.txt", "encode", "provenance.json", "summary.json", "pileup_stats.json",
        "read_qc", "metrics_summary", "saturation", "polya_sites", "star_Solo.out/Gene/raw/matrix.mtx.unused")


def opener(p):
    return gzip.open(p, "rb") if str(p).endswith(".gz") else open(p, "rb")


def digest(p):
    h = hashlib.md5()
    with opener(p) as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_mtx(p):
    with opener(p) as fh:
        txt = fh.read().decode()
    lines = [ln for ln in txt.split("\n") if ln and not ln.startswith("%")]
    body = pd.read_csv(io.StringIO("\n".join(lines[1:])), sep=r"\s+", header=None, names=["r", "c", "v"])
    return body


def cmp_mtx(a, b):
    A, B = read_mtx(a), read_mtx(b)
    m = A.merge(B, on=["r", "c"], how="outer", suffixes=("_a", "_b"), indicator=True)
    both = m[m._merge == "both"]
    diff = both[both.v_a != both.v_b]
    return (f"entries A {len(A):,} B {len(B):,}; only in A {int((m._merge == 'left_only').sum()):,}, only in B {int((m._merge == 'right_only').sum()):,}, "
            f"value differs {len(diff):,}; total count A {A.v.sum():,.0f} B {B.v.sum():,.0f} (difference {A.v.sum() - B.v.sum():+,.0f}); "
            f"rows(features) touched by a difference {int(pd.concat([m[m._merge != 'both'].r, diff.r]).nunique())}, columns(cells) {int(pd.concat([m[m._merge != 'both'].c, diff.c]).nunique())}")


def cmp_table(a, b):
    with opener(a) as fa, opener(b) as fb:
        la, lb = fa.read().decode().split("\n"), fb.read().decode().split("\n")
    n = min(len(la), len(lb))
    nd = sum(1 for i in range(n) if la[i] != lb[i])
    return f"lines A {len(la):,} B {len(lb):,}; lines that differ at the same position {nd:,}" + (f"; first: A={la[next(i for i in range(n) if la[i] != lb[i])][:160]!r}" if nd else "")


def main():
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    files = sorted(str(p.relative_to(A)) for p in A.rglob("*") if p.is_file() and (B / p.relative_to(A)).is_file())
    same, diffs = 0, 0
    for f in files:
        if any(s in f for s in SKIP) or f.endswith((".1pz", ".bam", ".bai")):
            continue
        pa, pb = A / f, B / f
        try:
            if digest(pa) == digest(pb):
                same += 1
                continue
        except Exception as e:  # unreadable gzip etc.
            print(f"{f}: cannot read ({e})")
            continue
        diffs += 1
        try:
            if ".mtx" in f:
                print(f"DIFF {f}: {cmp_mtx(pa, pb)}")
            else:
                print(f"DIFF {f}: {cmp_table(pa, pb)}")
        except Exception as e:
            print(f"DIFF {f}: (comparison failed: {e})")
    print(f"-- {same} files identical, {diffs} differ, {len(files)} common files ({A.name} vs {B.name})")


if __name__ == "__main__":
    main()
