#!/usr/bin/env python3
"""Check a protocol's barcode geometry against real reads.

For each whitelist, report the fraction of reads whose window at every offset is a whitelist member
(exact), which shows where the barcode really sits, and, for a stated geometry, the fraction of reads
whose segments are all exact / all within one mismatch.

  wl_scan.py --fq M2.fq --n 200000 --wl rd3=splitseq_bc_rd3.txt --wl rd2=... --maxoff 100
  wl_scan.py --fq M2.fq --wl ... --geom rd3@10,rd2@48,rd1@78

Pure Python, no numpy. Chance level for a window is len(whitelist) / 4^L.
"""
import argparse
import itertools


def load(path):
    out = set()
    for line in open(path):
        s = line.split()
        if s:
            out.add(s[0].upper())
    return out


def neighbours(bc):
    for i, c in enumerate(bc):
        for b in "ACGT":
            if b != c:
                yield bc[:i] + b + bc[i + 1:]


def with_mm1(wl):
    """Map every 1-substitution neighbour to its barcode; neighbours shared by two barcodes are dropped."""
    m = {}
    for bc in wl:
        for nb in neighbours(bc):
            if nb in wl:
                continue
            m[nb] = None if nb in m and m[nb] != bc else bc
    return {k: v for k, v in m.items() if v}


def reads(path, n):
    with open(path) as f:
        for i, line in enumerate(f):
            if i % 4 == 1:
                yield line.strip().upper()
                if (i // 4) + 1 >= n:
                    return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fq", required=True)
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--wl", action="append", required=True, help="name=path")
    ap.add_argument("--maxoff", type=int, default=0, help="scan offsets 0..maxoff (0 = skip scan)")
    ap.add_argument("--geom", default="", help="name@offset,... in segment order")
    a = ap.parse_args()
    wls = {}
    for w in a.wl:
        k, p = w.split("=", 1)
        wls[k] = load(p)
    rs = list(reads(a.fq, a.n))
    print(f"{a.fq}: {len(rs)} reads")
    if a.maxoff:
        for k, wl in wls.items():
            L = len(next(iter(wl)))
            chance = len(wl) / 4 ** L
            hits = [0] * (a.maxoff + 1)
            for r in rs:
                for o in range(a.maxoff + 1):
                    if r[o:o + L] in wl:
                        hits[o] += 1
            top = sorted(range(len(hits)), key=lambda o: -hits[o])[:4]
            print(f"scan {k} (len {L}, {len(wl)} entries, chance {chance:.5f}): " +
                  ", ".join(f"off {o}={hits[o] / len(rs):.4f}" for o in top))
    if a.geom:
        segs = [(s.split("@")[0], int(s.split("@")[1])) for s in a.geom.split(",")]
        m1 = {k: with_mm1(wls[k]) for k, _ in segs}
        exact = one = 0
        per = {k: 0 for k, _ in segs}
        for r in rs:
            ok_e = ok_1 = True
            for k, o in segs:
                L = len(next(iter(wls[k])))
                s = r[o:o + L]
                if s in wls[k]:
                    per[k] += 1
                elif s in m1[k]:
                    ok_e = False
                else:
                    ok_e = ok_1 = False
            exact += ok_e
            one += ok_1
        print(f"geometry {a.geom}: all segments exact {exact / len(rs):.4f}, all within 1MM {one / len(rs):.4f}; " +
              "per-segment exact " + ", ".join(f"{k}={v / len(rs):.4f}" for k, v in per.items()))


if __name__ == "__main__":
    main()
