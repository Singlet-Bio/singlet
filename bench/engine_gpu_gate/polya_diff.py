#!/usr/bin/env python3
"""Compare two SINGLET_POLYA_DUMP files (24-byte records: site u64, gene u32, bc u32, umi u64) as multisets.
usage: polya_diff.py CPU.bin GPU.bin"""
import sys
import numpy as np

dt = np.dtype([("site", "<u8"), ("gene", "<u4"), ("bc", "<u4"), ("umi", "<u8")])
a = np.fromfile(sys.argv[1], dtype=dt)
b = np.fromfile(sys.argv[2], dtype=dt)
print(f"records: cpu {len(a)} gpu {len(b)}")

def key(x):
    return np.core.records.fromarrays([x["site"], x["gene"], x["bc"], x["umi"]], names="site,gene,bc,umi")

def multiset_diff(x, y):
    """records of x not matched by y (multiset difference)"""
    xs = np.sort(x, order=["site", "gene", "bc", "umi"]); ys = np.sort(y, order=["site", "gene", "bc", "umi"])
    ix = np.zeros(len(xs), bool)
    j = 0
    # merge walk
    xk = xs.view("<u8").reshape(len(xs), 3)  # site, gene|bc, umi
    yk = ys.view("<u8").reshape(len(ys), 3)
    i = 0
    only = []
    while i < len(xs):
        if j >= len(ys): only.append(i); i += 1; continue
        tx = tuple(xk[i]); ty = tuple(yk[j])
        if tx == ty: i += 1; j += 1
        elif tx < ty: only.append(i); i += 1
        else: j += 1
    return xs[only]

oc = multiset_diff(a, b); og = multiset_diff(b, a)
print(f"cpu-only {len(oc)}, gpu-only {len(og)}")
for name, o in (("cpu-only", oc), ("gpu-only", og)):
    if not len(o): continue
    minus = (o["site"] >> 32) & 1
    print(f"{name}: minus strand {int(minus.sum())}, plus {len(o) - int(minus.sum())}; no UMI {int((o['umi'] == 0).sum())}; no gene {int((o['gene'] == 0xFFFFFFFF).sum())}")
# cpu-only records whose (bc, umi, gene) the GPU has at another site: a shifted site
gk = {(int(r["bc"]), int(r["umi"]), int(r["gene"])): int(r["site"]) for r in b}
shift = {}
same_mol = 0
for r in oc:
    g = gk.get((int(r["bc"]), int(r["umi"]), int(r["gene"])))
    if g is not None:
        same_mol += 1
        d = int(g) - int(r["site"]) if (g >> 33) == (int(r["site"]) >> 33) and ((g >> 32) & 1) == ((int(r["site"]) >> 32) & 1) else None
        shift[d] = shift.get(d, 0) + 1
print(f"cpu-only records whose (cell, UMI, gene) the GPU has: {same_mol}; site shift histogram (None = other contig/strand):",
      sorted(shift.items(), key=lambda t: -t[1])[:12])
for r in oc[:15]:
    print("cpu-only", int(r["site"] >> 33), int((r["site"] >> 32) & 1), int(r["site"] & 0xFFFFFFFF), int(r["gene"]), int(r["bc"]), hex(int(r["umi"])))
