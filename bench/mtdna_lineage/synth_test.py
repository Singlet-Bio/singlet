"""Smoke test of analyze.py on a small simulated ROOT (no BAMs): python synth_test.py WORKDIR"""
import os
import struct
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import mtlib as M  # noqa: E402

rng = np.random.default_rng(0)
root = os.path.abspath(sys.argv[1]); truth = f"{root}/truth"
os.makedirs(truth, exist_ok=True)
ref = "".join(rng.choice(list("ACGT"), M.L))
for d in ("bpdcn_3p", "pile/bpdcn_3p", "pile/bpdcn_maester"):
    os.makedirs(f"{root}/{d}", exist_ok=True)
open(f"{root}/bpdcn_3p/mt_reference.fa", "w").write(">chrM\n" + ref + "\n")
n = 400
cells = [f"{''.join(rng.choice(list('ACGT'), 16))}-1" for _ in range(n)]
vois = [(1000, 0), (2000, 1), (3000, 2), (4000, 3), (5000, 0)]
vois = [(p, b if ref[p] != "ACGT"[b] else (b + 1) % 4) for p, b in vois]
clone = rng.integers(0, 4, n)  # clone 3 = no variant
clone_vars = {0: [0, 1], 1: [2], 2: [3, 4]}


def pile(depth, path):
    recs = []
    for c in range(n):
        for p in range(0, M.L, 37):
            d = rng.poisson(depth)
            if d == 0:
                continue
            cnt = np.zeros(8, int); rb = "ACGT".index(ref[p])
            vaf = 0.0
            for j, (vp, vb) in enumerate(vois):
                if vp == p and j in clone_vars.get(clone[c], []):
                    vaf = 0.6
            alt = [vb for vp, vb in vois if vp == p]
            a = rng.binomial(d, vaf) if alt else 0
            cnt[rb] = (d - a) // 2; cnt[rb + 4] = d - a - (d - a) // 2
            if alt:
                cnt[alt[0]] += a // 2; cnt[alt[0] + 4] += a - a // 2
            recs.append((c, p, cnt))
    with open(path, "wb") as f:
        f.write(b"MTP1"); f.write(struct.pack("<IQ", n, len(recs)))
        for c, p, cnt in recs:
            f.write(struct.pack("<II8I", c, p, *cnt))


vois_list = vois  # sites must be on the 37-grid
vois = [(p - p % 37, b) for p, b in vois]
vois = [(p, b if ref[p] != "ACGT"[b] else (b + 1) % 4) for p, b in vois]
for nm, dp in (("bpdcn_3p", 0.8), ("bpdcn_maester", 30)):
    for lv in ("read", "umi1", "umi3"):
        pile(dp, f"{root}/pile/{nm}/mt.{lv}.bin")
    open(f"{root}/pile/{nm}/cells.txt", "w").write("\n".join(cells) + "\n")
vn = [f"{p + 1}_{ref[p]}>{'ACGT'[b]}" for p, b in vois]
open(f"{truth}/4.2_vois.txt", "w").write("\n".join(vn) + "\n")
with open(f"{truth}/4.4_positive_cells.txt", "w") as f:
    f.write("cell\tvariant\tcov\tclone\tCellType\n")
    for c in range(n):
        if clone[c] in clone_vars:
            lab = "-".join(vn[j] for j in clone_vars[clone[c]])
            f.write(f"{cells[c]}\t{vn[clone_vars[clone[c]][0]]}\t50\t{lab}\tT\n")
import gzip  # noqa: E402
with gzip.open(f"{truth}/gnomad.genomes.v3.1.sites.chrM.vcf.bgz", "wt") as f:
    f.write("#CHROM\n")
env = dict(os.environ, MTL_TRUTH=truth)
r = subprocess.run([sys.executable, f"{HERE}/analyze.py", "bpdcn", root, f"{root}/bpdcn.json"], env=env)
print("exit", r.returncode)
r = subprocess.run([sys.executable, f"{HERE}/analyze.py", "size", root, f"{root}/size.json"], env=env)
print("exit", r.returncode)
