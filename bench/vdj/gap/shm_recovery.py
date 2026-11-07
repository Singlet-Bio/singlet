# SPDX-License-Identifier: MIT
"""Recovery of Cell Ranger's contigs by flank mismatch load, for several Singlet runs of one library.

    python -P shm_recovery.py DATA_DIR OUT.json RUN_DIR1[=label] RUN_DIR2[=label] ...

DATA_DIR holds filtered_contig_annotations.csv and fastqs/ (a 10x V(D)J library). RUN_DIRs hold contigs.csv (bench/vdj/run_vdj.py).
For every reference contig (cell, chain, junction) the 40 nt of read sequence before the junction are compared with the germline V
gene Cell Ranger names (the segment's Cys anchor, offsets -3..+3, median over up to 20 reads of that barcode that contain the junction),
a local estimate of the mutation load that does not depend on any Singlet run. Cached beside OUT.json (shm_<library>.csv).
Prints, per run, the share of reference contigs recovered (same barcode, chain and junction) in the flank-mismatch bins
none / up to 3% / 3 to 6% / 6 to 10% / above 10%, and the sums over the bins up to 6% and above 6%.
"""
import glob
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
import compare_vdj as CV  # noqa: E402

data, out = sys.argv[1:3]
runs = [a.partition("=")[::2] for a in sys.argv[3:]]
lib = os.path.basename(os.path.normpath(data))
ref_csv = f"{data}/filtered_contig_annotations.csv"
ref = CV.load_cellranger(ref_csv)
cells = CV.cellranger_cells(ref_csv)
ref = ref[ref.barcode.isin(cells)].reset_index(drop=True)
cache = os.path.join(os.path.dirname(out) or ".", f"shm_{lib}.csv")
if os.path.exists(cache):
    shm = pd.read_csv(cache)["shm"].values
else:
    seg = {}
    VREF = os.environ.get("VDJ_REF", "/mnt/projects/debruinz_project/debruinz/singlet/state/vdjc/ref/vdj_human.tsv.gz")
    for l in gzip.open(VREF, "rt"):
        if l.startswith("#") or l.startswith("name"):
            continue
        p = l.rstrip("\n").split("\t")
        if p[3] == "V" and p[4].lstrip("-").isdigit() and int(p[4]) >= 40:
            seg[p[0].split("*")[0].upper()] = (p[-1], int(p[4]))
    comp = str.maketrans("ACGTN", "TGCAN")
    rc = lambda x: x.translate(comp)[::-1]  # noqa: E731
    want = {}
    for i, r in enumerate(ref.itertuples()):
        want.setdefault(r.barcode[:16], []).append((i, r.cdr3_nt))
    flanks = {}
    for p1 in sorted(glob.glob(f"{data}/fastqs/**/*_R1_*.fastq.gz", recursive=True)):
        with gzip.open(p1, "rt") as f1, gzip.open(p1.replace("_R1_", "_R2_"), "rt") as f2:
            for j, (x, y) in enumerate(zip(f1, f2)):
                if j % 4 != 1:
                    continue
                b = x[:16]
                if b not in want:
                    continue
                y = y.strip()
                for i, nt in want[b]:
                    for q in (y, rc(y)):
                        k = q.find(nt)
                        if k >= 24:
                            flanks.setdefault(i, []).append(q[max(0, k - 48):k])
                            break
    shm = np.full(len(ref), np.nan)
    for i, r in enumerate(ref.itertuples()):
        g = seg.get(r.v) or seg.get(r.v.replace("D", ""))
        fl = flanks.get(i, [])
        if g is None or not fl:
            continue
        seq, an = g
        best = []
        for q in fl[:20]:
            L = min(len(q), 40)
            qq = q[-L:]
            m = [sum(a != b for a, b in zip(qq, seq[e - L:e])) / L for e in range(an - 3, an + 4) if L <= e <= len(seq)]
            if m:
                best.append(min(m))
        if best:
            shm[i] = float(np.median(best))
    pd.DataFrame({"shm": shm}).to_csv(cache, index=False)
bins = [-0.01, 0.0, 0.03, 0.06, 0.10, 1.0]
names = ["none", "<=3%", "3-6%", "6-10%", ">10%"]
cat = pd.cut(pd.Series(shm), bins, labels=names)
res = {"library": lib, "n_reference_contigs": int(len(ref)), "n_with_estimate": int(np.isfinite(shm).sum()), "runs": {}}
for rd, label in runs:
    label = label or os.path.basename(rd)
    s = CV.load_singlet(f"{rd}/contigs.csv")
    keys = set(zip(s.barcode, s.chain, s.cdr3_nt))
    rec = np.array([(b, c, n) in keys for b, c, n in zip(ref.barcode, ref.chain, ref.cdr3_nt)])
    d = {}
    for nm in names:
        m = (cat == nm).values
        d[nm] = {"n": int(m.sum()), "recovered": int(rec[m].sum())}
    lo = np.isin(cat, names[:3])
    hi = np.isin(cat, names[3:])
    d["<=6%"] = {"n": int(lo.sum()), "recovered": int(rec[lo].sum())}
    d[">6%"] = {"n": int(hi.sum()), "recovered": int(rec[hi].sum())}
    d["all"] = {"n": int(len(rec)), "recovered": int(rec.sum())}
    res["runs"][label] = d
    print(lib, label, " ".join(f"{k}:{v['recovered']}/{v['n']}" for k, v in d.items()), flush=True)
json.dump(res, open(out, "w"), indent=1)
