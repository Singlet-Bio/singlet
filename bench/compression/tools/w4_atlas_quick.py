#!/usr/bin/env python
"""w4_atlas: reduced recoding check (zstd-19 and xz-9e vs deflate on a 6 MB slice) for 4 members, run in parallel at low priority."""
import subprocess, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = os.path.join(ROOT, "corpus_w4_atlas", "recode")
labs = ["star_Solo_out__Gene__raw__matrix_mtx__1", "star_Solo_out__Gene__filtered__matrix_mtx__1",
        "star_Solo_out__Gene__raw__barcodes_tsv__1", "ambient_profile_tsv__1"]
exe = os.path.join(ROOT, "bin", "w4_atlas_bench.exe")
out = os.path.join(ROOT, "results", "w4_atlas_recode_quick.tsv")
ps = []
for i, l in enumerate(labs):
    ps.append(subprocess.Popen([exe, "text", os.path.join(C, l + ".txt"), l, out + f".{i}", "6000000", "dzx"], cwd=ROOT,
                               creationflags=0x4000 if os.name == "nt" else 0))
for p in ps: p.wait()
rows = []
for i in range(len(labs)):
    f = out + f".{i}"
    L = open(f).read().splitlines()
    if i == 0: rows.append(L[0])
    rows.append(L[1]); os.remove(f)
open(out, "w").write("\n".join(rows) + "\n")
print("\n".join(rows))
