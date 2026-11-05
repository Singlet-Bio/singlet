#!/bin/bash
#SBATCH --job-name=gb1-scanq
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:28:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only: first 20000 reads of every raw read file in our project space: longest read, number of distinct quality values,
# and whether qualities below 10 occur on bases that are not N (the case that needs the per-base mask).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH
IN=$W/jobs/gpu-batch1/results/scan1fq-800344/raw_reads.tsv
OUT=$W/jobs/gpu-batch1/results/scanq-$SLURM_JOB_ID; mkdir -p $OUT
grep -av 'nonhost_reads.fq\|Unmapped\|/out/\|\.sra$' $IN | cut -f2 > $OUT/files.txt
echo "files: $(wc -l < $OUT/files.txt)"
python3 - $OUT <<'PY'
import sys, gzip, subprocess, collections
out = sys.argv[1]
rows = []
for p in open(out + "/files.txt").read().split("\n"):
    if not p:
        continue
    try:
        f = gzip.open(p, "rt", errors="replace") if p.endswith(".gz") else open(p, "r", errors="replace")
        mx = 0; n = 0; qs = collections.Counter(); lowN = 0; low = 0; i = 0
        seq = ""
        for l in f:
            l = l.rstrip("\n"); k = i % 4; i += 1
            if k == 1: seq = l
            elif k == 3:
                n += 1; mx = max(mx, len(l))
                for s, q in zip(seq, l):
                    qs[q] += 1
                    if ord(q) - 33 < 10:
                        low += 1
                        if s != "N": lowN += 1
                if n >= 20000: break
        f.close()
        rows.append((p, n, mx, len(qs), low, lowN))
    except Exception as e:
        rows.append((p, "ERR", str(e)[:60], "", "", ""))
with open(out + "/qscan.tsv", "w") as o:
    o.write("path\treads\tmax_len\tdistinct_q\tbases_q_lt10\tof_which_not_N\n")
    for r in rows:
        o.write("\t".join(str(x) for x in r) + "\n")
print("done", len(rows))
PY
echo output $OUT
