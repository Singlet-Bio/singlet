#!/bin/bash
# S15 store: first 100 M pairs of a kept pilot .sra as FASTQ (same names the earlier STARsolo run used).   pilot_fqdump.sh GSM
GSM=$1; W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/pilot/$GSM; ENV=$W/envs/pbench; CAPPAIRS=100000000
rm -rf $P/fq; mkdir -p $P/fq
python3 - <<PY || exit 4
import json, subprocess
r = json.load(open("$P/runs.json"))["runs"]; left = $CAPPAIRS
for acc, n in r:
    if left <= 0: break
    k = min(n, left); left -= k
    subprocess.run(["$ENV/bin/fastq-dump", "-X", str(k), "--split-files", "--gzip", "-O", "$P/fq", "$P/sra/%s.sra" % acc], check=True)
PY
