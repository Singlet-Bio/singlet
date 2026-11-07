#!/bin/bash
# S15 store: reference STARsolo (sorted BAM + Gene/GeneFull/GeneFull_Ex50pAS/Velocyto/SJ) for a deep pilot sample, on the first 100 M pairs of its kept .sra.
# Registers the run in the store. Run on bigmem (24 h); the downstream tools are queued by submit_big.sh.   pilot_big.sh GSM
set -uo pipefail
GSM=$1; W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; P=$W/pilot/$GSM; ENV=$W/envs/pbench; CAPPAIRS=100000000
export DSID=pilot_$GSM SMP=$GSM STORE=$W/benchmark-store STORE_RUN=1
rm -rf $P/fq; mkdir -p $P/fq
python3 - <<PY || exit 4
import json, subprocess
r = json.load(open("$P/runs.json"))["runs"]; left = $CAPPAIRS
for acc, n in r:
    if left <= 0: break
    k = min(n, left); left -= k
    subprocess.run(["$ENV/bin/fastq-dump", "-X", str(k), "--split-files", "--gzip", "-O", "$P/fq", "$P/sra/%s.sra" % acc], check=True)
PY
bash $D/pilot_stage.sh starsolo $GSM
