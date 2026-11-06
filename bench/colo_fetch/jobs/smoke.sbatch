#!/bin/bash
#SBATCH --job-name=colo-smoke
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Smoke test of colo_fetch.py and colo_summary.py without a GPU: 2 fetch slots for SMOKE_S seconds on small runs, then a fake drain point
# (one "engine run" spanning the fetch window) so the summary has something to compare. Results in $W/jobs/colo/results/smoke-<job>.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; CD=$W/jobs/colo/scripts
RES0=$W/jobs/colo/results/smoke-$SLURM_JOB_ID; P=$RES0/k1t1s1; mkdir -p $P/colo
SCR=/tmp/$USER-colosmoke-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists COLO_SLOT_MBPS=${COLO_SLOT_MBPS:-100}
python3 - $W/jobs/s9/runs.json 0.5 2.5 > $SCR/runs.txt <<'PYEOF'
import json, random, sys
rs = [r for r in json.load(open(sys.argv[1])) if r.get("full_s3") and float(sys.argv[2]) * 1e9 <= r.get("full_s3_size", 0) <= float(sys.argv[3]) * 1e9]
random.Random(7).shuffle(rs)
for r in rs: print(r["run"], r["full_s3"], r["full_s3_size"], r["reads"])
PYEOF
echo "runs in window: $(wc -l < $SCR/runs.txt)"; head -3 $SCR/runs.txt
T0=$(date +%s.%N)
python3 $CD/colo_fetch.py $P/colo 2 $SCR/colo $SCR/runs.txt > $P/colo/colo.log 2>&1 &
CP=$!
sleep ${SMOKE_S:-420}
touch $P/colo/STOP; wait $CP; echo "colo_fetch exit $?"
T1=$(date +%s.%N)
printf '{"tag":"k1t1s1","colo":2}\n' > $P/meta.json
printf "0\t0\tfake\t%s\t%s\t0\n" $T0 $T1 > $P/runs.tsv
mkdir -p $P/c0; printf "\tUser time (seconds): 600\n\tSystem time (seconds): 60\n" > $P/c0/time.txt
mkdir -p $RES0/k1t1s1_a; printf '{"tag":"k1t1s1_a","colo":0}\n' > $RES0/k1t1s1_a/meta.json
printf "0\t0\tfake\t%s\t%s\t0\n" $T0 $(python3 -c "print($T1+30)") > $RES0/k1t1s1_a/runs.tsv
cat $P/colo/colo.log; echo "--- runs.jsonl"; cat $P/colo/runs.jsonl; echo "--- net.tsv tail"; tail -3 $P/colo/net.tsv
ls -la $SCR/colo/* 2>/dev/null | head
echo "--- summary"; python3 $CD/colo_summary.py $RES0
