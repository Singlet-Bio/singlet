#!/bin/bash
#SBATCH --job-name=s9-fetchscale
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=48
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: fetch tier at scale on ONE node: NW fetch workers (decode in-process children) + a drain stub that frees the stage, cap small enough
# that backpressure triggers. Prints the NIC receive rate (bond0), staged bytes and load every 10 s, then totals.
# Env: NW (workers, default 12), STREAMS (per worker, default 8), RUNS (todo count, default 64), CAP_GB (default 120).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; R=$W/jobs/s9
NW=${NW:-12}; STREAMS=${STREAMS:-8}; RUNS=${RUNS:-64}; CAP_GB=${CAP_GB:-120}
BIN=${BIN:-$W/pipeline-benchmark/bin/mc99cpu}
T=$R/scale_$SLURM_JOB_ID; STAGE=$T/stage; mkdir -p $STAGE
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
echo "== node $(hostname) $(date -Is) NW=$NW streams=$STREAMS runs=$RUNS cap=${CAP_GB}GB; other running jobs on cluster: $(squeue -h -t R | wc -l)"
python3 $SRC/stagequeue.py status $STAGE >/dev/null
python3 - <<PY
import json
runs = [r for r in json.load(open("$R/runs.json")) if 1.0e9 < r["lite_ncbi_size"] < 9e9]
step = max(1, len(runs) // $RUNS)
with open("$T/todo.jsonl", "w") as f:
    for i, r in enumerate(runs[::step][:$RUNS]):
        f.write(json.dumps({"gsm": "SC%03d_%s" % (i, r["run"]), "gse": "NA", "organism": "Homo sapiens", "runs": [r["run"]]}) + "\n")
print(len(runs), "candidate runs;", sum(1 for _ in open("$T/todo.jsonl")), "queued; lite GB total", round(sum(r["lite_ncbi_size"] for r in runs[::step][:$RUNS]) / 1e9))
PY
python3 $SRC/stagequeue.py enqueue $STAGE $T/todo.jsonl
rx() { awk '/bond0:/ {print $2}' /proc/net/dev; }
(
  p=$(rx); t=$(date +%s)
  while sleep 10; do
    n=$(rx); u=$(date +%s)
    echo "MON $(date +%T) rx_MBps=$(( (n - p) / (u - t) / 1000000 )) load=$(cut -d' ' -f1 /proc/loadavg) stage=$(python3 $SRC/stagequeue.py status $STAGE)"
    p=$n; t=$u
  done
) > $T/monitor.log 2>&1 &
MON=$!
PIDS=""
for i in $(seq 1 $NW); do
  python3 -u $SRC/fetch_worker.py --stage $STAGE --cap-tb $(python3 -c "print($CAP_GB/1000)") --room-need-gb 5 --streams $STREAMS --quality binned \
      --decode-threads 2 --binary $BIN --whitelist-dir $SINGLET_WHITELIST_DIR --idle-exit-min 1 > $T/fetch$i.log 2>&1 &
  PIDS="$PIDS $!"; sleep 1
done
python3 $SRC/stagequeue.py drain-stub $STAGE --delay 8 --idle-exit-s 90 > $T/stub1.log 2>&1 &
PIDS="$PIDS $!"
python3 $SRC/stagequeue.py drain-stub $STAGE --delay 8 --idle-exit-s 90 > $T/stub2.log 2>&1 &
PIDS="$PIDS $!"
wait $PIDS
kill $MON 2>/dev/null
echo "== monitor (10 s samples)"; cat $T/monitor.log | cut -c1-230
echo "== staged samples (B/pair, ratio, decode core-s/M)"; cat $T/fetch*.log | grep "\[stage\].*ready" | cut -c1-210
echo "== parked / failed"; cat $T/fetch*.log | grep -E "parked|FAILED" | cut -c1-200 | sort | uniq -c | sort -rn | head
echo "== backpressure events"; cat $T/fetch*.log | grep -c "fetch paused"
echo "== fetch rates"; cat $T/fetch*.log | grep -E "\[fetch\].* lite " | awk '{gsub(/\(/,"",$0); print}' | head -3
python3 $SRC/stagequeue.py status $STAGE
echo "== done $(date -Is)"
