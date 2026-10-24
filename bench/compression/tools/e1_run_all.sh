#!/bin/bash
# usage: tools/e1_run_all.sh <res> <cap> <log> sample:member [sample:member ...]
# Runs the E1 ladder for each listed file sequentially, single-threaded (python -P).
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.." || exit 1
res=$1; cap=$2; log=$3; shift 3
for t in "$@"; do
  s=${t%%:*}; m=${t##*:}
  echo "=== $(date +%H:%M:%S) start $s $m $res" >> "$log"
  python -P tools/e1_split.py run "$s" "$m" --res "$res" --cap "$cap" >> "$log" 2>&1
  echo "=== $(date +%H:%M:%S) end $s $m $res rc=$?" >> "$log"
done
echo "=== $(date +%H:%M:%S) worker done" >> "$log"
