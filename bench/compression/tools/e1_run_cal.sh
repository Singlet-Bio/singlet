#!/bin/bash
# usage: tools/e1_run_cal.sh <res> <cap> <log> sample:member [...]   (runs tools/e1_cal.py run for each; needs the main e1_split run first)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$(dirname "$0")/.." || exit 1
res=$1; cap=$2; log=$3; shift 3
for t in "$@"; do
  s=${t%%:*}; m=${t##*:}
  echo "=== $(date +%H:%M:%S) start cal $s $m $res" >> "$log"
  python -P tools/e1_cal.py run "$s" "$m" --res "$res" --cap "$cap" >> "$log" 2>&1
  echo "=== $(date +%H:%M:%S) end cal $s $m $res rc=$?" >> "$log"
done
echo "=== $(date +%H:%M:%S) worker done" >> "$log"
