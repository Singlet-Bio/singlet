#!/usr/bin/env bash
# Run idea2_env under a list of parameter configurations on a list of files; one TSV + one log per sweep.
#   bash tools/idea2_sweep.sh <tag> <reps> "<file list>" "<config list>"
# config = space-free string of VAR=val pairs joined by ','  e.g. "I2_K=32,I2_MASK=18,I2_ORDER=1"
# Output: results/idea2_sweep_<tag>.log (stdout with "### cfg" markers) and results/idea2_sweep_<tag>.tsv
set -u
cd "$(dirname "$0")/.."
tag=$1; reps=$2; files=$3; cfgs=$4
log=results/idea2_sweep_${tag}.log
tsv=results/idea2_sweep_${tag}.tsv
: > "$log"
for cfg in $cfgs; do
  echo "### $cfg" >> "$log"
  for f in $files; do
    env $(echo "$cfg" | tr ',' ' ') ${EXE:-./bin/idea2_bench.exe} --no-file-check --reps "$reps" --codec idea2_env --out "$tsv" $f >> "$log" 2>&1
  done
done
echo "### done" >> "$log"
