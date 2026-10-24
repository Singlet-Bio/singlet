#!/usr/bin/env bash
# stage 1-3 size runs (reference decoder, 1 rep): bash tools/idea1_runbits.sh OUT.tsv file.1pz ...
cd "$(dirname "$0")/.."
OUT=$1; shift
for f in "$@"; do
  ./bin/idea1_bench.exe --reps 1 --no-file-check --codec idea1_s1,idea1_s2,idea1_s3,idea1_s3r,idea1_s3_ins,idea1_s3r_ins --out "$OUT" "$f"
done
