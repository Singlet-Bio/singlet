#!/usr/bin/env bash
# model-selection runs: bash tools/idea1_runs5.sh OUT.tsv CHOICE.tsv file.1pz ...   (codec idea1_s5r = v2 stream, 3 reps)
cd "$(dirname "$0")/.."
OUT=$1; CH=$2; shift; shift
for f in "$@"; do
  ./bin/idea1_bench.exe --reps 3 --no-file-check --choice "$CH" --codec idea1_s5r --out "$OUT" "$f"
done
