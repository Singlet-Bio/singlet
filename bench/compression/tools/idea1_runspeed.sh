#!/usr/bin/env bash
# stage 4 runs: bash tools/idea1_runspeed.sh OUT.tsv file.1pz ...   (fast v2 decoder, dense s4 and rank-1 s4r, 3 reps)
cd "$(dirname "$0")/.."
OUT=$1; shift
for f in "$@"; do
  ./bin/idea1_bench.exe --reps 3 --no-file-check --codec idea1_s4,idea1_s4r --out "$OUT" "$f"
done
