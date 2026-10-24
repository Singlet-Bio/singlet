#!/usr/bin/env bash
# usage: bash tools/idea4_pp_runjobs.sh <exe> <jobs.txt>      jobs.txt lines: FOLD SAMPLE K ADAPT
# runs the jobs whose model exists and that are not yet in results/idea4_pp_run.tsv (matrix roundtrip verified by the harness)
cd "$(dirname "$0")/.."
EXE=$1; JOBS=$2; OUT=${3:-results/idea4_pp_run.tsv}
export IDEA4_KC=${IDEA4_KC:-50000}
while read FOLD t k a; do
  [ -z "$FOLD" ] && continue
  M=results/idea4_pp_model_${FOLD}_k${k}.bin
  [ -f $M ] || { echo "[skip] $FOLD $t k=$k a=$a (no model yet)"; continue; }
  TAG=${FOLD}_k${k}_a${a}
  if [ "$k" = "0" ]; then CODEC=idea4_static_$TAG; export IDEA4_MODEL_STATIC=$M; else CODEC=idea4_nmf_$TAG; export IDEA4_MODEL_NMF=$M; fi
  if [ -f $OUT ] && grep -q "$CODEC" $OUT && grep "$CODEC" $OUT | grep -q "^$t"; then continue; fi
  echo "[run] $FOLD $t k=$k adapt=$a"
  IDEA4_ADAPT=$a IDEA4_TAG=$TAG $EXE --calls --no-file-check --reps 1 --codec $CODEC --out $OUT corpus/$t/exon_counts.1pz | grep -E "payload|FAIL|MISMATCH" | cut -c1-420
done < "$JOBS"
echo "[runjobs] done $JOBS"
