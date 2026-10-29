#!/usr/bin/env bash
# usage: bash tools/idea4_pp_run_fold.sh <fold> "<ks>" "<variants>" [exe]
#   variants: space separated list of  k:adapt   e.g. "0:0 0:1 16:0 16:1"
# builds the fold's models (if missing) and runs the harness (matrix roundtrip verified) with --calls on each test file
cd "$(dirname "$0")/.."
FOLD=$1; KS=$2; VARS=$3; EXE=${4:-bin/idea4_pp_bench.exe}
LINE=$(grep "^$FOLD|" tools/idea4_pp_folds.txt)
ORG=$(echo "$LINE" | cut -d'|' -f2); TRAIN=$(echo "$LINE" | cut -d'|' -f3); TEST=$(echo "$LINE" | cut -d'|' -f4)
need=""
for k in $(echo $KS | tr ',' ' '); do [ -f results/idea4_pp_model_${FOLD}_k${k}.bin ] || need="$need,$k"; done
need=${need#,}
if [ -n "$need" ]; then
  echo "[fold $FOLD] building models k=$need"
  OMP_WAIT_POLICY=PASSIVE python -P tools/idea4_pp_build.py --train "$TRAIN" --name $FOLD --ks $need --cap 2500 --iters 40 || exit 1
fi
for v in $VARS; do
  k=${v%%:*}; a=${v##*:}
  if [ "$k" = "0" ]; then CODEC=idea4_static; export IDEA4_MODEL_STATIC=results/idea4_pp_model_${FOLD}_k0.bin
  else CODEC=idea4_nmf; export IDEA4_MODEL_NMF=results/idea4_pp_model_${FOLD}_k${k}.bin; fi
  export IDEA4_ADAPT=$a; export IDEA4_KC=${IDEA4_KC:-50000}
  export IDEA4_TAG=${FOLD}_k${k}_a${a}
  for t in $(echo $TEST | tr ',' ' '); do
    echo "[fold $FOLD] $t k=$k adapt=$a"
    $EXE --calls --no-file-check --reps 1 --codec ${CODEC}_${IDEA4_TAG} --out results/idea4_pp_run.tsv corpus/$t/exon_counts.1pz
  done
done
