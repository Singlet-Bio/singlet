#!/usr/bin/env bash
# run every (fold, test sample, k, adapt) combination whose model exists and that is not yet in results/idea4_pp_run.tsv
#   bash tools/idea4_pp_run_all.sh [exe]
cd "$(dirname "$0")/.."
EXE=${1:-bin/idea4_pp_b3.exe}
OUT=results/idea4_pp_run.tsv
export IDEA4_KC=${IDEA4_KC:-50000}
# order: primary fold first, then the other folds, then in-sample / cross-domain
FOLDS="F142444 F244263 F152522 F145370 M285936 XH IShuman ISmouse"
KS="0 16 32 8"
for FOLD in $FOLDS; do
  LINE=$(grep "^$FOLD|" tools/idea4_pp_folds.txt); TEST=$(echo "$LINE" | cut -d'|' -f4)
  for k in $KS; do
    M=results/idea4_pp_model_${FOLD}_k${k}.bin
    [ -f $M ] || continue
    for a in 0 1; do
      for t in $(echo $TEST | tr ',' ' '); do
        TAG=${FOLD}_k${k}_a${a}
        if [ "$k" = "0" ]; then CODEC=idea4_static_$TAG; export IDEA4_MODEL_STATIC=$M; else CODEC=idea4_nmf_$TAG; export IDEA4_MODEL_NMF=$M; fi
        if [ -f $OUT ] && grep -q "^$t	exon_counts.1pz	$CODEC	" $OUT; then continue; fi
        echo "[run] $FOLD $t k=$k adapt=$a"
        IDEA4_ADAPT=$a IDEA4_TAG=$TAG $EXE --calls --no-file-check --reps 1 --codec $CODEC --out $OUT corpus/$t/exon_counts.1pz | grep -E "payload|FAIL|MISMATCH" | cut -c1-420
      done
    done
  done
done
echo "[run_all] done"
