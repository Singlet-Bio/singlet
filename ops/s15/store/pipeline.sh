#!/bin/bash
# S15 store: all missing CPU-side work for one dataset, run serially inside a node driver slot.   pipeline.sh DATASET_ID
# Skips what index.tsv already holds, fetches reads only when a missing tool needs them, runs STARsolo only when it or a BAM tool is missing, retries every stage once, cleans up only when everything delivered.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; PL=$W/pilot
ID=$1; BASEJOB=${SLURM_JOB_ID:-0}
eval "$(python3 $D/plan.py $STORE info $ID)"
echo "== $(date -Is) pipeline $ID kind=$KIND missing=[$MISSING] reads=$NEED_READS star=$NEED_STAR bam=$NEED_BAM species=$SP"
P=$PL/$DIR; mkdir -p $P
export DSID=$ID SMP=$DIR FIVE_ENV=$FIVE STORE STORE_RUN=1 SPECIES=$SP CHEM=$CHEM BAMSORT=${BAMSORT:-100000000000}
run() { name=$1; shift; for a in 1 2; do SLURM_JOB_ID=${BASEJOB}_${DIR}_$name "$@" && return 0; echo "[retry] $name attempt $a failed"; done; echo "[FAILED] $name"; return 1; }
fail=0
if [ "$NEED_READS" = 1 ]; then
  if [ "$KIND" = pilot ]; then
    if ! ls $P/sra/*.sra > /dev/null 2>&1; then run fetch env NODECODE=1 bash $ST fetch $DIR || fail=1; fi
    if [ "$NEED_STAR" = 1 ]; then rm -rf $P/starsolo_bam; [ -L $P/starsolo ] || { [ -d $P/starsolo ] && mv $P/starsolo $P/starsolo_old_$(date +%s); }; run star bash $D/pilot_big.sh $DIR || fail=1
    else run fqdump bash $D/pilot_fqdump.sh $DIR || fail=1; fi
  else
    case $FM in
      bam) run fetch env BAMURL=${ACC%%;*} ACC=${ACC##*;} bash $ST fetchbam $DIR || fail=1 ;;
      url) run fetch env URL=$(echo "$ACC" | tr ';' '\n' | grep http | head -1) bash $ST fetchurl $DIR || fail=1 ;;
      *) run fetch env SRRS=$(echo "$ACC" | tr ';' '\n' | grep -E '^SRR' | paste -sd,) NODECODE=1 bash $ST fetch $DIR || fail=1
         [ $fail = 0 ] && { run fq bash $ST fq $DIR || fail=1; } ;;
    esac
    if [ $fail = 0 ] && [ "$NEED_STAR" = 1 ]; then run star bash $ST starsolo $DIR || fail=1; fi
  fi
fi
if [ $fail = 0 ]; then
  for t in $MISSING; do
    ex=; [ -n "$K" ] && ex="KDON=$K"
    run $t env $ex bash $TL $t $DIR || fail=1
  done
fi
eval "$(python3 $D/plan.py $STORE info $ID)"
if [ -z "$MISSING" ] && [ $fail = 0 ]; then
  if [ "$KIND" = pilot ]; then rm -rf $P/fq $P/cap1fq; find $P \( -name '*.bam' -o -name '*.bai' \) -delete; else run clean bash $ST storeclean $DIR; fi
  echo "== $(date -Is) $ID complete, inputs cleaned"
else
  echo "== $(date -Is) $ID INCOMPLETE missing=[$MISSING]; inputs kept"; echo "$MISSING" > $P/INCOMPLETE
fi
