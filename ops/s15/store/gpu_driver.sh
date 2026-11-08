#!/bin/bash
# S15 store: one V100S node, two CellBender runs side by side (one per GPU). Takes datasets whose STARsolo output exists and CellBender is missing; waits for the CPU drivers to produce more.
# Submit: sbatch --partition=gpu --gres=gpu:tesla_v100s:2 --exclusive --nodes=1 --time=5-00:00:00 gpu_driver.sh
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; STORE=$W/benchmark-store; TL=$D/pilot_tools.sh; TRIED=$D/queue/cb_tried.tsv; touch $TRIED
echo "== $(date -Is) gpu driver job=$SLURM_JOB_ID node=$(hostname)"
export STORE STORE_RUN=1
pick() { flock $TRIED.lock bash -c "for id in \$(python3 $D/plan.py $STORE gpu-queue); do [ \$(grep -c -x \$id $TRIED) -lt 2 ] && { echo \$id >> $TRIED; echo \$id; exit 0; }; done"; }
slot() { g=$1; while :; do id=$(pick)
    if [ -z "$id" ]; then squeue -u $USER -h -t R -o "%j" | grep -q "^s15-node$" || break; sleep 600; continue; fi
    dir=${id#pilot_}; eval "$(python3 $D/plan.py $STORE info $id)"
    echo "== $(date -Is) gpu$g cellbender $id"
    CUDA_VISIBLE_DEVICES=$g DSID=$id SMP=$dir FIVE_ENV=$FIVE SPECIES=$SP SLURM_JOB_ID=${SLURM_JOB_ID}_cb${g}_$dir bash $TL cellbender $dir >> $W/logs/gpu-$SLURM_JOB_ID-g$g-$dir.log 2>&1 || echo "[cellbender failed] $id"
  done; echo "gpu slot $g done"; }
slot 0 & sleep 5; slot 1 & wait
echo "== $(date -Is) gpu driver done"
