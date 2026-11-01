#!/bin/bash
#SBATCH --job-name=e9-truth
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E9 truth (docs/gpu-engine/fable/E9-solo.md): pbmc1k_v3 through the binary of bench/fable/solo/jobs/build_singlet.sbatch in E8's gate mode
# (--engine gpu, SINGLET_GPU_PILEUP=on, SINGLET_ENGINE_GPU_E2E=1), twice:
#   run "dump": + SINGLET_ENGINE_GPU_DUMP (mapDistinct's output: sequences, blobs, readRecord, chrStart; E6) and SINGLET_SOLO_DUMP (per read
#               what STAR's Solo Gene feature saw; src/star/ReadAlign_outputAlignments.cpp). The dumps hold read-level data: they go to
#               $W/jobs/fable/solo/scratch/truth-<job>/ (deleted when E9 is done), not to results.
#   run "plain": no dump (the hook must not change any output; CellReads.stats' row order from a second STAR run).
# Kept per run in results/truth-<job>/<run>/: star_Solo.out/ (+ Gene/CellReads.stats, hard-linked as soon as STAR writes it because the
# pipeline deletes it after parsing), star_Log.out, star_Log.final.out, the run log.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/solo/bin}; T=${T:-8}; s=${SAMPLE:-pbmc1k_v3}; RUNS=(${RUNS:-dump plain})
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES=$W/jobs/fable/solo/results/truth-$SLURM_JOB_ID; mkdir -p $RES
SCRATCH=$W/jobs/fable/solo/scratch/truth-$SLURM_JOB_ID
SCR=/tmp/$USER-e9t-$SLURM_JOB_ID; mkdir -p $SCR
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(cat $BINDIR/commit) T=$T runs=${RUNS[*]} $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
for m in "${RUNS[@]}"; do
  O=$SCR/$s/$m/out; mkdir -p $O $SCR/keep-$m; K=$RES/$m; mkdir -p $K
  unset SINGLET_ENGINE_GPU_DUMP SINGLET_SOLO_DUMP
  export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1
  if [ $m = dump ]; then mkdir -p $SCRATCH/solo; export SINGLET_ENGINE_GPU_DUMP=$SCRATCH/engine.dump SINGLET_SOLO_DUMP=$SCRATCH/solo; fi
  # CellReads.stats: hard link it the moment it appears (same inode, so the link has the full file once STAR closes it)
  ( while [ ! -e $O/star_Solo.out/Gene/CellReads.stats ]; do sleep 0.2; done; ln $O/star_Solo.out/Gene/CellReads.stats $SCR/keep-$m/CellReads.stats ) &
  WATCH=$!
  T0=$(date +%s.%N)
  ( cd $SCR/$s/$m && /usr/bin/time -v -o $K/time.txt timeout -k 30 1500 $BINDIR/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $O/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 \
      | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $K/run.log.ts )
  T1=$(date +%s.%N); kill $WATCH 2>/dev/null
  echo "$s $m: wall $(echo "$T1 - $T0" | bc) s, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $K/time.txt)" | tee -a $RES/summary.txt
  grep -a -e "E8" -e "gpu_pileup\] on" -e "E6 dump" $K/run.log.ts | cut -c1-300 | head -20 | tee -a $RES/summary.txt
  cp -r $O/star_Solo.out $K/ 2>/dev/null; cp $SCR/keep-$m/CellReads.stats $K/star_Solo.out/Gene/ 2>/dev/null
  cp $O/star_Log.out $O/star_Log.final.out $K/ 2>/dev/null
  grep -a -i -e "solo" -e "ENGINE_GPU" $O/star_Log.out | grep -v "^solo\|RE-DEFINED" | cut -c1-300 | tee -a $RES/summary.txt
  ls -la $K/star_Solo.out $K/star_Solo.out/Gene | tee -a $RES/summary.txt
  [ $m = dump ] && { ls -la $SCRATCH $SCRATCH/solo | tee -a $RES/summary.txt; du -sh $SCRATCH | tee -a $RES/summary.txt; }
done
# the two runs (and E8's gpuE of gate 796010) agree on every Solo file? CellReads.stats: whole file and sorted rows
E8=$W/jobs/fable/e2e/results/gate-796010/gpuE/out/star_Solo.out
for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
         Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx Gene/CellReads.stats; do
  line="$f:"; for m in "${RUNS[@]}"; do line="$line $m=$(sha256sum < $RES/$m/star_Solo.out/$f 2>/dev/null | cut -c1-16)"; done
  [ -f $E8/$f ] && line="$line e8=$(sha256sum < $E8/$f | cut -c1-16)"
  echo "$line" | tee -a $RES/summary.txt
done
for m in "${RUNS[@]}"; do echo "CellReads.stats $m: $(wc -l < $RES/$m/star_Solo.out/Gene/CellReads.stats) lines, sorted rows $(tail -n +3 $RES/$m/star_Solo.out/Gene/CellReads.stats | LC_ALL=C sort | sha256sum | cut -c1-16)" | tee -a $RES/summary.txt; done
echo done $(date -Is) | tee -a $RES/summary.txt
