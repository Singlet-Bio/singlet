#!/bin/bash
#SBATCH --job-name=s8-probegate
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8: the 5' prefix probe reads the same records whatever the route. Per sample: .1fq (FIFO decode, CPU), .1fg2 (FIFO decode),
# and the two file-decode routes (SINGLET_ENGINE_GPU=1 / SINGLET_FILE_DECODE=1) on each; each run is stopped once the probe line is logged.
# Gate: the "5' prefix probe: N reads, ..." line is identical in all four runs of a sample (N = 12000 on these samples).
# Env: BINDIR (needs singlet-gpu).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?}; SAMPLES=(SRR19421372 SRR19573735); s=${SAMPLES[$SLURM_ARRAY_TASK_ID]}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
G=$REF/GRCh38-2024-A; [ $s = neuron1k_v3 ] && G=$REF/GRCm39-2024-A
FQ=$W/jobs/s9/fixtures/$s.1fq; [ -e $FQ ] || FQ=$W/data/s8/$s/$s.1fq; [ -e $FQ ] || FQ=$PB/data/$s/$s.1fq
SCR=/tmp/$USER-s8pg-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT
RES=$W/jobs/s8/results/probegate-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $RES
for m in fq fg2 fq_file fg2_file; do
  IN=$FQ; case $m in fg2*) IN=${FQ%.1fq}.1fg2 ;; esac
  unset SINGLET_ENGINE_GPU SINGLET_FILE_DECODE
  case $m in *_file) export SINGLET_FILE_DECODE=1 ;; esac
  mkdir -p $SCR/$m/out; L=$SCR/$m/run.log
  ( cd $SCR/$m && exec $BINDIR/singlet-gpu $IN --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/$m/out/ --threads 12 --pipeline --output-format 1pz > $L 2>&1 ) &
  P=$!
  for i in $(seq 1 1500); do grep -aq "5' prefix probe" $L 2>/dev/null && break; kill -0 $P 2>/dev/null || break; sleep 2; done
  pkill -P $P 2>/dev/null; kill $P 2>/dev/null; wait $P 2>/dev/null
  echo "$s $m: $(grep -a "5' prefix probe" $L | head -1)" | tee -a $RES/summary.txt
done
n=$(grep -a "probe:" $RES/summary.txt | sed 's/^[^:]*:[^:]*: *//' | sort -u | wc -l); c=$(grep -ac "probe:" $RES/summary.txt)
[ "$c" = 4 ] && [ "$n" = 1 ] && echo "PROBE GATE OK" | tee -a $RES/summary.txt || echo "!! PROBE GATE FAILED (lines=$c distinct=$n)" | tee -a $RES/summary.txt
