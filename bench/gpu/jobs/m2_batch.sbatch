#!/bin/bash
#SBATCH --job-name=w11-m2batch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:55:00
#SBATCH --array=0-1%2
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# M2 cross-sample sharing on same-study batches that are already staged (read-only):
#   0: Kang 2018 (GSE96583) libraries kang_b1A, kang_b1B, kang_b1C, kang_b2stim  (4 libraries, ~0.8 G reads)
#   1: GSM6996272, GSM6996273, GSM6996274 (adjacent samples of one study, shallow, 2-4 M reads each)
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=$W/pipeline-benchmark/bin/singlet-w3
IMG=$W/images/python-3.12-bookworm.sif; PY=$W/venvs/singlet/bin/python
OUT=$W/jobs/w11/results; mkdir -p $OUT
if [ "$SLURM_ARRAY_TASK_ID" = 0 ]; then
  LABEL=kang_batch; FILES=($W/bench-public/data/kang_b1A/kang_b1A.1fq $W/bench-public/data/kang_b1B/kang_b1B.1fq \
                          $W/bench-public/data/kang_b1C/kang_b1C.1fq $W/bench-public/data/kang_b2stim/kang_b2stim.1fq)
else
  LABEL=GSM699627x_batch; FILES=($W/data/atlas-rerun/perf/GSM6996272/sample.1fq $W/data/atlas-rerun/perf/GSM6996273/sample.1fq \
                                 $W/data/atlas-rerun/perf/GSM6996274/sample.1fq)
fi
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
PIPES=(); i=0
for f in "${FILES[@]}"; do
  p=$S/r2_$i.fq; mkfifo $p; PIPES+=($p)
  ( $BIN decode "$f" --r2 $p --threads 2 2>/dev/null ) &
  i=$((i+1))
done
# Note: the python script reads the FIFOs in the order given; the decoders block until their FIFO is opened.
T0=$(date +%s)
apptainer exec -B $W -B $S $IMG $PY -I $W/jobs/w11/distinct_fraction.py "${PIPES[@]}" --len 90 --json $OUT/m2_$LABEL.json
echo "exit=$? wall=$(( $(date +%s) - T0 ))s files=${FILES[*]}"
