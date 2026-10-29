#!/bin/bash
#SBATCH --job-name=w11-m6-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 M6 GPU run. env: TUP (set to ' ' to skip tuples), LABEL (pbmc2m|GSM4006843|pbmc1k_v3), BATCHES, EXTRA (m6 args), NW (4)
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; M=$W/jobs/w11/m6; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
declare -A FQ=( [pbmc2m]=$PB/data/pbmc2m/pbmc2m.1fq [GSM4006843]=$W/data/bam-stream/GSM4006843/sample.1fq [pbmc1k_v3]=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq )
LABEL=${LABEL:-pbmc2m}; NW=${NW:-4}; BATCHES=${BATCHES:-0}; OUT=$M/results; mkdir -p $OUT
echo "node=$(hostname) job=$SLURM_JOB_ID label=$LABEL $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,memory.total --format=csv
nvidia-smi -q -d CLOCK | grep -A4 -E "^    Clocks$|Max Clocks" | head -12
$PB/bin/singlet-w3 decode ${FQ[$LABEL]} --r2 /dev/stdout --threads 4 2>/dev/null \
 | apptainer exec --nv -B $W $IMG $M/bin/m6_nw$NW --r2 - ${TUP:---tuples $M/data/$LABEL.tuples --truth $M/data/$LABEL.tuples.truth} \
     --batches $BATCHES --label $LABEL --json $OUT/m6_${LABEL}${TAG:-}.json ${EXTRA:-}
echo "exit=$?"; ls -la $OUT/m6_${LABEL}${TAG:-}.json
