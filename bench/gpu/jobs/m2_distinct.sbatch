#!/bin/bash
#SBATCH --job-name=w11-m2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=3
#SBATCH --mem=48G
#SBATCH --time=03:30:00
#SBATCH --array=0-7%4
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# M2: distinct-sequence fraction on real staged reads. Read-only on the .1fq inputs; result = one small JSON per task.
# singlet decode streams R2 (cDNA) as FASTQ into distinct_fraction.py (no FASTQ is written to disk).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=$W/pipeline-benchmark/bin/singlet-w3
IMG=$W/images/python-3.12-bookworm.sif; PY=$W/venvs/singlet/bin/python
OUT=$W/jobs/w11/results; mkdir -p $OUT
# label | file | note
TASKS=(
"pbmc1k_v3|$W/pipeline-benchmark/data/pbmc1k_v3/pbmc1k_v3.1fq"
"GSM7656149|$W/data/atlas-rerun/perf/GSM7656149/SRR25376447.1fq"
"GSM7036403|$W/data/atlas-rerun/perf/GSM7036403/SRR23353990.1fq"
"pbmc5p_v1|$W/pipeline-benchmark/data/pbmc5p_v1/pbmc5p_v1.1fq"
"GSM6470208|$W/data/atlas-rerun/perf/GSM6470208/sample.1fq"
"nuclei2k_v2|$W/pipeline-benchmark/data/nuclei2k_v2/nuclei2k_v2.1fq"
"neuron1k_v3|$W/pipeline-benchmark/data/neuron1k_v3/neuron1k_v3.1fq"
"GSM5411066|$W/data/bam-stream/GSM5411066/SRR14999752.1fq"
)
IFS='|' read -r LABEL F <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
echo "task=$SLURM_ARRAY_TASK_ID label=$LABEL file=$F node=$(hostname)"
$BIN decode "$F" --header 2>&1 | grep -E 'Protocol|Reads|R1 length'
T0=$(date +%s)
$BIN decode "$F" --r2 /dev/stdout --threads 2 2>/dev/null \
  | apptainer exec -B $W $IMG $PY -I $W/jobs/w11/distinct_fraction.py - --len 90 --json $OUT/m2_$LABEL.json
echo "exit=$? wall=$(( $(date +%s) - T0 ))s"
