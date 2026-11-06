#!/bin/bash
#SBATCH --job-name=dmx2-vireo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --array=0-10%6
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Vireo (vireoSNP, Apache-2.0) on the mixed AD/DP matrices of one mixture: K given and auto, 3 seeds x 2 thread counts.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/demux-truth2
row=$(grep -v '^#' $J/repo/bench/demux-truth/mixtures.tsv | sed -n "$((SLURM_ARRAY_TASK_ID+1))p")
IFS=$'\t' read -r id donors rate mseed <<< "$row"
K=$(echo $donors | tr ',' '\n' | wc -l)
rm -f $J/results/$id/runs-vireo.jsonl $J/results/$id/failures-vireo.txt
echo "node=$(hostname) mix=$id K=$K $(date -Is)"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
$(cat $J/python_ceiling.txt) $J/repo/bench/demux-truth/ceilings.py vireo $J/results/$id $J/private/$id $K 2>&1 | grep -E '^\[ceil\]|rror|Traceback' | tail -60
cat $J/results/$id/failures-vireo.txt 2>/dev/null || true
echo "done $(date -Is)"
