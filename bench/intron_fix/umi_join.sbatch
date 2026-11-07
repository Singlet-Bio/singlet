#!/bin/bash
#SBATCH --job-name=if-join
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
REF=/mnt/projects/debruinz_project/cellarium/reference
DSS=(pbmc1k_v3 nuclei2k_v2 hgmm1k_v3); DS=${DSS[$SLURM_ARRAY_TASK_ID]}
case $DS in hgmm*) SP=GRCh38-GRCm39-barnyard ;; nuclei*|neuron*) SP=GRCm39-2024-A ;; *) SP=GRCh38-2024-A ;; esac
TAG=${TAG:-singlet-if-sweep}
mkdir -p $W/jobs/intronfix/results
$ENV/bin/python $W/jobs/intronfix/umi_join.py $PB $DS $TAG $REF/$SP/genes/genes.gtf $W/jobs/intronfix/results/join_${DS}_${TAG}.json
