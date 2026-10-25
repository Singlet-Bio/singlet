#!/bin/bash
#SBATCH --job-name=evp-breakdown
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Feature-layer breakdown of the cross-species UMIs (see breakdown.py).
#   sbatch --export=ALL,LAYOUT=evp,TAG=final breakdown.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; PB=$WORK/pipeline-benchmark
LAYOUT=${LAYOUT:-evp}; TAG=${TAG:-final}
if [ "$LAYOUT" = evp ]; then RUNS=$WORK/exact-vs-pseudo/runs; else RUNS=$PB/runs; fi
OUT=$WORK/exact-vs-pseudo/results/$TAG; mkdir -p $OUT
REF=/mnt/projects/debruinz_project/cellarium/reference
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
grep -P '\tgene\t' $REF/GRCh38-GRCm39-barnyard/genes/genes.gtf > $SCR/genes_only.gtf
$WORK/envs/pbench/bin/python -I $WORK/exact-vs-pseudo/code/breakdown.py --runs $RUNS --layout $LAYOUT --data $PB/data \
    --gtf $SCR/genes_only.gtf --out $OUT/breakdown_hgmm1k_v3.json
