#!/bin/bash
#SBATCH --job-name=evp-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Score the runs of run_dataset.sbatch (or, with LAYOUT=legacy, the earlier pipeline-benchmark runs) against real truth.
#   sbatch --export=ALL,LAYOUT=evp,DATASETS=hgmm1k_v3,pbmc1k_cite_v3,pbmc1k_v3,TAG=final score.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
PB=$WORK/pipeline-benchmark
LAYOUT=${LAYOUT:-evp}; TAG=${TAG:-final}; DATASETS=${DATASETS:-hgmm1k_v3,pbmc1k_cite_v3,pbmc1k_v3}
if [ "$LAYOUT" = evp ]; then RUNS=$WORK/exact-vs-pseudo/runs; else RUNS=$PB/runs; fi
OUT=$WORK/exact-vs-pseudo/results/$TAG; mkdir -p $OUT
REF=/mnt/projects/debruinz_project/cellarium/reference
SRC=${SRC:-$WORK/exact-vs-pseudo/code}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
# the barnyard GTF holds both species; keep only its gene lines (chrY genes are taken from the human, ENSG, genes)
grep -P '\tgene\t' $REF/GRCh38-GRCm39-barnyard/genes/genes.gtf > $SCR/genes_only.gtf
$WORK/envs/pbench/bin/python -I $SRC/score.py --runs $RUNS --data $PB/data --gtf $SCR/genes_only.gtf \
    --out $OUT --layout $LAYOUT --datasets $DATASETS
