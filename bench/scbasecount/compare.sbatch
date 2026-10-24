#!/bin/bash
#SBATCH --job-name=sbc-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --array=0-19%8
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# one array task per row of picks.tsv (header + rows: cat gse gsm srx org ...)
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
D=$WORK/jobs/sbc
row=$(awk -F'\t' -v n=$((SLURM_ARRAY_TASK_ID + 2)) 'NR==n' $D/picks.tsv)
GSE=$(cut -f2 <<<"$row"); GSM=$(cut -f3 <<<"$row"); SRX=$(cut -f4 <<<"$row"); ORG=$(cut -f5 <<<"$row")
echo "$GSE $GSM $SRX $ORG"
EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070
apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" --env "EDROP=$EDROP" \
  --env "WORK=$WORK" --env "HDF5_USE_FILE_LOCKING=FALSE" $WORK/images/python-3.12-bookworm.sif \
  python -P $D/compare.py $GSE $GSM $SRX $ORG $WORK/data/scbasecount/compare
