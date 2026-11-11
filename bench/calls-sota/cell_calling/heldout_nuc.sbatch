#!/bin/bash
#SBATCH --job-name=cc-nuc
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Held-out hashed nuclei (GSE129578 POC nuclei). Inputs were fetched by the probe job: nuc.mtx.gz, proc.csv.tar.gz, proc.tsv.tar.gz
# from https://ftp.ncbi.nlm.nih.gov/geo/series/GSE129nnn/GSE129578/suppl/ into $SRC/probe_nuc.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$W/bench/calls-sota/cell_calling; P=$SRC/probe_nuc; O=$SRC/heldout_nuc
mkdir -p $O; cd $P
[ -s POC_nuc_barcodes.tsv ] || tar xzf proc.tsv.tar.gz POC_nuc_barcodes.tsv
sha256sum nuc.mtx.gz POC_nuc_barcodes.tsv csv/POC_nuc_MULTI_matrix.csv
PY=$W/envs/gpool/venv/bin/python
NUMBA_NUM_THREADS=1 NUMBA_CACHE_DIR=/tmp/$USER-nb-$SLURM_JOB_ID $PY -P $SRC/heldout_nuc_prep.py $P/nuc.mtx.gz $P/POC_nuc_barcodes.tsv $P/csv/POC_nuc_MULTI_matrix.csv $SRC/cellcall_v21.py $O
$W/envs/ambient/bin/Rscript $SRC/emptydrops_ref.R $O
$PY -P $SRC/merge_ed.py $O
$PY -P $SRC/heldout_nuc_score.py $O/table.tsv.gz $O/heldout_scores.json
