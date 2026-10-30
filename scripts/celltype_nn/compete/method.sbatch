#!/bin/bash
# One competitor on one benchmark study per array task (index into studies.txt). METHOD = celltypist | scanvi | seurat | singler.
#   sbatch --array=0-15%8 --export=ALL,METHOD=seurat scripts/celltype_nn/compete/method.sbatch
#SBATCH --job-name=ctnn-comp
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:59:00
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__celltype-nn/scripts/celltype_nn/common.sh
O=$D/compete
G=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $SRC/scripts/celltype_nn/compete/studies.txt)
case "$METHOD" in
  seurat|singler)
    apptainer exec --bind $WORK,/tmp --env R_LIBS_USER=$WORK/envs/rlib-celltype --env SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK \
      $WORK/images/seurat-5.sif Rscript $SRC/scripts/celltype_nn/compete/run_r.R $METHOD $O $G ;;
  celltypist)
    in_runtime python -u $SRC/scripts/celltype_nn/compete/run_py.py predict --method celltypist --out $O --studies $G ;;
  scanvi)
    in_runtime_gpu env PYTHONPATH=$TORCH:$WORK/venvs/forge-scvi/lib/python3.12/site-packages:$PYTHONPATH \
      python -u $SRC/scripts/celltype_nn/compete/run_py.py predict --method scanvi --out $O --studies $G ;;
esac
