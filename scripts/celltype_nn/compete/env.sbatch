#!/bin/bash
# Runtimes for the competitor tools: Seurat 5 (R, official image) + SingleR (Bioconductor) in a project-space R library, and
# scvi-tools on the CUDA torch build. Run once.
#SBATCH --job-name=ctnn-cenv
#SBATCH --account=debruinz
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
mkdir -p $W/images $W/envs/rlib-celltype
export APPTAINER_CACHEDIR=/tmp/$USER-apcache-$SLURM_JOB_ID
[ -f $W/images/seurat-5.sif ] || apptainer pull $W/images/seurat-5.sif docker://satijalab/seurat:5.0.0
apptainer exec --bind $W $W/images/seurat-5.sif Rscript -e '
.libPaths(c("/mnt/projects/debruinz_project/debruinz/singlet/envs/rlib-celltype", .libPaths()))
options(repos=c(CRAN="https://cloud.r-project.org"), Ncpus=8)
if (!requireNamespace("BiocManager", quietly=TRUE)) install.packages("BiocManager", lib=.libPaths()[1])
BiocManager::install(c("SingleR","BiocParallel","zellkonverter"), lib=.libPaths()[1], update=FALSE, ask=FALSE)
for (p in c("arrow")) if (!requireNamespace(p, quietly=TRUE)) install.packages(p, lib=.libPaths()[1])
library(Seurat); library(SingleR); cat("Seurat", as.character(packageVersion("Seurat")), "SingleR", as.character(packageVersion("SingleR")), "\n")'
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
apptainer exec --nv --bind $W,/tmp --env PATH=$W/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin $W/images/python-3.12-bookworm.sif env \
  PYTHONPATH=$W/venvs/ctnn-torch:$W/venvs/forge-scvi/lib/python3.12/site-packages python -c "
import torch, scvi, anndata, numpy; print('torch', torch.__version__, torch.cuda.is_available(), 'scvi', scvi.__version__, 'numpy', numpy.__version__)"
