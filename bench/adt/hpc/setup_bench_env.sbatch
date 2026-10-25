#!/bin/bash
#SBATCH --job-name=adt-benv
#SBATCH --account=debruinz
#SBATCH --partition=debug-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Env for the HTO demultiplexing comparison: Seurat HTODemux / MULTIseqDemux (R), scanpy hashsolo, demuxEM.
# Kept apart from envs/adt so the production counting env never changes under the running arrays.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt-bench
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs HOME=$WORK/cache/condahome
source /etc/profile.d/lmod.sh; module load miniforge3
if [ ! -f $ENV/.ok ]; then rm -rf $ENV; mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 numpy scipy pandas pyarrow anndata scanpy h5py matplotlib r-base=4.4 r-seurat r-matrix r-jsonlite \
  r-data.table && touch $ENV/.ok; fi
$ENV/bin/pip install -q zstandard
$ENV/bin/pip install -q demuxEM==0.1.7.post1 || $ENV/bin/pip install -q demuxEM
$ENV/bin/python -c "import scanpy, scanpy.external as sce, demuxEM; print('py ok', scanpy.__version__)"
$ENV/bin/Rscript -e 'suppressMessages(library(Seurat)); cat("Seurat", as.character(packageVersion("Seurat")), "\n")'
