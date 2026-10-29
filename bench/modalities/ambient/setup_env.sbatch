#!/bin/bash
#SBATCH --job-name=amb-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# conda env with R + SoupX + celda (DecontX) + scDblFinder, and python + scrublet. CellBender uses cellbender-0.3.2.sif.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$WORK/envs/ambient; export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs
source /etc/profile.d/lmod.sh; module load miniforge3
export HOME=$WORK/cache/condahome; mkdir -p $HOME
[ -d $ENV ] || mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  r-base=4.4 r-soupx r-matrix bioconductor-celda bioconductor-scdblfinder bioconductor-singlecellexperiment \
  r-data.table python=3.12 scanpy anndata pandas numpy scipy h5py scikit-learn
$ENV/bin/pip install --no-cache-dir scrublet
$ENV/bin/Rscript -e 'library(SoupX); library(celda); library(scDblFinder); cat("R packages ok\n"); print(packageVersion("SoupX")); print(packageVersion("celda")); print(packageVersion("scDblFinder"))'
$ENV/bin/python -c "import scrublet, scanpy; print('scrublet ok')"
