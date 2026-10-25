#!/bin/bash
#SBATCH --job-name=as-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Comparator env for the ATAC / spatial benchmark: MACS3, SnapATAC2 (Python), Signac (R), plus the analysis stack.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/atacbench
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs HOME=$WORK/cache/condahome
source /etc/profile.d/lmod.sh; module load miniforge3
if [ ! -f $ENV/.ok ]; then rm -rf $ENV; mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  python=3.11 numpy scipy pandas pyarrow h5py anndata scanpy macs3 pysam bedtools htslib samtools \
  r-base=4.4 r-signac r-seurat bioconductor-rtracklayer bioconductor-genomicranges r-data.table r-jsonlite \
  && touch $ENV/.ok; fi
$ENV/bin/pip install -q snapatac2 || echo "snapatac2 pip failed"
$ENV/bin/python -c "import snapatac2, MACS3, scanpy; print('py ok', snapatac2.__version__)"
$ENV/bin/macs3 --version
$ENV/bin/Rscript -e 'suppressMessages(library(Signac)); cat("Signac", as.character(packageVersion("Signac")), "\n")'
