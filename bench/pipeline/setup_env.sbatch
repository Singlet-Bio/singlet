#!/bin/bash
#SBATCH --job-name=pb-env
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One-off: conda env with every tool the benchmark compares, plus build deps for singlet.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$WORK/envs/pbench; export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs
source /etc/profile.d/lmod.sh; module load miniforge3
mkdir -p $CONDA_PKGS_DIRS
export HOME=$WORK/cache/condahome; mkdir -p $HOME   # keep conda's rc/cache out of the real $HOME
[ -d $ENV ] || mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 star=2.7.11b samtools htslib ncbi-vdb sra-tools hdf5 zstd zlib cmake make pkg-config \
  gxx_linux-64 gcc_linux-64 sysroot_linux-64 salmon alevin-fry simpleaf piscem pigz \
  scanpy anndata pandas numpy scipy matplotlib leidenalg python-igraph pyroe scikit-learn
$ENV/bin/STAR --version; $ENV/bin/samtools --version | head -1; $ENV/bin/salmon --version
$ENV/bin/alevin-fry --version; $ENV/bin/simpleaf --version; $ENV/bin/piscem --version || true
$ENV/bin/python -c "import scanpy, anndata; print('scanpy', scanpy.__version__, 'anndata', anndata.__version__)"
