#!/bin/bash
#SBATCH --job-name=pb-env
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One-off: conda env with every tool the benchmark compares, plus build deps for singlet.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$WORK/envs/pbench; export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs
source /etc/profile.d/lmod.sh; module load miniforge3
mkdir -p $CONDA_PKGS_DIRS
export PKG_CONFIG_PATH=$ENV/lib/pkgconfig:$ENV/share/pkgconfig; export HOME=$WORK/cache/condahome; mkdir -p $HOME   # keep conda's rc/cache out of the real $HOME
[ -d $ENV ] || mamba create -y -p $ENV -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 star=2.7.11b samtools htslib ncbi-vdb sra-tools hdf5 zstd zlib cmake make pkg-config \
  gxx_linux-64 gcc_linux-64 sysroot_linux-64 salmon alevin-fry simpleaf piscem pigz \
  scanpy anndata pandas numpy scipy matplotlib leidenalg python-igraph pyroe scikit-learn
mamba install -y -p $ENV -c conda-forge liblzma-devel libdeflate bzip2 libcurl   # htslib.pc Requires.private
$ENV/bin/pkg-config --exists htslib && echo "htslib.pc ok"
$ENV/bin/STAR --version; $ENV/bin/samtools --version | head -1; $ENV/bin/salmon --version
$ENV/bin/alevin-fry --version; ALEVIN_FRY_HOME=$WORK/pipeline-benchmark/af_home $ENV/bin/simpleaf --version; $ENV/bin/piscem --version || true
$ENV/bin/python -c "import scanpy, anndata; print('scanpy', scanpy.__version__, 'anndata', anndata.__version__)"
