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
# competitor tools with old pins live in their own envs: CITE-seq-Count 1.4.5; salmon 1.10 (2.0 dropped --features) + alevin-fry
for spec in "adt-csc:python=3.9 cite-seq-count=1.4.5" "adt-afry:salmon=1.10.3 alevin-fry=0.8.2"; do
  E=$WORK/envs/${spec%%:*}; P=${spec#*:}
  if [ ! -f $E/.ok ]; then rm -rf $E; mamba create -y -p $E -c conda-forge -c bioconda --strict-channel-priority $P && touch $E/.ok; fi
done
$WORK/envs/adt-csc/bin/CITE-seq-Count --version | tail -1
$WORK/envs/adt-afry/bin/salmon --version; $WORK/envs/adt-afry/bin/alevin-fry --version
