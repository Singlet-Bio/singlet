#!/bin/bash
#SBATCH --job-name=adt-bench
#SBATCH --account=debruinz
#SBATCH --partition=debug-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Method benchmark on 10x pbmc_1k_protein_v3 (TotalSeq-B, Cell Ranger 3.0.0 truth). Inputs are the pipeline-benchmark
# session's downloads (read only). Results: $WORK/state/adt/benchmark/.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt
BRANCH=claude/adt-modality-0637a6
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
D=$WORK/pipeline-benchmark/data/pbmc1k_cite_v3
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
git -C $SRC checkout -q --detach "$BRANCH"; git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
export CXX=$ENV/bin/x86_64-conda-linux-gnu-g++ CPATH=$ENV/include LIBRARY_PATH=$ENV/lib
mkdir -p $WORK/state/adt/benchmark
/usr/bin/time -v python -P $SRC/bench/adt/benchmark.py --fastq-dir $D/pbmc_1k_protein_v3_fastqs/pbmc_1k_protein_v3_antibody_fastqs \
  --h5 $D/cellranger_pbmc_1k_protein_v3_filtered_feature_bc_matrix.h5 --out $WORK/state/adt/benchmark \
  --work $SCRATCH --src $SRC --threads 8 ${METHODS:+--methods $METHODS} 2>&1 | grep -v "^\s*$"
