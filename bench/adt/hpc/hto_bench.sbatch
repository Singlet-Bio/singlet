#!/bin/bash
#SBATCH --job-name=adt-htob
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Hashtag demultiplexing benchmark vs genotype truth (bench/adt/hto_benchmark.py) -> $WORK/state/adt/hto_bench/
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=$WORK/src/singlet-py/claude__adt-modality-0637a6
export PATH=$WORK/envs/adt-bench/bin:$PATH LD_LIBRARY_PATH=$WORK/envs/adt-bench/lib:$WORK/envs/adt/lib
export PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome OMP_NUM_THREADS=4
python -P $SRC/bench/adt/hto_benchmark.py --data $WORK/data/adt --truth $WORK/state/adt/hto_truth.tsv \
  --out $WORK/state/adt/hto_bench --workdir /tmp/$USER-htob-$SLURM_JOB_ID ${MAXPOOLS:+--max-pools $MAXPOOLS}
rm -rf /tmp/$USER-htob-$SLURM_JOB_ID
