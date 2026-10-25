#!/bin/bash
#SBATCH --job-name=adt-prof
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Profile the Singlet ADT counter on pbmc1k_cite_v3 (cProfile, top 60 by cumulative time), from a git archive in scratch.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
REV=${REV:-refs/adt/check}
SCR=/tmp/$USER-fast-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/claude/adt-modality-0637a6:refs/adt/check" || true
git -C $WORK/repos/singlet-py.git archive $REV | tar -x -C $SCR/src
cp $WORK/src/singlet-py/claude__adt-modality-0637a6/python/singlet/_pz*.so $SCR/src/python/singlet/ 2>/dev/null || true
export PATH=$WORK/envs/adt/bin:$PATH LD_LIBRARY_PATH=$WORK/envs/adt/lib PYTHONPATH=$SCR/src/python HOME=$WORK/cache/condahome
OUT=$WORK/bench-public/artifacts/adt_fastcheck; mkdir -p $OUT
cp $WORK/bench-public/artifacts/adt/adt_*_panel.tsv $OUT/
python -P -m cProfile -s cumtime $SCR/src/bench/adt/adt_public_run.py --inner --dataset pbmc1k_cite_v3 --tool singlet   --data $WORK/data/adt_bench --out $OUT --scratch $SCR --threads 8 2>&1 | head -60
