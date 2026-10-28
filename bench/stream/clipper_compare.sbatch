#!/bin/bash
# Streaming loaders on one Clipper GPU node: singlet.torch.AtlasCells (data.singlet.bio) and the CELLxGENE Census loader
# (tiledbsoma-ml), each into a GPU training step and as a bare stream. Code comes from GitHub (BRANCH), data from the two services.
#SBATCH --job-name=ml-loaders
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
BRANCH=${BRANCH:-claude/loader-compare}
SECONDS_PER_RUN=${SECONDS_PER_RUN:-600}
SINGLET_IMG=$WORK/images/python-3.12-bookworm.sif
TORCH=$WORK/venvs/ctnn-torch            # torch 2.6.0+cu124 (pip --target)
CENSUS=$WORK/venvs/ctnn-census          # cellxgene-census 1.18, tiledbsoma 2.3
EXTRA=$WORK/venvs/annotate-extra        # setuptools for the extension build
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
OUT=$WORK/out/ml_loaders/$SLURM_JOB_ID; mkdir -p $OUT
rt() { apptainer exec --nv --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "OMP_NUM_THREADS=8" --env "HOME=$SCRATCH" "$SINGLET_IMG" "$@"; }

echo "node $(hostname) $(date -Is)"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
git clone -q --depth 1 --branch "$BRANCH" https://github.com/Singlet-Bio/singlet.git $SCRATCH/singlet
SRC=$SCRATCH/singlet; git -C $SRC log --oneline -1
rt env PYTHONPATH=$EXTRA bash -c "cd $SRC && python setup.py build_ext --inplace -q 2>&1 | tail -3"
# tiledbsoma-ml (the Census PyTorch loader) next to the existing census env, without touching it
TSML=$SCRATCH/tsml; rt python -m pip install -q --no-deps --target $TSML tiledbsoma-ml 2>&1 | tail -2
PY_S="$SRC/python:$TORCH"
PY_C="$TSML:$CENSUS:$TORCH"
rt env PYTHONPATH=$PY_S python -c "import torch, singlet._pz2, singlet.torch; print('torch', torch.__version__, torch.cuda.is_available())"
rt env PYTHONPATH=$PY_C python -c "import tiledbsoma_ml, cellxgene_census, tiledbsoma; print('tsml', tiledbsoma_ml.__version__, 'soma', tiledbsoma.__version__)"

cd $SRC
run() { echo "== $*"; rt env PYTHONPATH=$1 python -u bench/stream/compare_loaders.py "${@:2}" --seconds $SECONDS_PER_RUN 2>&1 \
  | grep -v -E "UserWarning|sparse_c|^ *return" || true; }
run $PY_S --loader singlet --step train --workers 8 --prefetch 4 --out $OUT/singlet_gpu.json
run $PY_C --loader census --step train --workers 8 --out $OUT/census_gpu.json
run $PY_S --loader singlet --step none --workers 16 --prefetch 4 --seed 1 --out $OUT/singlet_stream.json
run $PY_C --loader census --step none --workers 16 --seed 1 --out $OUT/census_stream.json
ls -la $OUT; echo "done $(date -Is)"
