#!/bin/bash
# End-to-end run of bench/doublets/doublets_run.py on two real samples of the store (GSE138867, blood, 8.1k + 6.6k cells; timing, peak memory,
# sidecar guard), then parity_kang.py (the code against the cached study scores on Kang lane 2.1).
# The job assembles its python tree from the synced celltype-nn worktree (compiled readers) plus this branch's bench/doublets/legacy_doublets
# (CODE = a directory with bench/doublets/legacy_doublets, bench/doublets/doublets_run.py, bench/doublets). Submit:
#   sbatch --export=ALL,CODE=$WORK/jobs/dbl-train bench/doublets/e2e.sbatch
#SBATCH --job-name=dbl-e2e
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
CODE=${CODE:-$WORK/jobs/dbl-train}   # code.tgz: python/singlet/{doublets,features,celllabels,s1,celltype_nn}, scripts, bench/doublets
OUT=${OUT:-$WORK/bench-public/artifacts/doublets/e2e}
BASE=$WORK/src/singlet-py/claude__celltype-nn
SIF=$WORK/images/python-3.12-bookworm.sif; VENV=$WORK/venvs/singlet; TORCH=$WORK/venvs/ctnn-torch
MODEL=$WORK/data/celltype_nn/models/m1_notissue/model.pt; CEN=$WORK/latent_qc/dbl/atlas_centroids.npz
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH "$OUT"; trap 'rm -rf $SCRATCH' EXIT
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
mkdir -p $SCRATCH/code && tar xzf $CODE/code.tgz -C $SCRATCH/code; CODE=$SCRATCH/code
# 1. the engine's doublet harness (v7 scorer of prod/modality-fixes), built on the node
bash $CODE/bench/doublets/build_dbl_run.sh $SCRATCH/dbl
# 2. python tree: celltype-nn worktree + the doublets package and its combiner
PY=$SCRATCH/py; mkdir -p $PY; cp -r $BASE/python/singlet $PY/
mkdir -p $SCRATCH/repo; cp -r $CODE/bench $SCRATCH/repo/; ln -s $PY $SCRATCH/repo/python
rt() { apptainer exec --nv --bind "$WORK,/tmp" --env "PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin" \
  --env "PYTHONPATH=$PY:$TORCH" --env "OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK" --env "SINGLET_CACHE_DIR=$SCRATCH/cache" \
  --env "PYTHONWARNINGS=ignore" "$SIF" "$@"; }
nvidia-smi -L || true
rt python -u $SCRATCH/repo/bench/doublets/doublets_run.py run --gse GSE138867 --gsm GSM4120733 GSM4120734 --out $OUT/sidecars \
  --model $MODEL --centroids $CEN --dbl-run $SCRATCH/dbl/dbl_run --celltype-version m1_notissue --dbl-commit "$(cat $SCRATCH/dbl/dbl_run.commit)" \
  --threads $SLURM_CPUS_PER_TASK --force
cat $OUT/sidecars/GSE138867/GSM4120733.doublets.json $OUT/sidecars/GSE138867/GSM4120734.doublets.json
rt env PYTHONPATH=$PY:$TORCH:$CODE/bench/doublets python $CODE/bench/doublets/parity_kang.py $MODEL $CEN $SCRATCH/dbl/dbl_run $OUT/parity_kang.json
