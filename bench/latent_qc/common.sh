#!/bin/bash
# Sourced by every bench/latent_qc job on Clipper. Syncs this study's branch and a pinned prod/modality-fixes checkout (for the
# production doublet / ambient headers), and defines the runtimes:
#   rt   python 3.12 image + venvs/singlet (scanpy, sklearn, pyarrow, leidenalg)
#   rtt  rt + CUDA torch (venvs/ctnn-torch, pip --target dir) and the GPUs (apptainer --nv)
#   R    envs/ambient (R 4.4: scDblFinder, celda/DecontX, SoupX, scran), run outside the image
# Data:  $LQC (results, inputs) under $WORK/latent_qc; nothing is uploaded from the laptop and nothing is written to R2 / D1.
WORK=/mnt/projects/debruinz_project/debruinz/singlet
STUDY_BRANCH=${STUDY_BRANCH:-study/latent-qc}
REPO=$WORK/repos/singlet-py.git
SRC=$WORK/src/singlet-py/${STUDY_BRANCH//\//__}
PROD=$WORK/src/singlet-py/lqc-prod                     # detached at origin/prod/modality-fixes
EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070   # fetch.py + in-place _pz build (empty-droplet study)
LQC=${LQC:-$WORK/latent_qc}
SIF=$WORK/images/python-3.12-bookworm.sif
VENV=$WORK/venvs/singlet
TORCH=$WORK/venvs/ctnn-torch
RENV=$WORK/envs/ambient
CXX=$WORK/envs/pbench/bin/x86_64-conda-linux-gnu-g++
export XDG_CACHE_HOME=$WORK/cache PIP_CACHE_DIR=$WORK/cache/pip NUMBA_CACHE_DIR=$WORK/cache/numba-lqc MPLCONFIGDIR=$WORK/cache/mpl
mkdir -p "$LQC"
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }

lqc_sync() {
  (
    flock -w 3600 9
    git --git-dir="$REPO" fetch -q origin "+refs/heads/$STUDY_BRANCH:refs/heads/$STUDY_BRANCH" "+refs/heads/prod/modality-fixes:refs/lqc/prod"
    for pair in "$SRC:refs/heads/$STUDY_BRANCH" "$PROD:refs/lqc/prod"; do
      d=${pair%%:*}; ref=${pair#*:}; tip=$(git --git-dir="$REPO" rev-parse "$ref")
      [ -e "$d/.git" ] || git --git-dir="$REPO" worktree add -q --detach "$d" "$tip"
      [ "$(git -C "$d" rev-parse HEAD)" = "$tip" ] || git -C "$d" checkout -q --detach "$tip"
    done
  ) 9>"$WORK/state/lqc-sync.lock"
  echo "latent_qc @ $(git -C "$SRC" rev-parse --short HEAD), prod @ $(git -C "$PROD" rev-parse --short HEAD), host $(hostname)"
}

_rt_env=(--env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}"
         --env "HDF5_USE_FILE_LOCKING=FALSE" --env "WORK=$WORK" --env "LQC=$LQC" --env "TMPDIR=${TMPDIR:-/tmp}")
rt() {
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "rt: refusing to run outside a Slurm job" >&2; return 2; }
  apptainer exec --bind "$WORK,/tmp" --env "PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin" --env "PYTHONPATH=$SRC/bench/latent_qc" \
    "${_rt_env[@]}" "$SIF" "$@"
}
rtt() {
  [ -n "${SLURM_JOB_ID:-}" ] || { echo "rtt: refusing to run outside a Slurm job" >&2; return 2; }
  apptainer exec --nv --bind "$WORK,/tmp" --env "PATH=$VENV/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "PYTHONPATH=$SRC/bench/latent_qc:$TORCH:$WORK/src/singlet-py/claude__celltype-nn/python" "${_rt_env[@]}" "$SIF" "$@"
}
R() { PATH=$RENV/bin:$PATH HOME=$WORK/cache/condahome Rscript "$@"; }
