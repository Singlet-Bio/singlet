#!/bin/bash
# Sourced by the celltype_nn job scripts on Clipper: the runtime, the synced source tree and the extra package dirs (Census
# client, CUDA torch: pip --target, outside the shared venv). sync.sbatch puts this branch at $SRC first.
WORK=/mnt/projects/debruinz_project/debruinz/singlet
SRC=${SRC:-$WORK/src/singlet-py/claude__celltype-nn}
D=${D:-$WORK/data/celltype_nn}
source $WORK/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
CENSUS=$WORK/venvs/ctnn-census
TORCH=$WORK/venvs/ctnn-torch
export PYTHONPATH=$SRC/python OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1} SINGLET_CACHE_DIR=/tmp/$USER-ctnn-cache
echo "celltype_nn @ $(git -C $SRC rev-parse --short HEAD)"
# the same runtime with the node's GPUs (apptainer --nv)
in_runtime_gpu() {
  if [ -z "${SLURM_JOB_ID:-}" ]; then echo "in_runtime_gpu: refusing to run outside a Slurm job" >&2; return 2; fi
  apptainer exec --nv --bind "$WORK,/tmp" --env "PATH=$SINGLET_VENV/bin:/usr/local/bin:/usr/bin:/bin" \
    --env "OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}" --env "HDF5_USE_FILE_LOCKING=FALSE" --env "WORK=$WORK" "$SINGLET_IMG" "$@"
}
