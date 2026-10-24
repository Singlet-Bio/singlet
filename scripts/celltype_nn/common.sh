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
