#!/bin/bash
# Build the engine's doublet harness (dbl_run) from a pinned commit of prod/modality-fixes, whose doublet_detect.h is the v7 scorer
# the combiner was trained on (prod/campaign still has the v6 one). Usage: build_dbl_run.sh OUT_DIR [GIT_DIR] [COMMIT]
# Linked statically so it also runs inside the python image. Writes OUT_DIR/dbl_run and OUT_DIR/dbl_run.commit. Runs inside a Slurm job (g++ from the pbench env).
set -euo pipefail
OUT=${1:?OUT_DIR}
WORK=/mnt/projects/debruinz_project/debruinz/singlet
GIT=${2:-$WORK/repos/singlet-py.git}
COMMIT=${3:-4f110f283cd4b1bc4865d1103a08b56346027da3}
CXX=${CXX:-$WORK/envs/pbench/bin/x86_64-conda-linux-gnu-g++}
mkdir -p "$OUT"; T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
git --git-dir="$GIT" archive "$COMMIT" include/singlet/pileup bench/modalities/doublet_fix/dbl_run.cpp | tar -x -C "$T"
"$CXX" -O2 -std=c++17 -pthread -static -I"$T/include/singlet/pileup" "$T/bench/modalities/doublet_fix/dbl_run.cpp" -o "$OUT/dbl_run"
echo "$COMMIT" > "$OUT/dbl_run.commit"
echo "built $OUT/dbl_run from $COMMIT"
