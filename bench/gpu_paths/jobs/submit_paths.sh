#!/bin/bash
# One self-running chain for the code-path corpus (login node: only sbatch runs here).
#   submit_paths.sh BIN COMMIT [EXPECT_SCRIPTS]
# 1. gpup-refs: one short-cpu task per corpus id builds the kept CPU reference of BIN (skipped when it exists).
# 2. gpu-paths: one debug-gpu job with one H100 runs every input on the GPU engine, compares, checks and writes verdicts.tsv.
# Run jobs/build_corpus.sbatch once before the first chain.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=${1:?BIN}; COMMIT=${2:?COMMIT}; ES=${3:-}
J=$W/jobs/gpu-paths/jobs; N=${NIDS:-16}
EX="ALL,BIN=$BIN,COMMIT=$COMMIT${ES:+,EXPECT_SCRIPTS=$ES}"
REFS=$(sbatch --parsable --array=0-$((N-1))%${REFPAR:-6} -p short-cpu --gres=none --mem=64G --time=03:00:00 --job-name=gpup-refs --export=$EX,MODES=cpu $J/paths.sbatch)
GPU=$(sbatch --parsable --dependency=afterany:$REFS${EXTRA_DEP:+,afterany:$EXTRA_DEP} --job-name=gpu-paths --export=$EX,MODES=gpu $J/paths.sbatch)
echo "refs=$REFS gpu=$GPU"
