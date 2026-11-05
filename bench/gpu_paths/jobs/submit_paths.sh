#!/bin/bash
# One self-running chain for the code-path corpus (login node: only sbatch runs here).
#   submit_paths.sh BIN COMMIT [EXPECT_SCRIPTS]
# 1. gpup-refs: one short-cpu task per corpus id builds the kept CPU reference of BIN (skipped when it exists, or with SKIP_REFS=1).
# 2. gpu-paths: one debug-gpu job with one H100 runs every input on the GPU engine, compares, checks and writes verdicts.tsv.
# Environment: EXCLUDE=g050,g051  nodes the GPU job must not use (a node that holds a timing job of another session; check
#              `squeue -w <node>` before submitting). EXTRA_DEP=<job ids>  the GPU job also waits for them (afterany).
#              NIDS (default 16), REFPAR (default 6).
# Run jobs/build_corpus.sbatch once before the first chain.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=${1:?BIN}; COMMIT=${2:?COMMIT}; ES=${3:-}
J=$W/jobs/gpu-paths/jobs; N=${NIDS:-16}
EX="ALL,BIN=$BIN,COMMIT=$COMMIT${ES:+,EXPECT_SCRIPTS=$ES}"
DEP=""
if [ "${SKIP_REFS:-0}" != 1 ]; then
  REFS=$(sbatch --parsable --array=0-$((N-1))%${REFPAR:-6} -p short-cpu --gres=none --mem=64G --time=03:00:00 --job-name=gpup-refs --export=$EX,MODES=cpu $J/paths.sbatch)
  DEP="afterany:$REFS"
fi
[ -n "${EXTRA_DEP:-}" ] && DEP="${DEP:+$DEP,}afterany:${EXTRA_DEP//,/,afterany:}"
GPU=$(sbatch --parsable ${DEP:+--dependency=$DEP} ${EXCLUDE:+--exclude=$EXCLUDE} --job-name=gpu-paths --export=$EX,MODES=gpu $J/paths.sbatch)
echo "refs=${REFS:-none} gpu=$GPU"
