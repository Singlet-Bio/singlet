#!/bin/bash
#SBATCH --job-name=s9-bench
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9 Phase 1b/c/d: download throughput, one script for every experiment. Env: TAG, MODE (single|multi), SPECS (space separated
# source:tool:streams,...), NODES (>1: one srun task per node, started at the same wall-clock second), SECONDS_PER_TEST.
# Results: $W/jobs/s9/results/$TAG.jsonl (one line per test per node). Raw downloads live in node /tmp and are deleted per test.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
R=$W/jobs/s9; mkdir -p $R/results
TAG=${TAG:-single}; MODE=${MODE:-single}; SECS=${SECONDS_PER_TEST:-25}
SPECS=${SPECS:-"ncbi:curl:1,4,8,16,32,64 ncbi:aria2:1,4,8,16,32,64 s3:curl:1,4,8,16,32,64 s3:aria2:1,4,8,16,32,64 s3:s5cmd:1,4,8,16,32,64 gcs:curl:1,8"}
echo "== $TAG mode=$MODE nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ') $(date -Is)"
echo "== competing downloads: $(squeue -h -t R -o '%u %j' | grep -ci -E 'dl|download|fetch|prefetch|io-|sra' || true) matching jobs; $(squeue -h -t R | wc -l) running overall"
if [ ! -s $R/runs.json ]; then python3 $R/dlbench.py prep --runs $R/runs.json --n 80; fi
ARGS=""; for s in $SPECS; do ARGS="$ARGS --spec $s"; done
START=$(( $(date +%s) + 15 ))
srun --ntasks-per-node=1 python3 $R/dlbench.py matrix --runs $R/runs.json --out $R/results/$TAG.jsonl $ARGS --mode $MODE --seconds $SECS \
     --start-at $START --tag $TAG --node-count ${SLURM_NNODES:-1}
echo "== done $(date -Is)"
