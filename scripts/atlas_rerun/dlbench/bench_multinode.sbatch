#!/bin/bash
#SBATCH --job-name=s9-multinode
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9 Phase 1c: does the aggregate keep growing with node count? k = 1, 2, 4, 8 nodes start the same test at the same wall-clock second.
# Each result line is one node; the aggregate is the sum over nodes of a (tag, source, streams) group (see summarize.py).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/s9
TAG=${TAG:-nodes}
echo "== nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ')  $(date -Is)"
echo "== other running jobs on the cluster: $(squeue -h -t R | wc -l)"
for K in 1 2 4 8; do
  for SP in s3:curl:64 ncbi:curl:64 s3:curl:32 ncbi:curl:32; do
    START=$(( $(date +%s) + 8 ))
    srun --nodes=$K --ntasks=$K --ntasks-per-node=1 python3 $R/dlbench.py matrix --runs $R/runs.json --out $R/results/$TAG.jsonl \
         --spec $SP --seconds 30 --skip 10 --gap 1 --start-at $START --tag ${TAG}_k$K --node-count $K 2>&1 | grep -E " n="
  done
done
echo "== done $(date -Is)"
