#!/bin/bash
#SBATCH --job-name=s9-tod
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9 Phase 1d: the best configuration at another time of day. One node (96 curl streams, both sources), then 4 nodes (64 streams each).
# Submit with --begin=<time>; the tag records the hour. Total volume per run is about 120 GB.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/s9
TAG=tod_$(date +%a%H)
echo "== $TAG nodes: $(scontrol show hostnames $SLURM_JOB_NODELIST | tr '\n' ' ') running jobs on cluster: $(squeue -h -t R | wc -l)"
squeue -h -t R -o '%u %j' | awk '{print $1}' | sort | uniq -c | sort -rn | head -5
START=$(( $(date +%s) + 8 ))
srun --nodes=1 --ntasks=1 python3 $R/dlbench.py matrix --runs $R/runs.json --out $R/results/tod.jsonl --spec ncbi:curl:96 --spec s3:curl:96 \
     --seconds 30 --skip 10 --gap 1 --start-at $START --tag ${TAG}_k1 --node-count 1 | grep -E " n="
START=$(( $(date +%s) + 8 ))
srun --nodes=4 --ntasks=4 --ntasks-per-node=1 python3 $R/dlbench.py matrix --runs $R/runs.json --out $R/results/tod.jsonl --spec s3:curl:64 \
     --seconds 30 --skip 10 --gap 1 --start-at $START --tag ${TAG}_k4 --node-count 4 | grep -E " n="
echo "== done $(date -Is)"
