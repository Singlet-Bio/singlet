#!/bin/bash
#SBATCH --job-name=rg-gpudiff
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=120G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; RG=$W/release-gate
$W/envs/pbench/bin/python -P $RG/jobs/gpu_diff.py "$W/pipeline-benchmark/runs/{ds}/singlet-public" $RG/gpu $RG/results/gpu_vs_cpu.json
