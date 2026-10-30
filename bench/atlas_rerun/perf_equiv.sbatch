#!/bin/bash
#SBATCH --job-name=ar-equiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch perf_equiv.sbatch <dataset> [reference]
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/pbench/bin/python -I $W/jobs/atlas-rerun/perf_equiv.py $W/pipeline-benchmark/perf/$1 ${2:-t4_r1}
