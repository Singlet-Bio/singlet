#!/bin/bash
#SBATCH --job-name=txf-equiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch txf_equiv.sbatch <dataset> [a=off_r1] [b=on_r1]
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PYTHONPATH=$W/envs/pylib-pysam
$W/envs/pbench/bin/python $W/jobs/txfirst/txf_equiv.py $W/pipeline-benchmark/perf/$1_txf ${2:-off_r1} ${3:-on_r1}
