#!/bin/bash
#SBATCH --job-name=amb-perf
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Timing on a synthetic 15k-cell x 38k-gene library at 16 and 1 threads.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/jobs/modfix-ambient
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -std=c++20 -O2 -pthread -I$S/include $S/perf.cpp -o $S/perf
/usr/bin/time -v $S/perf 16 2>&1 | grep -E "perf|Maximum resident|Elapsed"
$S/perf 1 2>&1 | grep perf
