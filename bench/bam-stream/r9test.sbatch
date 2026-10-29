#!/bin/bash
#SBATCH --job-name=bs-r9test
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,REF=perf/bam-stream r9test.sbatch
# Builds and runs tests/test_umi_r9.cpp (order-free-rules R9 unit test) from a fresh clone of REF in job /tmp.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/pbench
S=/tmp/$USER-r9t-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch ${REF:-perf/bam-stream} git@github.com:Singlet-Bio/singlet.git $S/src
git -C $S/src log --oneline -1
$E/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -Wall -I$S/src/include $S/src/tests/test_umi_r9.cpp -o $S/t && $S/t
