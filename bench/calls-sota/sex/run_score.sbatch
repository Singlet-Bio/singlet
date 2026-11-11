#!/bin/bash
#SBATCH --job-name=sexcalls-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# usage: sbatch run_score.sbatch <sets> <tag>
export SETS=geo TAG=t
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/bpub/bin/python -u $W/bench/calls-sota/sex/score.py --sets "${SETS:-geo}" --tag "${TAG:-t}"
