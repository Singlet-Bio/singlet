#!/bin/bash
#SBATCH --job-name=sexcalls-guard
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/bpub/bin/python -u $W/bench/calls-sota/sex/guard_analysis.py
