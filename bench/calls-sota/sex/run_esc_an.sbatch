#!/bin/bash
#SBATCH --job-name=sexcalls-escan
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/bpub/bin/python -u $W/bench/calls-sota/sex/esc_analysis.py
