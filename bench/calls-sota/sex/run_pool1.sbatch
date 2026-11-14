#!/bin/bash
#SBATCH --job-name=sexcalls-pool1
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet
$W/envs/bpub/bin/python -u $W/bench/calls-sota/sex/score_pool.py lcl_pool osorio schihost
