#!/bin/bash
#SBATCH --job-name=gsota-detcmp
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet/bench-public
for rd in kang2018_b2_stim kang2018_b1_a; do echo "== $rd"; python3 $W/../jobs/gsota-det/det_compare.py $W/runs_tip/$rd/singlet_auto $W/runs_det/$rd/singlet_rep2 2>&1 | head -60; done
