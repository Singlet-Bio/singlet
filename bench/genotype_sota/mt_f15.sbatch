#!/bin/bash
#SBATCH --job-name=gsota-mt
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# mtDNA evaluation (haplotype, voi, linemix) on the tip runs: runs_f15 plus the stim run of runs_bs, counts only.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code_gsota/public/genotype/lib
A=$W/bench-public/artifacts/genotype-sota-f15/mt; mkdir -p $A
rm -rf $W/bench-public/runs_f15/kang2018_b2_stim/singlet_auto; ln -sfn $W/bench-public/runs_bs/kang2018_b2_stim/singlet_auto $W/bench-public/runs_f15/kang2018_b2_stim/singlet_auto
python -I $L/mt_eval.py $W/bench-public/runs_f15 $W/bench-public/data/kang_truth $A
ls -l $A
