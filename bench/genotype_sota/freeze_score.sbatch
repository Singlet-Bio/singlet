#!/bin/bash
#SBATCH --job-name=gsota-frzs
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Freeze scoring (TAG as in freeze_run.sbatch): donor-demux tables, SNP concordance, mt evaluation, counts only.
#   sbatch --export=ALL,TAG=<freeze> freeze_score.sbatch
set -u
: ${TAG:?}
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code_gsota/public/genotype/lib; R=$W/bench-public/runs_$TAG
A=$W/bench-public/artifacts/genotype-sota-$TAG; mkdir -p $A/snps $A/demux $A/mt
# competitor outputs are shared across commits (same BAM construction): link the dev lane and reuse
[ -e $R/kang2018_b2_ctrl ] || ln -sfn $W/bench-public/runs/kang2018_b2_ctrl $R/kang2018_b2_ctrl
python -I $L/export_demux.py $R $W/bench-public/data/kang_truth $A/demux
python -I $L/snp_concordance.py $R $W/bench-public/data/kang_truth $A/snps/snps.tsv kang2018_b1_a kang2018_b1_b kang2018_b1_c kang2018_stim osorio2019_lcl_mix
python -I $L/mt_eval.py $R $W/bench-public/data/kang_truth $A/mt
ls -l $A/*
