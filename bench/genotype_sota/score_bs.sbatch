#!/bin/bash
#SBATCH --job-name=gsota-sbs
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# stim pool (run on the bamsort-budget binary, 200G node): demux label table + SNP concordance.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code_gsota/public/genotype/lib
A=$W/bench-public/artifacts/genotype-sota-bs; mkdir -p $A/snps $A/demux
ln -sfn $W/bench-public/runs/kang2018_b2_ctrl $W/bench-public/runs_bs/kang2018_b2_ctrl
python -I $L/export_demux.py $W/bench-public/runs_bs $W/bench-public/data/kang_truth $A/demux
python -I $L/snp_concordance.py $W/bench-public/runs_bs $W/bench-public/data/kang_truth $A/snps/snps.tsv \
  kang2018_stim
