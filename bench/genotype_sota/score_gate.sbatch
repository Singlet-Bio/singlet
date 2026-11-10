#!/bin/bash
#SBATCH --job-name=gsota-sgate
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Gate-binary scoring: counts-only SNP concordance tables + donor-demux label tables from runs_gate.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code_gsota/public/genotype/lib
A=$W/bench-public/artifacts/genotype-sota-gate; mkdir -p $A/snps $A/demux
python -I $L/export_demux.py $W/bench-public/runs_gate $W/bench-public/data/kang_truth $A/demux
python -I $L/snp_concordance.py $W/bench-public/runs_gate $W/bench-public/data/kang_truth $A/snps/snps.tsv \
  kang2018_b1_a kang2018_b1_b kang2018_b1_c kang2018_stim osorio2019_lcl_mix
