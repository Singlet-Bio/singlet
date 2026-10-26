#!/bin/bash
#SBATCH --job-name=bpg-export
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch export.sbatch <step>      step: demux | snps | mt
# Builds the small public artifacts under $WORK/bench-public/artifacts/<capability>/ from the run directories.
# The snps and mt steps read private files and write counts only.
set -u
STEP=$1
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
L=$W/bench-public/code/public/genotype/lib; R=$W/bench-public/runs; T=$W/bench-public/data/kang_truth; A=$W/bench-public/artifacts
case $STEP in
  demux) mkdir -p $A/donor-demux; python -I $L/export_demux.py $R $T $A/donor-demux ;;
  snps)  mkdir -p $A/donor-snps; python -I $L/snp_concordance.py $R $T $A/donor-snps/donor_snps_counts.tsv kang2018_ctrl kang2018_stim kang2018_b1_a kang2018_b1_b kang2018_b1_c ;;
  mt)    mkdir -p $A/mtdna; python -I $L/mt_eval.py $R $T $A/mtdna ;;
esac
ls -la $A/donor-demux $A/donor-snps $A/mtdna 2>/dev/null
