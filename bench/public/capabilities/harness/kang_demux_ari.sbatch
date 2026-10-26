#!/bin/bash
#SBATCH --job-name=bp-ari
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Donor-label ARI and split/merge table of Singlet's estimated donor count on every Kang lane of the benchmark.
W=/mnt/projects/debruinz_project/debruinz/singlet; BP=$W/bench-public; H=$W/jobs/bp/harness; T=$BP/data/kang_truth
export PATH=$W/envs/ambient/bin:$PATH HOME=$W/cache/condahome
for spec in "kang2018_ctrl GSE96583_batch2 stim=ctrl" "kang2018_b1_a GSE96583_batch1 batch=A" "kang2018_b1_b GSE96583_batch1 batch=B" "kang2018_b1_c GSE96583_batch1 batch=C"; do
  set -- $spec; echo "=============== $1"
  python -I $H/kang_demux_ari.py $BP/runs/$1/singlet $T/$2.total.tsne.df.tsv.gz $3
done
