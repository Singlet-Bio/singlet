#!/bin/bash
#SBATCH --job-name=mtfix-inspect
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=24G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modfix-mt; O=$D/mt_fix
for r in "$@"; do
  $W/envs/gpool/venv/bin/python $J/inspect_sites.py $D/truth/GSE96583_batch2.total.tsne.df.tsv.gz $D/ref_k8/mt_pileup.npz $O/$r $J/results/sites_$r.tsv
done
