#!/bin/bash
#SBATCH --job-name=gp-dbl
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch doublets.sbatch <singlet-tag> [vireo donor_ids.tsv]
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modalities/genotype_pool
mkdir -p $D/results; cd $J
$W/envs/gpool/venv/bin/python doublets.py $D/truth/GSE96583_batch2.total.tsne.df.tsv.gz $D/singlet_$1 $D/results/doublets_$1.json ${2:-}
