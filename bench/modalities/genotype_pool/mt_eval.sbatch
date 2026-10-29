#!/bin/bash
#SBATCH --job-name=gp-mteval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch mt_eval.sbatch <singlet-tag>
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modalities/genotype_pool
mkdir -p $D/results; cd $J
$W/envs/gpool/venv/bin/python mt_eval.py $D/truth/GSE96583_batch2.total.tsne.df.tsv.gz $D/singlet_$1 $D/ref_$1/mt_pileup.npz $D/results/mt_$1.json
