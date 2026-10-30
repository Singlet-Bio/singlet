#!/bin/bash
#SBATCH --job-name=gp-evald
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch eval_demux.sbatch OUT.json name=path [name=path ...]
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modalities/genotype_pool
mkdir -p $D/results; cd $J; out=$1; shift
$W/envs/gpool/venv/bin/python eval_demux.py $D/truth/GSE96583_batch2.total.tsne.df.tsv.gz $D/results/$out "$@"
