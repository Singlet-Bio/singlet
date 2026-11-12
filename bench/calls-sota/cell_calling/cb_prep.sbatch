#!/bin/bash
#SBATCH --job-name=cc-cbprep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/bench/calls-sota/cell_calling
cd $S/probe_nuc; [ -s POC_nuc_genes.tsv ] || tar xzf proc.tsv.tar.gz POC_nuc_genes.tsv; $W/envs/gpool/venv/bin/python -P $S/cb_prep.py $S/cb_in
