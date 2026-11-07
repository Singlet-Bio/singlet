#!/bin/bash
#SBATCH --job-name=s15-titles
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
python3 /mnt/projects/debruinz_project/debruinz/singlet/jobs/s15/ena_titles.py GSE168859 GSE168163 GSE163781 GSE188429 GSE256089 GSE298047 GSE180268 > /mnt/projects/debruinz_project/debruinz/singlet/jobs/s15/ena_titles.txt
wc -l /mnt/projects/debruinz_project/debruinz/singlet/jobs/s15/ena_titles.txt
