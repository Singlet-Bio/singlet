#!/bin/bash
#SBATCH --job-name=sjnh-kraken-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Kraken2 env (micromamba, under singlet/envs) + prebuilt viral-only DB (Langmead indexes, k2_viral 2024-06-05).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet
export MAMBA_ROOT_PREFIX=$W/tools/mm/root
MM=$W/tools/mm/bin/micromamba
$MM create -y -p $W/envs/kraken2 -c conda-forge -c bioconda kraken2 seqkit samtools 2>&1 | tail -5
$W/envs/kraken2/bin/kraken2 --version | head -2
D=$W/modality-bench/sj_nonhost/db/k2_viral; mkdir -p $D; cd $D
curl -sSL --retry 3 -o k2_viral.tar.gz https://genome-idx.s3.amazonaws.com/kraken/k2_viral_20240605.tar.gz
tar xzf k2_viral.tar.gz && rm k2_viral.tar.gz; ls -la $D; cat $D/inspect.txt 2>/dev/null | head -5
