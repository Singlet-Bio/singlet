#!/bin/bash
#SBATCH --job-name=sjnh-sra-setup
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:29:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# sra-tools env + look at the read structure of the GSE166766 runs (ENA serves only one read of each spot).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
export MAMBA_ROOT_PREFIX=$W/tools/mm/root HOME=$W/tools/home; mkdir -p $HOME
$W/tools/mm/bin/micromamba create -y -p $W/envs/sratools -c conda-forge -c bioconda sra-tools 2>&1 | tail -3
export PATH=$W/envs/sratools/bin:$PATH
mkdir -p $HOME/.ncbi $W/modality-bench/sj_nonhost/ncbi_cache
printf '/repository/user/main/public/root = "%s"\n/repository/user/cache-disabled = "true"\n' $W/modality-bench/sj_nonhost/ncbi_cache > $HOME/.ncbi/user-settings.mkfg
fastq-dump --version
for r in SRR13711613 SRR13711615; do echo == $r; fastq-dump -X 3 --split-files -Z $r 2>&1 | head -30; done
