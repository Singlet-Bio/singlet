#!/bin/bash
#SBATCH --job-name=sjnh-mm2-setup
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# minimap2 into the kraken2 env + the SARS-CoV-2 reference (NC_045512.2) used as an independent per-read oracle.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost
export MAMBA_ROOT_PREFIX=$W/tools/mm/root
$W/tools/mm/bin/micromamba install -y -p $W/envs/kraken2 -c conda-forge -c bioconda minimap2 2>&1 | tail -3
mkdir -p $B/db; curl -s "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=nuccore&id=NC_045512.2&rettype=fasta" > $B/db/NC_045512.2.fa
head -c 200 $B/db/NC_045512.2.fa; grep -c ">" $B/db/NC_045512.2.fa; $W/envs/kraken2/bin/minimap2 --version
