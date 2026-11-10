#!/bin/bash
#SBATCH --job-name=lr-enc27
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
# Auto-detect check on the 27-nt-R1 (16 CB + 11 UMI) PBMC-E pbmc1 reads WITHOUT padding and WITHOUT --protocol, on a tip build that contains S5's fix (#124).
# sbatch --export=ALL,PIN8=<sha8> encode_unpadded.sbatch ; encodes the first 2 M read pairs only and prints the detected protocol.
set -u
source /mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa/code/common.sh
BIN=/mnt/projects/debruinz_project/debruinz/singlet/release-gate/bin/${PIN8}/singlet-cpu
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
f1=${R1%%,*}; f2=${R2%%,*}
zcat $f1 | head -8000000 | gzip -1 > $SCR/R1.fastq.gz; zcat $f2 | head -8000000 | gzip -1 > $SCR/R2.fastq.gz
echo "bin $BIN"; $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/x.1fq 2>&1 | grep -i -E "protocol|chemistry|detect|confidence|umi|whitelist|error" | head -12
