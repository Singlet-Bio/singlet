#!/bin/bash
#SBATCH --job-name=sjnh-r2comp
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Is there a constant 5' prefix in PBMC 1k v3 R2? Base frequencies at positions 1-14 over the first 200k reads.
F=/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark/data/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001_R2_001.fastq.gz
zcat $F | head -800000 | awk 'NR%4==2{for(i=1;i<=14;i++){c[i, substr($0,i,1)]++}; n++} END{for(i=1;i<=14;i++){printf "pos%d", i; split("A C G T N",b," "); for(k=1;k<=5;k++) printf " %s=%.2f", b[k], c[i,b[k]]/n; printf "\n"}}'
