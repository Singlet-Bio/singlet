#!/bin/bash
#SBATCH --job-name=sjnh-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch fetch_gse166766.sbatch SRR_ACCESSION LABEL N_SPOTS
# First N spots of one GSE166766 run (spot = I1 8 nt, R1 28 nt = CB+UMI, R2 91 nt) via fastq-dump streaming; R1/R2 gzip.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; B=$W/modality-bench/sj_nonhost
export HOME=$W/tools/home PATH=$W/envs/sratools/bin:$PATH
ACC=$1; LAB=$2; N=$3; O=$B/data/$LAB; rm -rf $O/raw; mkdir -p $O $B/ncbi_cache/tmp; cd $B/ncbi_cache/tmp
date; /usr/bin/time -v -o $O/fetch.time.txt fastq-dump -X $N --split-files -O $O/raw $ACC
ls -la $O/raw; date
rm -f $O/raw/${ACC}_1.fastq; mv $O/raw/${ACC}_2.fastq $O/R1.fastq; mv $O/raw/${ACC}_3.fastq $O/R2.fastq
head -2 $O/R1.fastq $O/R2.fastq
gzip -1 $O/R1.fastq & gzip -1 $O/R2.fastq & wait; rm -rf $O/raw; ls -la $O; grep -E "Elapsed|Maximum res" $O/fetch.time.txt
