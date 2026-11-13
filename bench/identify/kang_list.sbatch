#!/bin/bash
#SBATCH --job-name=identify-kang-list
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Metadata only: the files ENA lists for study PRJNA381100 (Kang 2018 author BAMs). One request.
set -eu
OUT=/mnt/projects/debruinz_project/debruinz/singlet/jobs/identify/bamhead/out; mkdir -p $OUT
curl -sf --max-time 60 "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA381100&result=read_run&fields=run_accession,sample_alias,library_strategy,submitted_ftp,submitted_format,submitted_bytes,fastq_ftp&format=tsv" > $OUT/kang_filereport.tsv
wc -l $OUT/kang_filereport.tsv; cut -c1-300 $OUT/kang_filereport.tsv | head -20
