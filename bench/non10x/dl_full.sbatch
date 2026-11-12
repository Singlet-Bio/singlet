#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=non10x-dl
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Stage every run of one accession ($ACC: SRR, GSM or SRS) as two FASTQ files, full depth, for run_items.sbatch (SPOTS=0).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/sratools/bin:$PATH
: "${ACC:?}"; O=$W/data/non10x/full/$ACC; mkdir -p $O; T=/tmp/$USER-dl-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
case $ACC in
  SRR*|ERR*) RUNS=$ACC ;;
  GSM*) RUNS=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/search?result=read_run&query=experiment_title=%22*$ACC*%22&fields=run_accession&format=tsv" | sed 1d | cut -f1) ;;
  *) RUNS=$(curl -s "https://www.ebi.ac.uk/ena/portal/api/filereport?accession=$ACC&result=read_run&fields=run_accession&format=tsv" | sed 1d | cut -f1) ;;
esac
echo "runs: $RUNS"; : > $O/R_1.fastq; : > $O/R_2.fastq
for R in $RUNS; do
  prefetch -q -O $T $R 2>&1 | tail -1
  fasterq-dump --split-files -e 8 -O $T -t $T $T/$R 2>&1 | tail -2 || fastq-dump --split-files -O $T $T/$R
  cat $T/${R}_1.fastq >> $O/R_1.fastq; cat $T/${R}_2.fastq >> $O/R_2.fastq; rm -f $T/${R}_*.fastq; rm -rf $T/$R
done
echo "done $(date -Is) reads: $(wc -l < $O/R_1.fastq)"
