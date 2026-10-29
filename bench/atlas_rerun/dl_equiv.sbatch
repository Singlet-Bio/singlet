#!/bin/bash
#SBATCH --job-name=ar-dlequiv
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch dl_equiv.sbatch <SRR>
# Is `prefetch` + `singlet download <local .sra>` the same as the campaign's remote
# `singlet download <SRR>`? Both on the first 2M reads; decoded FASTQ compared by checksum.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
BIN=$W/pipeline-benchmark/bin/singlet-fixed; r=$1
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/remote $SCR/local; trap 'rm -rf $SCR' EXIT; cd $SCR
$BIN download $r -o remote/$r.1fq --max-reads 2000000 > remote.log 2>&1
prefetch --max-size u -O sra $r > /dev/null 2>&1
$BIN download sra/$r/$r.sra -o local/$r.1fq --max-reads 2000000 > local.log 2>&1
for m in remote local; do
  $BIN decode $m/$r.1fq --r1 $m/R1.fq --r2 $m/R2.fq > /dev/null 2>&1
  echo "$m: 1fq $(md5sum < $m/$r.1fq | cut -c1-12)  R1 $(md5sum < $m/R1.fq | cut -c1-12)  R2 $(md5sum < $m/R2.fq | cut -c1-12)  reads $(( $(wc -l < $m/R1.fq) / 4 ))"
done
cmp -s remote/R1.fq local/R1.fq && cmp -s remote/R2.fq local/R2.fq && echo "DECODED READS IDENTICAL" || echo "DECODED READS DIFFER"
