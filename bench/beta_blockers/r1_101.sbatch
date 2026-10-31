#!/bin/bash
#SBATCH --job-name=bb-r1-101
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch r1_101.sbatch <SRR> <bin> [bin ...]
# A real 10x library whose barcode read R1 is 101 nt (GSE230239, GSM7208744: SRR24236331). Prefetch the run, then
# `singlet download --max-reads 2000000` with each binary; report reads written, R1 length and the .1fq header.
set -u
SRR=$1; shift
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
prefetch --max-size u -O $SCR/sra $SRR > $SCR/pf.log 2>&1 || { tail $SCR/pf.log; exit 1; }
SRA=$(ls $SCR/sra/$SRR/*.sra 2>/dev/null || ls $SCR/sra/*.sra)
echo "sra: $SRA $(du -h $SRA | cut -f1)"
for b in "$@"; do
  echo "=== $b"
  $PB/bin/$b download $SRA --max-reads 2000000 --vdb-threads 2 -o $SCR/$b.1fq > $SCR/$b.log 2>&1; echo "rc=$?"
  grep -i "r1\|read\|protocol\|drop\|warn\|error" $SCR/$b.log | head -20
  ls -la $SCR/$b.1fq 2>&1
  $PB/bin/$b decode $SCR/$b.1fq --head 3 2>&1 | head -12
done
