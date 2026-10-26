#!/bin/bash
#SBATCH --job-name=bp-dl-lossy
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch lossy_diagnose.sbatch [SRR ...]      default: SRR24098804 SRR25032524
# Decodes each run three ways (fasterq-dump, singlet default, singlet --no-trim --quality full) and runs lossy_diagnose.py on the
# first 8 M read pairs of the cDNA files (the .sra files of dl_reference.sbatch are reused).
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
verify_bin
RUNS=${*:-"SRR24098804 SRR25032524"}
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
for r in $RUNS; do
  SRA=$BP/download/reference/sra/$r/$r.sra
  $BIN download $SRA -o d.1fq --no-profile > d.log 2>&1
  $BIN download $SRA -o l.1fq --no-trim --quality full --no-profile > l.log 2>&1
  $BIN decode d.1fq --r1 d.R1.fq --r2 d.R2.fq --threads 8 > /dev/null 2>&1
  $BIN decode l.1fq --r1 l.R1.fq --r2 l.R2.fq --threads 8 > /dev/null 2>&1
  export PATH=$ENVDIR/bin:$PATH
  fasterq-dump --split-files -e 8 -t tmp -O ref $SRA > /dev/null 2>&1
  R=$(ls -S ref/${r}_*.fastq | head -1)   # the cDNA file is the one with the longest reads (largest)
  echo "=========== $r  reference file $R"; grep -hE "trim|Trim|quality|Quality|dedup|swap" d.log | head -8
  $PY -I $SRC/bench/public/harness/download/lossy_diagnose.py $R d.R2.fq l.R2.fq
  rm -rf ref d.* l.*
done 2>&1 | tee $BP/download/lossy_diagnose.txt
