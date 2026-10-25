#!/bin/bash
#SBATCH --job-name=bp-ap-prepdeep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch prep_deep.sbatch
# The deep sample of the performance benchmark: GSM5815984 (human adrenal, 10x 3' v3, two runs, 489 M read pairs). Both runs
# are fetched with prefetch and decoded with fasterq-dump (--split-files); the 28 nt barcode read and the 91 nt cDNA read are
# identified by length, runs are concatenated in accession order and written as R1.fastq.gz / R2.fastq.gz under
# $BP/data/gsm5815984/. These FASTQs are the shared input of every tool in the timing.
set -eu
source /mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet/bench/public/harness/common.sh
export PATH=$ENVDIR/bin:$PATH HOME=$WORK/cache/condahome
SRRS="SRR17615087 SRR17615088"
OUT=$BP/data/gsm5815984; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/sra $SCR/tmp $SCR/fq $SCR/gz; trap 'rm -rf $SCR' EXIT
cd $SCR
for r in $SRRS; do
  t0=$(date +%s)
  prefetch --max-size u -O $SCR/sra $r > /dev/null 2>&1
  echo "[prep] $r prefetch $(( $(date +%s) - t0 )) s, $(stat -c %s $SCR/sra/$r/$r.sra) bytes"
  t0=$(date +%s)
  fasterq-dump --split-files -e 16 -t $SCR/tmp -O $SCR/fq $SCR/sra/$r/$r.sra > $SCR/fqd.log 2>&1
  echo "[prep] $r fasterq-dump $(( $(date +%s) - t0 )) s"; ls -la $SCR/fq
  for f in $SCR/fq/${r}*.fastq; do
    len=$(sed -n 2p $f | tr -d '\n' | wc -c)
    if [ $len -ge 26 ] && [ $len -le 28 ]; then mate=R1; elif [ $len -ge 50 ]; then mate=R2; else mate=skip; fi
    echo "[prep] $f first-read length $len -> $mate"
    [ $mate = skip ] || pigz -p 16 -c $f > $SCR/gz/$r.$mate.gz
    rm -f $f
  done
  rm -rf $SCR/sra/$r
done
for m in R1 R2; do
  cat $(for r in $SRRS; do echo $SCR/gz/$r.$m.gz; done) > $OUT/$m.fastq.gz
done
for m in R1 R2; do echo "$m: $(stat -c %s $OUT/$m.fastq.gz) bytes, reads $(zcat $OUT/$m.fastq.gz | wc -l | awk '{print $1/4}')"; done | tee $OUT/summary.txt
ls -la $OUT
