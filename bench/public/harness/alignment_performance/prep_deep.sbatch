#!/bin/bash
#SBATCH --job-name=bp-ap-prepdeep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch prep_deep.sbatch
# The deep sample of the performance benchmark: GSM5815984 (human adrenal, 10x 3' v3, two runs, 489 M read pairs, all reads).
# The archive stores the barcode read as a technical read that fasterq-dump does not write (ENA has only the cDNA file), so the runs
# are decoded with Singlet itself: `prefetch`, then `singlet download <local .sra> --no-trim --quality full` (lossless: no poly(A) or
# adapter trimming, base qualities kept) and `singlet decode` to R1 (barcode + UMI, 28 nt) and R2 (cDNA, 91 nt), compressed with pigz.
# Runs are concatenated in accession order into $BP/data/gsm5815984/R1.fastq.gz and R2.fastq.gz, the shared input of every tool.
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
export PATH=$ENVDIR/bin:$PATH HOME=$WORK/cache/condahome
verify_bin
SRRS="SRR17615087 SRR17615088"
OUT=$BP/data/gsm5815984; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/sra $SCR/gz; trap 'rm -rf $SCR' EXIT
cd $SCR
for r in $SRRS; do
  t0=$(date +%s)
  prefetch --max-size u -O $SCR/sra $r > /dev/null 2>&1
  echo "[prep] $r prefetch $(( $(date +%s) - t0 )) s, $(stat -c %s $SCR/sra/$r/$r.sra) bytes"
  t0=$(date +%s)
  $BIN download $SCR/sra/$r/$r.sra -o $SCR/$r.1fq --no-trim --quality full --vdb-threads 8 --no-profile > $SCR/dl_$r.log 2>&1
  echo "[prep] $r singlet download $(( $(date +%s) - t0 )) s, $(stat -c %s $SCR/$r.1fq) bytes"; tail -3 $SCR/dl_$r.log
  rm -rf $SCR/sra/$r
  t0=$(date +%s)
  $BIN decode $SCR/$r.1fq --r1 >(pigz -p 8 > $SCR/gz/$r.R1.gz) --r2 >(pigz -p 8 > $SCR/gz/$r.R2.gz) --threads 8 > $SCR/dec_$r.log 2>&1
  wait
  echo "[prep] $r decode $(( $(date +%s) - t0 )) s"; tail -3 $SCR/dec_$r.log
  rm -f $SCR/$r.1fq
done
for m in R1 R2; do
  cat $(for r in $SRRS; do echo $SCR/gz/$r.$m.gz; done) > $OUT/$m.fastq.gz
done
for m in R1 R2; do echo "$m: $(stat -c %s $OUT/$m.fastq.gz) bytes, reads $(zcat $OUT/$m.fastq.gz | wc -l | awk '{print $1/4}'), first read length $(zcat $OUT/$m.fastq.gz | sed -n 2p | tr -d '\n' | wc -c)"; done | tee $OUT/summary.txt
ls -la $OUT
