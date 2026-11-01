#!/bin/bash
#SBATCH --job-name=gsota-gatenh
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=18
#SBATCH --mem=160G
#SBATCH --time=03:55:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# PR nonhost-confirmed-reads gate: base (prod/campaign) vs new binary, CPU engine, on SARS-CoV-2 dpi2 (infected), the HBEC mock and pbmc1k_v3
# (uninfected). Outputs compared byte for byte (det_compare.py); the new member must exist on dpi2 only.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SBASE=$W/jobs/gsota-r12/bin_nhbase; NEW=$W/jobs/gsota-r12/bin_nh
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
L=(dpi2 mock pbmc1k_v3); s=${L[$SLURM_ARRAY_TASK_ID]}
G=$REF/GRCh38-2024-A
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R=$W/jobs/gsota-gatenh/$s; rm -rf $R; mkdir -p $R
echo "base $(cat $SBASE/commit) new $(cat $NEW/commit)"
if [ $s = pbmc1k_v3 ]; then FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || FQ=$W/data/w11/$s/sample.1fq
else D=$W/modality-bench/sj_nonhost/data/$s; $NEW/singlet-gpu encode --reads $D/R1.fastq.gz $D/R2.fastq.gz -o $SCR/s.1fq > $R/encode.log 2>&1 || exit 1; FQ=$SCR/s.1fq; fi
for v in base new; do
  O=$R/$v; mkdir -p $O; B=$SBASE; [ $v = new ] && B=$NEW
  /usr/bin/time -v -o $O/time.txt $B/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads 18 --pipeline --output-format 1pz > $O/run.log 2>&1 || echo "EXIT $? $v"
  grep -E "Elapsed|Maximum resident" $O/time.txt; grep -a "confirmed-species reads" $O/run.log
done
echo "== member"; ls -l $R/*/nonhost/nonhost_confirmed_reads.bin 2>&1 | sed 's/.*gatenh//'
echo "== compare $s"; python3 $W/jobs/gsota-det/det_compare.py $R/base $R/new 2>&1 | head -40
