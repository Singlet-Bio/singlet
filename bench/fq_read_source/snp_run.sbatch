#!/bin/bash
#SBATCH --job-name=fqrs-snp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
#   sbatch --export=ALL,BIN=singlet-fqrs-<tag> snp_run.sbatch
# SNP false-alt on the donor-snps benchmark's Kang batch-2 IFN-beta lane (kang2018_stim; truth = the same donors'
# genotypes from the control lane, bench/public-genotype). The lane's FASTQs are encoded with this branch twice:
#   0 = --quality none   (constant 'F' downstream: today's production .1fq from `singlet download`)
#   1 = default          (ILLUMINA4 binned qualities)
# and run as bench/public/genotype/hpc/run_singlet.sbatch does (auto K, default 1000G panel, --pipeline).
# Output: $W/jobs/fq-read-source/snp/<variant>/kang2018_b2_stim/singlet_auto/ (private/ stays on Clipper).
set -u
V=( none ilmn4 ); QA=( none binned ); VAR=${V[$SLURM_ARRAY_TASK_ID]}
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
BIN=$W/jobs/fq-read-source/bin/$BIN; echo "bin $(sha256sum $BIN | cut -c1-16) $(cat $BIN.commit)"
FQ=$W/bench-public/data/kang_b2stim/fastq
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
O=$W/jobs/fq-read-source/snp/$VAR/kang2018_b2_stim/singlet_auto; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists SINGLET_STALL_TIMEOUT=10800 TMPDIR=$SCR
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $FQ/R1.fastq.gz $FQ/R2.fastq.gz -o $SCR/s.1fq \
  --quality ${QA[$SLURM_ARRAY_TASK_ID]} > $O/encode.log 2>&1 || { tail -5 $O/encode.log; exit 1; }
echo "1fq bytes $(stat -c %s $SCR/s.1fq)"; grep -E "Elapsed|Maximum" $O/encode.time.txt
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --n-donors -1 --tagged-bam $SCR/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; grep -E "auto-K|selected K" $O/run.log | head -5; ls $O/private
