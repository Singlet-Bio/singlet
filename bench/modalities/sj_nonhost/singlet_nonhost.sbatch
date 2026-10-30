#!/bin/bash
#SBATCH --job-name=sjnh-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch singlet_nonhost.sbatch LABEL
# Encode + full Singlet run with the non-host screen on a fetched GSE166766 sample ($B/data/LABEL/R{1,2}.fastq.gz).
# nonhost/ is created beforehand: the per-cell species matrix is written before the binary creates that directory (singlet.cpp:1613 vs 1621).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; B=$W/modality-bench/sj_nonhost
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; BIN=$PB/bin/singlet-fixed-det
L=$1; D=$B/data/$L; O=$B/singlet/$L; rm -rf $O; mkdir -p $O/nonhost
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap "rm -rf $SCR" EXIT
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $D/R1.fastq.gz $D/R2.fastq.gz -o $SCR/s.1fq > $O/encode.log 2>&1
ls -la $SCR/s.1fq; grep -E "Elapsed|Maximum" $O/encode.time.txt
export SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference SINGLET_STALL_TIMEOUT=10800
cd $SCR; /usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base $SINGLET_REF_BASE --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf \
  --out-prefix $O/ --threads 16 --pipeline --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum" $O/time.txt; grep -E "nonhost|cell_calling|metrics" $O/run.log | head -30; ls $O $O/nonhost
