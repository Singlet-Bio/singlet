#!/bin/bash
#SBATCH --job-name=gp-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_singlet.sbatch <tag> <n-donors>     (n-donors 8 = told K; -1 = auto)
# Singlet on the Kang batch-2 control lane with donor demux (--snps 1000G AF>=5% panel), doublets and mt heteroplasmy.
# Writes a tagged BAM (CB/UB) that the reference tools (cellsnp-lite, mgatk, pileup) read.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; TAG=$1; K=$2
BIN=$W/pipeline-benchmark/bin/singlet-fixed-det
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
O=$D/singlet_$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cp $D/kang21.1fq $SCR/s.1fq; cd $SCR
export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --snps $PANEL --n-donors $K --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
grep -E "Elapsed|Maximum resident" $O/time.txt; ls -la $O $O/donor 2>/dev/null | head -60
