#!/bin/bash
#SBATCH --job-name=hto-singlet
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# I1 truth, step 1: Singlet's own pipeline on the first 200 M read pairs of the Stoeckius 2018 PBMC RNA library
# (cell calls + tagged BAM). Same command as bench/modalities/genotype_pool/run_singlet.sbatch (Kang lane, K = 8).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/expression-sota/hto/data/stoeckius2018; O=$D/singlet
BIN=$W/pipeline-benchmark/bin/singlet-fixed-det
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
export PATH=$W/envs/pbench/bin:$PATH
rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
/usr/bin/time -v -o $O/encode.time.txt $BIN encode --reads $D/fastq/rna_R1.fastq.gz $D/fastq/rna_R2.fastq.gz -o $SCR/s.1fq --no-trim -v > $O/encode.log 2>&1
tail -15 $O/encode.log; ls -l $SCR/s.1fq; grep -E "Elapsed|Maximum" $O/encode.time.txt
cd $SCR; export SINGLET_STALL_TIMEOUT=10800
/usr/bin/time -v -o $O/time.txt $BIN $SCR/s.1fq --ref-base /mnt/projects/debruinz_project/cellarium/reference \
  --genome-dir $REF/star_2.7.11b --exons $REF/genes/genes.gtf --out-prefix $O/ --threads 16 --pipeline \
  --snps $PANEL --n-donors 8 --tagged-bam $O/possorted.bam --bam-sort-mode disk --output-format mtx --raw-matrix > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
tail -20 $O/run.log; grep -E "Elapsed|Maximum resident" $O/time.txt; ls -la $O | head -40
awk -F"\t" 'NR>1 && $2=="TRUE"' $O/cell_calls.tsv | wc -l
