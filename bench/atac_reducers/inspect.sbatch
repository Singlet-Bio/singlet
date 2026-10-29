#!/bin/bash
#SBATCH --job-name=ar-inspect
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Look at the formats of the 5k run outputs (read-only): BAM header contigs, a few records, fragments/peaks/metrics/mtx heads.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
O=$W/pipeline-benchmark/runs/atacsp/atac_pbmc_5k_nextgem
samtools view -H $O/atac_sorted.bam | head -3; samtools view -H $O/atac_sorted.bam | grep -c '^@SQ'
samtools view -H $O/atac_sorted.bam | grep -E 'SN:(chr22|chrM|chrY|chr1)\s'
samtools view $O/atac_sorted.bam | head -4 | cut -f1-12
echo ---frag; zcat $O/fragments.tsv.gz | head -4 | cut -c1-200; zcat $O/fragments.tsv.gz | wc -l
echo ---peaks; head -3 $O/peaks.bed; wc -l $O/peaks.bed
echo ---metrics; head -3 $O/atac_barcode_metrics.tsv | cut -c1-400
echo ---mtx; zcat $O/atac_peak_counts.mtx.gz | head -5; zcat $O/atac_peak_counts_barcodes.tsv.gz | head -2; zcat $O/atac_peak_counts_features.tsv.gz | head -2
echo ---cells; head -3 $O/atac_cells.tsv; cat $O/atac_summary.json
head -2 $W/pipeline-benchmark/whitelists/atacsp/737K-cratac-v1.txt
