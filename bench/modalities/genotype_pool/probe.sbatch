#!/bin/bash
#SBATCH --job-name=gp-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
D=$W/modality-bench/genotype_pool; mkdir -p $D/truth; cd $D/truth
U=https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398238/bam/2.1.bam
samtools view -H $U | head -30
samtools view $U 2>/dev/null | head -4
for f in GSE96583_batch2.total.tsne.df.tsv.gz GSE96583_batch2.genes.tsv.gz; do curl -sL -o $f ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/$f || curl -sL -o $f https://ftp.ncbi.nlm.nih.gov/geo/series/GSE96nnn/GSE96583/suppl/$f; done
ls -la; zcat GSE96583_batch2.total.tsne.df.tsv.gz | head -5
curl -sIL https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download | head -5
