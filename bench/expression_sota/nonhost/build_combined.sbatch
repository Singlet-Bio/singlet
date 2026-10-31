#!/bin/bash
#SBATCH --job-name=nht-combgenome
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --dependency=afterok:794702
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Combined host+viral STAR genome: GRCh38-2024-A genome.fa + chrom NC_045512.2; GENCODE GTF + viral GTF.
# STAR 2.7.11b, sjdbOverhang 100, SAindexNbases 14 as in the host index.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/nonhost; V=$X/ref/viral; C=$X/ref/GRCh38-2024-A_plus_SARS2
R=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
export PATH=$W/envs/pbench/bin:$PATH
mkdir -p $C/star; cd $C
cat $R/genes/genes.gtf $V/viral.gtf > genes_plus_viral.gtf
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
/usr/bin/time -v -o time.txt STAR --runMode genomeGenerate --runThreadN 16 --genomeDir $C/star --genomeFastaFiles $R/fasta/genome.fa $V/NC_045512.2.fna \
  --sjdbGTFfile genes_plus_viral.gtf --sjdbOverhang 100 --genomeSAindexNbases 14 --outTmpDir $SCR/gg --outFileNamePrefix $C/gen_ > gen.log 2>&1
grep -E "Elapsed|Maximum res" time.txt; tail -3 gen.log; tail -3 $C/star/chrNameLength.txt; du -sh $C/star
