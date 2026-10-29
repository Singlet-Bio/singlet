#!/bin/bash
#SBATCH --job-name=sjnh-solo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch starsolo_sj_pbmc.sbatch [TAG] [EXTRA STAR ARGS]   e.g.  clip11 '--clip5pNbases 11'
# STARsolo Gene+SJ on PBMC 1k v3 with the same STAR settings Singlet passes (see run.log of t4_r1).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A
TAG=${1:-}; EXTRA=${2:-}; O=$W/modality-bench/sj_nonhost/pbmc1k_v3/starsolo${TAG:+_$TAG}; rm -rf $O; mkdir -p $O
F=$PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1
STAR=$W/envs/pbench/bin/STAR
$STAR --version
/usr/bin/time -v -o $O/time.txt $STAR --runMode alignReads --genomeDir $REF/star_2.7.11b \
  --readFilesIn ${F}_L001_R2_001.fastq.gz,${F}_L002_R2_001.fastq.gz ${F}_L001_R1_001.fastq.gz,${F}_L002_R1_001.fastq.gz \
  --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist $PB/whitelists/3M-february-2018.txt \
  --soloCBlen 16 --soloCBstart 1 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 \
  --soloFeatures Gene SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts \
  --outSAMtype None --outFilterScoreMin 10 --outReadsUnmapped Fastx \
  --runThreadN 16 $EXTRA --outFileNamePrefix $O/
ls -R $O/Solo.out | head -30
