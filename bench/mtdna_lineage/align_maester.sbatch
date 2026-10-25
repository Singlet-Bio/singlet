#!/bin/bash
#SBATCH --job-name=mtl-maester
#SBATCH --account=debruinz
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch align_maester.sbatch <name>    (name = k562_maester | bpdcn_maester)
# MAESTER reads (R1 = 16 bp CB + 12 bp UMI, R2 = long mt-enriched cDNA read) aligned to the same GRCh38-2024-A STAR index
# with STARsolo so the BAM carries CB/UB (STAR writes CB/UB only into a coordinate-sorted BAM). Only chrM alignments are kept.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
NAME=$1; D=$W/data/maester/$NAME; O=$W/out/mtdna_lineage/$NAME
G=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; cd $SCR
# all barcodes in the 10x v3 whitelist (STARsolo 1-mismatch correction); cells are selected later, from the 3' library
WL=$W/pipeline-benchmark/whitelists/3M-february-2018.txt
STAR --runThreadN 48 --genomeDir $G --readFilesIn $D/${NAME}_R2.fastq.gz $D/${NAME}_R1.fastq.gz --readFilesCommand "pigz -dc -p 4" \
  --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBstart 1 --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 \
  --soloBarcodeReadLength 0 --soloFeatures Gene --outSAMattributes NH HI nM AS CR UR CB UB \
  --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 200000000000 --outBAMsortingThreadN 16 --outFileNamePrefix $SCR/star/ > $O/star.log 2>&1
cat $SCR/star/Log.final.out > $O/Log.final.out
samtools index -@ 16 $SCR/star/Aligned.sortedByCoord.out.bam
samtools idxstats $SCR/star/Aligned.sortedByCoord.out.bam | sort -k3,3nr | head -5
samtools view -@ 16 -b -o $O/chrM.bam $SCR/star/Aligned.sortedByCoord.out.bam chrM
cp -r $SCR/star/Solo.out $O/ 2>/dev/null || true
samtools index $O/chrM.bam
samtools idxstats $O/chrM.bam | head -3
ls -la $O
