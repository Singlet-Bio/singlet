#!/bin/bash
#SBATCH --job-name=d101-star
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=72G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Bulk RNA-seq of the same donor (GSE120446) aligned with STAR 2.7.11b to the same GRCh38-2024-A index the engine uses; sorted,
# indexed BAM with MD tags (REDItools reads them) -> $W/jobs/d101/bulkbam/<bulk_run>.bam. Row SLURM_ARRAY_TASK_ID of libs.tsv.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; REF=/mnt/projects/debruinz_project/cellarium/reference
LIBS=${LIBS:?}; T=${SLURM_CPUS_PER_TASK:-16}
read -r DONOR SC SCG BULK BULKG <<< "$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS)"
[ "$BULK" = "-" ] && { echo "donor $DONOR: no bulk"; exit 0; }
O=$W/jobs/d101/bulkbam; mkdir -p $O; [ -s $O/$BULK.bam.bai ] && { echo "have $BULK"; exit 0; }
SCR=/tmp/$USER-d101star-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
STAR --runThreadN $T --genomeDir $REF/GRCh38-2024-A/star_2.7.11b --readFilesIn $W/data/d101/bulk/${BULK}_1.fastq.gz $W/data/d101/bulk/${BULK}_2.fastq.gz \
  --readFilesCommand zcat --outSAMtype BAM SortedByCoordinate --outSAMattributes NH HI AS nM MD --limitBAMsortRAM 30000000000 \
  --outFileNamePrefix $SCR/ --outTmpDir $SCR/tmp
mv $SCR/Aligned.sortedByCoord.out.bam $O/$BULK.bam; samtools index -@ $T $O/$BULK.bam
cp $SCR/Log.final.out $O/$BULK.Log.final.out; grep -E "Uniquely mapped reads %|Number of input reads" $O/$BULK.Log.final.out
