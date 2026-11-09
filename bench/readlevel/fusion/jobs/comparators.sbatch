#!/bin/bash
#SBATCH --job-name=fus-comp
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# STAR-Fusion 1.15.1 and Arriba 2.5.1 on one droplet library, cDNA read (R2) as single-end reads, every lane. Each caller reports the
# names of its supporting reads, which readmap.py turns into cell barcodes and UMIs from R1. Default parameters as their READMEs give.
# FQDIR: the 10x-named FASTQ folder. SAMPLE: name. Output: $W/jobs/readlevel/fusion/$SAMPLE/{starfusion,arriba}/
set -uo pipefail
: "${FQDIR:?}" "${SAMPLE:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/fusion; SIF=$W/images/starfusion-1.15.1.sif
REF=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A; CTAT=$W/data/readlevel/ctat/GRCh38_gencode_v44_CTAT_lib_Oct292023.plug-n-play/ctat_genome_lib_build_dir
O=$W/jobs/readlevel/fusion/$SAMPLE; mkdir -p $O; T=${SLURM_CPUS_PER_TASK:-24}
SCR=/tmp/$USER-fc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
R2=$(ls $FQDIR/*_R2_001.fastq.gz | sort | paste -sd,); echo "R2 files: $R2"
cat $(ls $FQDIR/*_R2_001.fastq.gz | sort) > $SCR/r2.fq.gz
if [ "${RUN_SF:-1}" = 1 ] && [ ! -s $O/starfusion/star-fusion.fusion_predictions.tsv ]; then
  mkdir -p $O/starfusion
  /usr/bin/time -v -o $O/starfusion/time.txt apptainer exec -B $W -B $SCR $SIF STAR-Fusion --genome_lib_dir $CTAT --left_fq $SCR/r2.fq.gz \
      --CPU $T --output_dir $SCR/sf > $O/starfusion/log.txt 2>&1
  cp $SCR/sf/star-fusion.fusion_predictions.tsv $SCR/sf/star-fusion.fusion_predictions.abridged.tsv $O/starfusion/ 2>/dev/null
  cp $SCR/sf/star-fusion.preliminary/star-fusion.fusion_candidates.preliminary* $O/starfusion/ 2>/dev/null || true
  tail -3 $O/starfusion/log.txt
fi
if [ "${RUN_AR:-1}" = 1 ] && [ ! -s $O/arriba/fusions.tsv ]; then
  mkdir -p $O/arriba; DB=$(ls -d $ENV/var/lib/arriba 2>/dev/null || dirname $(ls $ENV/share/arriba*/database/blacklist_hg38_GRCh38_v*.tsv.gz | head -1))
  BL=$(ls $DB/blacklist_hg38_GRCh38_v*.tsv.gz | head -1); KF=$(ls $DB/known_fusions_hg38_GRCh38_v*.tsv.gz | head -1); PD=$(ls $DB/protein_domains_hg38_GRCh38_v*.gff3 | head -1)
  echo "arriba db $BL $KF $PD"
  cd $SCR && /usr/bin/time -v -o $O/arriba/time.txt bash -c "$ENV/bin/STAR --runThreadN $T --genomeDir $REF/star_2.7.11b --genomeLoad NoSharedMemory \
      --readFilesIn $SCR/r2.fq.gz --readFilesCommand zcat --outStd BAM_Unsorted --outSAMtype BAM Unsorted --outSAMunmapped Within --outBAMcompression 0 \
      --outFilterMultimapNmax 50 --peOverlapNbasesMin 10 --alignSplicedMateMapLminOverLmate 0.5 --alignSJstitchMismatchNmax 5 -1 5 5 \
      --chimSegmentMin 10 --chimOutType WithinBAM HardClip --chimJunctionOverhangMin 10 --chimScoreDropMax 30 --chimScoreJunctionNonGTAG 0 \
      --chimScoreSeparation 1 --chimSegmentReadGapMax 3 --chimMultimapNmax 50 --outFileNamePrefix $SCR/ar_ |
    $ENV/bin/arriba -x /dev/stdin -o $O/arriba/fusions.tsv -O $O/arriba/fusions.discarded.tsv -a $REF/fasta/genome.fa -g $REF/genes/genes.gtf \
      -b $BL -k $KF -t $KF -p $PD -I" > $O/arriba/log.txt 2>&1
  tail -3 $O/arriba/log.txt; cp $SCR/ar_Log.final.out $O/arriba/ 2>/dev/null || true
fi
wc -l $O/starfusion/*.tsv $O/arriba/*.tsv 2>/dev/null
