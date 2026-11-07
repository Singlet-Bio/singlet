#!/bin/bash
#SBATCH --job-name=evp-locus
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# What are the reads behind human LINC00486 in MOUSE cells (hgmm1k_v3)? Align the hgmm1k_v3 reads to the joint genome as the
# STARsolo run did, keep a CB-tagged BAM, take the reads at the locus whose cell is a (Cell Ranger) mouse cell, and re-map them
# to the mouse genome ALONE (GRCm39-2024-A). If they map there, the joint reference has a homologous mouse region and the
# exact aligner chose the human copy; if they do not, the mouse sequence is not in the mouse assembly/annotation (e.g. rDNA).
#   sbatch locus.sbatch [GENE]
set -euo pipefail
GENE=${1:-LINC00486}
WORK=/mnt/projects/debruinz_project/debruinz/singlet; PB=$WORK/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
export PATH=$WORK/envs/pbench/bin:$PATH
O=$WORK/exact-vs-pseudo/results/locus; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
D=$PB/data/hgmm1k_v3
R1=$(grep "_R1_" $D/gex_fastqs.txt | paste -sd,); R2=$(grep "_R2_" $D/gex_fastqs.txt | paste -sd,)
cd $SCR
STAR --runThreadN 16 --genomeDir $REF/GRCh38-GRCm39-barnyard/star_2.7.11b --readFilesIn $R2 $R1 --readFilesCommand zcat \
    --soloType CB_UMI_Simple --soloCBwhitelist $PB/whitelists/3M-february-2018.txt --soloCBlen 16 --soloUMIstart 17 \
    --soloUMIlen 12 --soloBarcodeReadLength 0 --soloStrand Forward --soloFeatures Gene GeneFull \
    --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR \
    --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype BAM SortedByCoordinate \
    --outSAMattributes NH HI AS nM CB UB --limitBAMsortRAM 60000000000 --outTmpDir $SCR/startmp --outFileNamePrefix $SCR/b_ > $SCR/star.log 2>&1
samtools index $SCR/b_Aligned.sortedByCoord.out.bam
grep -P '\tgene\t' $REF/GRCh38-GRCm39-barnyard/genes/genes.gtf > $SCR/genes_only.gtf
H5=$(ls $D/*filtered*.h5)
python -I $WORK/exact-vs-pseudo/code/locus.py --bam $SCR/b_Aligned.sortedByCoord.out.bam --h5 $H5 --gtf $SCR/genes_only.gtf --gene $GENE --out $O
# re-map the mouse-cell reads to mouse alone, default STAR scoring
mkdir -p $SCR/mm
STAR --runThreadN 16 --genomeDir $REF/GRCm39-2024-A/star_2.7.11b --readFilesIn $O/${GENE}_mouse_cells.fastq \
    --outSAMtype BAM Unsorted --outSAMattributes NH nM AS --outFileNamePrefix $SCR/mm/ > $SCR/mm/star.log 2>&1
cp $SCR/mm/Log.final.out $O/${GENE}_remap_mouse_only.Log.final.out
samtools view -F 0x100 $SCR/mm/Aligned.out.bam | awk '{n++; c[$3]++} END{print "mapped_alignments",n; for(k in c) print k,c[k]}' | sort -k2,2nr | head -12 > $O/${GENE}_remap_mouse_only.chroms.txt
# which mouse genes / biotypes do they land on?
grep -P '\tgene\t' $REF/GRCm39-2024-A/genes/genes.gtf | awk -F'\t' '{match($9,/gene_name "[^"]+"/); n=substr($9,RSTART+11,RLENGTH-12); match($9,/gene_(bio)?type "[^"]+"/); b=substr($9,RSTART,RLENGTH); print $1"\t"$4-1"\t"$5"\t"n"\t"b}' | sort -k1,1 -k2,2n > $SCR/mm/genes.bed
bedtools bamtobed -i $SCR/mm/Aligned.out.bam | sort -k1,1 -k2,2n | bedtools intersect -a - -b $SCR/mm/genes.bed -wao | awk -F'\t' '{print $10"\t"$11}' | sort | uniq -c | sort -k1,1nr | head -15 > $O/${GENE}_remap_mouse_only.genes.txt
echo done; cat $O/${GENE}_remap_mouse_only.Log.final.out | grep -E "input reads|Uniquely mapped reads %|multiple loci %|too short"; cat $O/${GENE}_remap_mouse_only.genes.txt
