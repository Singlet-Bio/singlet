#!/bin/bash
#SBATCH --job-name=nht-proto
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --dependency=afterok:794702
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Singlet-side prototype: the reads the non-host screen keeps (nonhost_reads.fq of the release-gate dpi2 run) paired with their barcode mates,
# aligned by STARsolo to the viral-only genome; CB corrected against Singlet's called-cell list (1 mismatch, unique), UMI dedup 1MM_CR,
# genes from the viral GTF. Variant "all_unmapped" runs the same on every STAR-unmapped read (what the k-mer screen leaves out).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/nonhost; V=$X/ref/viral
export PATH=$W/envs/pbench/bin:$PATH
S=$W/release-gate/perf/nonhost/gse166766_hbec_dpi2/singlet; CELLS=$W/release-gate/perf/nonhost/cells/gse166766_hbec_dpi2.txt
O=$X/proto/dpi2; rm -rf $O; mkdir -p $O; cd $O
python -I $X/code/prep_kept_reads.py $S/nonhost_reads.fq $S/star_Unmapped.out.mate2 $O/kept
run() { # label cdna bc
  mkdir -p $O/$1; STAR --runThreadN 8 --genomeDir $V/star_viral --readFilesIn $2 $3 \
    --soloType CB_UMI_Simple --soloCBwhitelist $CELLS --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 --soloStrand Forward \
    --soloFeatures Gene GeneFull --soloCellFilter None --soloCBmatchWLtype 1MM --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR \
    --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 4000000000 --outSAMattributes NH HI AS nM CB UB GX GN \
    --outFileNamePrefix $O/$1/ > $O/$1/run.log 2>&1
  samtools view -h $O/$1/Aligned.sortedByCoord.out.bam > $O/$1/aligned.sam
  python -I $X/code/sgrna_junctions.py $O/$1/aligned.sam $V/viral.trs.json $O/$1/sgrna
}
run kept $O/kept.cdna.fq $O/kept.bc.fq
run all_unmapped $S/star_Unmapped.out.mate1 $S/star_Unmapped.out.mate2
rm -f $O/*/Aligned.sortedByCoord.out.bam; for d in kept all_unmapped; do tail -n 20 $O/$d/Solo.out/Gene/Summary.csv | head -8; grep -E "Uniquely mapped reads %|Number of input reads" $O/$d/Log.final.out; done
