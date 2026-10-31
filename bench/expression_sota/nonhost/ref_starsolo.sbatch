#!/bin/bash
#SBATCH --job-name=nht-refsolo
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=03:59:00
#SBATCH --dependency=afterok:794705
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Reference ("truth-like") viral counts: STARsolo 2.7.11b on the FULL ravindra2021 dpi2 reads (GSE166766 HBEC, 10x v3) against the
# combined GRCh38-2024-A + NC_045512.2 genome, viral GTF genes as extra genes. Settings = bench/public starsolo_public.sbatch MODE=cr
# (Gene + GeneFull, EmptyDrops_CR, 1MM_multi_Nbase_pseudocounts, MultiGeneUMI_CR, 1MM_CR, CellRanger4 clipping, score >= 30), plus an unsorted
# BAM with CB/UB/GX tags so the leader-TRS junction reads can be extracted. Viral alignments are cut out of the BAM and the BAM is dropped.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/nonhost; PB=$W/pipeline-benchmark
export PATH=$W/envs/pbench/bin:$PATH
D=$W/modality-bench/sj_nonhost/data/dpi2; G=$X/ref/GRCh38-2024-A_plus_SARS2/star; WL=$PB/whitelists/3M-february-2018.txt
O=$X/ref_starsolo/dpi2; rm -rf $O; mkdir -p $O; cd $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
STAR --version > star_version.txt
CMD=(STAR --runThreadN 16 --genomeDir $G --readFilesIn $D/R2.fastq.gz $D/R1.fastq.gz --readFilesCommand zcat
     --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 --soloStrand Forward
     --soloFeatures Gene GeneFull --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts
     --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30
     --outSAMtype BAM SortedByCoordinate --limitBAMsortRAM 60000000000 --outBAMcompression 1 --outSAMattributes NH HI AS nM CB UB GX GN --outSAMunmapped None
     --outTmpDir $SCR/startmp --outFileNamePrefix $O/)
echo "${CMD[*]}" > command.txt
/usr/bin/time -v -o time.txt "${CMD[@]}" > run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" time.txt
B=Aligned.sortedByCoord.out.bam; samtools index -@ 8 $B
samtools view -@ 8 -h $B NC_045512.2 > viral_aligned.sam
wc -l viral_aligned.sam; ls -la $B; rm -f $B $B.bai
ls Solo.out/Gene Solo.out/Gene/raw Solo.out/GeneFull/raw; tail -n 25 Solo.out/Gene/Summary.csv
