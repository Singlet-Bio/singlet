#!/bin/bash
#SBATCH --job-name=as-solo-visium
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# STARsolo 2.7.11b on the Visium mouse brain FASTQs with the visium-v1 spot list (Space Ranger 1.x-equivalent counting:
# exonic genes, CB 1 mismatch, UMI 1MM_CR). Same reference and threads as Singlet.
set -u
#   [RUN=visium_mouse_brain_ant FQ=<fastq dir> GENOME=GRCm39-2024-A]
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
G=/mnt/projects/debruinz_project/cellarium/reference/${GENOME:-GRCm39-2024-A}
FQ=${FQ:-$W/data/atac-spatial/V1_Mouse_Brain_Sagittal_Anterior/V1_Mouse_Brain_Sagittal_Anterior_Section_1_fastqs}
O=$PB/runs/atacsp/starsolo_${RUN:-visium_mouse_brain_ant}; rm -rf $O; mkdir -p $O; cd $O
R1=$(ls $FQ/*_R1_*.fastq.gz | sort | paste -sd,); R2=$(ls $FQ/*_R2_*.fastq.gz | sort | paste -sd,)
/usr/bin/time -v -o $O/time.txt $W/envs/pbench/bin/STAR --runThreadN 16 --genomeDir $G/star_2.7.11b --readFilesCommand zcat \
  --readFilesIn $R2 $R1 --soloType CB_UMI_Simple --soloCBwhitelist $PB/whitelists/visium-v1.txt --soloCBstart 1 --soloCBlen 16 \
  --soloUMIstart 17 --soloUMIlen 12 --soloBarcodeReadLength 0 --soloFeatures Gene --soloCellFilter None \
  --soloUMIdedup 1MM_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR \
  --clipAdapterType CellRanger4 --outFilterScoreMin 30 --outSAMtype None --outFileNamePrefix $O/ > $O/star.log 2>&1
echo "exit=$? wall=$(grep 'Elapsed (wall' $O/time.txt | sed 's/.*: //') rss_kb=$(grep 'Maximum resident' $O/time.txt | sed 's/.*: //')"
ls $O/Solo.out/Gene/raw; cat $O/Solo.out/Gene/Summary.csv
