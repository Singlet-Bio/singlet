#!/bin/bash
#SBATCH --job-name=lr-starsolo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
# STARsolo 2.7.11b (STAR-velodump = stock + a per-UMI Velocyto class dump) on full pbmc1; Velocyto feature, EmptyDrops_CR, Cell Ranger 4 emulation
# exactly as starsolo_public.sbatch (branch bench/public-counts) with UMI length 11.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/expression-sota/longread_usa/code/common.sh
ENV=$W/envs/pbench
O=$X/starsolo; rm -rf $O; mkdir -p $O; cd $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
CMD="$PB/bin/STAR-velodump --runThreadN 16 --genomeDir $G/star_2.7.11b --readFilesIn ${R2} ${R1} --readFilesCommand zcat --soloType CB_UMI_Simple --soloCBwhitelist $WL --soloCBlen 16 --soloUMIstart 17 --soloUMIlen 11 --soloBarcodeReadLength 0 --soloStrand Forward --outSAMtype None --outTmpDir $SCR/startmp --soloFeatures Gene GeneFull Velocyto SJ --soloCellFilter EmptyDrops_CR --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --soloUMIfiltering MultiGeneUMI_CR --soloUMIdedup 1MM_CR --clipAdapterType CellRanger4 --outFilterScoreMin 30"
echo "$CMD" > command.txt
export LD_LIBRARY_PATH=$ENV/lib STAR_VELO_DUMP=$SCR/velo_umis.tsv
eval "/usr/bin/time -v -o time.txt $CMD" > run.log 2>&1
gzip -1 -c $SCR/velo_umis.tsv > velo_umis.tsv.gz
grep -E "Elapsed \(wall|Maximum resident" time.txt; zcat velo_umis.tsv.gz | cut -f4 | sort | uniq -c; ls Solo.out/*
