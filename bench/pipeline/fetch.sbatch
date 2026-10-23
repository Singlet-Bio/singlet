#!/bin/bash
#SBATCH --job-name=pb-fetch
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:59:00
#SBATCH --array=1-7%4
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Download one public 10x dataset (FASTQ tar + Cell Ranger matrix) per array task into
# $PB/data/<id>/ and record its GEX FASTQs. Row N of datasets.tsv (header = row 0).
set -euo pipefail
HERE=/mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline
source $HERE/common.sh
IFS=$'\t' read -r ID SPECIES CHEM TAR CRH5 NOTES < <(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" $HERE/datasets.tsv)
D=$PB/data/$ID; mkdir -p $D; cd $D
echo "[fetch] $ID $TAR"
if [ ! -f .extracted ]; then
  curl -sSL --retry 5 -C - -o fastqs.tar "$TAR"
  tar -xf fastqs.tar && rm -f fastqs.tar && touch .extracted
fi
curl -sSL --retry 5 -o "cellranger_$(basename $CRH5)" "$CRH5" || echo "[fetch] no Cell Ranger matrix"
find $D -name "*.fastq.gz" | sort > all_fastqs.txt
# GEX reads: drop antibody/protein/VDJ libraries
grep -viE "antibody|protein_fastqs/.*_(adt|ab)_|vdj|_t_|_b_|tcr|bcr" all_fastqs.txt | grep -E "_R[12]_" > gex_fastqs.txt || true
grep -iE "antibody|_adt|_ab_|feature" all_fastqs.txt > adt_fastqs.txt || true
wc -l all_fastqs.txt gex_fastqs.txt adt_fastqs.txt; du -sh $D
