#!/bin/bash
#SBATCH --job-name=pb-subsample
#SBATCH --account=debruinz
#SBATCH --partition=scavenger,short-cpu
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch subsample.sbatch <dataset_id> <read_pairs>
# Keep the first N read pairs of lane L001 (same reads for every tool); full list kept.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__molecule-labels/bench/pipeline/common.sh
D=$PB/data/$1; N=$2; mkdir -p $D/sub
[ -f $D/gex_fastqs_full.txt ] || cp $D/gex_fastqs.txt $D/gex_fastqs_full.txt
for R in R1 R2; do
  f=$(grep "_L001_${R}_" $D/gex_fastqs_full.txt)
  zcat $f | head -n $((4*N)) | pigz -p 6 > $D/sub/sub_S1_L001_${R}_001.fastq.gz
done
ls $D/sub/*.fastq.gz > $D/gex_fastqs.txt
echo "$1: first $N pairs of L001" > $D/sub/README; cat $D/gex_fastqs.txt
