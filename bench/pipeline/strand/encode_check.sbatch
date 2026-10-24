#!/bin/bash
#SBATCH --job-name=pb-strand-encode
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Protocol label `singlet encode` gives each benchmark dataset (first 2M read pairs).
set -u   # no -e/pipefail: zcat | head closes the pipe early
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/strand/common.sh
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
cd $SCR
for ID in pbmc1k_v2 pbmc1k_v3 neuron1k_v3 nuclei2k_v2 hgmm1k_v3 pbmc1k_cite_v3 pbmc5p_v1; do
  R1=$(grep "_R1_" $PB/data/$ID/gex_fastqs.txt | head -1); R2=$(grep "_R2_" $PB/data/$ID/gex_fastqs.txt | head -1)
  zcat $R1 2>/dev/null | head -8000000 | gzip -1 > r1.fq.gz; zcat $R2 2>/dev/null | head -8000000 | gzip -1 > r2.fq.gz
  echo "[enc] $ID: $(zcat r1.fq.gz | wc -l) R1 lines"
  $BIN_STRAND encode --reads r1.fq.gz r2.fq.gz -o $ID.1fq > $ID.log 2>&1 || echo "$ID encode exit $?"
  echo "== $ID"; grep -a -E "3'/5' check|protocol_detect\] 10x|protocol=|Detected|detected protocol" $ID.log | head -6
  rm -f $ID.1fq r1.fq.gz r2.fq.gz
done
