#!/bin/bash
#SBATCH --job-name=ar-kneeties
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,short-cpu,scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Uncalled barcodes whose total equals the knee total, per validation run: the only barcodes the "retain = exact
# knee total" change (vs 10 ** log10(total)) can affect.
set -euo pipefail
PB=/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark
for ID in pbmc1k_v3 pbmc5p_v1 nuclei2k_v2 neuron1k_v3 hgmm1k_v3; do
  R=$PB/runs/$ID/singlet-prod-t4
  K=$(grep -o '"retain_total": [0-9.e+-]*' $R/pileup_stats.json | head -n 1 | awk '{printf "%.0f", $2}')
  N=$(awk -F'\t' -v k=$K 'NR>1 && $3==0 && $6==k' $R/cell_calls.tsv | wc -l)
  echo "$ID knee_total=$K uncalled_at_knee_total=$N"
done
