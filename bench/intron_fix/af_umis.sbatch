#!/bin/bash
#SBATCH --job-name=if-afumi
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=3
#SBATCH --mem=150G
#SBATCH --time=03:00:00
#SBATCH --array=0-2
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; ENV=$W/envs/pbench
DSS=(pbmc1k_v3 nuclei2k_v2 hgmm1k_v3); DS=${DSS[$SLURM_ARRAY_TASK_ID]}
case $DS in hgmm*) SP=GRCh38-GRCm39-barnyard ;; nuclei*|neuron*) SP=GRCm39-2024-A ;; *) SP=GRCh38-2024-A ;; esac
O=$PB/runs/$DS/alevinfry-umis; mkdir -p $O
$ENV/bin/alevin-fry view --rad $PB/runs/$DS/alevinfry/af/af_map/map.rad 2>/dev/null | \
  $ENV/bin/python $W/jobs/intronfix/af_umis.py $PB/refs/$SP/af_idx/index/t2g_3col.tsv \
  $PB/runs/$DS/starsolo-public-cr/Solo.out/Gene/filtered/barcodes.tsv $O/af_umis.tsv.gz
zcat $O/af_umis.tsv.gz | cut -f4 | sort | uniq -c
