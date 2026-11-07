#!/bin/bash
#SBATCH --job-name=pb-strand-classify
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Per-protocol classification of the campaign's wrong-strand samples -> $RES_STRAND/campaign/
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/fix__strand-detection/bench/pipeline/strand/common.sh
# pyarrow lives in the shared runtime venv (python-3.12 image), not in the pbench env
apptainer exec --bind /mnt/projects $WORK/images/python-3.12-bookworm.sif $WORK/venvs/singlet/bin/python -I \
  $SRC_STRAND/bench/pipeline/analysis/strand_classify.py $PB/results/campaign/campaign_samples.csv $CAMPAIGN \
  /mnt/projects/debruinz_project/cellarium/catalog/all_gsm_metadata.parquet $RES_STRAND/campaign \
  $(ls $RES_STRAND/bam/*/probe.tsv 2>/dev/null)
