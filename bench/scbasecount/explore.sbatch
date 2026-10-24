#!/bin/bash
#SBATCH --job-name=sbc-explore
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Print the members and matrix shapes of one sample per study (lines "GSE GSM" in $1) of the legacy bundles.
set -uo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
EDROP=$WORK/src/singlet-py/claude__empty-droplet-check-624070
while read gse gsm; do
  echo "##### $gse $gsm"
  apptainer exec --bind "$WORK,/tmp" --env "PATH=$WORK/venvs/singlet/bin:/usr/local/bin:/usr/bin:/bin" $WORK/images/python-3.12-bookworm.sif \
    python -P $EDROP/bench/empty_droplets/fetch.py $gse --out /tmp/x --explore 2>&1 | grep -E "^(exon|intron) |samples:|Error|error" | cut -c1-200
done < $WORK/jobs/sbc/${1:-verify_empty.txt}
