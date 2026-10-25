#!/bin/bash
#SBATCH --job-name=bpub-ascore
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Score the ancestry benchmark (ancestry_score.py) and write results/genetic-ancestry.json.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
BPE=$WORK/envs/bpub; A=$BP/artifacts/genetic-ancestry; mkdir -p $A
export PYTHONPATH=$SRC/python
rt python -u $SRC/bench/public/sexancestry/ancestry_score.py --mage $BP/work/ancestry/mage \
  --af $BP/work/ancestry/af_leaveout.parquet --excluded $BP/work/ancestry/excluded.txt \
  --somalier-dir $WORK/tools/bpub/somalier --gtf $REF/GRCh38-2024-A/genes/genes.gtf \
  --artifacts $A --bcftools $BPE/bin/bcftools --bench-commit $(git -C $SRC rev-parse HEAD) --peddy-python $BPE/bin/python
rt python $SRC/bench/public/run_capability.py genetic-ancestry $A --out $BP/results/genetic-ancestry.json
