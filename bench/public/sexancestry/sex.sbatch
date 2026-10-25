#!/bin/bash
#SBATCH --job-name=bpub-sex
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Chromosomal-sex benchmark on the geo_sex21 cohort row (and any dataset with per-cell donor truth). Steps:
#   sbatch --array=1-N%8 sex.sbatch extract     per study (line N of $BP/work/sex/studies.txt): sex_extract.py
#   sbatch sex.sbatch cellxy                    cellXY on every extracted sample (R, cellxy.R)
#   sbatch sex.sbatch collect                   artifacts (sex_cells.parquet, sex_samples.tsv, provenance.json) + result
# submit_sex.sh chains the three.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/src/singlet-py/claude__bench-public-cohort-485cf7/bench/public/harness/common.sh
W=$BP/work/sex; A=$BP/artifacts/chromosomal-sex; mkdir -p $W $A
LABELS=$WORK/data/annotate/sex_labels.json
export PYTHONPATH=$SRC/python OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK SINGLET_CACHE_DIR=/tmp/$USER-$SLURM_JOB_ID-cache
case $1 in
extract)
  GSE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" $W/studies.txt)
  SCR=/tmp/$USER-sex-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
  rt python -u $SRC/bench/public/sexancestry/sex_extract.py --gse $GSE --sex-labels $LABELS --out $W --scratch $SCR
  ;;
cellxy)
  # species.txt for any sample extracted before the extractor wrote it (from samples.jsonl)
  rt python -c "
import json, pathlib
w = pathlib.Path('$W')
for l in open(w / 'samples.jsonl'):
    s = json.loads(l); f = w / 'mtx' / s['gsm'] / 'species.txt'
    if f.parent.exists() and not f.exists(): f.write_text(s['species'] + '
')
"
  export R_LIBS=$WORK/envs/rlib-benchpublic
  /usr/bin/time -v -o $W/cellxy.time.txt $WORK/envs/ambient/bin/Rscript $SRC/bench/public/sexancestry/cellxy.R $W
  ;;
collect)
  rt python -u $SRC/bench/public/sexancestry/sex_collect.py --work $W --labels $LABELS --artifacts $A \
    --singlet-binary $BP/singlet_binary.json --bench-commit $(git -C $SRC rev-parse HEAD)
  rt python $SRC/bench/public/run_capability.py chromosomal-sex $A --out $BP/results/chromosomal-sex.json
  ;;
*) echo "usage: sex.sbatch extract|cellxy|collect"; exit 2 ;;
esac
