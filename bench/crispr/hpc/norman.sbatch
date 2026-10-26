#!/bin/bash
#SBATCH --job-name=vdjc-norman
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Norman 2019 gem group 1: guide-barcode table learnt on the training half (prep_norman.py), counting + callers
# (run_crispr.py, 3' v2), comparison on the held-out half with target induction (CRISPRa).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
D=$WORK/data/vdjc/norman2019_gem1; O=$WORK/state/vdjc/runs/norman2019_gem1; mkdir -p $O
A=$WORK/bench-public/artifacts/crispr-guides
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
python -P $SRC/bench/crispr/prep_norman.py $D --gemgroup 1
cut -d, -f1 $D/reference_gg1.csv | tail -n +2 > $O/columns.txt
/usr/bin/time -v python -P $SRC/bench/crispr/run_crispr.py --fastq-dir $D/fastqs --guide-table $D/guides.tsv \
  --pattern "(BC)CCTAGAGGGCCCG" --whitelist $WORK/pipeline-benchmark/whitelists/737K-august-2016.txt \
  --cells $O/columns.txt --out $O/singlet --umi-len 10 2> $O/singlet.time | tail -5
grep -E "Elapsed|Maximum resident" $O/singlet.time
python -P $SRC/bench/crispr/compare_crispr.py --run $O/singlet --reference $D/reference_gg1.csv --ref-format norman \
  --mode a --gemgroup 1 --norman-dir $D --guides $D/guides.tsv --columns $O/columns.txt --out $O/metrics.json \
  --artifacts $A --dataset norman2019_gem1 | head -60
