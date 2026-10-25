#!/bin/bash
#SBATCH --job-name=vdjc-crispr
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One CRISPR guide library: counting + the three callers (run_crispr.py), then the comparison (compare_crispr.py).
#   sbatch --export=ALL,DS=a549 run.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
D=$WORK/data/vdjc/$DS
O=$WORK/state/vdjc/runs/$DS; mkdir -p $O
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
TIME="/usr/bin/time -v"
case $DS in
  a549)
    WL=$WORK/cache/adt/3M-february-2018.txt.gz; TR=$WORK/cache/adt/translation_3M-february-2018.txt.gz
    python -P $SRC/bench/crispr/h5_barcodes.py $D/raw_feature_bc_matrix.h5 $O/columns.txt --min-umis 1
    python -P $SRC/bench/crispr/h5_barcodes.py $D/count_sample_feature_bc_matrix.h5 $O/called.txt
    $TIME python -P $SRC/bench/crispr/run_crispr.py --fastq-dir $D/fastqs --r1-glob "*_crispr_*_R1_*.fastq.gz" \
      --guides $D/count_feature_reference.csv --whitelist $WL --translation $TR --cells $O/columns.txt \
      --called-file $O/called.txt --out $O/singlet --shards 16 --tmp $SCRATCH 2> $O/singlet.time | tail -5
    grep -E "Elapsed|Maximum resident" $O/singlet.time
    python -P $SRC/bench/crispr/compare_crispr.py --run $O/singlet \
      --reference $D/crispr_analysis/crispr_analysis/protospacer_calls_per_cell.csv --ref-format cellranger \
      --gex $D/count_sample_feature_bc_matrix.h5 --guides $D/count_feature_reference.csv --out $O/metrics.json | head -80
    ;;
  *) echo "unknown DS $DS"; exit 2;;
esac
