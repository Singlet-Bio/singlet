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
      --gex $D/count_sample_feature_bc_matrix.h5 --guides $D/count_feature_reference.csv --out $O/metrics.json       --cr-thresholds $D/crispr_analysis/crispr_analysis/protospacer_umi_thresholds.json       --cr-h5 $D/count_sample_feature_bc_matrix.h5 --columns $O/columns.txt       --artifacts $WORK/bench-public/artifacts/crispr-guides --dataset tenx_crispr_a549_5k | head -80
    ;;
  a375)
    WL=$WORK/cache/adt/3M-february-2018.txt.gz; TR=$WORK/cache/adt/translation_3M-february-2018.txt.gz
    FR=$(find $D/crispr_analysis -name feature_reference.csv | head -1)
    PC=$(find $D/crispr_analysis -name protospacer_calls_per_cell.csv | head -1)
    TH=$(find $D/crispr_analysis -name protospacer_umi_thresholds.json | head -1)
    python -P $SRC/bench/crispr/h5_barcodes.py $D/raw_feature_bc_matrix.h5 $O/columns.txt --min-umis 1
    python -P $SRC/bench/crispr/h5_barcodes.py $D/filtered_feature_bc_matrix.h5 $O/called.txt
    $TIME python -P $SRC/bench/crispr/run_crispr.py --fastq-dir $D/fastqs --r1-glob "*rispr*_R1_*.fastq.gz" \
      --guides $FR --whitelist $WL --translation $TR --cells $O/columns.txt \
      --called-file $O/called.txt --out $O/singlet --shards 16 --tmp $SCRATCH 2> $O/singlet.time | tail -5
    grep -E "Elapsed|Maximum resident" $O/singlet.time
    python -P $SRC/bench/crispr/compare_crispr.py --run $O/singlet --reference $PC --ref-format cellranger \
      --gex $D/filtered_feature_bc_matrix.h5 --guides $FR ${TH:+--cr-thresholds $TH} --cr-h5 $D/filtered_feature_bc_matrix.h5 \
      --columns $O/columns.txt --out $O/metrics.json --artifacts $WORK/bench-public/artifacts/crispr-guides \
      --dataset tenx_crispr_a375_10k | head -60
    ;;
  replogle_k562_gg41)
    # published cells of gem group 41 are the columns (the h5ad holds no empty droplets); molecules on other whitelist
    # barcodes are the ambient profile. Dual guides: protospacer A before GTTTCAGAGC, B before GTTTAAGAGC.
    WL=$WORK/cache/adt/3M-february-2018.txt.gz; TR=$WORK/cache/adt/translation_3M-february-2018.txt.gz
    cut -d, -f1 $D/reference_gg41.csv | tail -n +2 > $O/columns.txt
    $TIME python -P $SRC/bench/crispr/run_crispr.py --fastq-dir $D/fastqs --guide-table $D/guides.tsv       --pattern "(BC)GTTTCAGAGC|GTTTAAGAGC" --whitelist $WL --translation $TR --cells $O/columns.txt       --out $O/singlet --shards 16 --tmp $SCRATCH 2> $O/singlet.time | tail -5
    grep -E "Elapsed|Maximum resident" $O/singlet.time
    python -P $SRC/bench/crispr/compare_crispr.py --run $O/singlet --reference $D/K562_essential_raw_singlecell_01.h5ad       --ref-format replogle --gemgroup 41 --guides $D/guides.tsv --columns $O/columns.txt --out $O/metrics.json       --artifacts $WORK/bench-public/artifacts/crispr-guides --dataset replogle2022_k562_gg41 | head -60
    ;;
  *) echo "unknown DS $DS"; exit 2;;
esac
