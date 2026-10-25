#!/bin/bash
#SBATCH --job-name=vdjc-run
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One V(D)J library: Singlet and TRUST4 on the same FASTQs and threads, then the comparison with the reference contigs.
#   sbatch --export=ALL,DS=pbmc2_t,CHAINS=TCR[,METHODS="singlet trust4"] run.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=claude/vdj-crispr-benchmark-721d3a
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
git -C $WORK/repos/singlet-py.git fetch -q origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
git -C $SRC checkout -q --detach "$BRANCH"; git -C $SRC log --oneline -1
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
D=$WORK/data/vdjc/$DS
O=$WORK/state/vdjc/runs/$DS; mkdir -p $O
T=${THREADS:-16}
WL=$WORK/pipeline-benchmark/whitelists/737K-august-2016.txt
SCRATCH=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
METHODS=${METHODS:-"singlet trust4"}
TIME="/usr/bin/time -v"
if [[ " $METHODS " == *" singlet "* ]]; then
  $TIME python -P $SRC/bench/vdj/run_vdj.py --fastq-dir $D/fastqs --whitelist $WL --ref $WORK/state/vdjc/ref/vdj_human.tsv.gz \
    --out $O/singlet --threads $T ${MAX_READS:+--max-reads $MAX_READS} 2> $O/singlet.time | tail -20
  grep -E "Elapsed|Maximum resident" $O/singlet.time
fi
if [[ " $METHODS " == *" trust4 "* ]]; then
  cat $(ls $D/fastqs/*/*_R1_*.fastq.gz | sort) > $SCRATCH/R1.fq.gz
  cat $(ls $D/fastqs/*/*_R2_*.fastq.gz | sort) > $SCRATCH/R2.fq.gz
  r1len=$(zcat $SCRATCH/R1.fq.gz | head -2 | tail -1 | tr -d '\n' | wc -c)
  R4=$WORK/data/vdjc/trust4
  mkdir -p $O/trust4
  if [ "$r1len" -gt 60 ]; then  # R1 carries insert after barcode, UMI and TSO: use both mates
    ARGS="-1 $SCRATCH/R1.fq.gz -2 $SCRATCH/R2.fq.gz --readFormat r1:39:-1,bc:0:15,um:16:25"
  else
    ARGS="-u $SCRATCH/R2.fq.gz --readFormat bc:0:15,um:16:25"
  fi
  echo "trust4 $ARGS"
  ( cd $O/trust4 && $TIME run-trust4 -f $R4/hg38_bcrtcr.fa --ref $R4/human_IMGT+C.fa $ARGS --barcode $SCRATCH/R1.fq.gz \
      --UMI $SCRATCH/R1.fq.gz --barcodeWhitelist $WL -t $T -o TRUST --od $O/trust4 ) > $O/trust4.log 2> $O/trust4.time || tail -20 $O/trust4.time
  grep -E "Elapsed|Maximum resident" $O/trust4.time
fi
[ -s $O/gex_lineage.csv ] || python -P $SRC/bench/vdj/gex_lineage.py $D/gex_filtered.h5 $O/gex_lineage.csv
python -P $SRC/bench/vdj/compare_vdj.py --reference $D/filtered_contig_annotations.csv --chains $CHAINS \
  --singlet $O/singlet/contigs.csv --singlet-cells $O/singlet/cells.csv \
  $( [ -s $O/trust4/TRUST_barcode_report.tsv ] && echo --trust4 $O/trust4/TRUST_barcode_report.tsv ) \
  $( [ -s $O/gex_lineage.csv ] && echo --gex-labels $O/gex_lineage.csv ) --out $O/metrics.json
