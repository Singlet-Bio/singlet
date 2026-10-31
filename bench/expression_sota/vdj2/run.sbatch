#!/bin/bash
#SBATCH --job-name=vdj2-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# One library per array task: Singlet (and/or TRUST4) on the same FASTQs and 16 threads, into
# $WORK/expression-sota/vdj2/runs/<lib>/{singlet<TAG>,trust4}.
#   sbatch --array=0-2 --export=ALL,LIBSPEC="lib:TCR:cr3+lib:BCR:cr5",METHODS="singlet trust4",TAG=,PARAMS="--param x=y" run.sbatch
# cr3 / cr5 = the Cell Ranger version convention of the library's truth (PREREG.md rule 3): Cell Ranger >= 5 starts the
# junction at the first of two adjacent Cys codons (reference vdj_human_imgtcys.tsv.gz), 3.x / 4.x at the second
# (vdj_human.tsv.gz, the packaged default).
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-sota}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
IFS="+" read -ra SPECS <<< "$LIBSPEC"
IFS=":" read -r DS CHAINS CONV <<< "${SPECS[$SLURM_ARRAY_TASK_ID]}"
D=$WORK/data/vdjc/$DS
O=$WORK/expression-sota/vdj2/runs/$DS; mkdir -p $O
T=16
WL=$WORK/pipeline-benchmark/whitelists/737K-august-2016.txt
SCRATCH=/tmp/$USER-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCRATCH; trap 'rm -rf $SCRATCH' EXIT
METHODS=${METHODS:-"singlet"}
TAG=${TAG:-}
REF=$SRC/python/singlet/vdj/data/vdj_human.tsv.gz
[ "$CONV" = cr5 ] && REF=$SRC/python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz
echo "lib $DS $CHAINS $CONV ref $REF sha $SHA tag '$TAG' params '${PARAMS:-}'"
TIME="/usr/bin/time -v"
if [[ " $METHODS " == *" singlet "* ]]; then
  $TIME python -P $SRC/bench/vdj/run_vdj.py --fastq-dir $D/fastqs --whitelist $WL --ref $REF \
    --out $O/singlet$TAG --threads $T ${PARAMS:-} 2> $O/singlet$TAG.time | tail -3
  grep -E "Elapsed|Maximum resident" $O/singlet$TAG.time
  echo "$SHA" > $O/singlet$TAG/SHA
fi
if [[ " $METHODS " == *" trust4 "* ]]; then
  cat $(find $D/fastqs -name "*_R1_*.fastq.gz" | sort) > $SCRATCH/R1.fq.gz
  cat $(find $D/fastqs -name "*_R2_*.fastq.gz" | sort) > $SCRATCH/R2.fq.gz
  r1len=$( (zcat $SCRATCH/R1.fq.gz 2>/dev/null || true) | head -2 | tail -1 | tr -d '\n' | wc -c)
  R4=$WORK/data/vdjc/trust4
  mkdir -p $O/trust4
  if [ "$r1len" -gt 60 ]; then
    ARGS="-1 $SCRATCH/R1.fq.gz -2 $SCRATCH/R2.fq.gz --readFormat r1:39:-1,bc:0:15,um:16:25"
  else
    ARGS="-u $SCRATCH/R2.fq.gz --readFormat bc:0:15,um:16:25"
  fi
  echo "trust4 $ARGS"
  ( cd $O/trust4 && $TIME run-trust4 -f $R4/hg38_bcrtcr.fa --ref $R4/human_IMGT+C.fa $ARGS --barcode $SCRATCH/R1.fq.gz \
      --UMI $SCRATCH/R1.fq.gz --barcodeWhitelist $WL -t $T -o TRUST --od $O/trust4 ) > $O/trust4.log 2> $O/trust4.time || tail -20 $O/trust4.time
  grep -E "Elapsed|Maximum resident" $O/trust4.time
fi
if [ ! -s $O/gex_lineage.csv ] && [ -s $D/gex_filtered.h5 ]; then python -P $SRC/bench/vdj/gex_lineage.py $D/gex_filtered.h5 $O/gex_lineage.csv; fi
if [ ! -s $O/gex_cells.txt ] && [ -s $D/gex_filtered.h5 ]; then python -P $SRC/bench/crispr/h5_barcodes.py $D/gex_filtered.h5 $O/gex_cells.txt; fi
