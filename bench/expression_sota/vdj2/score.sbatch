#!/bin/bash
#SBATCH --job-name=vdj2-score
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Score Singlet variants against the published Cell Ranger contigs (and TRUST4): secondary metrics per library
# (compare_vdj.py -> runs/<lib>/metrics<TAG>.json) and the primary metric (../vdj/primary.py -> scores/<SET><TAG>.json).
#   sbatch --export=ALL,SET=dev|heldout,TAGS="_base _v1" score.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-sota}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
V=$WORK/expression-sota/vdj2
if [ "${SET:-dev}" = heldout ]; then
  LIBS="gse145370_S134A:TCR:cr3 gse145370_S134B:TCR:cr3 gse145370_S159A:TCR:cr3 gse145370_S159B:TCR:cr3 pbmc3_t:TCR:cr3 ngpbmc3_t:TCR:cr3 pbmc1k_t:TCR:cr5 pbmc3_b:BCR:cr3 ngpbmc3_b:BCR:cr3 pbmc1k_b:BCR:cr5 melanoma_b:BCR:cr3"
else
  LIBS="pbmc2_t:TCR:cr3 pbmc2_b:BCR:cr3 pbmc10k_t:TCR:cr5 pbmc10k_b:BCR:cr5 gse145370_S133A:TCR:cr3 gse145370_S158A:TCR:cr3 gse145370_S149B:TCR:cr3 gse145370_S150B:TCR:cr3 gse145370_S135B:TCR:cr3"
fi
mkdir -p $V/scores
for TAG in $TAGS; do
  spec=""
  for it in $LIBS; do
    IFS=":" read -r lib ch conv <<< "$it"
    REF=$SRC/python/singlet/vdj/data/vdj_human.tsv.gz
    [ "$conv" = cr5 ] && REF=$SRC/python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz
    D=$WORK/data/vdjc/$lib; O=$V/runs/$lib
    [ -s $O/singlet$TAG/contigs.csv ] || { echo "skip $lib$TAG (no run)"; continue; }
    python -P $SRC/bench/vdj/compare_vdj.py --reference $D/filtered_contig_annotations.csv --chains $ch --vdj-ref $REF \
      $( [ -s $O/gex_cells.txt ] && echo --gex-cells $O/gex_cells.txt ) \
      --singlet $O/singlet$TAG/contigs.csv --singlet-cells $O/singlet$TAG/cells.csv \
      $( [ -s $O/trust4/TRUST_barcode_report.tsv ] && echo --trust4 $O/trust4/TRUST_barcode_report.tsv ) \
      $( [ -s $O/gex_lineage.csv ] && echo --gex-labels $O/gex_lineage.csv ) --out $O/metrics$TAG.json > /dev/null
    spec="$spec${spec:+,}$lib:$ch"
  done
  python -P $SRC/bench/expression_sota/vdj/primary.py --runs $V/runs --data $WORK/data/vdjc --libs $spec \
    --singlet-dir singlet$TAG --out $V/scores/${SET:-dev}$TAG.json > /dev/null
  python -P $SRC/bench/expression_sota/vdj2/table.py --runs $V/runs --scores $V/scores --set ${SET:-dev} --tag "$TAG" \
    --libs $spec ${BASE:+--base $BASE} | tee $V/scores/${SET:-dev}$TAG.md
done
