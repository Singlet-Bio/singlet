#!/bin/bash
#SBATCH --job-name=vdjh-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Secondary metrics (bench/vdj/compare_vdj.py) for runs/<lib>/singlet<TAG> and the TRUST4 run, into runs/<lib>/metrics<TAG>.json.
#   sbatch --export=ALL,LIBSET=dev|heldout,TAG=_fix,REFV=<segment reference> compare.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-heldout}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
if [ "${LIBSET:-dev}" = heldout ]; then
  LIBS="ctrl10k_t:TCR ctrl10k_b:BCR gse145370_S135A:TCR gse145370_S149A:TCR gse145370_S150A:TCR gse145370_S133B:TCR"
else
  LIBS="pbmc2_t:TCR pbmc2_b:BCR pbmc10k_t:TCR pbmc10k_b:BCR gse145370_S133A:TCR gse145370_S158A:TCR"
fi
[ -n "${LIBSPEC:-}" ] && LIBS=$(echo $LIBSPEC | tr + " ")
TAG=${TAG:-}
for it in $LIBS; do
  lib=${it%%:*}; ch=${it##*:}
  D=$WORK/data/vdjc/$lib; O=$WORK/state/vdjc/runs/$lib
  python -P $SRC/bench/vdj/compare_vdj.py --reference $D/filtered_contig_annotations.csv --chains $ch \
    --vdj-ref ${REFV:-$WORK/state/vdjc/ref/vdj_human.tsv.gz} $( [ -s $O/gex_cells.txt ] && echo --gex-cells $O/gex_cells.txt ) \
    --singlet $O/singlet$TAG/contigs.csv --singlet-cells $O/singlet$TAG/cells.csv \
    $( [ -s $O/trust4/TRUST_barcode_report.tsv ] && echo --trust4 $O/trust4/TRUST_barcode_report.tsv ) \
    $( [ -s $O/gex_lineage.csv ] && echo --gex-labels $O/gex_lineage.csv ) --out $O/metrics$TAG.json > /dev/null
  echo done $lib
done
