#!/bin/bash
#SBATCH --job-name=vdjh-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Singlet only (TRUST4 outputs are reused) with a variant reference, into runs/<lib>/singlet<TAG>.
#   sbatch --array=0-5%3 --export=ALL,TAG=_fix,REF=$WORK/state/vdjh/vdj_human_fix.tsv.gz[,LIBSET=dev|heldout] run_fix.sbatch
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-bench/vdj-heldout}
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-vdjc
if [ "${LIBSET:-dev}" = heldout ]; then
  LIBS=(ctrl10k_t ctrl10k_b gse145370_S135A gse145370_S149A gse145370_S150A gse145370_S133B)
else
  LIBS=(pbmc2_t pbmc2_b pbmc10k_t pbmc10k_b gse145370_S133A gse145370_S158A)
fi
DS=${LIBS[$SLURM_ARRAY_TASK_ID]}
D=$WORK/data/vdjc/$DS
O=$WORK/state/vdjc/runs/$DS; mkdir -p $O
WL=$WORK/pipeline-benchmark/whitelists/737K-august-2016.txt
echo "lib $DS tag $TAG ref $REF"
/usr/bin/time -v python -P $SRC/bench/vdj/run_vdj.py --fastq-dir $D/fastqs --whitelist $WL --ref $REF \
  --out $O/singlet$TAG --threads 16 ${PARAMS:+$PARAMS} 2> $O/singlet$TAG.time | tail -5
grep -E "Elapsed|Maximum resident" $O/singlet$TAG.time
