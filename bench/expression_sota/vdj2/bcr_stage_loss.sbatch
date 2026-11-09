#!/bin/bash
#SBATCH --job-name=b12-stage
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# B12 per-stage loss of reference chains on the five BCR development libraries (bcr_stage_loss.py), code at gpu/frontier.
set -uo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$WORK/envs/vdjcrispr
BRANCH=${BRANCH:-gpu/frontier}; source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome NUMBA_CACHE_DIR=$WORK/cache/numba-b12
V=$WORK/jobs/b12; mkdir -p $V/stage
REF5=$SRC/python/singlet/vdj/data/vdj_human_imgtcys.tsv.gz; REF3=$SRC/python/singlet/vdj/data/vdj_human.tsv.gz
LIBS=${LIBS:-"chx10k_b:cr5 pbmc10k_b:cr5 pbmc2_b:cr3 nsclc22_b:cr3 melv11_b:cr5"}
for it in $LIBS; do IFS=: read L c <<< "$it"; R=$REF5; [ $c = cr3 ] && R=$REF3
  python -P $V/bcr_stage_loss.py $L $R $V/stage/$L.json > $V/stage/$L.log 2>&1 &
done
wait
for it in $LIBS; do L=${it%%:*}; echo "== $L"; grep -A40 '^{' $V/stage/$L.log | head -60; tail -3 $V/stage/$L.log; done
