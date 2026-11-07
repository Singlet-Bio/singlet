#!/bin/bash
#SBATCH --job-name=cp-score
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Score one count-parity Singlet run against STARsolo (bench/count_parity/parity_score.py) and, with a molecule dump, diagnose it.
#   sbatch --export=ALL,DS=hgmm1k_v3,TAG=<run tag>[,BR=count/starsolo-parity][,STARSOLO=<dir>] score_run.sbatch
# Code comes from the branch (git archive of the Singlet-Bio/singlet bare clone). Output next to the run: parity.json, diagnose.json.
set -euo pipefail
: "${DS:?}" "${TAG:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BR=${BR:-count/starsolo-parity}; BARE=$W/repos/singlet-py.git; URL="$(git -C $BARE config remote.origin.url)"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/cp/score ) 9>$BARE/.gn-fetch.lock
SHA=$(git -C $BARE rev-parse refs/cp/score); CODE=$W/count-parity/code/${SHA:0:8}
[ -d $CODE ] || { mkdir -p $CODE.tmp && git -C $BARE archive $SHA bench/count_parity bench/exact_vs_pseudo | tar -x -C $CODE.tmp && mv $CODE.tmp $CODE; }
R=$W/count-parity/runs/$DS/$TAG
STARSOLO=${STARSOLO:-$W/count-parity/runs/$DS/starsolo}; [ -d $STARSOLO ] || STARSOLO=$W/exact-vs-pseudo/runs/$DS/starsolo
H5=$(ls $PB/data/$DS/*filtered_feature_bc_matrix.h5 | head -1)
case $DS in hgmm*) KIND=barnyard; GTF0=$REF/GRCh38-GRCm39-barnyard/genes/genes.gtf ;; *) KIND=cite; GTF0=$REF/GRCh38-2024-A/genes/genes.gtf ;; esac
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
grep -P '\tgene\t' $GTF0 > $SCR/genes.gtf
PY=$W/envs/pbench/bin/python
echo "code $SHA run $R starsolo $STARSOLO"
$PY -I $CODE/bench/count_parity/parity_score.py --singlet $R --starsolo $STARSOLO --cr $H5 --gtf $SCR/genes.gtf --kind $KIND --out $R/parity_$(basename $STARSOLO).json || true; cp $R/parity_$(basename $STARSOLO).json $R/parity.json
if [ -s $R/molecules.tsv.gz ] && [ $KIND = barnyard ]; then
  $PY -I $CODE/bench/count_parity/diagnose.py --dump $R/molecules.tsv.gz --cr $H5 --gtf $SCR/genes.gtf --out $R/diagnose.json
fi
