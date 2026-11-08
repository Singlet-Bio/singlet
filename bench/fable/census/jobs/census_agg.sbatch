#!/bin/bash
#SBATCH --job-name=fable-census-agg
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Re-run census_agg.py (branch tip) on the raw records of earlier census runs (no sequences in them): LABELS="pbmc1k_v3 ..."
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; OUT=$W/jobs/fable/census; PY=$W/venvs/grantfig/bin/python3
SCR=/tmp/$USER-census-agg-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/census:refs/fable/census
echo "aggregation from $(git -C $BARE rev-parse refs/fable/census)"
git -C $BARE archive refs/fable/census bench/fable/census | tar -x -C $SCR
for L in ${LABELS:-pbmc1k_v3}; do
  R=$OUT/results/raw/census_$L
  [ -f $R.reads.bin.gz ] || { echo "no records for $L"; continue; }
  gunzip -c $R.reads.bin.gz > $SCR/t.reads.bin; gunzip -c $R.wins.bin.gz > $SCR/t.wins.bin; cp $R.meta.json $SCR/t.meta.json
  T0=$(date +%s)
  $PY $SCR/bench/fable/census/census_agg.py $SCR/t $OUT/results/census_$L.json --label $L > $OUT/results/census_$L.md 2> $OUT/results/census_$L.err
  echo "$L agg exit=$? $(( $(date +%s) - T0 )) s"; tail -3 $OUT/results/census_$L.err
done
