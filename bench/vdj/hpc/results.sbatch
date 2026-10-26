#!/bin/bash
#SBATCH --job-name=vdjc-results
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Benchmark artifacts and the vdj / crispr-guides capability documents (bench/public contract), computed inside a
# copy of bench-public's harness (contract.py, _common.py, cohort.tsv from its branch) with this branch's modules.
# Writes $WORK/bench-public/artifacts/{vdj,crispr-guides}/ and $WORK/state/vdjc/results/{vdj,crispr-guides}.json.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/vdjcrispr
source $WORK/jobs/vdjc/src.sh
export PATH=$ENV/bin:$PATH PYTHONPATH=$SRC/python HOME=$WORK/cache/condahome
BARE=$WORK/repos/singlet-py.git
BP=claude/bench-public-cohort-485cf7
git -C $BARE fetch -q origin "+refs/heads/$BP:refs/heads/$BP" || true
T=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $T; trap 'rm -rf $T' EXIT
git -C $BARE archive "refs/heads/$BP" bench/public | tar -x -C $T
mkdir -p $T/python $T/bench/vdj $T/bench/crispr
cp -r $SRC/python/singlet $T/python/; cp $SRC/bench/vdj/*.py $T/bench/vdj/; cp $SRC/bench/crispr/*.py $T/bench/crispr/
cp $SRC/bench/public/capabilities/vdj.py $SRC/bench/public/capabilities/crispr_guides.py $T/bench/public/capabilities/
# cohort rows this program still needs (until bench-public adds them): appended to the copy only
if ! grep -q "^replogle2022_k562_gg41" $T/bench/public/cohort.tsv; then
  awk -F'\t' -v OFS='\t' '$1=="replogle2022_k562_lane4"{$1="replogle2022_k562_gg41"; print}' $T/bench/public/cohort.tsv >> $T/bench/public/cohort.tsv
fi
export PYTHONPATH=$T/python
AV=$WORK/bench-public/artifacts/vdj; AC=$WORK/bench-public/artifacts/crispr-guides; mkdir -p $AV $AC $WORK/state/vdjc/results
python -P $T/bench/vdj/make_artifacts.py $WORK/state/vdjc/runs $WORK/data/vdjc $AV --commit $SHA \
  --vdj-ref $WORK/state/vdjc/ref/vdj_human.tsv.gz | tail -30
python -P $T/bench/crispr/make_provenance.py $WORK/state/vdjc/runs $AC --commit $SHA | tail -20
cd $T
python bench/public/run_capability.py vdj $AV --out $WORK/state/vdjc/results/vdj.json
python bench/public/run_capability.py crispr-guides $AC --out $WORK/state/vdjc/results/crispr-guides.json
du -sh $AV $AC
