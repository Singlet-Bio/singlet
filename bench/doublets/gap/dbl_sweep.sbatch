#!/bin/bash
#SBATCH --job-name=dbl-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Doublet-score sweep on development libraries, one whole CPU node (many dbl_run processes in parallel).
#   sbatch --export=ALL,BRANCH=fix/doublet-expression-score,VARIANTS=/path/variants.txt dbl_sweep.sbatch
# variants.txt: one "label key=value key=value ..." per line (dbl_run keys, see bench/modalities/doublet_fix/dbl_run.cpp).
# Inputs: the v1.3.4 gap-table run's called-cell matrices (jobs/gap134/amb/work/<ds>/cells.mtx). No genotype input anywhere.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/dblfix; BARE=$W/repos/singlet-py.git
BRANCH=${BRANCH:-fix/doublet-expression-score}; VARIANTS=${VARIANTS:-$J/variants.txt}
WORK=$W/jobs/gap134/amb/work; TRUTH=$W/modality-bench/genotype_pool/truth; DATA=$J/data
PY=$W/envs/pbench/bin/python; export PATH=$W/envs/pbench/bin:$PATH
mkdir -p $J/run $J/res $J/bin $DATA
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/dblfix/head || exit 2
COMMIT=$(git -C $BARE rev-parse refs/dblfix/head); echo "[sweep] $BRANCH $COMMIT"
T=$(mktemp -d /tmp/dblfix.XXXX); trap 'rm -rf $T' EXIT
git -C $BARE archive $COMMIT include/singlet/pileup bench/modalities/doublet_fix/dbl_run.cpp | tar -x -C $T
BINX=$J/bin/dbl_run.${COMMIT:0:8}
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$T/include/singlet/pileup $T/bench/modalities/doublet_fix/dbl_run.cpp -o $BINX || exit 3
git -C $BARE archive $COMMIT bench/doublets/gap | tar -x -C $T
[ -f $DATA/kang2018_ctrl.hdr ] || $PY -I $T/bench/doublets/gap/dbl_sweep.py prep $WORK $DATA
SW=$T/bench/doublets/gap/dbl_sweep.py
DS="kang2018_ctrl kang2018_b1_a kang2018_b1_b kang2018_b1_c hgmm1k_v3 hgmm1k_v2"
jobs_list=$T/jobs.txt; : > $jobs_list
while read -r label args; do
  [ -z "$label" ] && continue; case $label in \#*) continue;; esac
  mkdir -p $J/run/$label
  for d in $DS; do echo "$label $d $args" | sed "s/ *$//" >> $jobs_list; done
done < $VARIANTS
runone() { label=$1; d=$2; shift 2; $BINX $DATA/$d $J/run/$label/$d.tsv threads=4 nomap=1 "$@" 2> $J/run/$label/$d.err; }
export -f runone; export BINX DATA J
xargs -P 11 -L 1 bash -c 'runone "$@"' _ < $jobs_list
while read -r label args; do
  [ -z "$label" ] && continue; case $label in \#*) continue;; esac
  $PY -I $SW score $DATA $J/run $label $J/res/$label.json $TRUTH $WORK
done < $VARIANTS
echo "[sweep] done"
