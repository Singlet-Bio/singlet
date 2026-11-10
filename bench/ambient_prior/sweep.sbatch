#!/bin/bash
#SBATCH --job-name=amb-prior
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Ambient estimator variants on development barnyards, no engine rerun: compile bench/ambient_prior/harness.cpp against the branch's
# ambient_correction.h, feed it the counts of v1.3.4 runs, score against the cross-species truth by depth.
# Variants: $J/variants.txt, one "label key=value ..." per line (keys: see set_param in harness.cpp). Libraries: LIBS below.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/ambprior; BARE=$W/repos/singlet-py.git
BRANCH=${BRANCH:-fix/ambient-shallow-prior}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/ambp/head || exit 2
C=$(git -C $BARE rev-parse refs/ambp/head); echo "commit $C"
T=$(mktemp -d /tmp/ambp.XXXX); trap 'rm -rf $T' EXIT
git -C $BARE archive $C include bench/ambient_prior | tar -x -C $T
export PATH=$W/envs/pbench/bin:$PATH
x86_64-conda-linux-gnu-g++ -std=c++20 -O2 -pthread -I$T/include $T/bench/ambient_prior/harness.cpp -o $T/harness || exit 3
PY=$W/envs/ambient/bin/python; export HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
declare -A RUNS=( [hgmm1k_v3]=$W/jobs/gap134/runs/hgmm1k_v3 [hgmm1k_v2]=$W/jobs/gap134/runs/hgmm1k_v2 [hgmm10k_v3]=$W/jobs/ambprior/runs/hgmm10k_v3 )
LIBS=${LIBS:-"hgmm1k_v2 hgmm10k_v3 hgmm1k_v3"}
mkdir -p $J/in $J/out
for L in $LIBS; do
  [ -f $J/in/$L/dims.txt ] || $PY -I $T/bench/ambient_prior/dump_run.py ${RUNS[$L]} $J/in/$L > $J/in/$L.dump.log 2>&1 || { echo "dump failed $L"; tail -3 $J/in/$L.dump.log; }
done
while read -r label args; do
  [ -z "$label" ] && continue; case $label in \#*) continue;; esac
  for L in $LIBS; do
    [ -f $J/in/$L/dims.txt ] || continue
    O=$J/out/$label/$L; rm -rf $O; mkdir -p $O
    $T/harness $J/in/$L $O $args 2> $O/err.log
    $PY -I $T/bench/ambient_prior/score.py $J/in/$L $O "$label $L" | tail -n 11
  done
done < ${VARIANTS:-$J/variants.txt}
