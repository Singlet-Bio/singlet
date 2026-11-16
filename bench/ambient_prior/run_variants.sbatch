#!/bin/bash
# D-163: ambient estimator variants on the development barnyards from the pushed copy of the repo ($W/bpcalls/amb/src), CPU only.
# VARIANTS=<file of "label key=value ..." lines>, LIBS="hgmm1k_v3 hgmm1k_v2 hgmm10k_v3", TAG=<result folder>. Inputs: $W/jobs/ambprior/in (dumped runs of v1.3.4).
#SBATCH --job-name=ambvar
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/ambvar-%j.out
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
A=$W/bpcalls/amb; T=$A/src; TAG=${TAG:-run}; IN=$W/jobs/ambprior/in; OUT=$A/out/$TAG
export PATH=$W/envs/pbench/bin:$PATH
x86_64-conda-linux-gnu-g++ -std=c++20 -O2 -pthread -I$T/include $T/bench/ambient_prior/harness.cpp -o $A/harness_$TAG || exit 3
PY=$W/envs/ambient/bin/python; export HOME=$W/cache/condahome MPLCONFIGDIR=$W/cache/mpl
LIBS=${LIBS:-"hgmm1k_v3 hgmm1k_v2 hgmm10k_v3"}
mkdir -p $OUT
while read -r label args; do
  [ -z "$label" ] && continue; case $label in \#*) continue;; esac
  for L in $LIBS; do
    [ -f $IN/$L/dims.txt ] || continue
    O=$OUT/$label/$L; rm -rf $O; mkdir -p $O
    $A/harness_$TAG $IN/$L $O $args 2> $O/err.log
    $PY -I $T/bench/ambient_prior/score.py $IN/$L $O "$label $L" | tail -n 3
    [ -n "${DIAG:-}" ] && [ "$L" = hgmm1k_v3 ] && $PY -I $T/bench/ambient_prior/diag_cells.py $IN/$L $O
  done
done < ${VARIANTS:-$A/variants.txt}
