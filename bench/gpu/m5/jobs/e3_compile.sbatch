#!/bin/bash
#SBATCH --job-name=e3-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E3 (warp-window): build bench/gpu/m5 from branch gpu/fable/warp-window -> $W/jobs/fable/warp-window/bin/m5e3 (+ .commit,
# the full ptxas register/spill report ptxas_<sha>.txt and its summary for the phase-3 kernels)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif
E=$W/jobs/fable/warp-window; mkdir -p $E/bin $E/src $E/results
BARE=$W/repos/singlet-py.git; BR=${BR:-gpu/fable/warp-window}; NAME=${NAME:-m5e3}   # NAME: output binary name
DEFS=${DEFS:-}; CHECK=${CHECK:-}   # DEFS: extra -D flags of NAME; CHECK=1: also compile the other flavours (no binaries kept)
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/warp-window
SHA=$(git -C $BARE rev-parse --short=10 refs/fable/warp-window)
SRC=$E/src/$SHA; rm -rf $SRC; mkdir -p $SRC
git -C $BARE archive refs/fable/warp-window bench/gpu/m5 bench/fable/profile/tail.cuh | tar -x -C $SRC
echo "commit $SHA node $(hostname) $(date -Is)"
T0=$(date +%s)
NV="apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v"
if [ -n "$CHECK" ]; then   # the merge gate's flavours (E1 tail, E2 census, E5 baseline and KEEPIX, E3 statistics), in parallel
  for F in "e3stats:-DM5_E3STATS" "tail:-DM5_TAIL" "census:-DM5_CENSUS" "baseline:-DM5_BASELINE" "keepix:-DM5_KEEPIX"; do
    ( $NV ${F#*:} -o $SRC/m5_${F%%:*} $SRC/bench/gpu/m5/m5.cu > $E/bin/ptxas_${SHA}_${F%%:*}.txt 2>&1 && echo "[check ${F%%:*}] ok" \
      || { grep -E "error" $E/bin/ptxas_${SHA}_${F%%:*}.txt | head -20; echo "[check ${F%%:*}] FAILED"; } ) &
  done
fi
if ! apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v $DEFS \
     -o $E/bin/$NAME.new $SRC/bench/gpu/m5/m5.cu > $E/bin/ptxas_$SHA.txt 2>&1; then
  grep -E "error|warning" $E/bin/ptxas_$SHA.txt | head -40; echo "[compile] FAILED"; exit 1
fi
echo "[t] nvcc $(( $(date +%s) - T0 )) s"
# summary: entry, registers, stack frame and spills of every kernel
awk '/Compiling entry function/ {match($0, /_Z[^'"'"']*/); fn=substr($0, RSTART, RLENGTH)}
     /bytes stack frame/ {sf=$0} /Used [0-9]+ registers/ {print fn; print "   " sf; print "   " $0}' $E/bin/ptxas_$SHA.txt \
  | sed 's/ptxas info    ://' > $E/bin/ptxas_${SHA}_summary.txt
grep -A2 "k_p3\|k_p1\|k_p2" $E/bin/ptxas_${SHA}_summary.txt | grep -v "^--" | cut -c1-200 | head -60
mv $E/bin/$NAME.new $E/bin/$NAME; echo "$SHA $DEFS" > $E/bin/$NAME.commit
wait
if [ -n "$CHECK" ]; then for F in e3stats tail census baseline keepix; do echo "== $F"; grep -E "registers|spill" $E/bin/ptxas_${SHA}_$F.txt | sort | uniq -c | sort -rn | head -6 || true; done; rm -f $SRC/m5_*; fi
echo "[compile] ok $SHA -> $E/bin/$NAME"
