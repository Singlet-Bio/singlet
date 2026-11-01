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
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/warp-window
SHA=$(git -C $BARE rev-parse --short=10 refs/fable/warp-window)
SRC=$E/src/$SHA; rm -rf $SRC; mkdir -p $SRC
git -C $BARE archive refs/fable/warp-window bench/gpu/m5 | tar -x -C $SRC
echo "commit $SHA node $(hostname) $(date -Is)"
T0=$(date +%s)
if ! apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v \
     -o $E/bin/$NAME.new $SRC/bench/gpu/m5/m5.cu > $E/bin/ptxas_$SHA.txt 2>&1; then
  grep -E "error|warning" $E/bin/ptxas_$SHA.txt | head -40; echo "[compile] FAILED"; exit 1
fi
echo "[t] nvcc $(( $(date +%s) - T0 )) s"
# summary: entry, registers, stack frame and spills of every kernel
awk '/Compiling entry function/ {match($0, /_Z[^'"'"']*/); fn=substr($0, RSTART, RLENGTH)}
     /bytes stack frame/ {sf=$0} /Used [0-9]+ registers/ {print fn; print "   " sf; print "   " $0}' $E/bin/ptxas_$SHA.txt \
  | sed 's/ptxas info    ://' > $E/bin/ptxas_${SHA}_summary.txt
grep -A2 "k_p3\|k_p1\|k_p2" $E/bin/ptxas_${SHA}_summary.txt | grep -v "^--" | cut -c1-200 | head -60
mv $E/bin/$NAME.new $E/bin/$NAME; echo $SHA > $E/bin/$NAME.commit
echo "[compile] ok $SHA -> $E/bin/$NAME"
