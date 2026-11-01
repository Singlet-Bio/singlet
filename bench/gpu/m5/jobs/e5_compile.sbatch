#!/bin/bash
#SBATCH --job-name=e5-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E5 (compact): build m5 variants from branch gpu/fable/compact (fetched from GitHub through the bare mirror) in the CUDA 13.3
# image. VARIANTS="name:-DFLAG -DFLAG2;name2:" (default: baseline + copy statistics). Binaries -> $OUT/bin/<name>-<sha7>,
# ptxas -v output -> $OUT/results/ptxas_<name>-<sha7>.txt (registers, stack frame, spills per kernel).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/compact
BARE=$W/repos/singlet-py.git
mkdir -p $OUT/bin $OUT/results
SCR=/tmp/$USER-e5cc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/compact:refs/fable/compact
SHA=$(git -C $BARE rev-parse --short=7 refs/fable/compact); echo "commit $SHA $(git -C $BARE log -1 --format=%s refs/fable/compact)"
git -C $BARE archive refs/fable/compact bench/gpu/m5 | tar -x -C $SCR
VARIANTS=${VARIANTS:-"m5:;m5s:-DM5_COPYSTATS"}
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
  N=${v%%:*}; FL=${v#*:}
  ( apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v $FL \
      -o $OUT/bin/$N-$SHA $SCR/bench/gpu/m5/m5.cu > $OUT/results/ptxas_$N-$SHA.txt 2>&1 && echo "[cc] $N-$SHA ok ($FL)" \
    || { echo "[cc] $N-$SHA FAILED ($FL)"; grep -m 20 -i "error" $OUT/results/ptxas_$N-$SHA.txt; } ) &
done
wait
# lane A kernels (Caps<160,4,32,16,16,256,...>): registers / stack / spills
for v in "${VS[@]}"; do
  N=${v%%:*}; F=$OUT/results/ptxas_$N-$SHA.txt; [ -f $F ] || continue
  echo "== $N-$SHA"
  awk '/Compiling entry function/ {k=$0} /bytes stack frame/ {sf=$0} \
       /Used [0-9]+ registers/ { if (k ~ /ILi160ELi4ELi32ELi16ELi16ELi256E/ && k ~ /k_p1|k_p2|k_p3|k_lane/) {
         n = (k ~ /k_p1/) ? "k_p1" : (k ~ /k_p2/) ? "k_p2" : (k ~ /k_p3/) ? "k_p3" : "k_lane"; n = n ((k ~ /Lb1E/) ? "<local>" : (k ~ /Lb0E/) ? "<global>" : "");
         r = $0; sub(/.*Used/, "Used", r); s2 = sf; gsub(/^ +/, "", s2); print "  LaneA " n ": " r " | " s2 } }' $F
done
