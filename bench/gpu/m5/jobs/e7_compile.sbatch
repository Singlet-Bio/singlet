#!/bin/bash
#SBATCH --job-name=e7-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E7 (seeds): build m5 variants from branch gpu/fable/seeds (fetched from GitHub through the bare mirror) in the CUDA 13.3
# image. VARIANTS="name:-DFLAG -DFLAG2;name2:" (default below). Binaries -> $OUT/bin/<name>-<sha7> plus the link
# $OUT/bin/<name> -> <name>-<sha7>; ptxas -v output -> $OUT/results/ptxas_<name>-<sha7>.txt (registers, stack, spills).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/seeds
BARE=$W/repos/singlet-py.git
mkdir -p $OUT/bin $OUT/results
SCR=/tmp/$USER-e7cc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/seeds:refs/fable/seeds
SHA=$(git -C $BARE rev-parse --short=7 refs/fable/seeds); echo "commit $SHA $(git -C $BARE log -1 --format=%s refs/fable/seeds)"
git -C $BARE archive refs/fable/seeds bench/gpu/m5 | tar -x -C $SCR
VARIANTS=${VARIANTS:-"m5:;flatp:-DM5_SEEDFLAT -DM5_P1SPLIT;flatpni:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_CMPNI;flatpg:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_G2;step:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSTEP;stepg:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSTEP -DM5_G2;step8:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSTEP -DM5_P1MINB=8;stepg8:-DM5_SEEDFLAT -DM5_P1SPLIT -DM5_SEEDSTEP -DM5_G2 -DM5_P1MINB=8"}
IFS=';' read -ra VS <<< "$VARIANTS"
for v in "${VS[@]}"; do
  N=${v%%:*}; FL=${v#*:}
  ( apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v $FL \
      -o $OUT/bin/$N-$SHA $SCR/bench/gpu/m5/m5.cu > $OUT/results/ptxas_$N-$SHA.txt 2>&1 && ln -sfn $N-$SHA $OUT/bin/$N && echo "[cc] $N-$SHA ok ($FL)" \
    || { echo "[cc] $N-$SHA FAILED ($FL)"; grep -m 30 -i "error" $OUT/results/ptxas_$N-$SHA.txt; } ) &
done
wait
echo $SHA > $OUT/bin/commit
# phase-1 kernels and k_p3 of each lane (A = Caps<160,4,32,16,...>, M = <160,4,32,64,...>, B = <160,8,128,128,...>)
for v in "${VS[@]}"; do
  N=${v%%:*}; F=$OUT/results/ptxas_$N-$SHA.txt; [ -f $F ] || continue
  echo "== $N-$SHA"
  awk '/Compiling entry function/ {k=$0} /bytes stack frame/ {sf=$0} \
       /Used [0-9]+ registers/ { if (k ~ /k_p1|k_p3/) {
         ln = (k ~ /ILi160ELi4ELi32ELi16E/) ? "A" : (k ~ /ILi160ELi4ELi32ELi64E/) ? "M" : (k ~ /ILi160ELi8ELi128ELi128E/) ? "B" : "?";
         n = (k ~ /k_p1s/) ? "k_p1s" : (k ~ /k_p1w/) ? "k_p1w" : (k ~ /k_p1/) ? "k_p1" : "k_p3";
         r = $0; sub(/.*Used/, "Used", r); s2 = sf; gsub(/^ +/, "", s2); print "  Lane" ln " " n ": " r " | " s2 } }' $F
done
