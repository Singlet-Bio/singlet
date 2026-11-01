#!/bin/bash
#SBATCH --job-name=e4-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E4 (docs/gpu-engine/fable/E4-order.md): build bench/gpu/m5 from branch gpu/fable/order (GitHub, via the bare mirror) in the
# CUDA 13.3 image -> $W/jobs/fable/order/bin/m5-<sha>; the sha is compiled in (M5_GIT) and printed by every run.
# BRANCH (default gpu/fable/order) and REF may be overridden.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif
O=$W/jobs/fable/order; mkdir -p $O/bin $O/results
SCR=/tmp/$USER-e4cc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git; BRANCH=${BRANCH:-gpu/fable/order}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/fable/order
SHA=$(git -C $BARE rev-parse --short=10 ${REF:-refs/fable/order})
echo "building $BRANCH at $SHA on $(hostname) $(date -Is)"
git -C $BARE archive ${REF:-refs/fable/order} bench/gpu/m5 | tar -x -C $SCR
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v \
  -DM5_GIT="\"$SHA\"" -o $O/bin/m5-$SHA.new $SCR/bench/gpu/m5/m5.cu > $O/results/ptxas-$SHA.txt 2>&1 || { tail -40 $O/results/ptxas-$SHA.txt; exit 1; }
mv $O/bin/m5-$SHA.new $O/bin/m5-$SHA
echo "$SHA" > $O/bin/latest.sha
echo "[compile] ok in $(( $(date +%s) - T0 )) s: $O/bin/m5-$SHA sha256 $(sha256sum $O/bin/m5-$SHA | cut -c1-16)"
# registers / spills of the lane kernels (ptxas -v: "Compiling entry function" then "Used N registers")
awk '/Compiling entry function/ {f=$0; sub(/.*function ./,"",f); sub(/. for .*/,"",f)} /Used [0-9]+ registers/ {print f" :: "$0}' $O/results/ptxas-$SHA.txt \
  | grep -E "k_p1|k_p2|k_p3|k_fused|k_key" | sed -E 's/ptxas info *: //' | cut -c1-220
grep -E "spill" $O/results/ptxas-$SHA.txt | sort | uniq -c | sort -rn | head -5
