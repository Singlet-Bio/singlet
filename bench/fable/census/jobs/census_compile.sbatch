#!/bin/bash
#SBATCH --job-name=fable-census-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E2 census: build bench/gpu/m5 of branch gpu/fable/census (fetched from GitHub into the bare mirror) with the CUDA 13.3 image:
#   m5  = the default build (no census code), m5c = -DM5_CENSUS (per-read / per-window records, --census PREFIX)
# -> $W/jobs/fable/census/bin/{m5,m5c,commit}
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif
OUT=$W/jobs/fable/census; mkdir -p $OUT/bin $OUT/results
SCR=/tmp/$USER-census-cc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/census:refs/fable/census
SHA=$(git -C $BARE rev-parse refs/fable/census); echo "commit $SHA"
git -C $BARE archive refs/fable/census bench/gpu/m5 bench/fable/census | tar -x -C $SCR
cc() { apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v "$@" 2>&1 \
       | grep -E "error|warning: v|k_p3|registers|stack frame" | grep -B1 -A1 "k_p3\|error" | tail -30; }
cd $SCR/bench/gpu/m5
cc -o $OUT/bin/m5.new m5.cu > $SCR/cc0.txt &
cc -DM5_CENSUS -o $OUT/bin/m5c.new m5.cu > $SCR/cc1.txt &
wait
echo "--- default build"; cat $SCR/cc0.txt; echo "--- census build"; cat $SCR/cc1.txt
for b in m5 m5c; do mv $OUT/bin/$b.new $OUT/bin/$b; done && echo "$SHA" > $OUT/bin/commit && echo "[compile] ok"
cp -r $SCR/bench/fable/census/*.py $OUT/bin/ 2>/dev/null || true
# python with numpy for the aggregation (census_agg.py runs in the GPU job)
for py in "$W/venvs/grantfig/bin/python3" "apptainer exec -B $W $W/images/python-3.12-bookworm.sif $W/venvs/v2tools/bin/python" \
          "apptainer exec -B $W $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python"; do
  echo "py: $py -> $($py -c 'import numpy, sys; print(numpy.__version__, sys.version.split()[0])' 2>&1 | tail -1)"
done
