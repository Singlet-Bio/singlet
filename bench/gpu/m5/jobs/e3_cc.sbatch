#!/bin/bash
#SBATCH --job-name=e3-cc2
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E3 copy of bench/fable/engine/fable_cc.sbatch: the same builds of branch gpu/fable/warp-window into OUT=$W/jobs/fable/warp-window/engine
# (the integration's bin directory is left alone), plus the E3 flavours (-DM5_E3STATS, -DM5_KEEPIX, -DM5_NOCLK: m5s, m5k, m5n).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/warp-window}; OUT=${OUT:-$W/jobs/fable/warp-window/engine}
SCR=/tmp/$USER-fablecc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/e3cc && git -C $BARE archive refs/fable/e3cc bench/gpu/m5 bench/fable | tar -x -C $SCR
SHA=$(git -C $BARE rev-parse refs/fable/e3cc); echo "commit $SHA"; mkdir -p $OUT/bin; echo $SHA > $OUT/bin/commit
cc() { apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v "$@" 2>&1 | grep -E "error|registers|spill" | sort | uniq -c | head -20; }
cd $SCR/bench/gpu/m5
# the E3 flavours in the background (their output is printed after the four integration builds)
( cc -DM5_E3STATS -o $OUT/bin/m5s m5.cu > $SCR/e3stats.txt && echo "[e3stats ok]" >> $SCR/e3stats.txt ) &
( cc -DM5_KEEPIX -o $OUT/bin/m5k m5.cu > $SCR/keepix.txt && echo "[keepix ok]" >> $SCR/keepix.txt ) &
( cc -DM5_NOCLK -o $OUT/bin/m5n m5.cu > $SCR/noclk.txt && echo "[noclk ok]" >> $SCR/noclk.txt ) &
echo "== default"; cc -o $OUT/bin/m5 m5.cu && echo "[default ok]"
echo "== tail";    cc -DM5_TAIL -o $OUT/bin/m5t m5.cu && echo "[tail ok]"
echo "== census";  cc -DM5_CENSUS -o $OUT/bin/m5c m5.cu && echo "[census ok]"
echo "== baseline"; cc -DM5_BASELINE -o $OUT/bin/m5b m5.cu && echo "[baseline ok]"
wait; for f in e3stats keepix noclk; do echo "== $f"; cat $SCR/$f.txt; done
ls -la $OUT/bin/; echo "== done $(date)"
