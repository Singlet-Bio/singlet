#!/bin/bash
#SBATCH --job-name=fable-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile the merged gpu/fable-engine m5 harness in its build flavours (default, -DM5_TAIL, -DM5_CENSUS, -DM5_BASELINE,
# -DM5_P1NEST): the merge gate. BRANCH selects the branch, OUT the output directory (default $W/jobs/fable/engine).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable-engine}
SCR=/tmp/$USER-fablecc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/cc && git -C $BARE archive refs/fable/cc bench/gpu/m5 bench/fable | tar -x -C $SCR
OUT=${OUT:-$W/jobs/fable/engine}   # binaries in $OUT/bin (fable_gate.sbatch reads the same OUT)
echo "commit $(git -C $BARE rev-parse refs/fable/cc)"; mkdir -p $OUT/bin; git -C $BARE rev-parse --short refs/fable/cc > $OUT/bin/commit
cc() { apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v "$@" 2>&1 | grep -E "error|registers|spill" | sort | uniq -c | head -20; }
cd $SCR/bench/gpu/m5
echo "== default"; cc -o $OUT/bin/m5 m5.cu && echo "[default ok]"
echo "== tail";    cc -DM5_TAIL -o $OUT/bin/m5t m5.cu && echo "[tail ok]"
echo "== census";  cc -DM5_CENSUS -o $OUT/bin/m5c m5.cu && echo "[census ok]"
echo "== baseline"; cc -DM5_BASELINE -o $OUT/bin/m5b m5.cu && echo "[baseline ok]"
echo "== nest";    cc -DM5_P1NEST -o $OUT/bin/m5n m5.cu && echo "[nest ok]"   # E7: STAR's phase-1 nest with every other default
ls -la $OUT/bin/; echo "== done $(date)"
