#!/bin/bash
#SBATCH --job-name=bl-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu,short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Builds bench/barcode_layer/bl_harness.cu (nvcc 12.4, sm_90, -fmad=false, the image's gcc 11) and the CPU R11 comparator.
# BRANCH is fetched by URL into a private ref of $W/repos/singlet-py.git; the binaries go to $W/jobs/barcode-layer/bin.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BR=${BRANCH:-claude/gpu-barcode-layer-484c70}; OUT=$W/jobs/barcode-layer/bin; mkdir -p $OUT
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/w11/barcode-layer || exit 1
COMMIT=$(git -C $BARE rev-parse refs/w11/barcode-layer); echo "COMMIT=$COMMIT"
git -C $BARE archive refs/w11/barcode-layer src/engine_gpu/barcode_layer.cuh src/engine_gpu/barcode_layer_ref.py bench/barcode_layer | tar -x -C $SCR/src
cat > $SCR/b.sh <<EOS
set -uo pipefail
cd $SCR/src
nvcc --version | tail -2
nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -lineinfo -Isrc/engine_gpu bench/barcode_layer/bl_harness.cu -o $SCR/bl_harness 2>&1 | grep -v "^\$" | head -80; echo "nvcc rc=\${PIPESTATUS[0]}"
g++ -O2 -std=c++17 -Ibench/barcode_layer bench/barcode_layer/cb_check_cpu.cpp -o $SCR/cb_check_cpu 2>&1 | head -30; echo "g++ rc=\${PIPESTATUS[0]}"
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/b.sh
[ -x $SCR/bl_harness ] && cp $SCR/bl_harness $OUT/bl_harness.new && mv $OUT/bl_harness.new $OUT/bl_harness
[ -x $SCR/cb_check_cpu ] && cp $SCR/cb_check_cpu $OUT/cb_check_cpu
echo $COMMIT > $OUT/commit; ls -la $OUT
