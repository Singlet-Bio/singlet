#!/bin/bash
#SBATCH --job-name=bl-gpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --time=03:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Barcode layer harness on one H100: build bl_harness from BRANCH, run every protocol class and the feature kernels, compare with the
# reference (and STAR's BAM where it exists). CLASSES (default all) and MODE (gpu: run the harness; cpu: --reuse, compare only) are env vars.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
D=$W/data/barcode-layer; BR=${BRANCH:-claude/gpu-barcode-layer-484c70}
CLASSES=${CLASSES:-"pbmc1k_v3_F pbmc1k_v3_real dropseq bd splitseq indrop"}
RES=$W/jobs/barcode-layer/results/$SLURM_JOB_ID; mkdir -p $RES $D/work
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/w11/barcode-layer || exit 1
COMMIT=$(git -C $BARE rev-parse refs/w11/barcode-layer); echo "COMMIT=$COMMIT node=$(hostname) $(date -Is)" | tee $RES/summary.txt
git -C $BARE archive refs/w11/barcode-layer src/engine_gpu/barcode_layer.cuh src/engine_gpu/barcode_layer_ref.py bench/barcode_layer | tar -x -C $SCR/src
cat > $SCR/run.sh <<EOS
set -uo pipefail
export PATH=$W/envs/pbench/bin:/usr/local/cuda/bin:\$PATH NUMBA_CACHE_DIR=$SCR/numba
cd $SCR/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -lineinfo -Isrc/engine_gpu bench/barcode_layer/bl_harness.cu -o $SCR/bl_harness 2>&1 | head -60; echo "nvcc rc=\${PIPESTATUS[0]}"
g++ -O2 -std=c++17 -Ibench/barcode_layer bench/barcode_layer/cb_check_cpu.cpp -o $SCR/cb_check_cpu 2>&1 | head; echo "g++ rc=\${PIPESTATUS[0]}"
[ -x $SCR/bl_harness ] || exit 3
for c in $CLASSES; do
  echo "=================== \$c \$(date -Is)"
  BAM=$D/star/\$c/star.bam; STARARG=""; [ -e \$BAM ] && STARARG="--star-bam \$BAM"
  REUSE=""; [ "${MODE:-gpu}" = cpu ] && REUSE="--reuse"
  python -I bench/barcode_layer/run_class.py --cls \$c --exe $SCR/bl_harness --cpu-exe $SCR/cb_check_cpu --work $D/work/\$c --out $RES \$STARARG \$REUSE 2>&1 | tail -150
done
echo "=================== features \$(date -Is)"
[ "${MODE:-gpu}" = cpu ] || python -I bench/barcode_layer/run_features.py --exe $SCR/bl_harness --work $D/work/features --out $RES 2>&1 | tail -120
EOS
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/run.sh 2>&1 | tee -a $RES/log.txt
echo "done $(date -Is)"
