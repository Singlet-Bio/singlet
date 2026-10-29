#!/bin/bash
#SBATCH --job-name=bl-smoke
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:28:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile + smoke test on one H100 (debug-gpu, 30 min cap): 400k real pbmc_1k_v3 read pairs from the local public FASTQs, every harness mode that
# does not need ENA data (simple F/real, features). Finds CUDA bugs before the full run.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BR=${BRANCH:-claude/gpu-barcode-layer-484c70}; FQ=$W/pipeline-benchmark/data/pbmc1k_v3/pbmc_1k_v3_fastqs; N=${N:-400000}
RES=$W/jobs/barcode-layer/results/smoke-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/data/pbmc1k_v3; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/w11/barcode-layer || exit 1
COMMIT=$(git -C $BARE rev-parse refs/w11/barcode-layer); echo "COMMIT=$COMMIT node=$(hostname) $(date -Is)" | tee $RES/summary.txt
git -C $BARE archive refs/w11/barcode-layer src/engine_gpu/barcode_layer.cuh src/engine_gpu/barcode_layer_ref.py bench/barcode_layer | tar -x -C $SCR/src
export PATH=$W/envs/pbench/bin:$PATH
python -c "import numpy, numba; print('numpy', numpy.__version__, 'numba', numba.__version__)"; python --version
for m in R1 R2; do gzip -dc $FQ/pbmc_1k_v3_S1_L001_${m}_001.fastq.gz 2>/dev/null | head -n $((4*N)) | awk 'NR%4==1{print "@r" (NR-1)/4; next} {print}' > $SCR/data/pbmc1k_v3/$m.fq; done
wc -l $SCR/data/pbmc1k_v3/R1.fq
cat > $SCR/run.sh <<EOS
set -uo pipefail
export PATH=$W/envs/pbench/bin:/usr/local/cuda/bin:\$PATH NUMBA_CACHE_DIR=$SCR/numba
cd $SCR/src
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -lineinfo -Isrc/engine_gpu bench/barcode_layer/bl_harness.cu -o $SCR/bl_harness 2>&1 | head -80; echo "nvcc rc=\${PIPESTATUS[0]}"
g++ -O2 -std=c++17 -Ibench/barcode_layer bench/barcode_layer/cb_check_cpu.cpp -o $SCR/cb_check_cpu 2>&1 | head; echo "g++ rc=\${PIPESTATUS[0]}"
[ -x $SCR/bl_harness ] || exit 3
for c in pbmc1k_v3_F pbmc1k_v3_real; do
  echo "=================== \$c \$(date -Is)"
  python -I bench/barcode_layer/run_class.py --cls \$c --exe $SCR/bl_harness --cpu-exe $SCR/cb_check_cpu --work $SCR/work/\$c --out $RES --data $SCR/data --py-subset 50000 2>&1 | tail -80
done
echo "=================== features \$(date -Is)"
python -I bench/barcode_layer/run_features.py --exe $SCR/bl_harness --work $SCR/work/features --out $RES --n-adt 300000 --n-guide 300000 --r2 $SCR/data/pbmc1k_v3/R2.fq 2>&1 | tail -80
EOS
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/run.sh 2>&1 | tee $RES/log.txt
echo "done $(date -Is)"
