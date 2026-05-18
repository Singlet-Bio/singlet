#!/bin/bash
#SBATCH --job-name=sg_streaming_retest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69c_streaming_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle69c_streaming_%j.log

set +e

echo "=== cycle69c streaming retest ==="
echo "Job: $SLURM_JOB_ID  Node: $SLURMD_NODENAME  $(date -Iseconds)"

export FACTORNET_ROOT=/mnt/home/debruinz/factornet

cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo "--- cmake configure ---"
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

echo "--- force rebuild streaming_pipeline_correctness ---"
rm -f tests/CMakeFiles/streaming_pipeline_correctness.dir/*.o 2>/dev/null
make -j8 streaming_pipeline_correctness 2>&1 | tail -15
echo "Build exit: $?"

echo "--- run StreamingPipeline tests ---"
ctest -R "StreamingPipeline" -V --timeout 600 2>&1 || true
echo "Streaming ctest done"

echo "=== done $(date -Iseconds) ==="
