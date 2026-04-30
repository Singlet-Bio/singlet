#!/bin/bash
#SBATCH --job-name=sg_stream_tol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle74b_streaming_%j.log
set +e
export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build
# Touch source to force rebuild without breaking cmake state
touch ../tests/streaming_pipeline_correctness.cpp
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
make -j8 streaming_pipeline_correctness 2>&1 | tail -15
echo "Build: $?"
echo "=== Streaming tests ==="
ctest -R "StreamingPipeline" -V --timeout 600 --output-on-failure 2>&1 || true
echo "=== DONE ==="
