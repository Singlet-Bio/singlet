#!/bin/bash
#SBATCH --job-name=sg_hvg_fix2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle75b_hvg_%j.log
set +e
export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build
# Touch BOTH source and header to guarantee recompile
touch ../tests/streaming_pipeline_correctness.cpp
touch ../include/singlet-gpu/streaming/streamed_pipeline.h
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
make -j8 streaming_pipeline_correctness 2>&1 | grep -E "Compil|Link|Built|error" | tail -10
echo "Build: $?"
echo "=== Streaming HVG fix v2 ==="
ctest -R "Pipeline_HvgOnly" -V --timeout 600 --output-on-failure 2>&1 || true
echo "=== DONE ==="
