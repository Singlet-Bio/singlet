#!/bin/bash
#SBATCH --job-name=sg_cycle74_streaming_tol
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle74_streaming_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle74_streaming_%j.log

set +e

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

# Force rebuild of streaming correctness test
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
rm -rf tests/CMakeFiles/streaming_pipeline_correctness.dir/ 2>/dev/null
make -j8 streaming_pipeline_correctness 2>&1 | tail -10
echo "Build: $?"

# Run all streaming tests
ctest -R "StreamingPipeline" -V --timeout 600 --output-on-failure 2>&1 || true
