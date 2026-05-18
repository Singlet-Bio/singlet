#!/bin/bash
#SBATCH --job-name=cycle73_streaming
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle73_streaming_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle73_streaming_%j.log

set +e

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo "=== CLEAN REBUILD: streaming_pipeline_correctness ==="
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

# Force complete rebuild of streaming test
rm -rf tests/CMakeFiles/streaming_pipeline_correctness.dir/ 2>/dev/null
echo "Cached objects cleared."

make -j8 streaming_pipeline_correctness 2>&1 | tail -20
BUILD_EXIT=$?
echo "Build exit: $BUILD_EXIT"

if [ $BUILD_EXIT -ne 0 ]; then
    echo "BUILD FAILED — check preprocessor"
    exit 1
fi

echo ""
echo "=== RUNNING StreamingPipeline TESTS ==="
ctest -R "StreamingPipeline" -V --timeout 600 --output-on-failure 2>&1 || true

echo "=== DONE ==="
