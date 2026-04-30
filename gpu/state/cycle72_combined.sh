#!/bin/bash
#SBATCH --job-name=cycle72_combined
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle72_combined_%j.log

set +e

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

# 1. Rebuild t-test with async fix
echo "=== Build t-test ==="
rm -f tests/CMakeFiles/de_ttest_correctness.dir/*.o 2>/dev/null
make -j8 de_ttest_correctness 2>&1 | tail -10
echo "t-test build: $?"

# 2. Run t-test
echo "=== t-test tests ==="
ctest -R "TtestDE" -V --timeout 120 --output-on-failure 2>&1 || true

# 3. Rebuild streaming with write_pz
echo "=== Build streaming ==="
rm -f tests/CMakeFiles/streaming_pipeline_correctness.dir/*.o 2>/dev/null
make -j8 streaming_pipeline_correctness 2>&1 | tail -10
echo "streaming build: $?"

# 4. Run ONLY the 3 failing streaming equivalence tests with output-on-failure
echo "=== Streaming LognormOnly ==="
ctest -R "Pipeline_LognormOnly" -V --timeout 300 --output-on-failure 2>&1 || true

echo "=== Streaming HvgOnly ==="
ctest -R "Pipeline_HvgOnly" -V --timeout 300 --output-on-failure 2>&1 || true

echo "=== Streaming LognormHvg ==="
ctest -R "Pipeline_LognormHvg_Equivalent" -V --timeout 300 --output-on-failure 2>&1 || true
