#!/bin/bash
#SBATCH --job-name=cycle74_scanpy_parity
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle74_parity_%j.log
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -uo pipefail
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu

rm -rf build_cycle74_verify
cmake -S . -B build_cycle74_verify -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -10
cmake --build build_cycle74_verify -j8 --target de_wilcoxon_correctness 2>&1 | tail -15
echo "BUILD_EXIT=$?"
cd build_cycle74_verify

# Priority: TinyPlanted correctness (the target of this cycle)
ctest -R "Wilcoxon.*TinyPlanted" --output-on-failure 2>&1 | tee tinyplanted_output.txt
echo "TINY_EXIT=$?"

# Regression: Test83 must still PASS (no NaN, no crash)
ctest -R "Test83_RealDataSized_PostNormalize_NoCrash" --output-on-failure 2>&1 | tee test83_output.txt
echo "TEST83_EXIT=$?"

echo "=== KEY CORRECTNESS DELTAS ==="
grep -E "Jaccard|Spearman|rho|PASSED|FAILED" tinyplanted_output.txt | head -20
