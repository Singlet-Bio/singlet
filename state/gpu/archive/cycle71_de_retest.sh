#!/bin/bash
#SBATCH --job-name=cycle71_de_retest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle71_de_retest_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle71_de_retest_%j.log

set +e

export FACTORNET_ROOT=/mnt/home/debruinz/factornet

cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo "=== Build ==="
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3

rm -f tests/CMakeFiles/de_wilcoxon_correctness.dir/*.o \
      tests/CMakeFiles/de_ttest_correctness.dir/*.o 2>/dev/null

make -j8 de_wilcoxon_correctness de_ttest_correctness 2>&1 | tail -10
echo "Build exit: $?"

echo "=== Wilcoxon ==="
ctest -R "WilcoxonTest" -V --timeout 120 --output-on-failure 2>&1 || true

echo "=== t-test ==="
ctest -R "TtestDE" -V --timeout 120 --output-on-failure 2>&1 || true
