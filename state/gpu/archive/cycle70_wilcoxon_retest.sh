#!/bin/bash
#SBATCH --job-name=cycle70_wilcoxon_retest
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle70_wilcoxon_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/bench/logs/cycle70_wilcoxon_%j.log

set +e

export FACTORNET_ROOT=/mnt/home/debruinz/factornet
cd /mnt/home/debruinz/Singlet-AI/singlet-gpu/build

echo "=== Force rebuild de_wilcoxon_correctness ==="
cmake .. -DFACTORNET_ROOT=$FACTORNET_ROOT 2>&1 | tail -3
rm -f tests/CMakeFiles/de_wilcoxon_correctness.dir/*.o 2>/dev/null
make -j8 de_wilcoxon_correctness 2>&1 | tail -10
echo "Build exit: $?"

echo ""
echo "=== Running Wilcoxon tests ==="
ctest -R "WilcoxonTest" -V --timeout 120 --output-on-failure 2>&1 || true

echo ""
echo "=== Done ==="
