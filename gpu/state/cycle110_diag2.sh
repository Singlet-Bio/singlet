#!/bin/bash
#SBATCH --job-name=sg_c110_diag2
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle110_diag2_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle110_diag2_%j.log

set -uo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || true
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== Cycle 110 diag2 — verify __init__.py + manual import ==="

INSTALLED=$(python -c "import singlet_gpu, os; print(os.path.dirname(singlet_gpu.__file__))")
echo "installed at: $INSTALLED"
echo "--- installed __init__.py first 50 lines ---"
head -50 "$INSTALLED/__init__.py"
echo "--- diff vs source? ---"
diff -u "$INSTALLED/__init__.py" /mnt/home/debruinz/Singlet-AI/singlet-gpu/python/singlet_gpu/__init__.py | head -30
echo "--- direct __init__.py exec test ---"
python -W default <<'PY'
import warnings; warnings.simplefilter("default")
import singlet_gpu
print(f"version: {singlet_gpu.__version__}")
print(f"preprocess: {singlet_gpu.preprocess}")
print(f"reduce: {singlet_gpu.reduce}")
print(f"qc: {singlet_gpu.qc}")
print(f"_import_submodule still defined? {hasattr(singlet_gpu, '_import_submodule')}")
print(f"All attrs starting with letter: {[n for n in dir(singlet_gpu) if not n.startswith('_')]}")
PY
date
