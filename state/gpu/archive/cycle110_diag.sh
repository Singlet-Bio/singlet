#!/bin/bash
#SBATCH --job-name=sg_c110_diag
#SBATCH --partition=gpu
#SBATCH --time=00:05:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle110_diag_%j.log
#SBATCH --error=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle110_diag_%j.log

set -uo pipefail
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || true
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== Cycle 110 diag — _core import + submodule list ==="
python -W default <<'PY'
# Force the warning to print:
import warnings; warnings.simplefilter("default")
import sys
try:
    import singlet_gpu._core as _core
    print(f"_core OK: {dir(_core)[:10]}...")
    print(f"  load_pz callable: {callable(getattr(_core, 'load_pz', None))}")
except Exception as e:
    import traceback
    print(f"_core FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)

# What submodules exist in the package?
import os
import singlet_gpu
pkg_dir = os.path.dirname(singlet_gpu.__file__)
print(f"\n=== singlet_gpu package dir: {pkg_dir} ===")
for entry in sorted(os.listdir(pkg_dir)):
    if entry.startswith('_') or entry.startswith('.'):
        continue
    full = os.path.join(pkg_dir, entry)
    if os.path.isdir(full):
        has_init = os.path.exists(os.path.join(full, '__init__.py'))
        print(f"  dir/  {entry:25s}  __init__.py={has_init}")
    elif entry.endswith('.py'):
        print(f"  file  {entry}")

# Try explicit submodule imports — what works, what fails?
print("\n=== submodule imports ===")
for mod in ('preprocess', 'reduce', 'reduce.svd', 'reduce.nmf',
            'qc', 'pp', 'tools', 'io', 'streaming'):
    try:
        m = __import__(f'singlet_gpu.{mod}', fromlist=['*'])
        public = [n for n in dir(m) if not n.startswith('_')]
        print(f"  singlet_gpu.{mod}: OK ({len(public)} public names)")
    except Exception as e:
        print(f"  singlet_gpu.{mod}: FAIL — {type(e).__name__}: {e}")
PY
echo ""
echo "EXIT=$?"
date
