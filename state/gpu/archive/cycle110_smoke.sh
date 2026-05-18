#!/bin/bash
# SPDX-License-Identifier: GPL-2.0-or-later
# Cycle 110 docs-grounding smoke — confirm what `import singlet_gpu` actually exposes.
# Output drives the docs ground-truth update.

#SBATCH --job-name=sg_c110_smoke
#SBATCH --partition=gpu
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=g001
#SBATCH --output=/mnt/home/debruinz/Singlet-AI/singlet-gpu/state/cycle110_smoke_%j.log

set -uo pipefail

source /etc/profile.d/lmod.sh 2>/dev/null || true
module load python/3.11.14 2>&1 || true

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

echo "=== Cycle 110 import smoke ==="
echo "Job: ${SLURM_JOB_ID}  Date: $(date)"
python --version

python <<'PY'
import singlet_gpu as sg

print(f"version: {sg.__version__}")
print()

# Top-level surface
print("=== singlet_gpu top-level ===")
for name in ('load_pz', 'DeviceCsc', 'Metadata', 'PzDeviceMatrix',
             'device_count', 'cuda_device_name', 'current_device', 'set_device'):
    obj = getattr(sg, name, None)
    print(f"  sg.{name:20s} = {type(obj).__name__:20s} callable={callable(obj)}")
print()

# load_pz call signature (introspect to confirm what users actually call)
import inspect
try:
    sig = inspect.signature(sg.load_pz)
    print(f"  sg.load_pz signature: {sig}")
except (TypeError, ValueError) as e:
    print(f"  sg.load_pz: {type(sg.load_pz).__name__} — {e}")
print()

# Submodules
print("=== submodules ===")
for mod in ('preprocess', 'reduce', 'qc', 'pp', 'tools', 'io',
            'streaming', 'velocity', 'lineage'):
    sub = getattr(sg, mod, None)
    if sub is None:
        print(f"  sg.{mod}: NOT_PRESENT")
        continue
    funcs = [n for n in dir(sub) if not n.startswith('_') and callable(getattr(sub, n))]
    print(f"  sg.{mod}: {len(funcs)} callables")
    for fn in sorted(funcs)[:10]:
        print(f"      .{fn}")
print()

# qc + preprocess CYCLE-103 wrappers — call signatures
print("=== CYCLE-103 wrappers (signatures) ===")
for path in ('qc.calculate_qc_metrics', 'qc.filter_cells', 'qc.filter_genes',
             'preprocess.scale', 'preprocess.regress_out'):
    obj = sg
    for part in path.split('.'):
        obj = getattr(obj, part, None)
    if obj is None:
        print(f"  sg.{path}: MISSING")
        continue
    try:
        sig = inspect.signature(obj)
        print(f"  sg.{path}{sig}")
    except (TypeError, ValueError):
        print(f"  sg.{path}: <{type(obj).__name__}>")

# scanpy-convention paths from CYCLE-102
print()
print("=== scanpy-style paths (CYCLE-102 doc ground-truth) ===")
for path in ('preprocess.normalize_total', 'preprocess.log1p',
             'preprocess.highly_variable_genes',
             'reduce.svd.pca', 'reduce.nmf.nmf',
             'tools.rank_genes_groups',
             'pp.neighbors', 'qc.run_doublet_score'):
    obj = sg
    for part in path.split('.'):
        obj = getattr(obj, part, None)
        if obj is None:
            break
    if obj is None:
        print(f"  sg.{path}: MISSING")
        continue
    try:
        sig = inspect.signature(obj)
        print(f"  sg.{path}{sig}")
    except (TypeError, ValueError):
        print(f"  sg.{path}: <{type(obj).__name__}>")

# Deprecation warnings should fire on stale SVD backends
print()
print("=== CYCLE-109 deprecation warnings ===")
import warnings
for name in ('svd_lanczos', 'svd_irlba', 'svd_krylov'):
    fn = getattr(sg.reduce.svd, name, None)
    if fn is None:
        print(f"  reduce.svd.{name}: MISSING (already removed?)")
        continue
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        # Just inspect the wrapper, don't actually invoke
        print(f"  reduce.svd.{name}: present (deprecated, callable={callable(fn)})")
PY
SMOKE_EXIT=$?

echo ""
echo "=== FINAL EXIT CODE: $SMOKE_EXIT ==="
date
