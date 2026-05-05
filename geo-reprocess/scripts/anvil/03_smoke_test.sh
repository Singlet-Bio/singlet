#!/bin/bash
#SBATCH --job-name=smoke_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH -p shared
#
# Smoke test: process 2 small mouse 10xv3 samples to validate
# the full pipeline (download → detect → quantify → QC) on Anvil.
#
# Usage:
#   sbatch -A YOUR_ALLOCATION smoke_test.sh
#
# Memory: 64GB matches Clipper production config (cpu/short partitions).
# CPUs: 4 — matches Clipper; piscem peak RSS scales with thread count,
#   so 4 CPUs keeps peak under ~64GB. (8 CPUs risks 90GB+ on large samples.)
# SU cost: max(4/128, 64/257) × 128 ≈ 32 SU/hr (memory-dominated).
# Typically finishes in <30 min = ~16 SU.

set -euo pipefail

module load anaconda
conda activate "$PROJECT/envs/scgeo"

export SCGEO_BASE="$PROJECT/scgeo"
export SCGEO_WORKSPACE="$PROJECT/geo-reprocess"
export SCGEO_INDEX_DIR="$SCGEO_BASE/index"
export ALEVIN_FRY_HOME="$SCGEO_BASE/af_home"
export PYTHONUNBUFFERED=1

cd /tmp

echo "Memory: $(free -g | awk '/Mem:/{print $2}')GB total, $(free -g | awk '/Mem:/{print $7}')GB available"

echo "════════════════════════════════════════════════════"
echo "Anvil Smoke Test | $(hostname) | $(date)"
echo "SCGEO_BASE=$SCGEO_BASE"
echo "ALEVIN_FRY_HOME=$ALEVIN_FRY_HOME"
echo "════════════════════════════════════════════════════"

# Verify tools
echo "simpleaf: $(which simpleaf) — $(simpleaf --version 2>&1 | head -1)"
echo "fasterq-dump: $(which fasterq-dump)"
echo "Python: $(python3 --version)"
python3 -c "import scgeo; print(f'scgeo {scgeo.__version__}')"
python3 -c "import singlepress; print(f'singlepress OK')"

# Verify index exists
if [[ ! -d "$SCGEO_INDEX_DIR/mouse_splici" ]]; then
    echo "ERROR: Mouse index not found at $SCGEO_INDEX_DIR/mouse_splici"
    echo "Build index first (01_bootstrap.sh) or transfer from Clipper."
    exit 1
fi
echo "Mouse index: $(du -sh "$SCGEO_INDEX_DIR/mouse_splici" | cut -f1)"

echo ""
echo "▸ Running 2 small mouse 10xv3 samples..."
echo ""

python3 -B << 'PYEOF'
import logging, time, os
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
from scgeo.pipeline.api import process_samples

samples = [
    {
        'gsm_id': 'GSM3215649',
        'gse_id': 'GSE107527',
        'organism': 'Mus musculus',
        'protocol_hint': '10xv3',
        'catalog_confidence': 'high',
    },
    {
        'gsm_id': 'GSM9338988',
        'gse_id': 'GSE312733',
        'organism': 'Mus musculus',
        'protocol_hint': '10xv3',
        'catalog_confidence': 'high',
    },
]

t0 = time.time()
results = process_samples(samples=samples)
elapsed = time.time() - t0

print(f"\n{'═'*50}")
for r in results:
    cells = getattr(r, 'n_cells', 'N/A')
    print(f"  {r.gsm_id}: {r.status} — {cells} cells")
    if r.status == 'failed':
        print(f"    Error: {r.error}")
print(f"  Elapsed: {elapsed:.0f}s")
print(f"{'═'*50}")

# Summary
n_ok = sum(1 for r in results if r.status in ('success', 'qc_warn'))
if n_ok == len(results):
    print("\n✓ SMOKE TEST PASSED — pipeline works on Anvil!")
elif n_ok > 0:
    print(f"\n⚠ PARTIAL: {n_ok}/{len(results)} succeeded")
else:
    print("\n✗ SMOKE TEST FAILED — check logs above")

import resource
rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6
print(f"Peak RSS (children): {rss_gb:.1f} GB")
PYEOF

echo ""
echo "Peak memory: $(grep VmPeak /proc/self/status 2>/dev/null || echo 'N/A')"
echo "Smoke test complete | $(date)"
