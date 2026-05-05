#!/bin/bash
#SBATCH --job-name=rebuild-v9
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/cellarium/pipeline/logs/rebuild_v9_%j.out
#SBATCH --error=/mnt/projects/debruinz_project/cellarium/pipeline/logs/rebuild_v9_%j.err

# Step 1: Rebuild batches (filter + sort by protocol confidence)
# Step 2: Pre-screen unknown_sc/snRNA_unknown via FASTQ peek
# Step 3: Launch auto-grab workers
#
# Run: sbatch run_rebuild_and_launch.sh

set -euo pipefail

module load miniconda3/25.5.1
source /opt/gvsu/clipper/2025.12/spack/apps/linux-cascadelake/miniconda3-25.5.1-xe7kyofwhfxilia75rj5t63zf6wpzzcr/etc/profile.d/conda.sh
conda activate cellarium

WORKSPACE="/mnt/home/debruinz/Singlet-AI"
SCRIPTS="$WORKSPACE/scripts"

echo "════════════════════════════════════════════════════"
echo "Step 1: Rebuilding batches (v9 autograb)"
echo "════════════════════════════════════════════════════"
cd "$WORKSPACE/geo-reprocess"
python3 -u "$SCRIPTS/rebuild_batches_v9.py"

echo ""
echo "════════════════════════════════════════════════════"
echo "Step 2: Pre-screening unknown protocols via FASTQ peek"
echo "════════════════════════════════════════════════════"
python3 -u "$SCRIPTS/smartseq_prescreener.py"

echo ""
echo "════════════════════════════════════════════════════"
echo "Step 3: Launching auto-grab workers"
echo "════════════════════════════════════════════════════"
python3 -u << 'PYEOF'
import sys
sys.path.insert(0, "/mnt/home/debruinz/Singlet-AI/geo-reprocess")

from pathlib import Path
from scgeo.slurm.submit import generate_autograb_script, submit_job

batch_dir = Path("/mnt/projects/debruinz_project/cellarium/pipeline/batches_v9_autograb")

script = generate_autograb_script(
    batch_dir=batch_dir,
    job_name="v9-autograb",
    partition="cpu",
    cpus=38,
    memory="128G",
    time="24:00:00",
    n_workers=75,
    max_concurrent=20,
)

print(f"Generated script: {script}")

result = submit_job(script)
if result:
    print(f"Submitted job {result.job_id}")
else:
    print("Job submission failed!")
PYEOF

echo ""
echo "All steps complete."
