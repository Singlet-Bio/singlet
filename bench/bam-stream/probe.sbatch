#!/bin/bash
#SBATCH --job-name=bs-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only probe (~0.1 node-h): pinned binary identity and the atlas run tables' schema. No data is written.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
sha256sum $PB/bin/singlet-campaign
strings $PB/bin/singlet-campaign | grep -E -m5 '^[0-9a-f]{40}$|02aa9b93'
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python - <<'PY'
import pyarrow.parquet as pq, glob
for f in sorted(glob.glob('/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/*.parquet')):
    t = pq.ParquetFile(f); print(f, t.metadata.num_rows); print([ (n.name, str(n.type)) for n in t.schema_arrow])
PY
