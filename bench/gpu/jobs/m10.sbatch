#!/bin/bash
#SBATCH --job-name=w11-m10
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# M10: read stage timers from existing CPU run logs (read-only) -> one small JSON.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/python-3.12-bookworm.sif; PY=$W/venvs/singlet/bin/python
OUT=$W/jobs/w11/results; mkdir -p $OUT
apptainer exec -B $W $IMG $PY -I $W/jobs/w11/m10_hostwork.py $OUT/m10_runs.json \
  "$W/bam-stream/runs/*/*" "$W/pipeline-benchmark/perf/*/*" "$W/pipeline-benchmark/runs/*/*" "$W/deep-perf/runs/*/*"
# which slurm logs (logs/) carry the export timer, and does any log mention called_ed / a timer for non-host or demux?
echo "== logs/*.out with '[export] Total'"; grep -l '\[export\] Total' $W/logs/*.out 2>/dev/null | wc -l
echo "== any 'called_ed' in recent run logs"; grep -l 'called_ed' $W/bam-stream/runs/*/*/run.log $W/pipeline-benchmark/perf/*/*/run.log 2>/dev/null | head -3
