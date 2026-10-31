#!/bin/bash
#SBATCH --job-name=samprof-resolve
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Resolve samprof runs. Env: RUNS = space-separated dir:total_cpu_seconds under jobs/cpu-perf/samprof, CM = build commit dir, BINV = binary name (default singlet-v3-g).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; CM=${CM:?}; BINV=${BINV:-singlet-v3-g}
for rc in $RUNS; do r=${rc%%:*}; c=${rc##*:}; $ENV/bin/python $W/jobs/cpu-perf/samprof_resolve.py $W/jobs/cpu-perf/samprof/$r $W/jobs/cpu-perf/$CM/bin/$BINV $ENV/bin/x86_64-conda-linux-gnu-addr2line --top 40 --cpu $c > /dev/null; echo "resolved $r"; done
