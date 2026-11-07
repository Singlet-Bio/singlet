#!/bin/bash
#SBATCH --job-name=cs-resolve
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Resolve samprof samples into per-process function tables (samprof/resolve.py). Env: RUNS (space-separated run dirs that hold samprof/),
# BD (binary dir under $W/jobs/cpu-speed/bin). Writes <run>/samprof/profile.md.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; BIN=$W/jobs/cpu-speed/bin/${BD:?}/singlet-cpu
for r in ${RUNS:?}; do
  $ENV/bin/python $W/jobs/cpu-speed/resolve.py $r/samprof $BIN $ENV/bin/x86_64-conda-linux-gnu-addr2line --ms 5 --top 60 > /dev/null
  echo "resolved $r"; head -5 $r/samprof/profile.md
done
