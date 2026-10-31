#!/bin/bash
#SBATCH --job-name=samprof-lib
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build libsamprof.so next to the -g binary. Env: CM (commit dir). The source is read from the uploaded jobs/cpu-perf/samprof.c.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
apptainer exec -B /mnt/projects/debruinz_project $IMG $ENV/bin/x86_64-conda-linux-gnu-gcc -O2 -fPIC -shared -o $W/jobs/cpu-perf/${CM:?}/bin/libsamprof.so $W/jobs/cpu-perf/samprof.c -ldl -lrt
ls -la $W/jobs/cpu-perf/$CM/bin/libsamprof.so
