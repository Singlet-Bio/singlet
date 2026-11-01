#!/bin/bash
#SBATCH --job-name=amb-syntax
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# -fsyntax-only of include/singlet/pileup/export.h (needs htslib from the pbench env) against the pushed include tree.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; S=$W/jobs/modfix-ambient
P=$W/envs/pbench
printf '#include "singlet/pileup/export.h"\nint main(){return 0;}\n' > $S/exp_check.cpp
$P/bin/x86_64-conda-linux-gnu-g++ -std=c++20 -fsyntax-only -I$S/fullinc/include -I$P/include $S/exp_check.cpp 2>&1 | head -60
echo "exit=${PIPESTATUS[0]}"
