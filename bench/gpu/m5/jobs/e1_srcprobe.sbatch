#!/bin/bash
#SBATCH --job-name=e1-src
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1: probe how ncu 2026.2 prints per-source-line metrics from an existing report (CPU only). Usage: sbatch e1_srcprobe.sbatch REP
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; REP=$1; NCU=/usr/local/cuda/bin/ncu
X() { apptainer exec -B $W $IMG "$@"; }
X $NCU --help | grep -n -i -A3 "print-source\|page \|source-folders\|print-details\|--print-units\|print-metric-instances\|print-metric-attribution" | head -80
for ps in sass "cuda,sass" cuda; do
  echo "=== --print-source $ps"
  X $NCU --import $REP --kernel-name-base function --kernel-name regex:k_p3 --launch-count 1 --page source --csv --print-source $ps 2>&1 | head -4 | cut -c1-600
done
echo "=== details instance"
X $NCU --import $REP --kernel-name-base function --kernel-name regex:k_p3 --launch-count 1 --page details --print-metric-instances values --csv 2>&1 | grep -i "pcsamp\|source" | head -5 | cut -c1-400
X ls /usr/local/cuda/nsight-compute*/extras/python /opt/nvidia/nsight-compute/*/extras/python 2>&1 | head; X which python3; X python3 --version
