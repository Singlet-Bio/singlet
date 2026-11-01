#!/bin/bash
#SBATCH --job-name=s14-equiv
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14: full out_equiv listing of a multi_gate run (the gate log keeps only the last lines). Env: T (the gate dir, required), A_DIR (default keep/single), B_DIR (default keep/multi).
set -uo pipefail
T=${T:?}; A=${A_DIR:-$T/keep/single}; B=${B_DIR:-$T/keep/multi}
EQ=$(ls $T/repo/bench/star_accel/out_equiv.py)
for g in $(ls $B); do echo "-- $g"; python3 $EQ $A/$g/out $B/$g/out 2>&1 | grep -v "^    [ab]:" | cut -c1-300; done
echo "-- replicates (both concurrent, 12 threads)"
for g in PBMC1KB PBMC1KC; do python3 $EQ $B/PBMC1KA/out $B/$g/out 2>&1 | grep -v "^    [ab]:" | cut -c1-300; done
