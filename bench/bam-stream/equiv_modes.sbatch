#!/bin/bash
#SBATCH --job-name=bs-eq
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,A=<runs dir A>,B=<runs dir B>,SETS="a b",TR=t4_r1 equiv_modes.sbatch   (~0.1 node-h)
# Phase 5: out_equiv.py of two runs' output directories, per set: every file by sha256, then JSON key by key, timings ignored.
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
for s in $SETS; do echo "######## $s $A vs $B"; in_runtime python $W/jobs/deep-perf/out_equiv.py $W/bam-stream/$A/$s/${TR:-t4_r1}/out $W/bam-stream/$B/$s/${TRB:-${TR:-t4_r1}}/out 2>&1 | cut -c1-260; done
