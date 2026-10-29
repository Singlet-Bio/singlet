#!/bin/bash
#SBATCH --job-name=bs-ambh
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,RUNS="runs_r14 runs_r14s runs_r14d",ENS="1 8 16 32" amb_harness.sbatch     (~0.5 node-h)
# Reruns only the ambient fit (amb_harness.cpp) on stored run outputs for ensemble sizes ENS, then amb_harness_cmp.py.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/pbench; J=$W/jobs/bam-stream
S=/tmp/$USER-ambh-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch ${REF:-perf/bam-stream} git@github.com:Singlet-Bio/singlet.git $S/src
git -C $S/src log --oneline -1
$E/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -fopenmp -I$S/src/include -I$E/include $J/amb_harness.cpp -L$E/lib -Wl,-rpath,$E/lib -lzstd -lpthread -o $S/h
O=$W/bam-stream/amb_harness; mkdir -p $O
for SET in ${SETS:-GSM4006843 GSM7901264}; do
  for R in ${RUNS:-runs_r14 runs_r14s runs_r14d}; do
    for M in ${ENS:-1 8 16 32}; do
      $S/h $W/bam-stream/$R/$SET/t4_r1/out $M ${RESTARTS:-10} $O/${SET}_${R}_e$M.tsv
    done
  done
done
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python -I $J/amb_harness_cmp.py $O "${SETS:-GSM4006843 GSM7901264}" "${ENS:-1 8 16 32}" > $O/compare.txt
cat $O/compare.txt
