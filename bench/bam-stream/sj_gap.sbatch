#!/bin/bash
#SBATCH --job-name=bs-sjgap
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=GSM7901264 sj_gap.sbatch      (~0.05 node-h)
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; J=$W/jobs/bam-stream
SCR=/tmp/$USER-bssjgap-$SLURM_JOB_ID; mkdir -p $SCR/inc; trap 'rm -rf $SCR' EXIT
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
cp -r $W/src/singlet-py/perf__bam-stream/include/singlet $SCR/inc/
sed -i 's/throw PZReadError("rowname count does not match the number of rows");/rownames.resize(hdr.m);/' $SCR/inc/singlet/pileup/pz_reader.h
$ENV/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$SCR/inc -I$ENV/include $J/pz2mtx.cpp -L$ENV/lib -Wl,-rpath,$ENV/lib -lzstd -o $SCR/pz2mtx
A=$W/bam-stream/runs_r8/$SET/t4_r1/out; B=$W/bam-stream/runs_prod/$SET/t4_r1/out
$SCR/pz2mtx $A/sj_counts.1pz $SCR/a; $SCR/pz2mtx $B/sj_counts.1pz $SCR/b
in_runtime python -I $J/sj_gap.py $SCR/a $SCR/b $A $B
