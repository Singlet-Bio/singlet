#!/bin/bash
#SBATCH --job-name=donorfix_diag
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Build the donor harness from a branch of Singlet-Bio/singlet and run it on stored private allele counts.
#   sbatch --export=ALL,BRANCH=release/v1.4.0,TAG=base,LIBS="GSM5130155",RUNS="auto -1 -- screen -1" diag_k.sbatch
# RUNS = "TAG K [key=value ...] -- TAG K ..." ; a run named screen* sets SINGLET_DEMUX_SCREEN=1.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH; P=$W/envs/pbench
BRANCH=${BRANCH:-release/v1.4.0}; TAG=${TAG:-base}
O=$W/jobs/donorfix/$TAG; mkdir -p $O
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch $BRANCH https://github.com/Singlet-Bio/singlet.git $S/eng || exit 1
(cd $S/eng && git rev-parse --short HEAD)
x86_64-conda-linux-gnu-g++ -std=gnu++17 -O3 -DNDEBUG -march=x86-64-v3 -pthread -I$S/eng/include -I$P/include $S/eng/bench/modalities/demux_fix/demux_harness.cpp -o $S/harness -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz || exit 1
for g in $LIBS; do
  d=$W/pilot/$g/s1cpu_0ea48783/out
  [ -d $d ] || d=$W/pilot/$g/s1cpu_r/out
  mkdir -p $O/$g
  IFS=' ' read -r -a RA <<< "$RUNS"
  if [ -n "${SCREEN:-}" ]; then export SINGLET_DEMUX_SCREEN=1; fi
  $S/harness --1pz $d $O/$g 16 "${RA[@]}" > $O/$g/log.txt 2>&1
  grep -E "auto-K|selected|harness\]" $O/$g/log.txt | tail -40
done
