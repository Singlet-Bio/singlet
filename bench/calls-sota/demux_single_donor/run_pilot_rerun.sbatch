#!/bin/bash
#SBATCH --job-name=pilot_demux_v137
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Re-run only the donor demultiplexer (v1.3.7 headers, auto-K) on the stored private allele counts of the pilot libraries.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
O=$W/bench/calls-sota/demux_single_donor/pilot_v137; mkdir -p $O
export PATH=$W/envs/pbench/bin:$PATH; P=$W/envs/pbench
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
git clone -q --depth 1 --branch v1.3.7 https://github.com/Singlet-Bio/singlet.git $S/eng
x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -pthread -I$S/eng/include -I$P/include $S/eng/bench/modalities/demux_fix/demux_harness.cpp -o $S/harness -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz
for g in GSM5130155 GSM5130163 GSM8183378 GSM8164511; do
  mkdir -p $O/$g
  $S/harness --1pz $W/pilot/$g/s1cpu_0ea48783/out $O/$g 8 v137 -1 2>&1 | grep -E "harness\] (called|v137|.*K=)|k_sel|error|fail" | tail -5 || true
done
args=(); for g in GSM5130155 GSM5130163 GSM8183378 GSM8164511; do p=$(ls $O/$g/*/donor_assignments.tsv 2>/dev/null | head -1); [ -n "$p" ] && args+=("$p:$g/v1.3.7"); done
apptainer exec -B /mnt $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python -I $W/bench/calls-sota/demux_single_donor/score_single_split.py $O/pilot_v137_split.json "${args[@]}"
