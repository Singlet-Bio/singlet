#!/bin/bash
#SBATCH --job-name=donorfix_eval
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Single-donor false-split rate (8 gate libraries) and pooled ARI (Kang b1 lane A) for the donor demultiplexer of two branches,
# both run on the stored private allele counts with the standalone harness, auto-K, default settings.
#   sbatch --export=ALL,TAG=run1,BRANCHES="release/v1.4.0 fix/donor-single-split" run_donorfix_eval.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
export PATH=$W/envs/pbench/bin:$PATH; P=$W/envs/pbench
TAG=${TAG:-run}; BRANCHES=${BRANCHES:-"release/v1.4.0 fix/donor-single-split"}
O=$W/jobs/donorfix/$TAG; mkdir -p $O
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
PY="apptainer exec -B /mnt $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python -I"
D=$W/bench/calls-sota/demux_single_donor; BP=$W/bench-public; BPY=$W/envs/pbench/bin/python3
LIBS="GSM5130155 GSM5130163 GSM5171610 GSM7064234 GSM7884577 GSM8183378 GSM8164511 GSM9038586"
for b in $BRANCHES; do
  n=$(echo $b | tr '/' '_'); mkdir -p $O/$n
  git clone -q --depth 1 --branch $b https://github.com/Singlet-Bio/singlet.git $S/$n || exit 1
  echo "== $b $(cd $S/$n && git rev-parse --short HEAD)"
  x86_64-conda-linux-gnu-g++ -std=gnu++17 -O3 -DNDEBUG -march=x86-64-v3 -pthread -I$S/$n/include -I$P/include $S/$n/bench/modalities/demux_fix/demux_harness.cpp -o $S/$n/harness -L$P/lib -Wl,-rpath,$P/lib -lzstd -lz || exit 1
  args=()
  for g in $LIBS kang_b1A; do
    if [ $g = kang_b1A ]; then d=$W/bench/rescore-v135/kang_b1A/out; else d=$W/pilot/$g/s1cpu_0ea48783/out; fi
    mkdir -p $O/$n/$g
    $S/$n/harness --1pz $d $O/$n/$g 16 auto -1 > $O/$n/$g/log.txt 2>&1
    echo "$g: $(grep -E 'selected K|harness\] auto' $O/$n/$g/log.txt | tr '\n' ' ')"
    [ $g != kang_b1A ] && args+=("$O/$n/$g/auto/donor_assignments.tsv:$g")
  done
  $PY $D/score_single_split.py $O/$n/single_split.json "${args[@]}" | tail -12
  $BPY -I $D/score_kang_ari.py $BP/code_gsota/public/genotype/lib $BP/data/kang_truth $O/$n/kang_ari.json $O/$n/kang_b1A/auto/donor_assignments.tsv:$n
done
