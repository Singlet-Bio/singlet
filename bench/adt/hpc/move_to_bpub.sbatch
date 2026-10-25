#!/bin/bash
#SBATCH --job-name=adt-mv
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One-off: move the ADT/HTO public-benchmark artifacts from $WORK/bench-public (the perf harness's folder) to $WORK/bpub.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
mkdir -p $WORK/bpub/artifacts $WORK/state/adt/fastcheck
for c in adt hto; do
  if [ -d $WORK/bench-public/artifacts/$c ]; then
    mkdir -p $WORK/bpub/artifacts/$c
    cp -a $WORK/bench-public/artifacts/$c/. $WORK/bpub/artifacts/$c/ && rm -rf $WORK/bench-public/artifacts/$c
  fi
done
[ -d $WORK/bench-public/artifacts/adt_fastcheck ] && cp -a $WORK/bench-public/artifacts/adt_fastcheck/. $WORK/state/adt/fastcheck/ && rm -rf $WORK/bench-public/artifacts/adt_fastcheck
for f in adt hto; do [ -f $WORK/bench-public/results/$f.json ] && mv $WORK/bench-public/results/$f.json $WORK/state/adt/$f.prev.json; done
ls -la $WORK/bpub/artifacts/adt $WORK/bpub/artifacts/hto | head -60
