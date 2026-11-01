#!/bin/bash
#SBATCH --job-name=bp-collect
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch collect_and_compute.sbatch CAPABILITY        CAPABILITY = non-host | alignment-performance | download
# Builds the artifact tables of one capability from the raw run directories ($BP/...), writes them to $ART/<capability>/ and
# computes the capability document from those tables alone (the same compute() the notebook calls). The document is written to
# $BP/results/<capability>.json; bring it back with `clipper.py get`. The harness code is the checkout at $SRC (sync_src.sbatch).
set -eu
source ${BP_SRC:-/mnt/projects/debruinz_project/debruinz/singlet/bench-public/src/singlet}/bench/public/harness/perf_common.sh
CAP=$1; DIR=${CAP//-/_}
OUT=$ART/$CAP; rm -rf $OUT; mkdir -p $OUT $BP/results
H=$SRC/bench/public/harness
$PY -I $H/$DIR/collect.py $BP $OUT
[ -f $H/$DIR/reported.tsv ] && cp $H/$DIR/reported.tsv $OUT/
ls -la $OUT
$PY -I $SRC/bench/public/capabilities/$DIR.py --artifacts $OUT --out $BP/results/$CAP.json
ls -la $BP/results/$CAP.json
