#!/bin/bash
# Timed stages for one dataset per array task, on 8 cores of a shared bigmem node (no GPU, no whole node):
#   sbatch --array=0-N%1 -w <node> --mem=... --export=ALL ... timed.sbatch     with DS_LIST, S1_SHA (and STAGES) set
# One array per node with %1 keeps one timed job per node, so cold reads do not share a disk with another of ours.
# DS_LIST entries are dataset ids, or id@big for matrices above decode.big_nnz (config.yaml), for which only the files
# the timed stages read are copied to the node (the decode stage applies the reduced protocol itself).
# Files are read from three places: a copy on the node's own disk (data_dir; the `local` tier), the shared NFS project
# space in place (S1BENCH_NFS_DATA; the `nfs` tier), and `rclone serve s3` behind Toxiproxy (the network tiers).
# A cross-check on another node type sets S1_RESULTS to a results directory of its own.
#SBATCH --job-name=s1-timed
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%A_%a.out
set -euo pipefail
S1_ENV=${S1_ENV:-/mnt/projects/debruinz_project/debruinz/singlet/s1paper/env-gpu}
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
s1_activate
s1_netem_tools
s1_cpp
# S1_PKG_OVERLAY=1: the singlet package of this commit, built into this job's TMPDIR and put ahead of the env's copy, so a
# rerun can use newer package code without reinstalling into the env that other jobs are using.
if [ -n "${S1_PKG_OVERLAY:-}" ]; then
  pip install -q --no-deps --no-build-isolation --target "$TMPDIR/site" "$SRC"
  export PYTHONPATH="$TMPDIR/site:$PYTHONPATH"
  python -c "import singlet, singlet.torch; print('singlet', singlet.__version__, singlet.__file__)"
fi
IFS=, read -ra DS <<<"${DS_LIST:?set DS_LIST}"
E=${DS[${SLURM_ARRAY_TASK_ID:-0}]}
D=${E%@*}
BIG=$([ "$E" != "$D" ] && echo 1 || true)
STAGES=${STAGES:-"decode sweep minibatch eco"}
BIG_FORMATS=${BIG_FORMATS:-csc_i32,tenx_h5,h5ad_none,h5ad_gzip,zarr,soma,s1,s1_pack}
# A rerun: S1_FORCE=1 redoes stages whose results exist; S1_SET adds --set options, separated by '|'
# (e.g. 'minibatch.loaders=[s1_stream]', which replaces only those loaders' rows).
FORCE=${S1_FORCE:+--force}
SET=()
if [ -n "${S1_SET:-}" ]; then IFS='|' read -ra SETS <<<"$S1_SET"; for s in "${SETS[@]}"; do SET+=(--set "$s"); done; fi
# Ports of our own: several of these jobs (and other users' servers) can share a node.
P=$((20000 + (SLURM_JOB_ID % 4000) * 4))
SET+=(--set "netem.minio_port=$P" --set "netem.toxiproxy_api_port=$((P + 1))" --set "netem.proxy_port=$((P + 2))")
RES=${S1_RESULTS:-$S1/results}
mkdir -p "$RES"
if [ "$RES" != "$S1/results" ]; then  # a cross-check reads the sizes and descriptors of the main run
  for s in stats write; do ln -sfn "$S1/results/$s" "$RES/$s"; done
fi

# Stage this dataset's format files and canonical matrices onto the node's own disk (for @big, only the files the
# timed stages read: BIG_FORMATS and the minibatch loaders' formats).
LOCAL=$TMPDIR/data
mkdir -p "$LOCAL/formats" "$LOCAL/canon" "$LOCAL/tmp"
skip='^$'
[ -n "$BIG" ] && skip='^(mtx_gz|parquet_coo|rds_gz\.rds|rds_xz\.rds|loom\.loom|h5ad_lzf\.h5ad|csc_zstd19)$'
need=0
for u in "$S1/data/formats/$D".*; do
  for f in "$u"/*; do [[ $(basename "$f") =~ $skip ]] || need=$((need + $(du -sb "$f" | cut -f1))); done
done
have=$(df -B1 --output=avail "$LOCAL" | tail -1)
echo "local disk: $((have >> 30)) GiB free, $D needs $((need >> 30)) GiB"
[ "$have" -gt $((need / 5 * 6 + (1 << 30))) ] || { echo "not enough local disk for $D"; exit 1; }
t0=$SECONDS
for u in "$S1/data/formats/$D".*; do
  mkdir -p "$LOCAL/formats/$(basename "$u")"
  for f in "$u"/*; do
    [[ $(basename "$f") =~ $skip ]] || cp -r --preserve=timestamps "$f" "$LOCAL/formats/$(basename "$u")/"
  done
done
cp --preserve=timestamps "$S1/data/canon/$D".*.npz "$LOCAL/canon/" 2>/dev/null || true
echo "staged $(du -sh "$LOCAL" | cut -f1) for $D in $((SECONDS - t0)) s on $(hostname), $(nproc) cores"
export S1BENCH_DATA=$LOCAL S1BENCH_NFS_DATA=$S1/data
s1b env >/dev/null
mkdir -p "$RES/env" && cp "$RES/env.json" "$RES/env/$D.json"   # the host and versions of this dataset's run

for st in $STAGES; do
  t0=$SECONDS
  case $st in
    # matrices above config decode.big_nnz get the reduced protocol inside the decode stage itself (no sweep, no NFS tier,
    # decode.big_formats only); @big here only limits which files are copied to the node
    sweep) s1b "${SET[@]}" --set 'storages=[{name: local}]' decode --datasets "$D" --sweep --formats s1,h5ad_gzip,csc_i32 $FORCE ;;
    *) s1b "${SET[@]}" "$st" --datasets "$D" $FORCE ;;
  esac
  echo "stage $st $D done in $((SECONDS - t0)) s"
done
