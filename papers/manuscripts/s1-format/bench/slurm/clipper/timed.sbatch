#!/bin/bash
# Timed stages for one dataset per array task, on a whole node to itself. The headline host is an ordinary GPU node
# (H100, spinning RAID /tmp), not one with fast local storage:
#   sbatch --array=0-N%1 --export=ALL ... timed.sbatch       with DS_LIST, S1_SHA (and STAGES) in the environment
# Files are read from three places: a copy on the node's own disk (data_dir; the `local` tier), the shared NFS project
# space in place (S1BENCH_NFS_DATA; the `nfs` tier), and `rclone serve s3` behind Toxiproxy (the network tiers).
# A cross-check on another node type sets S1_RESULTS to a results directory of its own.
#SBATCH --job-name=s1-timed
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%A_%a.out
set -euo pipefail
S1_ENV=${S1_ENV:-/mnt/projects/debruinz_project/debruinz/singlet/s1paper/env-gpu}
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout
s1_activate
s1_netem_tools
s1_cpp
IFS=, read -ra DS <<<"${DS_LIST:?set DS_LIST}"
D=${DS[${SLURM_ARRAY_TASK_ID:-0}]}
STAGES=${STAGES:-"decode sweep minibatch eco"}
# A rerun: S1_FORCE=1 redoes stages whose results exist, S1_SET adds one --set (e.g. 'minibatch.loaders=[s1_stream]',
# which replaces only those loaders' rows).
FORCE=${S1_FORCE:+--force}
SET=()
if [ -n "${S1_SET:-}" ]; then SET=(--set "$S1_SET"); fi
RES=${S1_RESULTS:-$S1/results}
mkdir -p "$RES"
if [ "$RES" != "$S1/results" ]; then  # a cross-check reads the sizes and descriptors of the main run
  for s in stats write; do ln -sfn "$S1/results/$s" "$RES/$s"; done
fi

# Stage this dataset's format files and canonical matrices onto the node's own disk.
LOCAL=$TMPDIR/data
mkdir -p "$LOCAL/formats" "$LOCAL/canon" "$LOCAL/tmp"
t0=$SECONDS
for f in "$S1/data/formats/$D".*; do cp -r --preserve=timestamps "$f" "$LOCAL/formats/"; done
cp --preserve=timestamps "$S1/data/canon/$D".*.npz "$LOCAL/canon/" 2>/dev/null || true
echo "staged $(du -sh "$LOCAL" | cut -f1) for $D in $((SECONDS - t0)) s"
export S1BENCH_DATA=$LOCAL S1BENCH_NFS_DATA=$S1/data
if [ -n "${SLURM_JOB_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]; then  # a GPU job must time the device copy: stop if torch cannot see the GPU
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
  python -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available()); sys.exit(not torch.cuda.is_available())"
fi
s1b env >/dev/null

for st in $STAGES; do
  t0=$SECONDS
  case $st in
    sweep) s1b --set 'storages=[{name: local}]' decode --datasets "$D" --sweep --formats s1,h5ad_gzip,csc_i32 ;;
    *) s1b "${SET[@]}" "$st" --datasets "$D" $FORCE ;;
  esac
  echo "stage $st $D done in $((SECONDS - t0)) s"
done
