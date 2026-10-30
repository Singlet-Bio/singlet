#!/bin/bash
# Research Alpha QC: atlas-wide sweep of manifests, orphans, withheld studies and the v2/ prefix (HTTPS reads only).
#   sbatch --export=ALL,QC_COMMIT=<sha> alpha_qc_sweep.sbatch
#SBATCH --job-name=aqc-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
QCB=/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/build-${QC_COMMIT:0:10}
( flock -w 3600 9; [ -e "$QCB/.git" ] || { git clone -q --no-checkout git@github.com:Singlet-Bio/singlet.git "$QCB" && git -C "$QCB" fetch -q origin claude/alpha-qc && git -C "$QCB" checkout -q --detach "$QC_COMMIT"; } ) 9>/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/clone.lock
source "$QCB/scripts/alpha_qc/qc_common.sh"
qc_build
cd /tmp
rt $PY "$SRC/scripts/alpha_qc/qc_sweep.py" --policy "$QC/alpha-policy.json" --lists "$QC/out/list" --out "$QC/out/sweep.json" --threads 24
