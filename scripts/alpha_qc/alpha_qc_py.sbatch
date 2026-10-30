#!/bin/bash
# Research Alpha QC: run one small QC script (SCRIPT, ARGS) in the runtime, e.g. qc_manifest_rows.py or qc_aggregate.py.
#   sbatch --export=ALL,QC_COMMIT=<sha>,SCRIPT=qc_aggregate.py,ARGS="..." alpha_qc_py.sbatch
#SBATCH --job-name=aqc-py
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
QCB=/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/build-${QC_COMMIT:0:10}
( flock -w 3600 9; [ -e "$QCB/.git" ] || { git clone -q --no-checkout git@github.com:Singlet-Bio/singlet.git "$QCB" && git -C "$QCB" fetch -q origin claude/alpha-qc && git -C "$QCB" checkout -q --detach "$QC_COMMIT"; } ) 9>/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/clone.lock
source "$QCB/scripts/alpha_qc/qc_common.sh"
qc_build
cd /tmp
eval "rt $PY $SRC/scripts/alpha_qc/${SCRIPT:?} ${ARGS:-}"
