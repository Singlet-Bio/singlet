#!/bin/bash
# Research Alpha QC step 0: S3 listings of the public prefixes (read-only) + runtime probe (R available?).
#SBATCH --job-name=aqc-list
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
# bootstrap: job-local clone of the QC commit (git only), then the shared setup from that clone
QCB=/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/build-${QC_COMMIT:0:10}
( flock -w 3600 9; [ -e "$QCB/.git" ] || { git clone -q --no-checkout git@github.com:Singlet-Bio/singlet.git "$QCB" && git -C "$QCB" fetch -q origin claude/alpha-qc && git -C "$QCB" checkout -q --detach "$QC_COMMIT"; } ) 9>/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/clone.lock
source "$QCB/scripts/alpha_qc/qc_common.sh"
qc_build
set -a; . "$HOME/.config/singlet-r2.env"; set +a
rt $PY "$SRC/scripts/alpha_qc/qc_list.py" "$QC/out/list"
echo "--- R probe"; source /etc/profile.d/lmod.sh 2>/dev/null || true; module -t avail 2>&1 | grep -i -E "^r/|^R/|^r-|R-" | head; command -v Rscript || true; rt bash -c "command -v Rscript || echo no-R-in-image"
