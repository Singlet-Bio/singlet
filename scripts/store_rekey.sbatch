#!/bin/bash
# Copy the published v2/ tree to store/ in R2 (singlet-bio scripts/store_rekey.py: plan, apply, verify). Writes store/ keys only.
#   On Clipper: COMMIT=<singlet-bio sha on main> MODE=plan|apply|verify sbatch -A debruinz scripts/store_rekey.sbatch (copy of this file)
#SBATCH --job-name=storerekey
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
COMMIT=${COMMIT:?set COMMIT}
MODE=${MODE:-plan}
REPO=$WORK/repos/singlet-py.git
SRC=$WORK/src/singlet-py/rekey__${COMMIT:0:10}
OUT=$WORK/state/store/rekey
SIF=$WORK/images/python-3.12-bookworm.sif
PY=$WORK/venvs/v2tools/bin/python
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
rt() { apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$@"; }

git --git-dir="$REPO" fetch -q origin "+refs/heads/main:refs/heads/main" "+refs/heads/claude/rename-to-1-0-447a9c:refs/heads/claude/rename-to-1-0-447a9c" || true
git --git-dir="$REPO" cat-file -e "$COMMIT^{commit}"
[ -d "$SRC" ] || git --git-dir="$REPO" worktree add -q --detach "$SRC" "$COMMIT"
echo "commit=$(git -C "$SRC" rev-parse HEAD) mode=$MODE host=$(hostname)"

set +x; source ~/.config/singlet-r2.env   # R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY (never printed)
mkdir -p "$OUT"
run() { rt env PYTHONPATH="$SRC/python" R2_ACCOUNT_ID="$R2_ACCOUNT_ID" R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
          R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" "$PY" -P "$SRC/scripts/store_rekey.py" "$@" --out "$OUT"; }
case "$MODE" in
  plan)  run plan; cat "$OUT/catalog.sql" ;;
  apply) run apply && run verify ;;
  verify) run verify ;;
  *) echo "MODE must be plan, apply or verify"; exit 2 ;;
esac
