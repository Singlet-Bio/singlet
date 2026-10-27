#!/bin/bash
# Final accounting of a .s1 transcode campaign (scripts/s1_account.py; read-only: R2 reads, nothing written to R2 or D1):
#   sbatch --export=ALL,COMMIT=<sha>,LISTING=<plan.tsv>,ALPHA_POLICY=<json>,RUNS="<run> <run> ...",CIDX_RUN=<cidx run> scripts/s1_account.sbatch
# RUNS are directory names under $WORK/state/s1, oldest first (later results rows win). Writes $WORK/state/s1/account-<jobid>.{json,tsv}.
# Needs the build of COMMIT from s1_migrate.sbatch (MODE=build) and ~/.config/singlet-r2.env.
#SBATCH --job-name=s1acct
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
WORK=${WORK:-/mnt/projects/debruinz_project/debruinz/singlet}
COMMIT=${COMMIT:?set COMMIT}
SRC=$WORK/src/singlet-py/v2mig__${COMMIT:0:10}
SIF=$WORK/images/python-3.12-bookworm.sif
VENV=${VENV:-$WORK/venvs/v2tools}
[ -e "$SRC/.v2build.ok" ] || { echo "no build of ${COMMIT:0:10}: run s1_migrate.sbatch MODE=build first"; exit 2; }
command -v apptainer >/dev/null 2>&1 || { source /etc/profile.d/lmod.sh 2>/dev/null || true; module load apptainer 2>/dev/null || true; }
set -a; . "$HOME/.config/singlet-r2.env"; set +a
export PYTHONPATH=$SRC/python SINGLET_S1_CACHE=/tmp/${USER:-x}-${SLURM_JOB_ID:-x}-cache XDG_CACHE_HOME=$WORK/cache
trap 'rm -rf "$SINGLET_S1_CACHE"' EXIT
runs=(); for r in ${RUNS:?set RUNS}; do runs+=("$WORK/state/s1/$r"); done
OUT=$WORK/state/s1/account-${SLURM_JOB_ID:-x}
apptainer exec --bind "$WORK" --bind /tmp "$SIF" "$VENV/bin/python" -P "$SRC/scripts/s1_account.py" \
  --listing "${LISTING:?set LISTING}" --policy "${ALPHA_POLICY:?set ALPHA_POLICY}" --runs "${runs[@]}" \
  ${CIDX_RUN:+--cidx-dir "$WORK/state/s1/$CIDX_RUN"} --sample-objects "${SAMPLE_OBJECTS:-300}" --out "$OUT.json" --tsv "$OUT.tsv"
