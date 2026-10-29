#!/bin/bash
# Research Alpha QC: one array task = one line of TASKS (GSE <tab> shard k/N <tab> organism): qc_study.py on that shard.
#   sbatch --array=0-<n-1>%24 --export=ALL,QC_COMMIT=<sha>,TASKS=<file>,RUN=<name> alpha_qc_study.sbatch
#SBATCH --job-name=aqc-study
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
set -euo pipefail
QCB=/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/build-${QC_COMMIT:0:10}
( flock -w 3600 9; [ -e "$QCB/.git" ] || { git clone -q --no-checkout git@github.com:Singlet-Bio/singlet.git "$QCB" && git -C "$QCB" fetch -q origin claude/alpha-qc && git -C "$QCB" checkout -q --detach "$QC_COMMIT"; } ) 9>/mnt/projects/debruinz_project/debruinz/singlet/qc/alpha_qc/clone.lock
source "$QCB/scripts/alpha_qc/qc_common.sh"
qc_build
TASK=${SLURM_ARRAY_TASK_ID:-0}
LINE=$(sed -n "$((TASK + 1))p" "${TASKS:?}")
GSE=$(echo "$LINE" | cut -f1); SHARD=$(echo "$LINE" | cut -f2); ORG=$(echo "$LINE" | cut -f3)
OUTD=$QC/out/${RUN:?}; mkdir -p "$OUTD"
OUT=$OUTD/${GSE}_${SHARD/\//of}.json
[ -s "$OUT" ] && [ "${REDO:-0}" = 0 ] && { echo "done already: $OUT"; exit 0; }
SCR=/tmp/$USER-aqc-$SLURM_JOB_ID-$TASK; mkdir -p "$SCR"; trap 'rm -rf "$SCR"' EXIT
# CL ontology (once, shared): only to check that label ids exist
CL=$QC/cache/cl-basic.obo
if [ ! -s "$CL" ]; then ( flock -w 600 8; [ -s "$CL" ] || { mkdir -p "$QC/cache"; curl -fsSL --retry 4 -o "$CL.part" https://github.com/obophenotype/cell-ontology/releases/latest/download/cl-basic.obo && mv "$CL.part" "$CL"; } ) 8>"$QC/cl.lock" || true; fi
cd "$SCR"
echo "task $TASK: $GSE shard $SHARD ($ORG) on $(hostname)"
rt $PY "$SRC/scripts/alpha_qc/qc_study.py" "$GSE" --policy "$QC/alpha-policy.json" --out "$OUT" --scratch "$SCR/w" \
   --shard "$SHARD" --sizes "$QC/out/list/list_s1_s.tsv.gz" --organism "$ORG" --cl "$CL" --full-get-max "${FULL_GET_MAX:-150000000}"
