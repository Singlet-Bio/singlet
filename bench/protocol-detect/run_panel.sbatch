#!/bin/bash
#SBATCH --job-name=pdetect-panel
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --array=0-7
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Head dump (identify --dump-spots, every read slot) and ranked detection for each panel row, part ARRAY_TASK of N.
#   sbatch run_panel.sbatch PANEL.tsv OUTNAME          (PANEL columns as bench/protocol-detect/panel.tsv)
# Writes $P/out/OUTNAME/<id>.json (protocol_detect output plus the row id) and keeps no reads.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/jobs/protocol-detect; J=$W/jobs/identify; ENV=$W/envs/pbench
PANEL=$1; OUT=$P/out/$2; SPOTS=${SPOTS:-50000}
N=$(( ${SLURM_ARRAY_TASK_MAX:-0} + 1 )); I=${SLURM_ARRAY_TASK_ID:-0}
export LD_LIBRARY_PATH=$ENV/lib PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID-$I; mkdir -p $SCR/home $OUT; trap 'rm -rf $SCR' EXIT
export HOME=$SCR/home
tail -n +2 $PANEL | tr -d '\r' | awk -v n=$N -v i=$I 'NR%n==i' > $SCR/part.tsv
while IFS=$'\t' read -r id src rest; do
  [ -s $OUT/$id.json ] && continue
  s=$(echo "$src" | sed "s#\$DATA#$W/data#g; s#\$HL#$W/data/heldout-launch#g")
  d=$SCR/d/$id; mkdir -p $d
  t0=$(date +%s)
  timeout 900 ${BIN:-$J/bin/singlet-afdec2de} identify "$s" --reads $SPOTS --dump $d --dump-spots $SPOTS --dump-only --whitelist-dir $J/wl > $d/identify.log 2>&1
  segs=$(ls $d/*.seg*.fq 2>/dev/null | sort -V | sed 's/^/--seg /' | tr '\n' ' ')
  if [ -z "$segs" ]; then
    printf '{"id": "%s", "error": "no head spots", "log": "%s"}\n' "$id" "$(tail -3 $d/identify.log | tr '"\n' "' " | cut -c1-400)" > $OUT/$id.json
  else
    python3 -I $P/protocol_detect.py --wl-dir $J/wl $segs --json $d/r.json > $d/py.log 2>&1 \
      && python3 -c "import json,sys; r=json.load(open('$d/r.json')); r['id']='$id'; r['head_s']=$(( $(date +%s) - t0 )); json.dump(r, open('$OUT/$id.json','w'))" \
      || printf '{"id": "%s", "error": "detect failed", "log": "%s"}\n' "$id" "$(tail -3 $d/py.log | tr '"\n' "' " | cut -c1-400)" > $OUT/$id.json
  fi
  rm -rf $d
  echo "$id done $(( $(date +%s) - t0 )) s"
done < $SCR/part.tsv
