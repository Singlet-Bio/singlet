#!/bin/bash
#SBATCH --job-name=pdetect-trial
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=60G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Trial mapping for rows of a list (id TAB source TAB tags TAB genome), one row per array task:
#   sbatch --array=0-K trial.sbatch LIST.tsv OUTNAME
# Dumps 1 M head spots (every slot), finds the barcode and cDNA slots with protocol_detect, runs trial_map.py under each tag with
# the campaign build's CPU engine and writes $P/out/OUTNAME/<id>.trial.json.  Reads stay in node-local scratch.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; P=$W/jobs/protocol-detect; J=$W/jobs/identify; ENV=$W/envs/pbench
LIST=$1; OUT=$P/out/$2; I=${SLURM_ARRAY_TASK_ID:-0}
BUILD=${BUILD:-$W/builds/925aea80}; BIN=${BIN:-$J/bin/singlet-9c99ca2f}; SPOTS=${SPOTS:-1000000}
export LD_LIBRARY_PATH=$ENV/lib PATH=$ENV/bin:$PATH
SCR=/tmp/$USER-$SLURM_JOB_ID-$I; mkdir -p $SCR/home $OUT; trap 'rm -rf $SCR' EXIT
export HOME=$SCR/home TMPDIR=$SCR
IFS=$'\t' read -r id src tags genome < <(tr -d '\r' < $LIST | awk -v i=$I 'NR==i+1')
s=$(echo "$src" | sed "s#\$DATA#$W/data#g; s#\$HL#$W/data/heldout-launch#g")
mkdir -p $SCR/d
t0=$(date +%s)
timeout 1800 $BIN identify "$s" --reads $SPOTS --dump $SCR/d --dump-spots $SPOTS --dump-only --whitelist-dir $J/wl > $SCR/identify.log 2>&1 || true
ls -la $SCR/d
python3 -I - "$SCR/d" "$J/wl" "$P" > $SCR/slots.txt <<'PY'
import glob, json, os, re, sys
sys.path.insert(0, sys.argv[3])
import protocol_detect as pd, diagnose
segs = sorted(glob.glob(os.path.join(sys.argv[1], "*.seg*.fq")), key=lambda p: int(re.search(r"\.seg(\d+)\.fq$", p).group(1)))
det = pd.detect([pd.read_fastq(p) for p in segs], sys.argv[2])
bc, cd = diagnose._bc_cdna(det, segs)
print(bc or "", cd or "")
PY
read -r BC CD < $SCR/slots.txt
echo "id=$id tags=$tags bc=$BC cdna=$CD dump_s=$(( $(date +%s) - t0 ))"
if [ -z "$BC" ] || [ -z "$CD" ]; then
  printf '{"id": "%s", "error": "no barcode/cDNA slot"}\n' "$id" > $OUT/$id.trial.json; exit 0
fi
python3 -I $P/trial_map.py --build $BUILD --seg-bc $BC --seg-cdna $CD --tags "$tags" --genome "${genome:-GRCh38-2024-A}" \
  --work $SCR/trial --threads ${SLURM_CPUS_PER_TASK:-4} --id "$id" --json $OUT/$id.trial.json > $SCR/trial.log 2>&1 || tail -20 $SCR/trial.log
python3 -c "import json; r=json.load(open('$OUT/$id.trial.json')); print(json.dumps(r['referee'])); [print(x['tag'], x.get('usable'), x.get('genes_per_cell'), x.get('run_exit'), x.get('seconds')) for x in r['results']]"
echo "total_s=$(( $(date +%s) - t0 ))"
