#!/bin/bash
#SBATCH --job-name=s9-pubdry
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S9: one REAL sample (GSM6470208 / GSE200743, 61 M pairs, SRR21102187, full .sra from S3) through fetch -> stage -> drain (--engine gpu, release-gate
# CUDA build) -> per-sample producer (s1_producer.py of #97 inside the runtime) -> Beta uploader checks, in DRY RUN: no token, nothing written to Cloudflare.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; SRC=$W/jobs/s9/src; J=$W/jobs/s6
BIN=$W/release-gate/bin/d9834b8a/singlet-gpu
T=$W/jobs/s9/pubdry_$SLURM_JOB_ID; STAGE=$T/stage; mkdir -p $T/root
export SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
RT=$W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
python3 - <<PY
import json
c = json.load(open("$SRC/config.json"))
k = c["clusters"]["clipper"]
k.update(root="$T/root", binary="$BIN", whitelist_dir="$SINGLET_WHITELIST_DIR", scratch="/tmp")
c["beta"].update(enabled=True, uploader="$J/beta_ingest.py", catalog="$W/data/atlas-rerun/samples.parquet",
                 producer=["python3", "{here}/s1_producer.py", "{gse}", "{handoff}", "{work}", "--catalog", "{catalog}"])
json.dump(c, open("$T/config.json", "w"))
PY
python3 -c "import sys; sys.path.insert(0,'$SRC'); import stagequeue as Q; Q.init('$STAGE')"
echo '{"gsm": "GSM6470208", "gse": "GSE200743", "organism": "Homo sapiens", "runs": ["SRR21102187"]}' > $STAGE/todo/GSM6470208.json
echo "== fetch"; python3 -u $SRC/fetch_worker.py --stage $STAGE --config $T/config.json --streams 16 --idle-exit-min 0.3 --max-samples 1 2>&1 | cut -c1-200
echo "== drain gpu + s1-dry"
python3 -u $SRC/drain_worker.py --stage $STAGE --engine gpu --config $T/config.json --threads 12 --idle-exit-min 0.5 --max-samples 1 \
   --publish s1-dry --s1-repo $J/s1repo --s1-runtime $RT 2>&1 | cut -c1-400
echo "== events"; cat $T/root/inbox/*.json | tr '}' '\n' | grep -E '"event": "(done|failed)"' | cut -c1-600
echo "== stage after"; ls $STAGE/ready $STAGE/claimed | head; ls /tmp | grep s9- | head
echo "== done"
