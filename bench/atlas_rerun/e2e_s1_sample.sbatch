#!/bin/bash
#SBATCH --job-name=s6-e2e
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --nice=100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,GSM=GSM6470208,BIN=singlet-pc-tip perrun_vs_pooled.sbatch      (code in $W/jobs/s6/atlas_rerun)
# End to end for one sample: orchestrator io + compute stages (1pz, per-run input) -> s1_producer.py (s1_publish.py, local store, verify) -> $R/work, ready for beta-publish.
#   A  io-stage keeps the per-run .1fq files; compute-stage passes --runs-dir + screen.json   (new default)
#   B  the same staged runs pooled by stages.pool_runs into sample.1fq; compute-stage on it    (pipeline.pooled_input)
# Then bench/star_accel/out_equiv.py and bench/fq_read_source/matrix_diff.py on A/out vs B/out (mtx, as in PR #82's table).
# Reads, BAMs and staged files are deleted at the end; logs and the comparison stay in $W/jobs/s6/<GSM>.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
GSM=${GSM:-GSM5287666}; J=$W/jobs/s6; CODE=$J/atlas_rerun; BINP=$PB/bin/${BIN:-singlet-pc-tip}
R=$J/e2e_$GSM; rm -rf $R; mkdir -p $R
export PATH=$W/envs/pbench/bin:$PATH
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
SRC=$W/src/singlet-py/perf__fq-read-source
SCR=/tmp/$USER-s6-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
SHA=$(sha256sum $BINP | cut -d' ' -f1); echo "bin=$BINP sha=$SHA"
# the sample's accession, runs and read counts (the campaign's own source table)
in_runtime python - <<PY
import json, pandas as pd
s = pd.read_parquet('$W/data/atlas-rerun/samples.parquet').set_index('gsm_id').loc['$GSM']
print(s.to_dict())
PY
in_runtime python - <<PY > $R/row.json
import pandas as pd, json
s = pd.read_parquet('$W/data/atlas-rerun/samples.parquet').set_index('gsm_id').loc['$GSM']
runs = str(s['srr_accessions']).replace(',', ' ').replace(';', ' ').split()
print(json.dumps({"gsm": "$GSM", "gse": str(s.get('gse_id', 'GSE0')), "runs": runs, "run_reads": [0] * len(runs),
                  "organism": None, "attempt": 0, "threads": 8}))
PY
cat $R/row.json
mk() {
python3 - <<PY
import json
c = json.load(open('$CODE/config.json'))
c['binary'] = {'path': '$BINP', 'sha256': '$SHA'}
k = c['clusters']['clipper']
k.update(root='$R/root', stage_root='$SCR/stage', out_root='$R/out', private_root='$R/private', scratch='$SCR',
         binary='$BINP', whitelist_dir='$PB/whitelists')
json.dump(c, open('$R/config.json', 'w'))
PY
mkdir -p $R/root/logs $R/root/inbox $R/root/handoff
}
mk
export SINGLET_REF_BASE=
O="python3 $CODE/orchestrator.py"
cp $R/row.json $R/batch.jsonl
$O io-stage --config $R/config.json --cluster clipper --batch $R/batch.jsonl --index 0
$O compute-stage --config $R/config.json --cluster clipper --batch $R/batch.jsonl --index 0 --threads 8
GSE=$(python3 -c "import json;print(json.load(open('$R/row.json'))['gse'])")
OUTD=$R/out/$GSE/$GSM; ls $OUTD | head -40
echo "{\"samples\": [{\"gsm\": \"$GSM\", \"state\": \"done\", \"out_dir\": \"$OUTD\"}]}" > $R/root/handoff/$GSE.json
REPO=$J/s1repo
echo "== producer"
in_runtime python3 $CODE/s1_producer.py $GSE $R/root/handoff/$GSE.json $R/work --catalog $W/data/atlas-rerun/samples.parquet --repo $REPO --python python3 --scratch $SCR
find $R/work -type f | xargs ls -l
find $R -name '*.bam' -o -name '*.1fq' | xargs -r rm -f
echo done
