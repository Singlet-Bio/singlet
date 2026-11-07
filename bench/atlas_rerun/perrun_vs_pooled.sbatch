#!/bin/bash
#SBATCH --job-name=s6-perrun
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,short-cpu
#SBATCH --nice=100
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,GSM=GSM6470208,BIN=singlet-pc-tip perrun_vs_pooled.sbatch      (code in $W/jobs/s6/atlas_rerun)
# One multi-run sample through the orchestrator's own stages twice, with R11 (the engine default) in both:
#   A  io-stage keeps the per-run .1fq files; compute-stage passes --runs-dir + screen.json   (new default)
#   B  the same staged runs pooled by stages.pool_runs into sample.1fq; compute-stage on it    (pipeline.pooled_input)
# Then bench/star_accel/out_equiv.py and bench/fq_read_source/matrix_diff.py on A/out vs B/out (mtx, as in PR #82's table).
# Reads, BAMs and staged files are deleted at the end; logs and the comparison stay in $W/jobs/s6/<GSM>.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
GSM=${GSM:-GSM6470208}; J=$W/jobs/s6; CODE=$J/atlas_rerun; BINP=$PB/bin/${BIN:-singlet-pc-tip}
R=$J/$GSM; rm -rf $R; mkdir -p $R
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
mk() {   # config <tag> <pooled true|false>
python3 - <<PY
import json
c = json.load(open('$CODE/config.json'))
c['binary'] = {'path': '$BINP', 'sha256': '$SHA'}
c['pipeline']['pooled_input'] = $2
c['pipeline']['extra_args'] = ['--output-format', 'mtx']
k = c['clusters']['clipper']
k.update(root='$R/$1', stage_root='$SCR/stage_$1', out_root='$R/$1/out', private_root='$R/$1/private', scratch='$SCR',
         binary='$BINP', whitelist_dir='$PB/whitelists')
json.dump(c, open('$R/config_$1.json', 'w'))
PY
mkdir -p $R/$1/logs $R/$1/inbox
}
mk A False; mk B True   # B's config is only used for its paths; pooling is done below from A's staged runs
export SINGLET_REF_BASE=
O="python3 $CODE/orchestrator.py"
echo "== io-stage (per-run)"; echo '{}' > /dev/null
cp $R/row.json $R/batch.jsonl
$O io-stage --config $R/config_A.json --cluster clipper --batch $R/batch.jsonl --index 0
ls -l $SCR/stage_A/$GSM
mkdir -p $SCR/stage_B; cp -r $SCR/stage_A/$GSM $SCR/stage_B/$GSM    # B starts from the very same staged runs
python3 - <<PY
import sys, os, glob
sys.path.insert(0, '$CODE')
import common as C, stages as S
cfg = C.load_config('$R/config_B.json'); ctx = S.Ctx(cfg, C.cluster_cfg(cfg, 'clipper'))
d = '$SCR/stage_B/$GSM'; runs = sorted(f for f in glob.glob(d + '/*.1fq'))
import time; t = time.time()
S.pool_runs(ctx, runs, d + '/sample.1fq', '$SCR', '$R/pool.log')
for f in runs: os.remove(f)
print('pooled', len(runs), 'runs in', round(time.time() - t), 's')
PY
for T in A B; do
  echo "== compute-stage $T"
  /usr/bin/time -f "$T wall %e s maxrss %M KB" $O compute-stage --config $R/config_$T.json --cluster clipper --batch $R/batch.jsonl --index 0 --threads 8
  grep -h "runs-dir\|sample.1fq" $R/$T/logs/$GSM/compute*.log | head -3 | cut -c1-300
done
echo "== compare"
GSE=$(python3 -c "import json;print(json.load(open('$R/row.json'))['gse'])")
in_runtime python $SRC/bench/star_accel/out_equiv.py $R/A/out/$GSE/$GSM $R/B/out/$GSE/$GSM | tee $R/out_equiv.txt
in_runtime python $SRC/bench/fq_read_source/matrix_diff.py $R/A/out/$GSE/$GSM $R/B/out/$GSE/$GSM | tee $R/matrix_diff.txt
find $R -name '*.bam' -o -name '*.1fq' | xargs -r rm -f
echo done
