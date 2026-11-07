#!/bin/bash
#SBATCH --job-name=s6-publish
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# beta-publish of the produced sample (job produce_s1.sbatch) from Clipper, then read-back and the withheld-layer check.
# The ingest token is read from ~/.singlet/beta_ingest_token (mode 600) into the job's environment only: it is never an argument,
# never in this script, and nothing here prints it (no set -x).
#   uploader: scripts/beta/beta_ingest.py of singlet-website PR #109, put at $J/beta_ingest.py
set -eo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/s6; GSM=${GSM:-GSM5287666}; R=$J/e2e_$GSM
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
GSE=$(python3 -c "import json;print(json.load(open('$R/row.json'))['gse'])")
mkdir -p $J/beta_work; ln -sfn $R/work $J/beta_work/$GSE
python3 - <<PY
import json
c = json.load(open('$J/atlas_rerun/config.json'))
c['clusters']['clipper']['root'] = '$R/root'
c['beta'].update(enabled=True, uploader='$J/beta_ingest.py', work_root='$J/beta_work', producer=['true'])
json.dump(c, open('$R/beta_config.json', 'w'))
PY
export SINGLET_INGEST_TOKEN="$(tr -d '\r\n ' < $HOME/.singlet/beta_ingest_token)"
O="python3 $J/atlas_rerun/orchestrator.py beta-publish --config $R/beta_config.json --cluster clipper --gse $GSE"
echo "== dry run"; $O --dry-run
echo "== publish"; $O
echo "== publish again (must write 0 rows)"; $O
unset SINGLET_INGEST_TOKEN
echo "== read back through beta-data.singlet.bio"
python3 - <<PY
import hashlib, json, urllib.request
m = json.load(open('$R/work/$GSE.manifest.json')); s = m['samples'][0]
get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 's6-readback'})).read()
live = json.loads(get('https://beta-data.singlet.bio/s1/m/$GSE.json?rb=%d' % __import__('time').time()))
print('manifest rev', live['rev'], 'n_samples', len(live['samples']), 'expected', m['rev'])
h = hashlib.sha256(get('https://beta-data.singlet.bio/' + s['key'])).hexdigest()
print('object sha256 matches manifest:', h == s['sha256'], s['key'])
PY
echo "== withheld layers in the object's members"
in_runtime env PYTHONPATH=$J/s1repo/python python - <<PY
import json
from singlet.s1 import policy
m = json.load(open('$R/work/$GSE.manifest.json'))['samples'][0]['members']
bad = {k: policy.omitted_layer(k) for k in m if policy.omitted_layer(k)}
print('members', len(m), 'withheld layers present:', bad or 'none')
PY
