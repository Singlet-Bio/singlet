#!/bin/bash
#SBATCH --job-name=s6-produce
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# s1_producer.py on an existing orchestrator out/ (job e2e_s1_sample.sbatch), with the built copy of the s1 code and the Beta policy.
#   sbatch --export=ALL,GSM=GSM6470208 produce_s1.sbatch      -> $W/jobs/s6/e2e_<GSM>/work
set -eo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/s6; GSM=${GSM:-GSM6470208}; R=$J/e2e_$GSM
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
GSE=$(python3 -c "import json;print(json.load(open('$R/row.json'))['gse'])")
rm -rf $R/work; SCR=/tmp/$USER-s6p-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
set +e
in_runtime python3 $J/atlas_rerun/s1_producer.py $GSE $R/root/handoff/$GSE.json $R/work \
  --catalog $W/data/atlas-rerun/samples.parquet --repo $J/s1repo --python python3 --scratch $SCR --keep \
  --af $W/data/annotate/1kgp_superpop_af.parquet
RC=$?; set -e
mkdir -p $R/logs_s1; find $SCR -name '*.log' -exec cp {} $R/logs_s1/ \; || true
[ $RC -eq 0 ] || { cp -r $SCR/s1prod-*/logs $R/logs_s1/ 2>/dev/null; find $SCR -name '*.singlet' -exec cp {} $R/ \; ; exit $RC; }
find $R/work -type f | xargs ls -l
in_runtime python3 -c "
import json; m = json.load(open('$R/work/$GSE.manifest.json')); s = m['samples'][0]
print('n_samples', m['n_samples'], 'n_columns', s['n_columns'], 'n_called', s['n_called'], 'policy', m.get('release_policy'))
print('members', sorted(s['members']))"
# the Beta manifest carries the per-cell labels: the cells-table columns plus an s1/a sidecar per sample, and no private file
in_runtime python3 - "$R/work" "$GSE" <<'PY'
import json, os, sys
work, gse = sys.argv[1], sys.argv[2]
m = json.load(open(os.path.join(work, gse + ".manifest.json")))
assert m.get("transcode", {}).get("labels") == "cells-table+sidecar/1", m.get("transcode")
for s in m["samples"]:
    ann = s.get("annotations") or {}
    assert ann, "no annotations for %s" % s["gsm"]
    for key in ann.values():
        assert os.path.isfile(os.path.join(work, key)), key
    print(s["gsm"], "annotations", ann)
bad = [os.path.relpath(os.path.join(d, f), work) for d, _, fs in os.walk(work) for f in fs
       if "private" in os.path.join(d, f) or f.endswith(("_coverage.tsv", ".vcf", ".vcf.gz", ".zip"))]
assert not bad, bad
print("labels ok:", m["transcode"]["labels"], len(m["samples"]), "samples")
PY
