#!/bin/bash
#SBATCH --job-name=bs-map
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only (~0.1 node-h): metadata of the staged perf samples, plus the atlas depth histogram by protocol.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python - <<'PY'
import pandas as pd, glob
A='/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/'
s=pd.read_parquet(A+'samples.parquet')
print(list(s.columns))
st=['GSM6250705','GSM6470208','GSM6996272','GSM6996273','GSM6996274','GSM7036403','GSM7656149']
c=[x for x in ['gsm_id','gse_id','organism','protocol_inferred','target_assay','catalog_read_count','n_runs','processing_status'] if x in s.columns]
print(s[s.gsm_id.isin(st)][c].to_string())
print(s.protocol_inferred.value_counts().head(15).to_string())
print(s.organism.value_counts().head(5).to_string())
print(s.catalog_read_count.describe(percentiles=[.5,.9,.99]).to_string())
PY
for g in GSM6250705 GSM6470208 GSM6996272 GSM6996273 GSM6996274 GSM7036403 GSM7656149; do echo $g $(ls -l $W/data/atlas-rerun/perf/$g/*.1fq | awk '{print $5}' | tr '\n' ' '); done
