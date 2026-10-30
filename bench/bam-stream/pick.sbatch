#!/bin/bash
#SBATCH --job-name=bs-pick
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Read-only (~0.1 node-h): candidate atlas samples for the protocol datasets missing from staging.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet
ls -l $W/pipeline-benchmark/data/{pbmc5p_v1,pbmc10m,pbmc1k_v3}/*.1fq; ls -d $W/pipeline-benchmark/data/*kang* $W/data/*/*kang* 2>/dev/null; ls $W/pipeline-benchmark/runs/kang2018_ctrl | head
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime python - <<'PY'
import pandas as pd
s=pd.read_parquet('/mnt/projects/debruinz_project/debruinz/singlet/data/atlas-rerun/samples.parquet')
s=s[(s.processing_status=='done')]
c=['gsm_id','gse_id','organism','protocol_inferred','reads','catalog_read_count','n_runs','fastq_bytes','title_lib_type','avg_read_length']
def show(name,q):
    print('##',name); print(q[c].head(8).to_string())
r=s.reads.fillna(s.catalog_read_count)
s=s.assign(r=r)
show('5p human 100-140M', s[(s.protocol_inferred=='10xv3_5prime')&(s.organism=='Homo sapiens')&s.r.between(100e6,140e6)].sort_values('r'))
show('mouse v3 60-100M', s[(s.protocol_inferred=='10xv3')&(s.organism=='Mus musculus')&s.r.between(60e6,100e6)&(s.n_runs==1)])
show('human v3 55-75M single run', s[(s.protocol_inferred=='10xv3')&(s.organism=='Homo sapiens')&s.r.between(55e6,75e6)&(s.n_runs==1)])
show('kang', s[s.gse_id=='GSE96583'])
print(s.assign(b=pd.cut(s.r,[0,1e7,3e7,6e7,1e8,2e8,4e8,1e12])).groupby(['b','protocol_inferred'],observed=True).size().unstack(fill_value=0).to_string())
PY
