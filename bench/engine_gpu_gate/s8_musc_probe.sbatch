#!/bin/bash
#SBATCH --job-name=s8-muscprobe
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:25:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: what the MuSC truth table and the ONT alignments hold (schema, tiers, tags, barcode/gene spelling), before scoring kFlank against it.
set -u
G=/mnt/projects/debruinz_project/debruinz/ground-truth; PY=/mnt/projects/debruinz_project/debruinz/envs/gt/bin/python
ls -l $PY $G/work/MuSC/aln/chunk_000 | head
$PY - <<'PYEOF'
import pyarrow.parquet as pq
G="/mnt/projects/debruinz_project/debruinz/ground-truth"
f=pq.ParquetFile(G+"/truth/MuSC/truth.v2.parquet")
print(f.schema_arrow); print("row groups",f.metadata.num_row_groups,"rows",f.metadata.num_rows)
t=f.read_row_group(0).to_pandas()
print(t.head(3).T)
print(t.tier.value_counts(), t.has_truth.value_counts())
print(t.CB.str.len().value_counts().head(), t.UB.str.len().value_counts().head())
import pysam
b=pysam.AlignmentFile(G+"/work/MuSC/aln/chunk_000/aligned.bam","rb",check_sq=False)
for i,r in enumerate(b.fetch(until_eof=True)):
    if i<3: print(r.reference_name,r.flag,r.cigarstring[:60] if r.cigarstring else None,dict(r.get_tags()))
    if i>3: break
PYEOF
