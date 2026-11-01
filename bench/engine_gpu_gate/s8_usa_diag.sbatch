#!/bin/bash
#SBATCH --job-name=s8-usadiag
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: why does the unspliced UMI-ratio distance stay above the tools' mutual distance on pbmc1k_v3 / pbmc1k_cite_v3 (USA labels, no kFlank)?
# Reads the extracted artifacts of the bisect runs (release-gate/bisect/<pin>/<ds>/art/spliced-unspliced): per-tool S/U/A totals over the called cells and the metric tables.
set -uo pipefail
RG=/mnt/projects/debruinz_project/debruinz/singlet/release-gate; PY=/mnt/projects/debruinz_project/debruinz/singlet/envs/pbench/bin/python
for ds in pbmc1k_v3 pbmc1k_cite_v3; do for pin in d9834b8a 72ac6555 2c030ed9; do
  A=$RG/bisect/$pin/$ds/art/spliced-unspliced; echo "=== $ds $pin"; ls $A | head -8
  $PY -I - <<PY
import pandas as pd, glob, json
A="$A"; ds="$ds"
c = pd.read_csv(f"{A}/{ds}.cells.tsv.gz", sep="\t", index_col=0)
print("columns:", list(c.columns)[:40])
tot = {col: float(c[col].sum()) for col in c.columns if col.endswith(("_S","_U","_A","_SA"))}
tools = sorted({k.rsplit("_",1)[0] for k in tot})
for t in tools:
    s,u,a = (tot.get(f"{t}_{k}",0.0) for k in "SUA")
    n = s+u+a
    if n: print(f"{t:12s} S {s/n:.3f}  U {u/n:.3f}  A {a/n:.3f}   (total {n:.0f})")
PY
done; done
