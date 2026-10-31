#!/bin/bash
#SBATCH --job-name=s6-check
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compare the produced manifest with the pipeline's own numbers (summary.json, cell_calls.tsv, exon_counts.1pz).
W=/mnt/projects/debruinz_project/debruinz/singlet; J=$W/jobs/s6; R=$J/e2e_${GSM:-GSM6470208}
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
in_runtime env PYTHONPATH=$J/s1repo/python python - <<PY
import glob, json, csv
from singlet.s1.legacy import decode_1pz_bytes
d = glob.glob("$R/out/*/*")[0]
s = json.load(open(d + "/summary.json"))
print("summary n_cells_called", s.get("n_cells_called"), "n_input_reads", s.get("n_input_reads"))
rows = list(csv.DictReader(open(d + "/cell_calls.tsv"), delimiter="\t"))
print("cell_calls rows", len(rows), {k: sum(1 for r in rows if r[k] in ("1", "TRUE")) for k in ("is_cell", "called", "called_v2")})
c = decode_1pz_bytes(open(d + "/exon_counts.1pz", "rb").read())
print("exon", c.m, c.n, "nnz", c.nnz, "sum", int(c.data.sum()))
m = json.load(open("$R/work/" + glob.glob("$R/work/*.manifest.json")[0].split("/")[-1]))["samples"][0]
print("manifest n_columns", m["n_columns"], "n_called", m["n_called"], "stats", m["stats"])
print("called_ed", {k: v for k, v in m.get("members", {}).get("cells", {}).items() if "called" in k})
PY
