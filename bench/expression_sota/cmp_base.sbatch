#!/bin/bash
#SBATCH -J ub-cmp
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:50:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# new build vs the base build of the same prod/campaign tip (SHA_NEW, SHA_BASE); what changed besides the label split
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code; PY=$W/envs/pbench/bin/python3
for d in pbmc1k_v3 hgmm1k_v3 nuclei2k_v2; do echo "== $d new vs base"; $PY -I $X/cmp_gate.py $W/release-gate/runs/ec/$SHA_NEW-cpu/$d $W/release-gate/runs/ec/$SHA_BASE-cpu/$d 2>&1 | head -1 | cut -c1-900
$PY -I - <<EOF
import csv
a=list(csv.DictReader(open("$W/release-gate/runs/ec/$SHA_BASE-cpu/$d/cell_calls.tsv"),delimiter="\t")); b=list(csv.DictReader(open("$W/release-gate/runs/ec/$SHA_NEW-cpu/$d/cell_calls.tsv"),delimiter="\t"))
n=lambda r,k:sum(1 for x in r if x[k] in ("1","TRUE"))
print("  cell_calls base: is_cell %d knee %d | new: is_cell %d knee %d ; differ in is_cell on %d rows"%(n(a,"is_cell"),n(a,"called_knee"),n(b,"is_cell"),n(b,"called_knee"),sum(1 for x,y in zip(a,b) if x["is_cell"]!=y["is_cell"])))
EOF
done
