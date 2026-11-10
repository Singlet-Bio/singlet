#!/bin/bash
#SBATCH -J ul-cmp
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:50:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code; PY=$W/envs/pbench/bin/python3; N=$W/release-gate/runs/ec/${SHA8}-cpu
for d in pbmc1k_v3 hgmm1k_v3 nuclei2k_v2; do echo "== $d vs pin"; $PY -I $X/cmp_gate.py $N/$d $W/pipeline-benchmark/runs/$d/singlet-public 2>&1 | head -1 | cut -c1-700; done
echo "== gpu (unmirrored) vs cpu pbmc1k_v3"; $PY -I $X/cmp_gate.py $W/release-gate/runs/ec/${SHA8}-gpu/pbmc1k_v3 $N/pbmc1k_v3 2>&1 | head -1 | cut -c1-700
