#!/bin/bash
#SBATCH -J amb-hgmm
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; A=$W/bench-public/artifacts/ambient-rna; X=$W/expression-sota
zcat $A/hgmm1k_v3_cells.tsv.gz > /tmp/hc.$$.tsv
$W/envs/pbench/bin/python3 -I $X/code/ambient_hgmm_eval.py $W/release-gate/runs/ec/92095e29-cpu/hgmm1k_v3 /tmp/hc.$$.tsv $A/hgmm1k_v3_pool.json $X/ambient/hgmm_pin.json
