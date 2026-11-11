#!/bin/bash
#SBATCH -J ah-cmp3
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 2
#SBATCH --mem=48G
#SBATCH -t 00:40:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; X=$W/expression-sota/code; PY=$W/envs/pbench/bin/python3; N=$W/release-gate/runs/ec/2c5563e9-cpu
echo "== nuclei2k corrected vs raw"; $PY -I $X/corr_vs_raw.py $N/nuclei2k_v2 2>&1 | tail -1
echo "== gaublomme corrected vs raw"; $PY -I $X/corr_vs_raw.py $W/expression-sota/nucleus_hash/singlet_ah 2>&1 | tail -1
echo "== donor0.vcf diff pbmc1k_v3 (pin vs new)"; diff <(head -20 $W/pipeline-benchmark/runs/pbmc1k_v3/singlet-public/private/donor/donor0.vcf) <(head -20 $N/pbmc1k_v3/private/donor/donor0.vcf) | head -6
echo "== same, vs the earlier is_cell run of the same lineage (92095e29)"; cmp $W/release-gate/runs/ec/92095e29-cpu/pbmc1k_v3/private/donor/donor0.vcf $N/pbmc1k_v3/private/donor/donor0.vcf && echo same
