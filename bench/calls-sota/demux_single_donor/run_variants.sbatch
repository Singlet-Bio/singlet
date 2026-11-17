#!/bin/bash
#SBATCH --job-name=single_split_var
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
PY="apptainer exec -B /mnt $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python -I"
O=$W/bench/calls-sota/demux_single_donor
for g in GSM5130155 GSM5130163 GSM5171610 GSM7064234 GSM7884577 GSM8183378 GSM8164511 GSM9038586; do
  args=()
  for s in s1cpu_0ea48783 s1gpu_0ea48783 s1cap_0ea48783 s1cpu s1gpu s1cap s1cpu_r s1gpu_r; do
    p=$W/pilot/$g/$s/out/donor_assignments.tsv
    [ -f $p ] && args+=("$p:$g/$s")
  done
  $PY $O/score_single_split.py $O/single_split_variants_$g.json "${args[@]}" | grep -v pooled
done
grep -h -i -E "autoK|k_selected|screen" $W/pilot/GSM5130155/s1cpu_0ea48783/*.log $W/pilot/GSM5130155/s1cpu_0ea48783/out/provenance.json 2>/dev/null | head -10
