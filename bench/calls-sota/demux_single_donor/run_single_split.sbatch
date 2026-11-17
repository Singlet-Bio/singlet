#!/bin/bash
#SBATCH --job-name=single_split
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
PY="apptainer exec -B /mnt $W/images/python-3.12-bookworm.sif $W/venvs/singlet/bin/python -I"
O=$W/bench/calls-sota/demux_single_donor; mkdir -p $O
cp $O/score_single_split.py /tmp/ss_$SLURM_JOB_ID.py 2>/dev/null || true
for pass in r 0ea48783; do
  args=()
  for g in GSM5130155 GSM5130163 GSM5171610 GSM7064234 GSM7884577 GSM8183378 GSM8164511 GSM9038586; do
    if [ $pass = r ]; then p=$W/pilot/$g/s1cpu_r/out/donor_assignments.tsv; else p=$W/pilot/$g/s1cpu_0ea48783/out/donor_assignments.tsv; fi
    [ -f $p ] && args+=("$p:$g") || echo "missing $pass $g"
  done
  $PY $O/score_single_split.py $O/single_split_$pass.json "${args[@]}"
done
