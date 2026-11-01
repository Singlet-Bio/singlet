#!/bin/bash
#SBATCH --job-name=e1-ncuexp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:40:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1: (re)export the per-source-line view of every ncu_<L>.ncu-rep in DIR (ncu 2026.2 prints line metrics only with
# --print-source cuda,sass) and run ncu_summary.py from the branch tip. CPU only. Usage: sbatch e1_ncu_export.sbatch DIR [DIR...]
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; PY=$W/images/python-3.12-bookworm.sif; NCU=/usr/local/cuda/bin/ncu
SCR=/tmp/$USER-e1x-$SLURM_JOB_ID; mkdir -p $SCR/py; trap 'rm -rf $SCR' EXIT
BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/profile:refs/fable/profile && git -C $BARE archive refs/fable/profile bench/fable/profile bench/gpu/m5/stitch.cuh | tar -x -C $SCR/py
echo "job=$SLURM_JOB_ID node=$(hostname) $(date -Is) branch=$(git -C $BARE rev-parse --short=8 refs/fable/profile)"
for D in "$@"; do
  for REP in $D/ncu_*.ncu-rep; do
    [ -f "$REP" ] || continue
    L=$(basename $REP .ncu-rep); L=${L#ncu_}
    [ -s $D/ncu_${L}_raw.csv ] || apptainer exec -B $W $IMG $NCU --import $REP --page raw --csv > $D/ncu_${L}_raw.csv 2>&1
    for k in k_p1 k_p2 k_p3 k_lane; do
      grep -q "$k" $D/ncu_${L}_raw.csv || continue
      grep -q '^"File Path"' $D/ncu_${L}_${k}_src.csv 2>/dev/null || apptainer exec -B $W $IMG $NCU --import $REP --kernel-name-base function --kernel-name regex:"$k" --page source --csv \
        --print-source cuda,sass > $D/ncu_${L}_${k}_src.csv 2>&1
      echo "$L $k: $(wc -c < $D/ncu_${L}_${k}_src.csv) bytes, $(grep -c '^"File Path"' $D/ncu_${L}_${k}_src.csv) file blocks"
    done
  done
  apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/ncu_summary.py $D > $D/summary_ncu.txt 2>&1
  apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/e1_lines_by_func.py $SCR/py/bench/gpu/m5/stitch.cuh $D/e1_ncu_lines.csv > $D/summary_by_func.txt 2>&1
  echo "== $D"; cut -c1-1500 $D/summary_ncu.txt; cat $D/summary_by_func.txt
done
