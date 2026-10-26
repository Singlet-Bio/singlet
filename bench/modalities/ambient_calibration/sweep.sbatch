#!/bin/bash
#SBATCH --job-name=af-sweep
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch sweep.sbatch TAG LANES CONFIG_FILE [BARNYARD=1]
# Runs every config line ("name key=value ...") of CONFIG_FILE on the Kang lanes (dumps from prep.sbatch; a config with
# cells=knee uses the knee-filter cells) and, if BARNYARD=1, on the 7 barnyard libraries; scores into $AF/res/TAG_*.jsonl.
set -euo pipefail
source /mnt/projects/debruinz_project/debruinz/singlet/jobs/ambfix/common.sh
af_sync
TAG=$1; LANES=$2; CFG=$3; BY=${4:-1}
B=$SRC/bench/modalities/ambient_calibration; H=$AF/harness-$TAG; af_build $H
O=$AF/est/$TAG; mkdir -p $O
for L in $LANES; do
  D=$AF/dump/$L; specs=()
  while read -r name args; do
    [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
    a=(${args}); extra=()
    for kv in "${a[@]}"; do
      if [ "$kv" = "cells=knee" ]; then extra+=("cells=$D/knee.idx"); cp $D/knee.txt $O/${L}_$name.cells.txt; else extra+=("$kv"); fi
    done
    $H $D $O/${L}_$name threads=8 "${extra[@]}"
    specs+=("$name=$O/${L}_$name")
  done < $CFG
  py $B/score_kang.py score $D $AF/res/${TAG}_kang_$L.jsonl "${specs[@]}"
done
if [ "$BY" = "1" ]; then
  for lib in $BARNYARD; do
    D=$W/latent_qc/ambient_truth/work/${lib}_full/in; specs=()
    while read -r name args; do
      [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
      echo "$args" | grep -q "cells=knee" && continue
      $H $D $O/by_${lib}_$name threads=8 $args
      specs+=("$name=$O/by_${lib}_$name")
    done < $CFG
    py $B/score_barnyard.py $LQC_SRC/bench/latent_qc/ambient_truth $AF/res/${TAG}_barnyard.jsonl $lib "${specs[@]}"
  done
fi
echo done
