#!/bin/bash
# S15: queue the competitor tools for every sample (login node: sbatch only), chained on the job ids in jobids.tsv.   submit_tools.sh [GSM ...]   (default: all samples in jobids.tsv)
# Every sample ends with finalclean (afterany of its tools): it drops KEEP and deletes the reads unless the GPU lane has not passed the sample yet.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; mkdir -p $W/logs
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
LOG=--output=$W/logs/%x-%j.out
only=" $* "
kbi=$(sb --job-name=s15-kbidx --partition=short-cpu --cpus-per-task=16 --mem=100G --time=03:59:00 $LOG $TL kbidx); echo "kbidx $kbi" | tee -a $D/toolids.tsv
grep -v '^gpu-lane' $D/jobids.tsv | while IFS=$'\t' read -r g fe fq so sc cp; do
  case $g in GSM*) ;; *) continue;; esac
  [ "$only" = "  " ] || [[ "$only" == *" $g "* ]] || continue
  s=${so#starsolo=}; c=${cp#scpu=}; row=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); five=$(echo "$row" | cut -f3)
  ids=()
  add() { n=$1; shift; j=$(sb --job-name=s15-$n-$g "$@" $LOG $TL $n $g) && ids+=($j) && printf "%s\t%s\t%s\n" $g $n $j | tee -a $D/toolids.tsv; }
  add rtools   --partition=short-cpu --cpus-per-task=8 --mem=96G --time=03:59:00 --dependency=afterok:$s
  add edrops   --partition=short-cpu --cpus-per-task=4 --mem=64G --time=03:59:00 --dependency=afterok:$s
  add cellbender --partition=short-gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=48G --time=03:59:00 --dependency=afterok:$s,singleton
  add kraken   --partition=short-cpu --cpus-per-task=16 --mem=64G --time=03:59:00 --dependency=afterok:$s
  add mgatk    --partition=short-cpu --cpus-per-task=8 --mem=64G --time=03:59:00 --dependency=afterok:$c
  add alevin   --partition=short-cpu --cpus-per-task=16 --mem=100G --time=03:59:00 --dependency=afterok:$s
  add kb       --partition=short-cpu --cpus-per-task=16 --mem=110G --time=03:59:00 --dependency=afterok:$s:$kbi
  [ "$five" = 1 ] && add trust4 --partition=short-bigmem --cpus-per-task=16 --mem=128G --time=03:59:00 --dependency=afterok:$s
  dep=$(IFS=:; echo "${ids[*]}")
  add finalclean --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterany:$dep
done
