#!/bin/bash
# S15 store: add the CellBender chain (primary on a V100S, retry, gate) to datasets whose chain was submitted without it (gres typo), and hook the gate into their cleanup job.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes --requeue "$@"; }
squeue -u debruinz -h -o "%i %j" > /tmp/fixcb.$$
cut -f1 $D/storeids.tsv | while read id; do
  grep -q " s15-s-cellbender-$id\$" /tmp/fixcb.$$ && continue
  gs=$(grep " s15-s-gate-star-$id\$" /tmp/fixcb.$$ | awk '{print $1}' | head -1); cl=$(grep " s15-s-clean-$id\$" /tmp/fixcb.$$ | awk '{print $1}' | head -1)
  [ -n "$gs" ] && [ -n "$cl" ] || continue
  row=$(awk -F'\t' -v i=$id '$1==i{print $6"\t"$10}' $STORE/datasets.tsv); chem=${row%%	*}; sp=${row##*	}; five=0; case $chem in *5p*) five=1;; esac
  E="ALL,DSID=$id,SMP=$id,FIVE_ENV=$five,STORE=$STORE,STORE_RUN=1,SPECIES=$sp"
  j=$(sb --export=$E --job-name=s15-s-cellbender-$id --partition=gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=96G --time=24:00:00 --dependency=afterok:$gs $LOG $TL cellbender $id) || continue
  r=$(sb --export=$E --job-name=s15-s-cellbender-retry-$id --partition=gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=192G --time=48:00:00 --dependency=afternotok:$j $LOG $TL cellbender $id)
  g=$(sb --job-name=s15-s-gate-cellbender-$id --partition=short-cpu --cpus-per-task=1 --mem=100M --time=00:05:00 --dependency="afterok:$j?afterok:$r" $LOG --wrap=true)
  cur=$(scontrol show job $cl | grep -oE "Dependency=[^ ]*" | cut -d= -f2-); scontrol update JobId=$cl Dependency="$cur,afterok:$g"
  echo "$id cellbender=$j gate=$g clean=$cl"
done
rm -f /tmp/fixcb.$$
