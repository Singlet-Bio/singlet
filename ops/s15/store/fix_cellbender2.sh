#!/bin/bash
# S15 store: retry + gate for CellBender primaries that were submitted without them (GPU retry asked for more memory than a GPU node has); gate joins the cleanup dependency.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes --requeue "$@"; }
squeue -u debruinz -h -o "%i %j" > /tmp/fixcb2.$$
grep -E " s15-s-cellbender-[A-Za-z0-9_]+$" /tmp/fixcb2.$$ | grep -v retry | while read j nm; do id=${nm#s15-s-cellbender-}
  grep -q " s15-s-gate-cellbender-$id\$" /tmp/fixcb2.$$ && continue
  cl=$(grep " s15-s-clean-$id\$" /tmp/fixcb2.$$ | awk '{print $1}' | head -1); [ -n "$cl" ] || continue
  row=$(awk -F'\t' -v i=$id '$1==i{print $6"\t"$10}' $STORE/datasets.tsv); chem=${row%%	*}; sp=${row##*	}; five=0; case $chem in *5p*) five=1;; esac
  E="ALL,DSID=$id,SMP=$id,FIVE_ENV=$five,STORE=$STORE,STORE_RUN=1,SPECIES=$sp"
  r=$(sb --export=$E --job-name=s15-s-cellbender-retry-$id --partition=gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=96G --time=48:00:00 --dependency=afternotok:$j $LOG $TL cellbender $id) || { echo "retry failed $id"; continue; }
  g=$(sb --job-name=s15-s-gate-cellbender-$id --partition=short-cpu --cpus-per-task=1 --mem=100M --time=00:05:00 --dependency="afterok:$j?afterok:$r" $LOG --wrap=true) || { echo "gate failed $id"; continue; }
  cur=$(scontrol show job $cl | grep -oE "Dependency=[^ ]*" | cut -d= -f2-); scontrol update JobId=$cl Dependency="$cur,afterok:$g"; echo "$id retry=$r gate=$g clean=$cl"
done
rm -f /tmp/fixcb2.$$
