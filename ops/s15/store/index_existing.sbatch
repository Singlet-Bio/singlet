#!/bin/bash
#SBATCH --job-name=s15-storeidx
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S15: register the pilot's finished competitor runs (first pass, STARsolo + tools) in the benchmark store, once. Outputs are linked, not copied.
W=/mnt/projects/debruinz_project/debruinz/singlet; export STORE=$W/benchmark-store; R=$W/jobs/s15/store_register.py
for P in $W/pilot/GSM*; do g=$(basename $P)
  [ -d $P/starsolo/Solo.out ] && [ ! -L $P/starsolo ] && python3 $R pilot_$g $g STARsolo "2.7.11b-velodump" $P/starsolo/Solo.out --cmd-file $P/starsolo/command.txt --time-file $P/starsolo/time.txt --input-dir $P
  for T in $P/tools/*; do t=$(basename $T); [ -f $T/time.txt ] || continue; case $t in kbidx|finalclean) continue;; esac
    v=$(awk -F'\t' -v t=$t '$1==t{print $2}' $STORE/tool_versions.tsv 2> /dev/null); python3 $R pilot_$g $g $t "${v:-unrecorded}" $T --cmd "pilot_tools.sh $t $g (first pass, ops/s15)" --input-dir $P
  done
done; wc -l $STORE/index.tsv
