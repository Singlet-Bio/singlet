#!/bin/bash
# S15: kb-python lives in envs/adt (not mgatk_extra): rerun the shared nac index and every sample's kb job, and add the new kb job to the sample's finalclean dependency.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
kbi=$(sb --job-name=s15-kbidx --partition=short-cpu --cpus-per-task=16 --mem=100G --time=03:59:00 $LOG $TL kbidx); echo "kbidx $kbi" | tee -a $D/toolids.tsv
grep '^GSM' $D/jobids.tsv | while IFS=$'\t' read -r g fe fq so sc cp; do
  s=${so#starsolo=}; fc=$(awk -F'\t' -v g=$g '$1==g && $2=="finalclean"{print $3}' $D/toolids.tsv | tail -1)
  k=$(sb --job-name=s15-kb-$g --partition=short-cpu --cpus-per-task=16 --mem=110G --time=03:59:00 --dependency=afterok:$s:$kbi $LOG $TL kb $g) || continue
  printf "%s\tkb\t%s\n" $g $k >> $D/toolids.tsv
  old=$(scontrol show job $fc 2> /dev/null | grep -o 'Dependency=[^ ]*' | grep -o '[0-9]\+' | tr '\n' ':')
  [ -n "$old" ] && scontrol update JobId=$fc Dependency=afterany:${old}$k && echo "$g kb=$k finalclean=$fc updated" || echo "$g kb=$k (finalclean $fc not pending: not updated)"
done
