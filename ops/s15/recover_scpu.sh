#!/bin/bash
# S15: multi-run samples need `--runs-used` with `--runs-dir`; the first scpu/sgpu jobs of every multi-run sample failed at once (rc=1). Resubmit the failed scpu jobs and their afterok dependents (login node, sbatch only).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
declare -A K=( [GSM4987148]=4 [GSM4987144]=4 [GSM8085472]=10 )
: > $D/jobids.new
while IFS=$'\t' read -r g fe fq so sc cp; do
  case $g in GSM*) ;; *) echo "$g $fe $fq $so $sc $cp" | tr ' ' '\t' >> $D/jobids.new; continue;; esac
  q=${fq#fq=}; s=${so#starsolo=}; c=${cp#scpu=}; st=$(sacct -j $c -X -n -o State | tr -d ' ')
  if [ "$st" != FAILED ] && [ "$st" != CANCELLED ]; then printf "%s\t%s\t%s\t%s\t%s\t%s\n" $g "$fe" "$fq" "$so" "$sc" "$cp" >> $D/jobids.new; continue; fi
  r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pc=$(echo "$r" | cut -f7); tc=$(echo "$r" | cut -f8)
  [ -d $W/pilot/$g/stage ] || { echo "$g: no staged reads, skipped"; printf "%s\t%s\t%s\t%s\t%s\t%s\n" $g "$fe" "$fq" "$so" "$sc" "$cp" >> $D/jobids.new; continue; }
  n=$(sb --job-name=s15-scpu-$g --partition=$pc --cpus-per-task=16 --mem=${mem}G --time=$tc $LOG $ST scpu $g) || continue
  scontrol update JobId=$s Dependency=afterok:$q,afterany:$n 2> /dev/null
  printf "%s\t%s\t%s\t%s\t%s\tscpu=%s\n" $g "$fe" "$fq" "$so" "$sc" $n >> $D/jobids.new
  printf "%s\tscpu-rerun\t%s\n" $g $n >> $D/toolids.tsv
  new=()
  m=$(sb --job-name=s15-mgatk-$g --partition=short-cpu --cpus-per-task=8 --mem=64G --time=03:59:00 --dependency=afterok:$n $LOG $TL mgatk $g) && { new+=($m); printf "%s\tmgatk\t%s\n" $g $m >> $D/toolids.tsv; }
  if [ -n "${K[$g]:-}" ]; then for t in vireo:64G souporcell:96G; do tn=${t%:*}; tm=${t#*:}
    v=$(sb --export=ALL,KDON=${K[$g]} --job-name=s15-$tn-$g --partition=cpu --cpus-per-task=16 --mem=$tm --time=2-00:00:00 --dependency=afterok:$n $LOG $TL $tn $g) && { new+=($v); printf "%s\t%s\t%s\n" $g $tn $v >> $D/toolids.tsv; }; done; fi
  fc=$(awk -F'\t' -v g=$g '$1==g && $2=="finalclean"{print $3}' $D/toolids.tsv | tail -1)
  old=$(scontrol show job $fc 2> /dev/null | grep -o 'Dependency=[^ ]*' | grep -o '[0-9]\+' | tr '\n' ':'); add=$(IFS=:; echo "${new[*]}")
  [ -n "$old" ] && scontrol update JobId=$fc Dependency=afterany:${old}$add
  echo "$g scpu $c -> $n; new tools: ${new[*]}; finalclean $fc ${old:+updated}"
done < <(cat $D/jobids.tsv)
cp $D/jobids.tsv $D/jobids.before_recover; mv $D/jobids.new $D/jobids.tsv
# GPU runs that failed on the same bug are redone through the shared extra queue (picked up when a lane ends)
for g in GSM7884577 GSM5171610; do l=$(awk -F'\t' -v g=$g '$1==g' $D/jobids.tsv); f=$(echo "$l" | cut -f2 | sed 's/fetch=//'); s=$(echo "$l" | cut -f4 | sed 's/starsolo=//'); p=$(echo "$l" | cut -f5 | sed 's/scap=//'); c=$(echo "$l" | cut -f6 | sed 's/scpu=//')
  echo "$g:$f:$c:$s:$p" >> $D/lane/extra.queue; done
echo "extra queue:"; cat $D/lane/extra.queue
