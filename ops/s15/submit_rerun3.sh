#!/bin/bash
# S15 (login node: sbatch only): the whole Singlet side again on a binary built at SHA (default 0ea48783 = prod/campaign tip at the time), after the 2a3b0d61 pass was judged not to be the Beta candidate.
#   BUILD=<job id> submit_rerun3.sh        per sample: rfetch (decode with the new binary into stage_<c8>) -> rcpu, rcap; GPU in 3 serial lanes (heads here, the rest chained by the lane jobs)
# rstar / mgatk / vireo / souporcell (STARsolo BAM side) are not touched: they do not use the Singlet binary.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; RL=$D/pilot_rerun.sh; mkdir -p $D/lane
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
LOG=--output=$W/logs/%x-%j.out; : > $D/rerunids3.tsv; : > $D/lane/extra.queue
order=$(awk -F'\t' 'NR>1 && $1!="GSM8958776"{print $4"\t"$1}' $D/samples.tsv | sort -rn | cut -f2)
declare -A F C
for g in $order; do
  r=$(awk -F'\t' -v g=$g '$1==g' $D/samples.tsv); pc=$(echo "$r" | cut -f7); tc=$(echo "$r" | cut -f8); mem=$(echo "$r" | cut -f6)
  F[$g]=$(sb --job-name=s15-rfetch-$g --partition=cpu --cpus-per-task=8 --mem=48G --time=2-00:00:00 --dependency=afterok:$BUILD $LOG $RL rfetch $g)
  C[$g]=$(sb --job-name=s15-rcpu-$g --partition=$pc --cpus-per-task=16 --mem=${mem}G --time=$tc --dependency=afterok:${F[$g]} $LOG $RL rcpu $g)
  cp=$(sb --job-name=s15-rcap-$g --partition=short-cpu --cpus-per-task=16 --mem=96G --time=03:59:00 --dependency=afterok:${F[$g]} $LOG $RL rcap $g)
  fin=$(sb --job-name=s15-rfinal-$g --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterany:${C[$g]}:$cp $LOG $RL rclean $g)
  printf "%s\trfetch=%s\trcpu=%s\trcap=%s\trfinal=%s\n" $g ${F[$g]} ${C[$g]} $cp $fin | tee -a $D/rerunids3.tsv
done
lane0=; lane1=; lane2=; t0=0; t1=0; t2=0
for g in $order; do p=$(awk -F'\t' -v g=$g '$1==g{print $4}' $D/samples.tsv)
  if [ $t0 -le $t1 ] && [ $t0 -le $t2 ]; then l=0; elif [ $t1 -le $t2 ]; then l=1; else l=2; fi
  eval "lane$l=\"\${lane$l:+\$lane$l,}$g\"; t$l=\$((t$l+p))"
done
for l in 0 1 2; do eval "ids=\$lane$l"; echo "lane $l: $ids"
  head=${ids%%,*}; rest=; for g in ${ids//,/ }; do [ "$g" = "$head" ] && continue; rest="${rest:+$rest,}$g:${F[$g]}:${C[$g]}"; done
  printf 'LANE_REST=%s\nSELF_C=%s\n' "$rest" ${C[$head]} > $D/lane/r_$head.env
  r=$(awk -F'\t' -v g=$head '$1==g' $D/samples.tsv); mem=$(echo "$r" | cut -f6); pg=$(echo "$r" | cut -f9); tg=$(echo "$r" | cut -f10)
  j=$(sb --job-name=s15-rgpu-$head --partition=$pg --gres=gpu:nvidia_h100_nvl:1 --cpus-per-task=16 --mem=${mem}G --time=$tg --dependency=afterany:${F[$head]} $LOG $RL rgpu $head); echo "lane $l head $head gpu=$j" | tee -a $D/rerunids3.tsv
done
