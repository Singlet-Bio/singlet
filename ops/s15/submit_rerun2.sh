#!/bin/bash
# S15 (login node: sbatch only): STARsolo sorted-BAM run (rstar), Singlet CPU on the 100 M-pair prefix (rcap), and the BAM tools (mgatk; vireo + souporcell for the known mixtures) per sample,
# chained on the sample's rfetch. the .sra is dropped by whichever of rclean / rbamclean runs last once everything has delivered (drop_sra_if_done in pilot_rerun.sh).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; RL=$D/pilot_rerun.sh; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=no "$@"; }
declare -A K=( [GSM4987148]=4 [GSM4987144]=4 [GSM8085472]=10 )
: > $D/rerunids2.tsv
for g in $(cut -f1 $D/rerunids.tsv | grep ^GSM); do
  f=$(awk -F'\t' -v g=$g '$1==g{sub("rfetch=","",$2);print $2}' $D/rerunids.tsv)
  st=$(sb --job-name=s15-rstar-$g --partition=short-cpu --cpus-per-task=16 --mem=140G --time=03:59:00 --dependency=afterok:$f $LOG $RL rstar $g)
  cp=$(sb --job-name=s15-rcap-$g --partition=short-cpu --cpus-per-task=16 --mem=96G --time=03:59:00 --dependency=afterok:$f $LOG $RL rcap $g)
  mg=$(sb --job-name=s15-mgatk-$g --partition=short-cpu --cpus-per-task=8 --mem=64G --time=03:59:00 --dependency=afterok:$st $LOG $TL mgatk $g); ids="$mg"
  if [ -n "${K[$g]:-}" ]; then
    for n in vireo:64G souporcell:96G; do t=${n%:*}; m=${n#*:}
      j=$(sb --export=ALL,KDON=${K[$g]} --job-name=s15-$t-$g --partition=short-cpu --cpus-per-task=16 --mem=$m --time=03:59:00 --dependency=afterok:$st $LOG $TL $t $g); ids="$ids:$j"; printf "%s\t%s\t%s\n" $g $t $j >> $D/toolids.tsv
    done; fi
  bc=$(sb --job-name=s15-rbamclean-$g --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterany:$ids $LOG $RL rbamclean $g)
  printf "%s\trstar=%s\trcap=%s\tmgatk=%s\trbamclean=%s\n" $g $st $cp $mg $bc | tee -a $D/rerunids2.tsv
done
