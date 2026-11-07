#!/bin/bash
# S15 amendment 2 (login node, sbatch only): HTO library GSM8085473 of the pilot GEX sample GSM8085472 -> Singlet HTO + open callers on the pilot's called cells.
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
L=GSM8085473; G=GSM8085472; mkdir -p $W/pilot/$L
scpu=$(awk -F'\t' -v g=$G '$1==g{print $6}' $D/jobids.tsv | sed 's/scpu=//' | tail -1)
f=$(sb --export=ALL,NODECODE=1 --job-name=s15-fetch-$L --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 $LOG $ST fetch $L)
c=$(sb --export=ALL,GEXGSM=$G,ATLASGSE=GSE256089,HTOWL=737K-august-2016.txt --job-name=s15-htocount-$L --partition=short-bigmem --cpus-per-task=8 --mem=128G --time=03:59:00 --dependency=afterok:$f:$scpu $LOG $TL htocount $L)
k=$(sb --job-name=s15-htocallers-$L --partition=short-cpu --cpus-per-task=4 --mem=32G --time=03:59:00 --dependency=afterok:$c $LOG $TL htocallers $L)
x=$(sb --job-name=s15-htoclean-$L --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterany:$k $LOG $TL vdjclean $L)
printf "%s\thto-fetch=%s\thtocount=%s\thtocallers=%s\thtoclean=%s\t(after scpu %s)\n" $L $f $c $k $x $scpu | tee -a $D/toolids.tsv
