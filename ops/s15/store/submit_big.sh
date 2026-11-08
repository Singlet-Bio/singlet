#!/bin/bash
# S15 store (login node: sbatch only): reference tools for the deep pilot samples whose chains lost their reads/BAM.   submit_big.sh GSM [K]
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; LOG=--output=$W/logs/%x-%j.out
g=$1; K=${2:-}; E="ALL,DSID=pilot_$g,SMP=$g,STORE=$W/benchmark-store,STORE_RUN=1"
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
s=$(sb --export=$E --job-name=s15-b-star-$g --partition=bigmem --cpus-per-task=16 --mem=200G --time=24:00:00 $LOG $D/pilot_big.sh $g); ids=()
add() { t=$1; shift; j=$(sb --export=$E "$@" --job-name=s15-b-$t-$g $LOG $TL $t $g) && ids+=($j); }
add rtools   --partition=short-cpu --cpus-per-task=8 --mem=96G --time=03:59:00 --dependency=afterok:$s
add edrops   --partition=short-cpu --cpus-per-task=4 --mem=64G --time=03:59:00 --dependency=afterok:$s
add cellbender --partition=short-gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=48G --time=03:59:00 --dependency=afterok:$s,singleton
add kraken   --partition=short-cpu --cpus-per-task=16 --mem=64G --time=03:59:00 --dependency=afterok:$s
add alevin   --partition=short-cpu --cpus-per-task=16 --mem=100G --time=03:59:00 --dependency=afterok:$s
add kb       --partition=short-cpu --cpus-per-task=16 --mem=110G --time=03:59:00 --dependency=afterok:$s
add mgatk    --partition=cpu --cpus-per-task=8 --mem=64G --time=24:00:00 --dependency=afterok:$s
if [ -n "$K" ]; then for t in vireo:64G souporcell:96G; do v=$(sb --export=$E,KDON=$K --job-name=s15-b-${t%:*}-$g --partition=cpu --cpus-per-task=16 --mem=${t#*:} --time=24:00:00 --dependency=afterok:$s $LOG $TL ${t%:*} $g) && ids+=($v); done; fi
# reads and BAM go once every tool has ended OK; the .sra stays (the Singlet GPU run on these samples is still to be redone)
c=$(sb --export=$E --job-name=s15-b-clean-$g --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterok:$(IFS=:; echo "${ids[*]}") --wrap="rm -rf $W/pilot/$g/fq $W/pilot/$g/cap1fq; find $W/pilot/$g \( -name '*.bam' -o -name '*.bai' \) -delete")
printf "%s\tstar=%s\ttools=%s\tclean=%s\n" $g $s "${ids[*]}" $c | tee -a $D/bigids.tsv
