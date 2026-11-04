#!/bin/bash
# S15 store (login node or Slurm job: sbatch only): fill missing BAM-based cells for a pilot sample: refetch the .sra if it was dropped, STARsolo sorted BAM (bigmem, 24 h, registers), then the listed tools with a retry and gate, cleanup afterok.
#   submit_gap.sh GSM tool[,tool..] [K]      e.g.  submit_gap.sh GSM5171610 mgatk
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; TL=$D/pilot_tools.sh; ST=$D/pilot_stage.sh; LOG=--output=$W/logs/%x-%j.out
g=$1; tools=$2; K=${3:-}; E="ALL,DSID=pilot_$g,SMP=$g,STORE=$W/benchmark-store,STORE_RUN=1,SPECIES=human"
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes --requeue "$@"; }
gate() { sb --job-name=s15-g-gate-$1-$g --partition=short-cpu --cpus-per-task=1 --mem=100M --time=00:05:00 --dependency="afterok:$2?afterok:$3" $LOG --wrap=true; }
dep=
if ! ls $W/pilot/$g/sra/*.sra > /dev/null 2>&1; then
  f=$(sb --export=$E,NODECODE=1 --job-name=s15-g-fetch-$g --partition=cpu --cpus-per-task=8 --mem=32G --time=24:00:00 $LOG $ST fetch $g)
  f2=$(sb --export=$E,NODECODE=1 --job-name=s15-g-fetch-retry-$g --partition=cpu --cpus-per-task=8 --mem=64G --time=48:00:00 --dependency=afternotok:$f $LOG $ST fetch $g)
  dep="--dependency=afterok:$(gate fetch $f $f2)"
fi
s1=$(sb --export=$E $dep --job-name=s15-g-star-$g --partition=bigmem --cpus-per-task=16 --mem=200G --time=24:00:00 $LOG $D/pilot_big.sh $g)
s2=$(sb --export=$E --job-name=s15-g-star-retry-$g --partition=bigmem --cpus-per-task=16 --mem=400G --time=48:00:00 --dependency=afternotok:$s1 $LOG $D/pilot_big.sh $g); s=$(gate star $s1 $s2)
gates=()
for t in ${tools//,/ }; do ex=; [ -n "$K" ] && ex=",KDON=$K"
  j=$(sb --export=$E$ex --job-name=s15-g-$t-$g --partition=cpu --cpus-per-task=16 --mem=96G --time=24:00:00 --dependency=afterok:$s $LOG $TL $t $g)
  r=$(sb --export=$E$ex --job-name=s15-g-$t-retry-$g --partition=bigmem --cpus-per-task=16 --mem=192G --time=48:00:00 --dependency=afternotok:$j $LOG $TL $t $g); gates+=($(gate $t $j $r)); done
c=$(sb --job-name=s15-g-clean-$g --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterok:$(IFS=:; echo "${gates[*]}") $LOG --wrap="rm -rf $W/pilot/$g/fq $W/pilot/$g/cap1fq; find $W/pilot/$g \( -name '*.bam' -o -name '*.bai' \) -delete")
echo "$g tools=$tools star=$s1 clean=$c"
