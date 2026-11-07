#!/bin/bash
# S15 store (run as a Slurm job): Kraken2 with the open PlusPF index on every store dataset that has a kraken row, and on the 13 pilot samples.
# Per dataset: resumable refetch (store: fetch route from datasets.tsv; pilot: kept .sra) -> [fq] -> kraken_pf (retry + gate) -> cleanup afterok. Ladder of THROTTLE datasets; the first kraken_pf job also waits for the database download (DBJOB).
#   DBJOB=<job id> THROTTLE=4 submit_kraken_pf.sh [id ...]
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; PL=$W/pilot; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes --requeue "$@"; }
gate() { sb --job-name=s15-k-gate-$1-$2 --partition=short-cpu --cpus-per-task=1 --mem=100M --time=00:05:00 --dependency="afterok:$3?afterok:$4" $LOG --wrap=true; }
THROTTLE=${THROTTLE:-4}; only=" $* "; declare -a done_ids
dbdep=; [ -n "${DBJOB:-}" ] && dbdep=",afterok:$DBJOB"
ids=$( { awk -F'\t' '$3=="kraken"{print $1}' $STORE/index.tsv | sort -u; } | grep -v '^dataset$' )
for id in $ids; do
  [ "$only" = "  " ] || [[ "$only" == *" $id "* ]] || continue
  grep -qP "^$id\t" $D/krakenpf_ids.tsv 2> /dev/null && { echo "skip $id"; continue; }
  case $id in pilot_*) dir=${id#pilot_}; mode=pilot;; *) dir=$id; mode=store;; esac
  P=$PL/$dir; mkdir -p $P; E="ALL,DSID=$id,SMP=$dir,STORE=$STORE,STORE_RUN=1"
  dep=; [ ${#done_ids[@]} -ge $THROTTLE ] && dep="--dependency=afterany:${done_ids[$((${#done_ids[@]}-THROTTLE))]}"
  if [ $mode = pilot ]; then
    if ls $P/sra/*.sra > /dev/null 2>&1; then
      q=$(sb $dep --export=$E --job-name=s15-k-fq-$dir --partition=cpu --cpus-per-task=4 --mem=16G --time=24:00:00 $LOG $D/pilot_fqdump.sh $dir)
    else
      f=$(sb $dep --export=$E,NODECODE=1 --job-name=s15-k-fetch-$dir --partition=cpu --cpus-per-task=8 --mem=32G --time=24:00:00 $LOG $ST fetch $dir)
      f2=$(sb --export=$E,NODECODE=1 --job-name=s15-k-fetch-retry-$dir --partition=cpu --cpus-per-task=8 --mem=64G --time=48:00:00 --dependency=afternotok:$f $LOG $ST fetch $dir)
      q=$(sb --export=$E --job-name=s15-k-fq-$dir --partition=cpu --cpus-per-task=4 --mem=16G --time=24:00:00 --dependency=afterok:$(gate fetch $dir $f $f2) $LOG $D/pilot_fqdump.sh $dir)
    fi
    clean="rm -rf $P/fq $P/cap1fq"
  else
    row=$(awk -F'\t' -v i=$id '$1==i{print $2"\t"$13}' $STORE/datasets.tsv); acc=${row%%$'	'*}; fm=${row##*$'	'}
    case $fm in sra|url|bam) ;; *) echo "no fetch route for $id ($fm)"; continue;; esac
    if [[ $acc == *.bam* ]]; then bu=${acc%%;*}; srr=${acc##*;}; fe="$E,BAMURL=$bu,ACC=$srr"; fs=fetchbam
    elif [[ $acc == *http* ]]; then fe="$E,URL=$(echo "$acc" | tr ';' '\n' | grep http | head -1)"; fs=fetchurl
    else fe="$E,SRRS=$(echo "$acc" | tr ';' '\n' | grep -E '^SRR' | paste -sd,),NODECODE=1"; fs=fetch; fi
    f=$(sb $dep --export=$fe --job-name=s15-k-fetch-$dir --partition=cpu --cpus-per-task=6 --mem=24G --time=24:00:00 $LOG $ST $fs $dir)
    f2=$(sb --export=$fe --job-name=s15-k-fetch-retry-$dir --partition=cpu --cpus-per-task=6 --mem=48G --time=48:00:00 --dependency=afternotok:$f $LOG $ST $fs $dir); q=$(gate fetch $dir $f $f2)
    if [ $fs = fetch ]; then
      q1=$(sb --export=$E --job-name=s15-k-fq-$dir --partition=cpu --cpus-per-task=4 --mem=16G --time=24:00:00 --dependency=afterok:$q $LOG $ST fq $dir)
      q2=$(sb --export=$E --job-name=s15-k-fq-retry-$dir --partition=cpu --cpus-per-task=4 --mem=32G --time=48:00:00 --dependency=afternotok:$q1 $LOG $ST fq $dir); q=$(gate fq $dir $q1 $q2); fi
    clean="rm -rf $P/sra $P/fq $P/tar $P/tar.dl $P/bamdl $P/stage $P/cap1fq"
  fi
  # kraken_pf: primary + retry (twice the memory) + gate; the database job gates the first one
  j=$(sb --export=$E --job-name=s15-k-kraken-$dir --partition=cpu --cpus-per-task=16 --mem=170G --time=24:00:00 --dependency=afterok:$q$dbdep $LOG $TL kraken_pf $dir)
  r=$(sb --export=$E --job-name=s15-k-kraken-retry-$dir --partition=bigmem --cpus-per-task=16 --mem=340G --time=48:00:00 --dependency=afternotok:$j $LOG $TL kraken_pf $dir); g=$(gate kraken $dir $j $r)
  c=$(sb --job-name=s15-k-clean-$dir --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterok:$g $LOG --wrap="$clean")
  printf "%s\tkraken=%s\tclean=%s\n" $id $j $c | tee -a $D/krakenpf_ids.tsv; done_ids+=($c)
done
