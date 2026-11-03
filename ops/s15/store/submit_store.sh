#!/bin/bash
# S15 benchmark store, wave 1 (login node: sbatch only). Reads $STORE/datasets.tsv rows with fetch=sra|url and queues, per dataset: fetch -> fq -> STARsolo (sorted BAM, GeneFull_Ex50pAS, Velocyto, SJ, EmptyDrops_CR, 1MM_CR)
# -> SoupX/DecontX/scDblFinder, DropletUtils EmptyDrops, CellBender, alevin-fry (USA), kb nac, mgatk, [vireo + souporcell when K is known], [TRUST4 for 5' V(D)J] -> storeclean.
#   submit_store.sh [dataset_id ...]     env: THROTTLE (max datasets in flight, default 6 via --dependency on the dataset THROTTLE places earlier)
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; PL=$W/pilot; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes "$@"; }
only=" $* "; THROTTLE=${THROTTLE:-6}
declare -A K=( [kang2018_b1_a]=4 [kang2018_b1_b]=4 [kang2018_b1_c]=8 [kang2018_ctrl]=8 )
KBI=${KBI:-$(awk '$1=="kbidx"{print $2}' $D/toolids.tsv | tail -1)}
n=0; declare -a done_ids
awk -F'\t' 'NR>1 && ($13=="sra"||$13=="url"||$13=="bam"){print $11"\t"$1"\t"$2"\t"$6"\t"$7}' $STORE/datasets.tsv | sort -t$'\t' -k1,1g | cut -f2- | while IFS=$'\t' read -r id acc chem mods; do
  [ "$only" = "  " ] || [[ "$only" == *" $id "* ]] || continue
  grep -qP "^$id\t" $D/storeids.tsv 2> /dev/null && { echo "skip $id (already queued)"; continue; }
  P=$PL/$id; mkdir -p $P; five=0; case $chem in *5p*) five=1;; esac
  ENVX="ALL,DSID=$id,SMP=$id,FIVE_ENV=$five,STORE=$STORE,STORE_RUN=1"
  dep=; [ ${#done_ids[@]} -ge $THROTTLE ] && dep="--dependency=afterany:${done_ids[$((${#done_ids[@]}-THROTTLE))]}"
  if [[ $acc == ftp*.bam* ]]; then bu=${acc%%;*}; srr=${acc##*;}
    f=$(sb $dep --export=$ENVX,BAMURL=$bu,ACC=$srr --job-name=s15-s-fetch-$id --partition=short-cpu --cpus-per-task=6 --mem=16G --time=03:59:00 $LOG $ST fetchbam $id); q=$f
  elif [[ $acc == *http* ]]; then url=$(echo "$acc" | tr ';' '\n' | grep http | head -1)
    f=$(sb $dep --export=$ENVX,URL=$url --job-name=s15-s-fetch-$id --partition=short-cpu --cpus-per-task=2 --mem=8G --time=03:59:00 $LOG $ST fetchurl $id); q=$f
  else srr=$(echo "$acc" | tr ';' '\n' | grep -E '^SRR' | paste -sd,)
    f=$(sb $dep --export=$ENVX,SRRS=$srr,NODECODE=1 --job-name=s15-s-fetch-$id --partition=short-cpu --cpus-per-task=8 --mem=32G --time=03:59:00 $LOG $ST fetch $id)
    q=$(sb --export=$ENVX --job-name=s15-s-fq-$id --partition=short-cpu --cpus-per-task=2 --mem=16G --time=03:59:00 --dependency=afterok:$f $LOG $ST fq $id); fi
  s=$(sb --export=$ENVX --job-name=s15-s-star-$id --partition=cpu --cpus-per-task=16 --mem=140G --time=12:00:00 --dependency=afterok:$q $LOG $ST starsolo $id)
  ids=()
  add() { t=$1; shift; j=$(sb --export=$ENVX "$@" --job-name=s15-s-$t-$id $LOG $TL $t $id) && ids+=($j); }
  add rtools   --partition=short-cpu --cpus-per-task=8 --mem=96G --time=03:59:00 --dependency=afterok:$s
  add edrops   --partition=short-cpu --cpus-per-task=4 --mem=64G --time=03:59:00 --dependency=afterok:$s
  add kraken   --partition=short-cpu --cpus-per-task=16 --mem=64G --time=03:59:00 --dependency=afterok:$s
  add cellbender --partition=short-gpu --gres=gpu:tesla_v100s:1 --cpus-per-task=8 --mem=48G --time=03:59:00 --dependency=afterok:$s,singleton
  [[ $chem == *3p-v1* ]] || add alevin   --partition=short-cpu --cpus-per-task=16 --mem=100G --time=03:59:00 --dependency=afterok:$s
  [[ $chem == *3p-v1* ]] || add kb       --partition=short-cpu --cpus-per-task=16 --mem=110G --time=03:59:00 --dependency=afterok:$s
  add mgatk    --partition=cpu --cpus-per-task=8 --mem=64G --time=24:00:00 --dependency=afterok:$s
  if [ -n "${K[$id]:-}" ]; then for t in vireo:64G souporcell:96G; do v=$(sb --export=$ENVX,KDON=${K[$id]} --job-name=s15-s-${t%:*}-$id --partition=short-cpu --cpus-per-task=16 --mem=${t#*:} --time=03:59:00 --dependency=afterok:$s $LOG $TL ${t%:*} $id) && ids+=($v); done; fi
  [ $five = 1 ] && [[ $mods == *vdj* ]] && add trust4 --partition=short-bigmem --cpus-per-task=16 --mem=128G --time=03:59:00 --dependency=afterok:$s
  c=$(sb --export=$ENVX --job-name=s15-s-clean-$id --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterok:$(IFS=:; echo "${ids[*]}") $LOG $ST storeclean $id)
  printf "%s\tfetch=%s\tstar=%s\ttools=%s\tclean=%s\n" $id $f $s "${ids[*]}" $c | tee -a $D/storeids.tsv; done_ids+=($c)
done
