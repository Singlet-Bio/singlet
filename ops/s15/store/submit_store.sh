#!/bin/bash
# S15 benchmark store (login node: sbatch only, nothing watches it afterwards). Reads $STORE/datasets.tsv rows with fetch = sra | url | bam and queues, per dataset:
#   fetch (resumable) -> [fq] -> STARsolo (sorted BAM, GeneFull_Ex50pAS, Velocyto, SJ, EmptyDrops_CR, 1MM_CR; species reference)
#   -> rtools (SoupX, DecontX, scDblFinder) | edrops | cellbender | kraken | alevin (human, not v1) | kb nac (human, not v1) | mgatk (human) | vireo + souporcell (known K, human) | trust4 (5' V(D)J, human)
#   -> storeclean, only after every tool delivered (afterok on the per-tool gates).
# Every tool has one automatic retry on the scheduler (afternotok: twice the memory, 48 h, cpu/bigmem/gpu) and a gate job (afterok:primary ? afterok:retry); time limits scale with read pairs.
#   submit_store.sh [dataset_id ...]     env: THROTTLE (datasets in flight, default 5; a dependency ladder: dataset N waits for the cleanup of dataset N-THROTTLE), TOOLS (comma list: only these tools), SKIPSTAR=1 (reuse Solo.out already there)
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s15; ST=$D/pilot_stage.sh; TL=$D/pilot_tools.sh; STORE=$W/benchmark-store; PL=$W/pilot; LOG=--output=$W/logs/%x-%j.out
sb() { sbatch --parsable --account=debruinz --kill-on-invalid-dep=yes --requeue "$@"; }
only=" $* "; THROTTLE=${THROTTLE:-5}; TOOLS=${TOOLS:-all}
declare -A K=( [kang2018_b1_a]=4 [kang2018_b1_b]=4 [kang2018_b1_c]=8 [kang2018_ctrl]=8 [kang2018_stim]=8 )
declare -a done_ids
awk -F'\t' 'NR>1 && ($13=="sra"||$13=="url"||$13=="bam"){print $11"\t"$1"\t"$2"\t"$6"\t"$7"\t"$10}' $STORE/datasets.tsv | sort -t$'\t' -k1,1g | cut -f2- | while IFS=$'\t' read -r id acc chem mods sp; do
  [ "$only" = "  " ] || [[ "$only" == *" $id "* ]] || continue
  grep -qP "^$id\t" $D/storeids.tsv 2> /dev/null && { echo "skip $id (already queued)"; continue; }
  pm=$(awk -F'\t' -v i=$id '$1==i{print $11}' $STORE/datasets.tsv); pm=${pm%.*}; [ -n "$pm" ] || pm=100; [ $pm -gt 100 ] && pm=100   # the store caps reads at 100 M pairs
  big=0; [ $pm -gt 60 ] && big=1
  P=$PL/$id; mkdir -p $P; five=0; case $chem in *5p*) five=1;; esac
  ENVX="ALL,DSID=$id,SMP=$id,FIVE_ENV=$five,STORE=$STORE,STORE_RUN=1,SPECIES=$sp"
  dep=; [ ${#done_ids[@]} -ge $THROTTLE ] && dep="--dependency=afterany:${done_ids[$((${#done_ids[@]}-THROTTLE))]}"
  gate() { sb --job-name=s15-s-gate-$1-$id --partition=short-cpu --cpus-per-task=1 --mem=100M --time=00:05:00 --dependency="afterok:$2?afterok:$3" $LOG --wrap=true; }
  # fetch: primary + one resumable retry; downstream waits on the gate
  if [[ $acc == ftp*.bam* ]]; then bu=${acc%%;*}; srr=${acc##*;}; fe="$ENVX,BAMURL=$bu,ACC=$srr"; fs=fetchbam
  elif [[ $acc == *http* ]]; then url=$(echo "$acc" | tr ';' '\n' | grep http | head -1); fe="$ENVX,URL=$url"; fs=fetchurl
  else srr=$(echo "$acc" | tr ';' '\n' | grep -E '^SRR' | paste -sd,); fe="$ENVX,SRRS=$srr,NODECODE=1"; fs=fetch; fi
  f=$(sb $dep --export=$fe --job-name=s15-s-fetch-$id --partition=cpu --cpus-per-task=6 --mem=24G --time=24:00:00 $LOG $ST $fs $id)
  f2=$(sb --export=$fe --job-name=s15-s-fetch-retry-$id --partition=cpu --cpus-per-task=6 --mem=48G --time=48:00:00 --dependency=afternotok:$f $LOG $ST $fs $id)
  gf=$(gate fetch $f $f2); q=$gf
  if [ $fs = fetch ]; then
    q=$(sb --export=$ENVX --job-name=s15-s-fq-$id --partition=cpu --cpus-per-task=4 --mem=16G --time=24:00:00 --dependency=afterok:$gf $LOG $ST fq $id)
    q2=$(sb --export=$ENVX --job-name=s15-s-fq-retry-$id --partition=cpu --cpus-per-task=4 --mem=32G --time=48:00:00 --dependency=afternotok:$q $LOG $ST fq $id); q=$(gate fq $q $q2); fi
  # STARsolo: 24 h / 140 G, retry on bigmem with twice the memory
  s1=$(sb --export=$ENVX --job-name=s15-s-star-$id --partition=cpu --cpus-per-task=16 --mem=140G --time=24:00:00 --dependency=afterok:$q $LOG $ST starsolo $id)
  s2=$(sb --export=$ENVX --job-name=s15-s-star-retry-$id --partition=bigmem --cpus-per-task=16 --mem=280G --time=48:00:00 --dependency=afternotok:$s1 $LOG $ST starsolo $id); s=$(gate star $s1 $s2)
  gates=(); ids=()
  # tool NAME PARTITION CPUS MEM_GB HOURS [GRES] [EXTRA_EXPORT]: primary + retry + gate
  tool() { t=$1; pt=$2; c=$3; m=$4; hrs=$5; gres=$6; ex=$7; [ -n "$ex" ] && ex=",$ex"
    [ "$TOOLS" = all ] || [[ ",$TOOLS," == *",$t,"* ]] || return
    tl=$(printf "%02d:00:00" $hrs); [ $hrs -le 3 ] && tl=03:59:00
    gr=; rp=cpu; rg=; [ -n "$gres" ] && { gr="--gres=$gres"; rp=gpu; rg="--gres=$gres"; }
    m2=$((m * 2)); [ $m2 -ge 200 ] && [ -z "$gres" ] && rp=bigmem; [ $m2 -gt 480 ] && m2=480; [ -n "$gres" ] && m2=$m   # GPU nodes: same memory, 48 h
    single=; [ "$t" = cellbender ] && single=",singleton"
    j=$(sb --export=$ENVX$ex --job-name=s15-s-$t-$id --partition=$pt $gr --cpus-per-task=$c --mem=${m}G --time=$tl --dependency=afterok:$s$single $LOG $TL $t $id) || return
    r=$(sb --export=$ENVX$ex --job-name=s15-s-$t-retry-$id --partition=$rp $rg --cpus-per-task=$c --mem=${m2}G --time=48:00:00 --dependency=afternotok:$j$single $LOG $TL $t $id)
    g=$(gate $t $j $r); gates+=($g); ids+=($j); }
  # short-* when the job fits in 4 h with 2x headroom (<= 60 M pairs), otherwise cpu with 24 h
  sp1=short-cpu; h1=3; [ $big = 1 ] && { sp1=cpu; h1=24; }
  tool rtools $sp1 8 96 $h1
  tool edrops $sp1 4 64 $h1
  tool kraken $sp1 16 64 $h1
  tool cellbender gpu 8 96 24 gpu:tesla_v100s:1
  [[ $chem == *3p-v1* || $sp != human ]] || tool alevin $sp1 16 100 $h1
  [[ $chem == *3p-v1* || $sp != human ]] || tool kb $sp1 16 110 $h1
  [ "$sp" = human ] && tool mgatk cpu 8 64 24
  if [ -n "${K[$id]:-}" ] && [ "$sp" = human ]; then tool vireo cpu 16 64 24 "" "KDON=${K[$id]}"; tool souporcell cpu 16 96 24 "" "KDON=${K[$id]}"; fi
  [ $five = 1 ] && [ "$sp" = human ] && [[ $mods == *vdj* ]] && tool trust4 bigmem 16 128 24
  [ ${#gates[@]} -gt 0 ] || { echo "no tools queued for $id"; continue; }
  c=$(sb --export=$ENVX --job-name=s15-s-clean-$id --partition=short-cpu --cpus-per-task=1 --mem=2G --time=00:30:00 --dependency=afterok:$(IFS=:; echo "${gates[*]}") $LOG $ST storeclean $id)
  printf "%s\tfetch=%s\tstar=%s\ttools=%s\tclean=%s\n" $id $f $s1 "${ids[*]}" $c | tee -a $D/storeids.tsv; done_ids+=($c)
done
