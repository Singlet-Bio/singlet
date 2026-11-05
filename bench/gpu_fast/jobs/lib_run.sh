# Run helpers of the fast lane. Source this file. Modified copy of bench/gpu_batch1/jobs/tp.sbatch at 0fdec57f, lines 86, 95 to 131.
# Differences from the batch 1 text:
#   1. run_item takes the binary from BIN and adds ARM_ENV (array of NAME=value) to the run environment and ARM_ARGS (array) to the command line.
#   2. worker stops taking items when the file $SCR/<point>/STOP exists. The tier 2 watcher creates it.
#   3. PIN is defined as empty. The fast lane never pins.
# Needs: W PB REF DATA2 SCR RES0 BIN, arrays ARM_ENV and ARM_ARGS (possibly empty), array ITEMS.
PIN=""
ARM_ENV=(); ARM_ARGS=()

genome() { case $1 in neuron1k_v3|nuclei2k_v2) echo $REF/GRCm39-2024-A ;; hgmm1k_v3) echo $REF/GRCh38-GRCm39-barnyard ;; *) echo $REF/GRCh38-2024-A ;; esac; }

stage_input() {
  [ -f $SCR/data/$1.1fq ] && return
  local src=$PB/data/$1/$1.1fq; [ -f $src ] || src=$DATA2/$1/$1.1fq
  cp $src $SCR/data/$1.1fq.part && mv $SCR/data/$1.1fq.part $SCR/data/$1.1fq
}

next_item() { ( flock -x 9; n=$(cat $1); echo $((n + 1)) > $1; echo $n ) 9>>$1.lock; }

run_item() {   # point item worker sample T
  local p=$1 i=$2 w=$3 s=$4 T=$5
  local D=$SCR/$p/c$i O=$RES0/$p/c$i G; G=$(genome $s)
  mkdir -p $D/out $O
  local XE=(E18_COPY=c$i)
  [ "${DUMP:-0}" = 1 ] && [ $i = 0 ] && XE+=(SINGLET_ENGINE_GPU_DUMP=$SCR/dump-$p-c$i.bin)
  XE+=(${ARM_ENV[@]+"${ARM_ENV[@]}"})
  local CRS=$D/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $D/CellReads.stats ) 2>/dev/null &
  local WATCH=$!
  local T0; T0=$(date +%s.%N)
  ( cd $D && env "${XE[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $SCR/data/$s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz --engine gpu ${ARM_ARGS[@]+"${ARM_ARGS[@]}"} 2>&1 \
      | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  local rc=$? T1; T1=$(date +%s.%N); kill $WATCH 2>/dev/null
  printf "%d\t%d\t%s\t%s\t%s\t%d\n" $i $w $s $T0 $T1 $rc >> $RES0/$p/runs.tsv
  cp $D/out/*Log.final.out $D/out/*Log.out $O/ 2>/dev/null
  cp $D/out/read_stats.tsv $D/out/summary.json $D/out/pileup_stats.json $O/ 2>/dev/null
  [ -f $D/CellReads.stats ] && cp $D/CellReads.stats $O/CellReads.stats
}

worker() {   # point worker T
  local p=$1 w=$2 T=$3 i
  while true; do
    [ -e $SCR/$p/STOP ] && break
    i=$(next_item $SCR/$p/queue)
    [ "$i" -ge ${#ITEMS[@]} ] && break
    run_item $p $i $w ${ITEMS[$i]} $T
  done
}

# identity of run c<i> of point p against a reference sample directory REFDIR (out/, CellReads.stats, readhash.txt).
# Writes <point>/c<i>/identity.txt and a line "i<TAB>cpu<TAB>1|0" in <point>/ident.tsv. The tag "cpu" is the one tp_point.py counts as the CPU reference.
# Needs lib_ident.sh. Keeps nothing: the outputs of the run are deleted afterwards when DELETE_AFTER=1.
ident_ref() {   # point item sample REFDIR
  local p=$1 i=$2 s=$3 R=$4 r
  local b=$SCR/$p/c$i/out cb=$RES0/$p/c$i/CellReads.stats d=$RES0/$p/c$i/identity.txt
  : > $d
  readhash $b > $RES0/$p/c$i/readhash.txt; prune $b
  if [ -d $R/out ]; then
    compare_out reference $R/out $R/CellReads.stats $R/readhash.txt $b $cb $RES0/$p/c$i/readhash.txt $d; r=$?
    printf "%d\tcpu\t%d\n" $i $([ $r = 0 ] && echo 1 || echo 0) >> $RES0/$p/ident.tsv
  else
    echo "  NO REFERENCE at $R" >> $d
  fi
  echo "  c$i ($s): $(grep -a 'IDENTICAL' $d | tr '\n' ' ')" >> $RES0/$p/identity.lines
  [ "${DELETE_AFTER:-1}" = 1 ] && rm -rf $SCR/$p/c$i
}
