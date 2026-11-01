#!/bin/bash
#SBATCH --job-name=e18-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:55:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E18 (docs/gpu-engine/fable/E18-throughput.md): several samples in flight on the job's GPU(s), level 5 + ingest (the command line and
# environment of E15's bench/fable/posterior/jobs/gate.sbatch mode gpuE5i), with the engine's device slots. Structure of
# bench/atlas_rerun/perf_run.sbatch's x4_private mode (k concurrent copies, each with its own private genome), plus a sampler, the
# lock logs, identity checks and the engine dumps. Binary of bench/fable/throughput/jobs/build.sbatch in $BINDIR (singlet-gpu).
#
# POINTS="tag ..." run in order. A tag is [mix]k<K>t<T>s<S>[m<GB>][r<R>][p]:
#   k<K>   samples in flight (workers; each runs its items one after another, taking the next item from a shared queue)
#   t<T>   threads per sample (--threads)
#   s<S>   SINGLET_ENGINE_GPU_SLOTS=S (S processes may hold the device at once; S=1: every device phase of every sample is serialised)
#   m<GB>  a device memory split for S > 1 with the existing knobs (only m43 is defined: see budget() below)
#   r<R>   rounds: K*R items (default R=1: K samples started together); the "stream" form, K workers busy for R samples each
#   p      pin worker w to device w mod <GPUs> (SINGLET_ENGINE_GPU_DEVICE); default: the engine's first-free-slot rule
#   mix    the items are MIX (default "pbmc1k_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2"), else K*R copies of SAMPLE (pbmc1k_v3); with k1
#          the items run alone one after another (each the reference of its sample at that T)
# Per point $RES/<tag>/: meta.json, runs.tsv, samples.jsonl (bench/fable/throughput/e18_sampler.py: cgroup CPU and memory, per run CPU,
# RSS, PSS and device memory, the lock holders and waiters from /proc/locks, nvidia-smi), per run c<item>/ (time.txt, run.log.ts,
# star_Log.out, star_Log.final.out, CellReads.stats, read_stats.tsv, summary.json), the analysis (e18_point.py -> point.txt, point.json)
# and identity.txt. Identity: every run against the reference of its sample at the same T (the first k=1 run of that sample and T in
# this job; REF_<sample>_t<T>=<dir> names a kept one), with out_equiv.py, star_Log.final.out's statistics, SJ.out.tab, the read-order
# files by sorted records and CellReads.stats by sorted rows. DUMP=1 (default): item 0 of each point writes mapDistinct's records (E6
# dump, SINGLET_ENGINE_GPU_DUMP), hashed and removed; the hashes per sample are compared at the end.
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,SAMPLE=pbmc2m,POINTS="k1t4s1 k2t4s1" run.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/throughput/bin}; BIN=$BINDIR/singlet-gpu
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/throughput}
POINTS=(${POINTS:-k1t28s1 k2t14s1 k3t9s1 k4t7s1}); SAMPLE=${SAMPLE:-pbmc1k_v3}; MIX=(${MIX:-pbmc1k_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2})
DUMP=${DUMP:-1}; IDENT=${IDENT:-1}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES0=${RES0:-$W/jobs/fable/throughput/results/run-$SLURM_JOB_ID}; mkdir -p $RES0
SCR=/tmp/$USER-e18-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data $SCR/ref
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/throughput-run
git -C $BARE archive refs/fable/throughput-run bench/star_accel/out_equiv.py bench/fable/throughput | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
NGPU=$(nvidia-smi -L | wc -l)
S=$RES0/summary.txt
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK gpus=$NGPU bin=$BIN commit=$(cat $BINDIR/commit-gpu 2>/dev/null) ($(sha256sum $BIN | cut -c1-16)) scripts=$(git -C $BARE rev-parse --short=8 refs/fable/throughput-run) points=${POINTS[*]} $(date -Is)" | tee $S
{ nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)|^NUMA node[0-9]"; free -g | head -2
  df -h /tmp /dev/shm | tail -n +1; findmnt -n -o SOURCE,FSTYPE,SIZE /tmp 2>/dev/null; echo "cgroup $CG memory.max $(cat $CG/memory.max 2>/dev/null)"
  cat /sys/kernel/mm/transparent_hugepage/enabled; ls -l /tmp/singlet_engine_gpu_*.lock 2>/dev/null; } 2>&1 | sed 's/^/  /' | tee -a $S

genome() { case $1 in neuron1k_v3|nuclei2k_v2) echo $REF/GRCm39-2024-A ;; hgmm1k_v3) echo $REF/GRCh38-GRCm39-barnyard ;; *) echo $REF/GRCh38-2024-A ;; esac; }
# the device memory split for slots > 1 (pbmc1k_v3 at level 5 + ingest; device marks of a run alone: index 16.4 GB, reads + clip 9.5,
# context 0.5 -> 26.4 GB before the lanes; Work per read in flight 36 / 68 / 550 KB for lanes A / M / B; outputs per sequence of a chunk
# 640 / 2048 / 4096 B): chunk 4 M sequences, each lane's Work within the budget, the pileup arena below it
budget() { case $1 in
  43) echo "SINGLET_ENGINE_GPU_CHUNK=4194304 SINGLET_ENGINE_GPU_NB=274432 SINGLET_ENGINE_GPU_NBM=155648 SINGLET_ENGINE_GPU_NBB=21504 SINGLET_GPU_PILEUP_ARENA_GB=38" ;;
  *) echo "!! budget m$1 undefined" >&2; echo "" ;;
esac; }

stage_input() { [ -f $SCR/data/$1.1fq ] || { cp $PB/data/$1/$1.1fq $SCR/data/$1.1fq.part && mv $SCR/data/$1.1fq.part $SCR/data/$1.1fq; }; }

next_item() { ( flock -x 9; n=$(cat $1); echo $((n + 1)) > $1; echo $n ) 9>>$1.lock; }

run_item() {   # point item worker sample T
  local p=$1 i=$2 w=$3 s=$4 T=$5
  local D=$SCR/$p/c$i O=$RES0/$p/c$i G; G=$(genome $s)
  mkdir -p $D/out $O
  local XE=(E18_COPY=c$i)
  [ "$DUMP" = 1 ] && [ $i = 0 ] && XE+=(SINGLET_ENGINE_GPU_DUMP=$SCR/dump-$p-c$i.bin)
  [ -n "$PIN" ] && XE+=(SINGLET_ENGINE_GPU_DEVICE=$((w % NGPU)))
  local CRS=$D/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $D/CellReads.stats ) 2>/dev/null &
  local WATCH=$!
  local T0; T0=$(date +%s.%N)
  ( cd $D && env "${XE[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $SCR/data/$s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 \
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
    i=$(next_item $SCR/$p/queue)
    [ "$i" -ge ${#ITEMS[@]} ] && break
    run_item $p $i $w ${ITEMS[$i]} $T
  done
}

# identity of run c<i> of point p against the reference of its sample at the same T
ident() {   # point item sample T
  local p=$1 i=$2 s=$3 T=$4 a b ca cb v
  b=$SCR/$p/c$i/out; cb=$RES0/$p/c$i/CellReads.stats
  v=REF_${s}_t$T; a=${!v:-$SCR/ref/$s-t$T/out}; ca=$(dirname $a)/CellReads.stats
  [ -f $ca ] || ca=$(dirname $(dirname $a))/CellReads.stats
  if [ ! -d $a ]; then echo "  c$i ($s): no reference at T=$T" | tee -a $RES0/$p/identity.txt; return; fi
  local I=$RES0/$p/identity.txt
  echo "=== $p c$i ($s, T=$T) vs $a" >> $I
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES0/$p/c$i/equiv.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $RES0/$p/c$i/equiv.txt | cut -c1-220 >> $I
  local st sj recs crs
  if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then st="stats identical"; else st="STATS DIFFER"; fi
  if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then sj="SJ identical"; else sj="SJ DIFFERS"; fi
  recs=""
  for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    [ -f $a/$f ] && [ -f $b/$f ] || { recs+=" $f:absent"; continue; }
    if cmp -s $a/$f $b/$f; then recs+=" $f:byte-identical"; continue; fi
    ha=$(paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=4 | sha256sum | cut -c1-16); hb=$(paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=4 | sha256sum | cut -c1-16)
    [ "$ha" = "$hb" ] && recs+=" $f:same-records" || recs+=" $f:RECORDS-DIFFER"
  done
  if [ -f "$ca" ] && [ -f "$cb" ]; then
    if cmp -s $ca $cb; then crs="CellReads.stats byte-identical"
    elif cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then crs="CellReads.stats same rows (order differs)"
    else crs="CellReads.stats ROWS DIFFER"; fi
  else crs="CellReads.stats not compared"; fi
  echo "  c$i ($s): $(grep -a '^files:' $RES0/$p/c$i/equiv.txt | cut -c8-) [$(grep -a '^  DIFF' $RES0/$p/c$i/equiv.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/' | tr '\n' ' ')]; $st; $sj;$recs; $crs" | tee -a $I | tee -a $S
}

for p in "${POINTS[@]}"; do
  if ! [[ $p =~ ^(mix)?k([0-9]+)t([0-9]+)s([0-9]+)(m([0-9]+))?(r([0-9]+))?(p)?$ ]]; then echo "!! bad point tag $p" | tee -a $S; continue; fi
  MIXP=${BASH_REMATCH[1]}; K=${BASH_REMATCH[2]}; T=${BASH_REMATCH[3]}; SL=${BASH_REMATCH[4]}; BG=${BASH_REMATCH[6]}; R=${BASH_REMATCH[8]:-1}; PIN=${BASH_REMATCH[9]}
  if [ -n "$MIXP" ]; then ITEMS=("${MIX[@]}"); else ITEMS=(); for ((j = 0; j < K * R; j++)); do ITEMS+=($SAMPLE); done; fi
  for s in $(printf "%s\n" "${ITEMS[@]}" | sort -u); do stage_input $s; done
  XENV="SINGLET_ENGINE_GPU_SLOTS=$SL"; [ -n "$BG" ] && XENV+=" $(budget $BG)"
  unset SINGLET_ENGINE_GPU_SLOTS SINGLET_ENGINE_GPU_CHUNK SINGLET_ENGINE_GPU_NB SINGLET_ENGINE_GPU_NBM SINGLET_ENGINE_GPU_NBB SINGLET_GPU_PILEUP_ARENA_GB SINGLET_ENGINE_GPU_DEVICE
  export $XENV
  mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv
  printf '{"tag":"%s","k":%d,"T":%d,"slots":%d,"rounds":%d,"pin":"%s","items":"%s","env":"%s","node":"%s","gpus":%d}\n' $p $K $T $SL $R "$PIN" "${ITEMS[*]}" "$XENV" "$(hostname)" $NGPU > $RES0/$p/meta.json
  echo "######## $p: $K in flight, T=$T, items ${ITEMS[*]}; $XENV${PIN:+; pinned}; $(date -Is)" | tee -a $S
  python3 $SCR/repo/bench/fable/throughput/e18_sampler.py $RES0/$p/samples.jsonl $CG 0.5 2 2>$RES0/$p/sampler.err &
  SMP=$!
  sleep 1
  WP=()
  for ((w = 0; w < K; w++)); do worker $p $w $T & WP+=($!); done
  wait "${WP[@]}"
  sleep 1; kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
  du -sh $SCR/$p 2>/dev/null | sed 's/^/  \/tmp use of the point: /' | tee -a $S; df -h /tmp | tail -1 | sed 's/^/  /' | tee -a $S
  python3 $SCR/repo/bench/fable/throughput/e18_point.py $RES0/$p > $RES0/$p/point.txt 2>&1; cut -c1-400 $RES0/$p/point.txt | tee -a $S
  # identity, the dump, references
  while IFS=$'\t' read -r i w s t0 t1 rc; do
    [ "$i" = item ] && continue
    if [ $K = 1 ] && [ ! -d $SCR/ref/$s-t$T ] && [ $rc = 0 ]; then
      mkdir -p $SCR/ref/$s-t$T && mv $SCR/$p/c$i/out $SCR/ref/$s-t$T/out && cp $RES0/$p/c$i/CellReads.stats $SCR/ref/$s-t$T/ 2>/dev/null
      echo "  c$i ($s): kept as the reference for $s at T=$T" | tee -a $S
    elif [ "$IDENT" = 1 ]; then ident $p $i $s $T; fi
  done < <(sort -n $RES0/$p/runs.tsv)
  for f in $SCR/dump-$p-c*.bin; do
    [ -f $f ] || continue
    c=$(basename $f .bin); c=${c##*-}; s=$(awk -F'\t' -v i=${c#c} '$1==i{print $3}' $RES0/$p/runs.tsv)
    printf "%s\t%s\t%s\t%s\t%s\n" $p $c $s $(stat -c %s $f) $(sha256sum $f | cut -c1-64) >> $RES0/dumps.tsv; rm -f $f
    echo "  E6 dump $p $c ($s): $(tail -1 $RES0/dumps.tsv | cut -f4) bytes, sha256 $(tail -1 $RES0/dumps.tsv | cut -f5 | cut -c1-16)" | tee -a $S
  done
  rm -rf "${SCR:?}/${p:?}"
done
echo "=== mapDistinct records (E6 dumps) per sample" | tee -a $S
[ -f $RES0/dumps.tsv ] && for s in $(cut -f3 $RES0/dumps.tsv | sort -u); do
  n=$(awk -F'\t' -v s=$s '$3==s' $RES0/dumps.tsv | wc -l); u=$(awk -F'\t' -v s=$s '$3==s{print $5}' $RES0/dumps.tsv | sort -u | wc -l)
  [ $u = 1 ] && echo "  $s: $n dumps, all IDENTICAL (sha256 $(awk -F'\t' -v s=$s '$3==s{print substr($5,1,16); exit}' $RES0/dumps.tsv))" | tee -a $S \
             || echo "  $s: $n dumps, $u DIFFERENT hashes" | tee -a $S
done
python3 $SCR/repo/bench/fable/throughput/e18_point.py --table $RES0 > $RES0/table.md 2>&1; tee -a $S < $RES0/table.md
echo done $(date -Is) | tee -a $S
