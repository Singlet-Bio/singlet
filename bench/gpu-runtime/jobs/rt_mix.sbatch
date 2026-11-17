#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=700G --time=02:00:00 --exclude=g050 -J rt-mix -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-mix: node throughput of one build on an H100 node (2 GPUs, 54 cores) for a MIX of samples, measured rolling like rt_node.sbatch
# (one gpu-pool per GPU, K in flight, NUMA binding, cgroup memory/cores sampling, dmon) but the queue is a sample list instead of N copies
# of one sample: the campaign is deep and mixed (about 75 % of its pairs are in samples above 300M pairs), so rate, memory and
# widen/unified-memory behaviour of deep and mixed work show here, and the outputs are compared sample by sample between arms.
# Parameters (sbatch --export=ALL,NAME=value): MIX (required): "s1 s2*3 s3" = sample names, "s*n" is n copies of s; the input is
#   $W/data/gpu-deep/<s>/<s>.1fq, else $PB/data/<s>/<s>.1fq. MIXORDER=spread (default): copy k of c of a sample sits at position k/c of the
#   list (ties: order of MIX), so the deep samples come early and the copies of one sample are spread evenly; MIXORDER=list: MIX as written.
#   Item i of the list goes to the queue of GPU i%2. The order is fixed (no randomness): two binaries given the same MIX see the same queues.
#   BINNAME or BIN, K, T, TAG, SLOTS, MEMGB, MPS, NUMA, XENV, LIM, STALLSTOP, REFMIX (a mixident.tsv of another arm: per-sample comparison).
# Per item: wall, core-s, RSS, engine lines and widen/unified-memory/fallback lines of STAR's log, md5 of the output set (then the output is removed).
# Per run: RUN line (all items), per-sample table, rate in the steady state and in the middle of the run, memory, card peaks, IDENT per sample.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINNAME=${BINNAME:-rel138-e0824b21}; K=${K:-4}; T=${T:-6}; TAG=${TAG:-base}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}
BIN=${BIN:-$W/jobs/gpu-pr2/bin2/$BINNAME/singlet-gpu}; GD=$REF/GRCh38-2024-A; NCPU=$(nproc); MEMSTOP=${MEMSTOP:-660}; LIM=${LIM:-6000}
O=$W/jobs/gpu-runtime/results/rt-node-$SLURM_JOB_ID-$TAG; A=$O; mkdir -p $O; SCR=/tmp/$USER-rtnode-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run
R=$O/results.txt; T0JOB=$(date +%s)
# node state at start (runs 815809/815810, 815871/815872 were disturbed by memory outside the job): MemFree, shmem, /dev/shm and /tmp use
{ echo "node at start: $(hostname) $(awk '/MemFree|^Shmem:|^Cached:/{printf "%s %d GB; ", $1, $2/1048576}' /proc/meminfo)"; df -h /dev/shm /tmp 2>/dev/null | awk 'NR>1{printf "  %s used %s of %s\n", $6, $3, $2}'; } > $O/node_start.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_ALLSITE_TIMING=1
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70
export SINGLET_GPU_SHARED_INDEX=1 SINGLET_ENGINE_GPU_SLOTS=$SLOTS SINGLET_ENGINE_GPU_MEM_GB=$MEMGB
for kv in ${XENV:-}; do export "$kv"; done
# PROD=1: the campaign's pipeline arguments (singlet-agents hpc/clipper/campaign/config.py pipeline_args): the patched T1K and no RNA
# editing. Without it the runs type HLA with the unpatched T1K of <ref-base>/hla/t1k and write the editing matrices (the arms of
# 2026-11-12 before job 813019). PARGS adds more pipeline arguments.
PARGS=${PARGS:-}; [ "${PROD:-0}" = 1 ] && PARGS="--hla-t1k ${T1K:-$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b} --no-editing $PARGS"
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() {  # signal pattern : only processes of this job's cgroup
  local p; for p in $(mypids); do [ "$p" = "$$" ] && continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
buskeys() { nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | sed -E 's/^0000//; s/[:.]/_/g'; }
stopholders() { local b; killmine TERM gpu-holder; sleep 3; for b in $(buskeys); do rm -f /tmp/singlet_gpu_holder.${b}.* 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; killmine TERM "nvidia-smi dmon"; stopholders; [ -n "${MPS_ON:-}" ] && echo quit | nvidia-cuda-mps-control > /dev/null 2>&1; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM

echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$NCPU tag=$TAG bin=$BIN ($(cat $(dirname $BIN)/commit-gpu 2>/dev/null)) mix='${MIX:-}' K=$K T=$T slots=$SLOTS memgb=$MEMGB mps=${MPS:-0} numa=${NUMA:-0} prod=${PROD:-0} pargs='${PARGS:-}' xenv='${XENV:-}' $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R; df -h /tmp | tail -1 | tee -a $R
[ -x $BIN ] || { echo "NOBIN $BIN" | tee -a $R; exit 2; }
if [ "${MPS:-0}" = 1 ]; then   # a user-level CUDA MPS daemon for the job: kernels of the processes sharing a GPU run at the same time instead of time-slicing
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/rtmps-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=$O/mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && MPS_ON=1; echo "MPS daemon: ${MPS_ON:+started}${MPS_ON:-FAILED} ($CUDA_MPS_PIPE_DIRECTORY)" | tee -a $R; [ -n "${MPS_ON:-}" ] || exit 2
fi
[ -n "${MIX:-}" ] || { echo "NO MIX" | tee -a $R; exit 2; }
# the item list: MIX expanded (spread: copy k of c at k/c), item i -> GPU i%2
declare -a EXP
if [ "${MIXORDER:-spread}" = spread ]; then
  mapfile -t EXP < <(r=0; for t in $MIX; do s=${t%\**}; c=1; case $t in *\**) c=${t##*\*};; esac; r=$((r+1)); echo "$s $c $r"; done |
    awk '{ for(k=0;k<$2;k++) printf "%.6f %d %s\n", k/$2, $3, $1 }' | sort -s -k1,1n -k2,2n | awk '{print $3}')
else
  mapfile -t EXP < <(for t in $MIX; do s=${t%\**}; c=1; case $t in *\**) c=${t##*\*};; esac; for ((k = 0; k < c; k++)); do echo $s; done; done)
fi
echo "mix: ${#EXP[@]} items: ${EXP[*]}" | tee -a $R
inp() { local f=$W/data/gpu-deep/$1/$1.1fq; [ -s $f ] || f=$PB/data/$1/$1.1fq; echo $f; }
t=$(date +%s); for s in $(printf '%s\n' "${EXP[@]}" | sort -u); do
  F=$(inp $s); [ -s $F ] || { echo "MISSING INPUT $s ($F)" | tee -a $R; exit 2; }
  cp $F $SCR/data/$s.1fq & done
wait; echo "input copy to node-local disk: $(( $(date +%s) - t )) s, $(du -sh $SCR/data | cut -f1) ($(ls $SCR/data | wc -l) samples)" | tee -a $R

sampler() {  # every 10 s: cgroup memory (anon+shmem, memory.current), cores busy (cgroup cpu.stat), device memory per GPU
  local u0 u1 t0 t1 as cur
  u0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); t0=$(date +%s.%N)
  while :; do
    sleep 10
    as=$(awk '$1=="anon"||$1=="shmem"{t+=$2} END{printf "%.1f", t/1073741824}' $CG/memory.stat)
    cur=$(awk '{printf "%.1f", $1/1073741824}' $CG/memory.current)
    u1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); t1=$(date +%s.%N)
    echo -e "$(date +%s)\t$as\t$cur\t$(awk -v a=$u0 -v b=$u1 -v s=$t0 -v e=$t1 'BEGIN{printf "%.2f", (b-a)/1e6/(e-s)}')\t$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{printf "%s%.1f", (NR>1?"\t":""), $1/1024}')" >> $A/mon.tsv
    u0=$u1; t0=$t1
    # the node's MemFree (GB), the cgroup's memory pressure (full avg10, %), system CPU seconds so far, the node's compact_stall count (direct compaction)
    echo -e "$(date +%s)\t$(awk '/^MemFree/{printf "%.1f", $2/1048576}' /proc/meminfo)\t$(awk '/^full/{sub("avg10=","",$2); print $2}' $CG/memory.pressure 2>/dev/null)\t$(awk '$1=="system_usec"{printf "%.0f", $2/1e6}' $CG/cpu.stat)\t$(awk '$1=="compact_stall"{print $2}' /proc/vmstat)" >> $A/mem2.tsv
    awk -v a=$as -v l=$MEMSTOP 'BEGIN{exit !(a>l)}' && echo "anon+shmem $as GB > $MEMSTOP" > $A/MEMSTOP
  done
}
procsnap() {  # every 5 s, per process of the job's cgroup: pid ppid comm utime+stime (ticks, with its exited threads) rss (pages)
  local p s c
  while :; do
    echo "S $(date +%s)"
    for p in $(mypids); do s=$(cat /proc/$p/stat 2>/dev/null) || continue; c=${s#*(}; c=${c%)*}; set -- ${s##*) }; echo "$p $2 ${c// /_} $(( ${12} + ${13} )) ${22}"; done
    sleep 5
  done >> $A/procs.snap
}
hashset() {  # out_dir -> per-file md5 list on stdout (the wave-4 set)
  local d=$1 f; for f in private/allsite/allele_delta.bin private/allsite/cells.tsv.gz private/allsite/sites.tsv.gz private/allsite/store.json \
      private/genotype/cells.vcf.gz private/genotype/cells_allsite.vcf.gz private/genotype/donors_allsite.vcf.gz star_Solo.out/Gene/raw/matrix.mtx \
      star_Solo.out/Gene/raw/barcodes.tsv star_Solo.out/Gene/filtered/matrix.mtx star_SJ.out.tab; do
    [ -f $d/$f ] && echo "$(md5sum < $d/$f | cut -c1-32) $f" || echo "missing $f"; done
  (cd $d && find . -maxdepth 3 -type f \( -name '*.pz' -o -name '*.pz2' -o -name '*.1pz' \) | sort | while read -r f; do echo "$(md5sum < $f | cut -c1-32) $f"; done)
}
reap() {  # d j : a finished item's engine lines, md5 set and size; then its output goes
  local d=$1 j=$2 dd=$SCR/run/c$1.$2/out
  [ -d $dd ] || return 0
  grep -a -i "engine_gpu" $dd/star_Log.out > $A/engine.c$d.$j.log 2>/dev/null
  grep -a -i -E "widen|unified.?mem|managed.?mem|fallback|fell back|out of memory|OOM|cudaMalloc" $dd/star_Log.out > $A/flags.c$d.$j.log 2>/dev/null
  cp $dd/read_stats.tsv $A/read_stats.c$d.$j.tsv 2>/dev/null
  du -sb $dd 2>/dev/null | cut -f1 > $A/bytes.c$d.$j
  nice -n 19 bash -c "$(declare -f hashset); hashset $dd" > $A/files.c$d.$j
  echo "c$d.$j $(md5sum < $A/files.c$d.$j | cut -c1-32)" > $A/h.c$d.$j
  rm -rf "${SCR:?}/run/c$d.$j"
}

stopholders; rm -f $A/MEMSTOP
for d in 0 1; do mkdir -p $A/pool$d; : > $SCR/run/queue$d; done
nj=(0 0); : > $A/items.tsv
for i in "${!EXP[@]}"; do d=$((i % 2)); j=${nj[$d]}; s=${EXP[$i]}; mkdir -p $SCR/run/c$d.$j
  echo "$SCR/data/$s.1fq $SCR/run/c$d.$j/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" >> $SCR/run/queue$d
  echo -e "$d\t$j\t$s" >> $A/items.tsv; nj[$d]=$((j + 1)); done
NMAX=${nj[0]}; [ ${nj[1]} -gt $NMAX ] && NMAX=${nj[1]}
echo "######## $TAG: 2 GPUs x K=$K in flight, T=$T, ${nj[0]}+${nj[1]} items, kill at ${LIM}s, $(date -Is)" | tee -a $R
nvidia-smi dmon -s pucm -d 1 -o DT > $A/dmon.txt 2>&1 &
sampler & SMP=$!
procsnap & PSN=$!
st=$(date +%s); pids=()
for d in 0 1; do
  PRE=""; [ "${NUMA:-0}" = 1 ] && PRE="numactl --cpunodebind=$d --preferred=$d"   # NUMA=1: the pool of GPU d and its samples on NUMA node d (cores, and memory by preference)
  setsid bash -c 'exec '"$PRE"' "$0" gpu-pool --device '$d' --queue '$SCR/run/queue$d' --slots '$SLOTS' --workers '$K' --threads '$T' --report '$A/pool$d' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $A/pool$d/stdout.txt 2>&1 &
  pids+=($!)
done
fl=ok
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  for d in 0 1; do for ((j = 0; j < ${nj[$d]}; j++)); do
    if [ -s $A/pool$d/$j.time ] && [ ! -e $A/reaped.c$d.$j ]; then : > $A/reaped.c$d.$j; reap $d $j & fi; done; done
  [ $alive = 0 ] && break
  el=$(( $(date +%s) - st ))
  if [ -f $A/MEMSTOP ]; then fl="MEMSTOP: $(cat $A/MEMSTOP)"; elif [ $el -gt $LIM ]; then fl="TIMEOUT ${el}s > ${LIM}s"
  elif [ $el -gt ${STALLSTOP:-999999} ] && ! ls $A/pool[01]/[0-9]*.time > /dev/null 2>&1; then fl="STALLED: no item finished in ${el}s"; fi
  if [ "$fl" != ok ]; then echo "  abort: $fl" | tee -a $R; killmine TERM "singlet-gpu"; sleep 20; killmine KILL "singlet-gpu"; break; fi
  sleep 5
done
wait "${pids[@]}" 2>/dev/null
en=$(date +%s); sleep 2; kill $SMP $PSN 2>/dev/null; killmine TERM "nvidia-smi dmon"
for d in 0 1; do for ((j = 0; j < ${nj[$d]}; j++)); do [ -e $A/reaped.c$d.$j ] || { : > $A/reaped.c$d.$j; reap $d $j & }; done; done
for p in $(jobs -p); do [ "$p" = "$SMP" ] || [ "$p" = "$PSN" ] || wait $p 2>/dev/null; done

# ---------------------------------------------------------------- summary
cat $A/pool0/samples.tsv > $A/samples.all 2>/dev/null; tail -n +2 $A/pool1/samples.tsv >> $A/samples.all 2>/dev/null
# samples.tsv: item slot sample out start end wall_s user_s sys_s core_s maxrss_gb exit signal pairs pid
awk -F'\t' -v tag=$TAG -v k=$K -v T=$T -v fl="$fl" 'NR>1{ nit++; if($12==0) ok++; p+=$14; if(mn==""||$5<mn) mn=$5; if($6>mx) mx=$6; if($11>rs) rs=$11; w+=$7; cs+=$10 }
  END{ wl=mx-mn; printf "RUN %s K=%d T=%d items=%d ok=%d wall=%.0fs pairs=%.1fM rate_overall=%.3f Mpairs/s item_wall_mean=%.0fs core_s_per_Mpair=%.2f max_item_rss=%.0fGB status=%s\n", tag,k,T,nit,ok,wl,p/1e6,(wl>0? p/1e6/wl:0),w/(nit>0?nit:1),(p>0? cs/(p/1e6):0),rs,fl }' $A/samples.all | tee -a $R
: > $A/ss.txt
for d in 0 1; do
  tail -n +2 $A/pool$d/samples.tsv 2>/dev/null | sort -t$'\t' -k6,6n | awk -F'\t' -v g=$d -v k=$K -v f=$A/ss.txt '
    { n++; e[n]=$6; pr[n]=$14; if(mn==""||$5<mn) mn=$5; p+=$14; w+=$7 }
    END{ if(n==0) exit; wl=e[n]-mn; for(i=k+1;i<=n;i++) sp+=pr[i]; sw=e[n]-e[k];
      printf "  GPU%d items=%d wall=%.0fs rate_overall=%.3f Mpairs/s item_wall_mean=%.0fs; steady state (after the first %d completions): %d items in %.0fs, %.3f Mpairs/s\n", g,n,wl,(wl>0?p/1e6/wl:0),w/n,k,n-k,sw,(sw>0?sp/1e6/sw:0)
      if(sw>0) print sp/1e6/sw >> f }' | tee -a $R; done
awk '{s+=$1} END{printf "NODE steady-state rate: %.3f Mpairs/s (sum of the two GPUs)\n", s}' $A/ss.txt | tee -a $R
awk -F'\t' -v a=$(( st + (en-st)/5 )) -v b=$(( en - (en-st)/5 )) '{ if($2>m1)m1=$2; if($3>m2)m2=$3; if($5>g0)g0=$5; if($6>g1)g1=$6; if($1>=a&&$1<=b){c+=$4;n++} }
  END{printf "  host peak: anon+shmem %.0f GB, memory.current %.0f GB; card peak GPU0 %.1f GB, GPU1 %.1f GB; cores busy (middle 60%%): %.1f\n", m1,m2,g0,g1,(n>0?c/n:0)}' $A/mon.tsv | tee -a $R
awk '/^#/ && !h { for(i=1;i<=NF;i++){ if($i=="gpu")cg=i; if($i=="sm")cs=i; if($i=="pwr")cp=i }; h=1; next } /^#/ {next}
  h { g=$cg; n[g]++; s[g]+=$cs; if($cs>0) b[g]++; pw[g]+=$cp }
  END{ for(g in n) printf "  dmon GPU%s: %d s, sm%% mean %.1f, seconds with a kernel running %.0f%%, power mean %.0f W\n", g, n[g], s[g]/n[g], 100*b[g]/n[g], pw[g]/n[g] }' $A/dmon.txt | sort | tee -a $R
awk -F'\t' -v w=$(( en - st )) 'NR==1{s0=$4; c0=$5} { if(f==""||$2<f)f=$2; if($3>p)p=$3; s1=$4; c1=$5 }
  END{printf "  memory: node MemFree min %.1f GB; cgroup memory pressure (full avg10) max %.0f%%; cores in the kernel (mean) %.1f; direct compaction stalls %d\n", f,p,(w>0?(s1-s0)/w:0),c1-c0}' $A/mem2.tsv | tee -a $R
# ---- the mix: mid-window rate, per-sample table, flags, IDENT per sample
# items finished in the middle 60 % of the RUN wall: their pairs over that window (an item that straddles the edge counts when it ends inside)
awk -F'\t' 'NR>1{ if(mn==""||$5<mn) mn=$5; if($6>mx) mx=$6; e[NR]=$6; p[NR]=$14 } END{ a=mn+(mx-mn)*0.2; b=mn+(mx-mn)*0.8; for(i in e) if(e[i]>=a&&e[i]<=b) s+=p[i];
  printf "NODE mid-window rate (items ending in the middle 60%% of the RUN wall): %.3f Mpairs/s (%.0f Mpairs in %.0f s; indicative only, items are 0.1-1.3 Gpairs)\n", (b>a?s/1e6/(b-a):0), s/1e6, b-a }' $A/samples.all | tee -a $R
# join: items.tsv (pool, j, sample) with samples.tsv (out path .../c<pool>.<j>/out/)
echo "  per sample: n items, pairs/item, item wall mean/min/max (s), core-s mean, core-s per Mpair, max RSS (GB), ok" | tee -a $R
awk -F'\t' 'NR==FNR{ it[$1" "$2]=$3; next } FNR>1{ match($4,/\/c[01]\.[0-9]+\//); k=substr($4,RSTART+2,RLENGTH-3); split(k,x,"."); s=it[x[1]" "x[2]]; if(s=="") s="?";
  n[s]++; w[s]+=$7; c[s]+=$10; pr[s]=$14; if(mn[s]==""||$7<mn[s]) mn[s]=$7; if($7>mxw[s]) mxw[s]=$7; if($11>rs[s]) rs[s]=$11; if($12==0) ok[s]++ }
  END{ for(s in n) printf "  SAMPLE %-14s n=%-3d pairs=%7.1fM wall mean %5.0f min %5.0f max %5.0f core_s %7.0f core_s/Mpair %5.2f rss %5.1f ok=%d/%d\n", s,n[s],pr[s]/1e6,w[s]/n[s],mn[s],mxw[s],c[s]/n[s],c[s]/n[s]/(pr[s]/1e6),rs[s],ok[s],n[s] }' $A/items.tsv $A/samples.all | sort | tee -a $R
# per depth class (pairs per item: shallow < 200M, mid 200M to < 600M, deep >= 600M): the rate the node would have if it ran only that class at
# this concurrency (2 GPUs x K items in flight: pairs / (sum of item walls / (2K))), item wall and core-s per item and per Mpair
awk -F'\t' -v k=$K 'FNR>1{ c=($14<200e6?"shallow":($14<600e6?"mid":"deep")); n[c]++; p[c]+=$14; w[c]+=$7; cs[c]+=$10; tp+=$14 }
  END{ for(c in n) printf "  CLASS %-8s items=%-3d pairs=%8.1fM (%4.1f%% of pairs) item wall mean %5.0f s, core-s per item %6.0f, core-s/Mpair %5.2f, implied node rate at 2xK=%d in flight %6.2f Mpairs/s (pairs/item-wall %.3f Mpairs/s per item)\n", c,n[c],p[c]/1e6,100*p[c]/tp,w[c]/n[c],cs[c]/n[c],cs[c]/(p[c]/1e6),2*k,2*k*p[c]/1e6/w[c],p[c]/1e6/w[c] }' $A/samples.all | sort -r | tee -a $R
# widen / unified memory / fallback lines of the items' logs (star_Log.out), per sample
for d in 0 1; do for ((j = 0; j < ${nj[$d]}; j++)); do f=$A/flags.c$d.$j.log; [ -s $f ] && echo "$(awk -F'\t' -v d=$d -v j=$j '$1==d&&$2==j{print $3}' $A/items.tsv) c$d.$j $(wc -l < $f)"; done; done > $A/flags.items
echo "  flag lines (widen|unified memory|managed|fallback|OOM|cudaMalloc) in item logs: $(awk '{s+=$3} END{print s+0}' $A/flags.items) in $(wc -l < $A/flags.items) items" | tee -a $R
awk '{n[$1]++; l[$1]+=$3} END{for(s in n) printf "    %s: %d items, %d lines\n", s, n[s], l[s]}' $A/flags.items | sort | tee -a $R
cat $A/flags.c*.log 2>/dev/null | sed -E 's/[0-9]+(\.[0-9]+)?/N/g' | sort | uniq -c | sort -rn | head -12 | cut -c1-220 | tee -a $R
echo "  pool stdout widen/fallback lines: $(cat $A/pool[01]/stdout.txt 2>/dev/null | grep -a -c -i -E 'widen|unified|fallback|fell back|out of memory')" | tee -a $R
cat $A/h.c* > $A/hashes.txt 2>/dev/null
# IDENT per sample: the set hash of each item's output; the items of one sample must agree, and the arms must agree per sample
awk 'NR==FNR{ it["c"$1"."$2]=$3; next } { s=it[$1]; n[s]++; if(!(s" "$2 in seen)){ seen[s" "$2]=1; u[s]++; if(f[s]=="") f[s]=$2 } }
  END{ for(s in n) printf "%s\t%s\t%d\t%d\n", s, f[s], n[s], u[s] }' $A/items.tsv $A/hashes.txt | sort > $O/mixident.tsv
awk -F'\t' -v tag=$TAG '{ printf "  IDENT %s %-14s items=%d distinct=%d set hash %s\n", tag,$1,$3,$4,$2; if($4!=1) bad++ } END{ print (bad?"  IDENT "tag": INCONSISTENT within "bad" samples":"  IDENT "tag": every sample has one output set hash across its items") }' $O/mixident.tsv | tee -a $R
echo "  IDENT $TAG: $(wc -l < $A/hashes.txt) outputs hashed; files per set $(wc -l < $A/files.c0.0 2>/dev/null), missing $(grep -c '^missing' $A/files.c0.0 2>/dev/null)" | tee -a $R
if [ -s "${REFMIX:-/nonexistent}" ]; then
  awk -F'\t' 'NR==FNR{ r[$1]=$2; next } { if(!($1 in r)) { print "  IDENT vs REFMIX: " $1 " not in reference"; bad++ } else if(r[$1]==$2) print "  IDENT vs REFMIX: " $1 " IDENTICAL"; else { print "  IDENT vs REFMIX: " $1 " DIFFERS (" r[$1] " vs " $2 ")"; bad++ } }
    END{ print (bad ? "  IDENT vs REFMIX: MISMATCH in " bad " samples" : "  IDENT vs REFMIX: all samples identical") }' $REFMIX $O/mixident.tsv | tee -a $R
fi
cp $A/files.c0.0 $O/ref-files.txt 2>/dev/null
echo "done $(date -Is) elapsed $(( $(date +%s) - T0JOB )) s" | tee -a $R
