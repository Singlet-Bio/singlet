#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=700G --time=01:30:00 --exclude=g050 -J rt-node -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-node: node throughput of one build on an H100 node (2 GPUs, 54 cores), measured rolling: N items of one sample per GPU
# go through one gpu-pool per GPU with K in flight, so the middle of the run is a steady state (the wave-4 arms started K
# items at once and stopped). Derived from jobs/gpu-campaign/w4int-h6.sbatch (run_arm) and w4prof.sbatch (cgroup-scoped kills).
# Parameters (sbatch --export=ALL,NAME=value): BINNAME or BIN, S, K, T, N, TAG, SLOTS, MEMGB, MPS (1: CUDA MPS daemon), NUMA (1: pool d bound to NUMA node d), XENV ("A=1 B=2": extra exports).
# Per item: wall, core-s, RSS, the engine's lines of STAR's log, md5 of the output set (then the output is removed).
# Per run: rate overall and in the steady state, per GPU; cgroup memory and cores every 10 s; nvidia-smi dmon every 1 s.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINNAME=${BINNAME:-rel138-e0824b21}; S=${S:-d300_3p}; K=${K:-4}; T=${T:-6}; N=${N:-12}; TAG=${TAG:-base}; SLOTS=${SLOTS:-2}; MEMGB=${MEMGB:-31}
BIN=${BIN:-$W/jobs/gpu-pr2/bin2/$BINNAME/singlet-gpu}; GD=$REF/GRCh38-2024-A; NCPU=$(nproc); MEMSTOP=${MEMSTOP:-660}; LIM=${LIM:-4500}
O=$W/jobs/gpu-runtime/results/rt-node-$SLURM_JOB_ID-$TAG; A=$O; mkdir -p $O; SCR=/tmp/$USER-rtnode-$SLURM_JOB_ID; mkdir -p $SCR/data $SCR/run
R=$O/results.txt; T0JOB=$(date +%s)
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

echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$NCPU tag=$TAG bin=$BIN ($(cat $(dirname $BIN)/commit-gpu 2>/dev/null)) sample=$S K=$K T=$T N=$N slots=$SLOTS memgb=$MEMGB mps=${MPS:-0} numa=${NUMA:-0} prod=${PROD:-0} pargs='${PARGS:-}' xenv='${XENV:-}' $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv,noheader | tee -a $R; free -g | head -2 | tee -a $R; df -h /tmp | tail -1 | tee -a $R
[ -x $BIN ] || { echo "NOBIN $BIN" | tee -a $R; exit 2; }
if [ "${MPS:-0}" = 1 ]; then   # a user-level CUDA MPS daemon for the job: kernels of the processes sharing a GPU run at the same time instead of time-slicing
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/rtmps-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=$O/mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && MPS_ON=1; echo "MPS daemon: ${MPS_ON:+started}${MPS_ON:-FAILED} ($CUDA_MPS_PIPE_DIRECTORY)" | tee -a $R; [ -n "${MPS_ON:-}" ] || exit 2
fi
F=$W/data/gpu-deep/$S/$S.1fq; [ -s $F ] || { echo "MISSING INPUT $F" | tee -a $R; exit 2; }
t=$(date +%s); cp $F $SCR/data/$S.1fq || exit 2; echo "input copy to node-local disk: $(( $(date +%s) - t )) s" | tee -a $R

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
memsnap() {  # SMAPS=1: every ${SMAPS_EVERY:-20} s the cgroup's memory.stat and, per process above 4 GB RSS, its mappings above 1 GB (addr perms path rss pss anon, kB) and the rest summed
  local p r
  while :; do
    { echo "S $(date +%s)"; echo "C $(awk '$1=="anon"||$1=="file"||$1=="shmem"||$1=="file_mapped"||$1=="kernel"||$1=="pagetables"||$1=="slab"||$1=="unevictable"{printf "%s=%.1f ", $1, $2/1073741824}' $CG/memory.stat)"
      echo "M $(awk '/^(Shmem|Cached|Mapped|AnonPages|Mlocked|Unevictable|HugePages_Total|Hugepagesize):/{printf "%s%s ", $1, $2}' /proc/meminfo)"
      for p in $(mypids); do
        r=$(awk '/^VmRSS/{print $2}' /proc/$p/status 2>/dev/null); [ "${r:-0}" -gt 4000000 ] || continue
        echo "P $p $(cat /proc/$p/comm) ppid=$(awk '/^PPid/{print $2}' /proc/$p/status) rss_kB=$r"
        awk 'function fl() { if (hdr != "") { if (rss > 1048576) print "m", hdr, rss, pss, anon; else { oR += rss; oP += pss; oA += anon } } }
             /^[0-9a-f]+-[0-9a-f]+ / { fl(); hdr = $1 " " $2 " " (NF >= 6 ? $6 : "-"); rss = pss = anon = 0; next }
             /^Rss:/ { rss = $2 } /^Pss:/ { pss = $2 } /^Anonymous:/ { anon = $2 }
             END { fl(); print "o", oR, oP, oA }' /proc/$p/smaps 2>/dev/null
      done; } >> $A/smaps.agg
    sleep ${SMAPS_EVERY:-20}
  done
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
  cp $dd/read_stats.tsv $A/read_stats.c$d.$j.tsv 2>/dev/null
  du -sb $dd 2>/dev/null | cut -f1 > $A/bytes.c$d.$j
  nice -n 19 bash -c "$(declare -f hashset); hashset $dd" > $A/files.c$d.$j
  echo "c$d.$j $(md5sum < $A/files.c$d.$j | cut -c1-32)" > $A/h.c$d.$j
  rm -rf "${SCR:?}/run/c$d.$j"
}

stopholders; rm -f $A/MEMSTOP
for d in 0 1; do mkdir -p $A/pool$d; : > $SCR/run/queue$d
  for ((j = 0; j < N; j++)); do mkdir -p $SCR/run/c$d.$j; echo "$SCR/data/$S.1fq $SCR/run/c$d.$j/out/ --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf" >> $SCR/run/queue$d; done; done
echo "######## $TAG: 2 GPUs x K=$K in flight, T=$T, $N items per GPU, kill at ${LIM}s, $(date -Is)" | tee -a $R
nvidia-smi dmon -s pucm -d 1 -o DT > $A/dmon.txt 2>&1 &
sampler & SMP=$!
procsnap & PSN=$!
[ "${SMAPS:-0}" = 1 ] && { memsnap & MSN=$!; }
st=$(date +%s); pids=()
for d in 0 1; do
  PRE=""; [ "${NUMA:-0}" = 1 ] && PRE="numactl --cpunodebind=$d --preferred=$d"   # NUMA=1: the pool of GPU d and its samples on NUMA node d (cores, and memory by preference)
  setsid bash -c 'exec '"$PRE"' "$0" gpu-pool --device '$d' --queue '$SCR/run/queue$d' --slots '$SLOTS' --workers '$K' --threads '$T' --report '$A/pool$d' --timeout '$LIM' --args "--ref-base '$REF' --pipeline --output-format 1pz '"$PARGS"' --engine gpu"' "$BIN" > $A/pool$d/stdout.txt 2>&1 &
  pids+=($!)
done
fl=ok
while :; do
  alive=0; for p in "${pids[@]}"; do kill -0 $p 2>/dev/null && alive=1; done
  for d in 0 1; do for ((j = 0; j < N; j++)); do
    if [ -s $A/pool$d/$j.time ] && [ ! -e $A/reaped.c$d.$j ]; then : > $A/reaped.c$d.$j; reap $d $j & fi; done; done
  [ $alive = 0 ] && break
  el=$(( $(date +%s) - st ))
  if [ -f $A/MEMSTOP ]; then fl="MEMSTOP: $(cat $A/MEMSTOP)"; elif [ $el -gt $LIM ]; then fl="TIMEOUT ${el}s > ${LIM}s"
  elif [ $el -gt ${STALLSTOP:-999999} ] && ! ls $A/pool[01]/[0-9]*.time > /dev/null 2>&1; then fl="STALLED: no item finished in ${el}s"; fi
  if [ "$fl" != ok ]; then echo "  abort: $fl" | tee -a $R; killmine TERM "singlet-gpu"; sleep 20; killmine KILL "singlet-gpu"; break; fi
  sleep 5
done
wait "${pids[@]}" 2>/dev/null
en=$(date +%s); sleep 2; kill $SMP $PSN ${MSN:-} 2>/dev/null; killmine TERM "nvidia-smi dmon"
for d in 0 1; do for ((j = 0; j < N; j++)); do [ -e $A/reaped.c$d.$j ] || { : > $A/reaped.c$d.$j; reap $d $j & }; done; done
for p in $(jobs -p); do [ "$p" = "$SMP" ] || [ "$p" = "$PSN" ] || [ "$p" = "${MSN:-}" ] || wait $p 2>/dev/null; done

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
cat $A/h.c* > $A/hashes.txt 2>/dev/null
nh=$(wc -l < $A/hashes.txt); nu=$(awk '{print $2}' $A/hashes.txt | sort -u | wc -l)
echo "  IDENT $TAG: $nh outputs hashed, $nu distinct; set hash $(awk 'NR==1{print $2}' $A/hashes.txt); files per set $(wc -l < $A/files.c0.0 2>/dev/null), missing $(grep -c '^missing' $A/files.c0.0 2>/dev/null)" | tee -a $R
[ -n "${REFSET:-}" ] && { if [ "$(awk '{print $2}' $A/hashes.txt | sort -u)" = "$REFSET" ]; then echo "  IDENT $TAG: every output equals the reference set $REFSET" | tee -a $R; else echo "  IDENT MISMATCH $TAG vs reference set $REFSET" | tee -a $R; fi; }
cp $A/files.c0.0 $O/ref-files.txt 2>/dev/null
echo "done $(date -Is) elapsed $(( $(date +%s) - T0JOB )) s" | tee -a $R
