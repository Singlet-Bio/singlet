#!/bin/bash
#SBATCH --job-name=e21-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:55:00
#SBATCH --nice=10000
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E21 (docs/gpu-engine/fable/E21-pool.md): `singlet gpu-pool` against the samples run alone, in E18's form (bench/fable/throughput/jobs/run.sbatch:
# the same command line and environment, level 5 + ingest, the sampler, the per-run files and E18's analysis), with E19's gate checks
# (bench/fable/release/jobs/gate.sbatch cmp2) per run. Binary of bench/fable/pool/jobs/build.sbatch in $BINDIR (singlet-gpu).
#
# POINTS="tag ..." run in order. A tag is <form>t<T>s<S>[n0][g0][_name]:<items>, items = comma-separated samples, sample*N = N copies:
#   a          alone: each item on its own, one after the other, with the plain command (E18's run_item; pinned to device 0); the first
#              successful run of a sample at T is the job's reference for that sample and T
#   p<P>k<K>   P pools (`singlet gpu-pool --device d`, d = 0 .. P-1, all at once), K samples in flight each; item i goes to pool i mod P
#   t<T> s<S>  --threads T, SINGLET_ENGINE_GPU_SLOTS=S;   n0: --nonhost-server off;   g0: --genome-preload off;   _name: a distinct tag
# Per point $RES/<tag>/: meta.json, runs.tsv (item, worker = d*K + the pool's slot, sample, start, end, exit), samples.jsonl
# (bench/fable/pool/e21_sampler.py: E18's sampler, processes by the pools' pids.tsv), per run c<item>/ (time.txt, run.log.ts, star_Log.out,
# star_Log.final.out, CellReads.stats, read_stats.tsv, summary.json, pileup_stats.json, readhash.txt, identity.txt), per pool pool<d>/ (the
# pool's report: pool.log, samples.tsv, pids.tsv, nhsrv.log, <j>.log, <j>.time; gmap.tsv: j -> item), the analysis (e18_point.py,
# e18_locks.py, e18_steady.py, e21_mem.py -> point.txt) and identity.txt.
# Identity, per pool run (exit 0) against (1) the job's alone run of its sample at the same T ($SCR/ref/<sample>-t<T>, REF_<sample>_t<T>=<dir>
# for a kept one) and (2) E19's kept CPU reference ($CPUREFD/<sample>: out/, readhash.txt, CellReads.stats; IDCPU=0 skips): out_equiv.py with
# E11's run-dependent files (EXPECT_DIFF), the 10 Solo files, star_Log.final.out's statistics, SJ.out.tab, the read files by sorted records
# (KNOWN_nuclei2k_v2: E15's non-host writer bug, same count) and CellReads.stats by sorted rows. VERDICT lines in summary.txt, verdicts.tsv.
# KILLTEST=<tag>:<item>:<s>: SIGKILL that item's process <s> s after it starts (a crashed sample: the pool must record it and go on).
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,POINTS="at4s2:pbmc2m,pbmc10m p1k2t4s2:pbmc2m,pbmc10m" run.sbatch
# node:      sbatch --gres=gpu:nvidia_h100_nvl:2 -c 54 --mem=730G --export=ALL,POINTS="at7s2:pbmc1k_v3 p2k8t7s2:pbmc1k_v3*32" run.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/pool/bin}; BIN=$BINDIR/singlet-gpu
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/pool}
set -f; POINTS=(${POINTS:-at7s2:pbmc1k_v3 p1k4t7s2:pbmc1k_v3*4}); set +f   # no globbing of the tags' "*"
CPUREFD=${CPUREFD:-$W/jobs/fable/release/ref/T18}; DATA2=$W/jobs/fable/release/data
IDENT=${IDENT:-1}; IDCPU=${IDCPU:-1}; IDPAR=${IDPAR:-6}; SDT=${SDT:-0.5}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES0=${RES0:-$W/jobs/fable/pool/results/run-$SLURM_JOB_ID}; mkdir -p $RES0
SCR=/tmp/$USER-e21-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data $SCR/ref
trap 'kill $(jobs -p) 2>/dev/null; sleep 2; rm -rf "${SCR:?}"' EXIT INT TERM
for i in 1 2 3; do git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/pool-run && break; sleep 5; done
git -C $BARE archive refs/fable/pool-run bench/star_accel/out_equiv.py bench/fable/throughput bench/fable/pool | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
NGPU=$(nvidia-smi -L | wc -l)
TB=$SCR/repo/bench/fable/throughput; PO=$SCR/repo/bench/fable/pool
S=$RES0/summary.txt
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK gpus=$NGPU bin=$BIN commit=$(cat $BINDIR/commit-gpu 2>/dev/null) ($(sha256sum $BIN | cut -c1-16)) scripts=$(git -C $BARE rev-parse --short=8 refs/fable/pool-run) points=${POINTS[*]} $(date -Is)" | tee $S
{ nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)|^NUMA node[0-9]"; free -g | head -2
  df -h /tmp /dev/shm | tail -n +1; echo "cgroup $CG memory.max $(cat $CG/memory.max 2>/dev/null)"; uname -r
  cat /sys/kernel/mm/transparent_hugepage/enabled; ls -l /tmp/singlet_engine_gpu_*.lock 2>/dev/null; ls -l /dev/shm/singlet_nonhost $REF/nonhost 2>&1 | head -12; } 2>&1 | sed 's/^/  /' | tee -a $S

genome() { case $1 in neuron1k_v3|nuclei2k_v2) echo $REF/GRCm39-2024-A ;; hgmm1k_v3) echo $REF/GRCh38-GRCm39-barnyard ;; *) echo $REF/GRCh38-2024-A ;; esac; }
fq_of() { local f=$PB/data/$1/$1.1fq; [ -f $f ] || f=$DATA2/$1/$1.1fq; echo $f; }
stage_input() { [ -f $SCR/data/$1.1fq ] || { cp $(fq_of $1) $SCR/data/$1.1fq.part && mv $SCR/data/$1.1fq.part $SCR/data/$1.1fq; }; }
expand() { local x j xs; IFS=, read -ra xs <<< "$1"; for x in "${xs[@]}"; do if [[ $x == *'*'* ]]; then for ((j = 0; j < ${x##*\*}; j++)); do echo "${x%%\**}"; done; else echo "$x"; fi; done; }

# ---- identity (E19's gate cmp2, for two out/ dirs whose read files were hashed and set aside)
EXPECT_DIFF=" ${EXPECT_DIFF:-pileup_stats.json read_stats.tsv star_Log.std.out summary.json} "
KNOWN_nuclei2k_v2=${KNOWN_nuclei2k_v2-nonhost_reads.fq}
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }
# the files E19's gate leaves out of out_equiv (its rsync EXCL list): removed from a run's out/ once its read files are hashed
strip() { find "$1" \( -name '*.bam' -o -name '*.sam' -o -name '*.cram' -o -name '*.fq*' -o -name '*.fastq*' -o -name '*.1fq' -o -name '.gpu_handoff*' \
                     -o -name '*Unmapped*' -o -name '_STARtmp*' -o -name '*.reads' -o -name '*.reads.q' \) -prune -exec rm -rf {} + 2>/dev/null; }
# cmpout NAME SAMPLE A_OUT A_READHASH A_CRS B_OUT B_READHASH B_CRS FILE: the checks into FILE, the verdict line on stdout
cmpout() {
  local nm=$1 s=$2 a=$3 ra=$4 ca=$5 b=$6 rb=$7 cb=$8 F=$9 ok=1 kn="" FL UNX f ns ha hb
  echo "=== $nm: $a vs $b" >> $F
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $F.equiv-$nm 2>&1
  grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $F.equiv-$nm | cut -c1-200 | head -40 >> $F
  FL=$(grep -a '^files:' $F.equiv-$nm | sed -E 's/files: ([0-9]+) identical, ([0-9]+) differ only in timings\/paths, ([0-9]+) differ/\1\/\2\/\3/')
  UNX=""; for f in $(grep -a '^  DIFF ' $F.equiv-$nm | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac; done
  grep -aq '^  DIFF provenance.json: /references/whitelist/name: [^;]*$' $F.equiv-$nm && UNX=${UNX/ provenance.json/} && echo "  provenance.json: the whitelist's path only (the binaries' directories)" >> $F
  for f in $(grep -a '^  only in ' $F.equiv-$nm | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
  [ -n "$FL" ] || UNX="$UNX out_equiv-failed"
  if [ -n "$UNX" ]; then echo "  out_equiv: UNEXPECTED:$UNX" >> $F; ok=0; else echo "  out_equiv: $FL (identical/timing-path/differ), every DIFF in E11's run-dependent set" >> $F; fi
  ns=0
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
           Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    if cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f; then ns=$((ns + 1)); else echo "  star_Solo.out/$f: DIFFERS" >> $F; ok=0; fi
  done
  echo "  star_Solo.out: $ns of 10 files identical" >> $F
  if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then echo "  star_Log.final.out: mapping statistics identical" >> $F
  else echo "  star_Log.final.out: MAPPING STATISTICS DIFFER" >> $F; ok=0; fi
  if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then echo "  star_SJ.out.tab: identical ($(wc -l < $a/star_SJ.out.tab) junctions)" >> $F; else echo "  star_SJ.out.tab: DIFFERS" >> $F; ok=0; fi
  if [ -f "$ra" ] && [ -f "$rb" ]; then
    local kv=KNOWN_$s; local KN=" ${!kv:-} "
    for f in $READF; do ha=$(awk -v f=$f '$1==f{print $2" "$3}' $ra); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rb)
      if [ "$ha" = "$hb" ]; then echo "  $f: same records ($ha)" >> $F
      elif [[ "$KN" == *" $f "* ]] && [ -n "$ha" ] && [ "${ha##* }" = "${hb##* }" ]; then echo "  $f: other records, same count (E15's known non-host writer bug: $ha vs $hb)" >> $F; kn="$kn $f"
      else echo "  $f: RECORDS DIFFER ($ha vs $hb)" >> $F; ok=0; fi
    done
  else echo "  read files: NOT COMPARED (no sorted hashes)" >> $F; ok=0; fi
  if [ -f "$ca" ] && [ -f "$cb" ]; then
    if cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then
      echo "  Gene/CellReads.stats: same rows ($(( $(wc -l < $ca) - 1 )) rows; byte-identical: $(cmp -s $ca $cb && echo yes || echo no))" >> $F
    else echo "  Gene/CellReads.stats: ROWS DIFFER" >> $F; ok=0; fi
  else echo "  Gene/CellReads.stats: NOT COMPARED (no copy)" >> $F; ok=0; fi
  if [ $ok = 1 ]; then echo "IDENTICAL vs $nm (except E11's run-dependent files${kn:+; the known non-host writer bug of E15 in$kn}; out_equiv $FL)"
  else echo "NOT IDENTICAL vs $nm (see identity.txt)"; fi
}

# one pool run against the references
ident() {   # point item sample T
  local p=$1 i=$2 s=$3 T=$4 b=$SCR/$1/c$2/out O=$RES0/$1/c$2 v A V
  readhash $b > $O/readhash.txt; strip $b
  : > $O/identity.txt; : > $O/verdict.txt
  v=REF_${s}_t$T; A=${!v:-$SCR/ref/$s-t$T}
  if [ -d $A/out ]; then V=$(cmpout alone-t$T $s $A/out $A/readhash.txt $A/CellReads.stats $b $O/readhash.txt $O/CellReads.stats $O/identity.txt); echo "alone-t$T $V" >> $O/verdict.txt; fi
  if [ "$IDCPU" = 1 ] && [ -f $CPUREFD/$s/DONE ]; then
    V=$(cmpout cpu $s $CPUREFD/$s/out $CPUREFD/$s/readhash.txt $CPUREFD/$s/CellReads.stats $b $O/readhash.txt $O/CellReads.stats $O/identity.txt); echo "cpu $V" >> $O/verdict.txt
  fi
  [ -s $O/verdict.txt ] || echo "none NO REFERENCE (no alone run at T=$T, no CPU reference)" >> $O/verdict.txt
  rm -rf $b
}

# one sample alone (E18's run_item: the plain command, pinned to device 0)
run_alone() {   # point item sample T
  local p=$1 i=$2 s=$3 T=$4 D=$SCR/$1/c$2 O=$RES0/$1/c$2 G rc T0 T1 WATCH
  G=$(genome $s); mkdir -p $D/out $O
  local CRS=$D/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $D/CellReads.stats ) 2>/dev/null &
  WATCH=$!
  T0=$(date +%s.%N)
  ( cd $D && env E18_COPY=c$i SINGLET_ENGINE_GPU_DEVICE=0 /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $SCR/data/$s.1fq --ref-base $REF \
      --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 \
      | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  rc=$(awk -F': ' '/Exit status/{print $2}' $O/time.txt 2>/dev/null); rc=${rc:-1}
  T1=$(date +%s.%N); kill $WATCH 2>/dev/null
  printf "%d\t%d\t%s\t%s\t%s\t%d\n" $i 0 $s $T0 $T1 $rc >> $RES0/$p/runs.tsv
  cp $D/out/*Log.final.out $D/out/*Log.out $D/out/read_stats.tsv $D/out/summary.json $D/out/pileup_stats.json $O/ 2>/dev/null
  [ -f $D/CellReads.stats ] && cp $D/CellReads.stats $O/CellReads.stats
  if [ "$rc" = 0 ] && [ ! -d $SCR/ref/$s-t$T ]; then
    mkdir -p $SCR/ref/$s-t$T; readhash $D/out > $SCR/ref/$s-t$T/readhash.txt; cp $SCR/ref/$s-t$T/readhash.txt $O/; strip $D/out
    mv $D/out $SCR/ref/$s-t$T/out; cp $O/CellReads.stats $SCR/ref/$s-t$T/ 2>/dev/null
    echo "  c$i ($s): exit 0, wall $(echo "$T1 - $T0" | bc | cut -c1-6) s; kept as the reference for $s at T=$T" | tee -a $S
  else echo "  c$i ($s): exit $rc, wall $(echo "$T1 - $T0" | bc | cut -c1-6) s" | tee -a $S; fi
  rm -rf $D
}

for p in "${POINTS[@]}"; do
  if ! [[ $p =~ ^(a|p([0-9]+)k([0-9]+))t([0-9]+)s([0-9]+)(n0)?(g0)?(_[A-Za-z0-9]+)?:(.+)$ ]]; then echo "!! bad point tag $p" | tee -a $S; continue; fi
  FORM=${BASH_REMATCH[1]}; P=${BASH_REMATCH[2]:-1}; K=${BASH_REMATCH[3]:-1}; T=${BASH_REMATCH[4]}; SL=${BASH_REMATCH[5]}; NH=${BASH_REMATCH[6]}; GP=${BASH_REMATCH[7]}
  ITEMS=($(expand "${BASH_REMATCH[9]}")); p=${p%%:*}
  if [ $FORM != a ] && [ $P -gt $NGPU ]; then echo "!! $p: $P pools but $NGPU GPUs" | tee -a $S; continue; fi
  # MEMGUARD=1 (default): skip a pool point whose host memory, predicted from the last pool point's mem.json (fixed per pool + per
  # sample in flight at its peak; only from a point of the same samples), would exceed 92 % of the cgroup's limit
  US=$(printf "%s\n" "${ITEMS[@]}" | sort -u | tr '\n' ' ')   # the point's samples (the guard compares like with like)
  if [ $FORM != a ] && [ "${MEMGUARD:-1}" = 1 ] && [ -n "${LASTMEM:-}" ] && [ -f "$LASTMEM" ] && [ "$US" = "${LASTMEMS:-}" ]; then
    PRED=$(python3 -c "import json; m=json.load(open('$LASTMEM')); f=m['fixed_gb']/max(1,len(m['fixed'])); s=max(m.get('per_sample_at_peak_gb') or 0, m.get('per_sample_window_gb') or 0); print(round($P*f + $((P * K))*s))" 2>/dev/null)
    LIM=$(( $(cat $CG/memory.max 2>/dev/null || echo 0) / 1073741824 ))
    if [ -n "$PRED" ] && [ "$LIM" -gt 0 ] && [ "$PRED" -gt $((LIM * 92 / 100)) ]; then
      echo "######## $p: SKIPPED by MEMGUARD: predicted $PRED GB (from $LASTMEM) > 92 % of the cgroup's $LIM GB" | tee -a $S; continue
    fi
    [ -n "$PRED" ] && echo "  $p: predicted host peak $PRED GB of the cgroup's $LIM GB (from $LASTMEM)" | tee -a $S
  fi
  for s in $(printf "%s\n" "${ITEMS[@]}" | sort -u); do stage_input $s; done
  unset SINGLET_ENGINE_GPU_SLOTS SINGLET_ENGINE_GPU_DEVICE
  export SINGLET_ENGINE_GPU_SLOTS=$SL
  mkdir -p $SCR/$p $RES0/$p; : > $RES0/$p/runs.tsv
  [ $FORM = a ] && W_=1 || W_=$((P * K))
  printf '{"tag":"%s","form":"%s","k":%d,"pools":%d,"k_per_pool":%d,"T":%d,"slots":%d,"nonhost_server":"%s","genome_preload":"%s","items":"%s","env":"SINGLET_ENGINE_GPU_SLOTS=%d","node":"%s","gpus":%d}\n' \
    $p $FORM $W_ $P $K $T $SL "$([ -n "$NH" ] && echo off || echo on)" "$([ -n "$GP" ] && echo off || echo on)" "${ITEMS[*]}" $SL "$(hostname)" $NGPU > $RES0/$p/meta.json
  echo "######## $p: $([ $FORM = a ] && echo "alone, one after the other" || echo "$P pool(s) x $K in flight${NH:+, non-host server off}${GP:+, genome preload off}"), T=$T, slots $SL, ${#ITEMS[@]} items: ${ITEMS[*]}; $(date -Is)" | tee -a $S
  REPS=(); [ $FORM != a ] && for ((d = 0; d < P; d++)); do REPS+=($RES0/$p/pool$d); mkdir -p $RES0/$p/pool$d; : > $RES0/$p/pool$d/gmap.tsv; : > $SCR/$p/queue$d; done
  # the queues (absolute paths; the pool runs each sample in its out dir's parent, as E18's runs)
  if [ $FORM != a ]; then
    for ((i = 0; i < ${#ITEMS[@]}; i++)); do
      s=${ITEMS[$i]}; d=$((i % P)); j=$((i / P)); G=$(genome $s); mkdir -p $SCR/$p/c$i $RES0/$p/c$i
      echo "$SCR/data/$s.1fq $SCR/$p/c$i/out/ --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf" >> $SCR/$p/queue$d
      printf "%d\t%d\n" $j $i >> $RES0/$p/pool$d/gmap.tsv
    done
    for ((d = 0; d < P; d++)); do cp $SCR/$p/queue$d $RES0/$p/pool$d/queue.txt; done
  fi
  python3 $PO/e21_sampler.py $RES0/$p/samples.jsonl $CG $SDT 4 2 ${REPS[@]+"${REPS[@]}"} 2>$RES0/$p/sampler.err &
  SMP=$!
  sleep 1
  if [ $FORM = a ]; then
    for ((i = 0; i < ${#ITEMS[@]}; i++)); do run_alone $p $i ${ITEMS[$i]} $T; done
  else
    # CellReads.stats is deleted by the pipeline after it parses it: hard-link each the moment it appears (E18's watcher, for all items)
    ( n=${#ITEMS[@]}
      while :; do left=0
        for ((i = 0; i < n; i++)); do [ -e $SCR/$p/c$i/CellReads.stats ] && continue; left=1
          [ -e $SCR/$p/c$i/out/star_Solo.out/Gene/CellReads.stats ] && ln $SCR/$p/c$i/out/star_Solo.out/Gene/CellReads.stats $SCR/$p/c$i/CellReads.stats 2>/dev/null; done
        [ $left = 0 ] && break; sleep 0.2; done ) &
    WATCH=$!
    if [ -n "${KILLTEST:-}" ] && [ "${KILLTEST%%:*}" = "$p" ]; then
      ( ki=$(echo $KILLTEST | cut -d: -f2); ks=$(echo $KILLTEST | cut -d: -f3); d=$((ki % P)); j=$((ki / P)); kp=""
        while [ -z "$kp" ]; do kp=$(awk -F'\t' -v l=c$j '$2==l{print $1}' $RES0/$p/pool$d/pids.tsv 2>/dev/null); sleep 0.5; done
        sleep $ks; kill -9 $kp && echo "  KILLTEST: item $ki (pool $d, pid $kp) killed with SIGKILL after $ks s, $(date -Is)" | tee -a $S ) &
    fi
    PP=()
    for ((d = 0; d < P; d++)); do
      $BIN gpu-pool --device $d --queue $SCR/$p/queue$d --slots $SL --workers $K --threads $T --report $RES0/$p/pool$d --timeout 3000 \
        --args "--ref-base $REF --pipeline --output-format 1pz --engine gpu" ${NH:+--nonhost-server off} ${GP:+--genome-preload off} \
        > $RES0/$p/pool$d/stdout.txt 2>&1 &
      PP+=($!)
    done
    for ((d = 0; d < P; d++)); do wait ${PP[$d]}; echo "  pool $d: exit $?; $(grep -a 'queue drained' $RES0/$p/pool$d/pool.log | tail -1 | cut -d' ' -f4-)" | tee -a $S; done
    sleep 1; kill $WATCH 2>/dev/null
    for ((d = 0; d < P; d++)); do
      grep -a -e "genome " -e "non-host server" -e "ready in" $RES0/$p/pool$d/pool.log | cut -d' ' -f4- | cut -c1-400 | sed "s/^/  pool $d: /" | tee -a $S
      grep -a -e "resident" -e "context on" -e "declined" -e "start-up" -e "stops" $RES0/$p/pool$d/nhsrv.log 2>/dev/null | head -5 | cut -c1-300 | sed "s/^/  pool $d nhsrv: /" | tee -a $S
      [ -f $RES0/$p/pool$d/nhsrv.log ] && echo "  pool $d nhsrv: $(grep -ac '^\[nhsrv\] request' $RES0/$p/pool$d/nhsrv.log 2>/dev/null) requests ($(grep -ac 'index kept' $RES0/$p/pool$d/nhsrv.log 2>/dev/null) with the index kept on the device)" | tee -a $S
      # E18's files per run, from the pool's report
      awk -F'\t' -v d=$d -v K=$K 'FNR==NR{g[$1]=$2; next} FNR>1{n=split($3,a,"/"); s=a[n]; sub(/\.1fq$/,"",s); rc=($12>=0)?$12:128+$13;
           printf "%s\t%d\t%s\t%s\t%s\t%d\n", g[$1], d*K+$2, s, $5, $6, rc}' $RES0/$p/pool$d/gmap.tsv $RES0/$p/pool$d/samples.tsv >> $RES0/$p/runs.tsv
      while IFS=$'\t' read -r j i; do
        D=$SCR/$p/c$i; O=$RES0/$p/c$i
        cp $RES0/$p/pool$d/$j.log $O/run.log.ts 2>/dev/null; cp $RES0/$p/pool$d/$j.time $O/time.txt 2>/dev/null
        cp $D/out/*Log.final.out $D/out/*Log.out $D/out/read_stats.tsv $D/out/summary.json $D/out/pileup_stats.json $O/ 2>/dev/null
        [ -f $D/CellReads.stats ] && cp $D/CellReads.stats $O/CellReads.stats
      done < $RES0/$p/pool$d/gmap.tsv
    done
    sort -n -o $RES0/$p/runs.tsv $RES0/$p/runs.tsv
  fi
  sleep 1; kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
  du -sh $SCR/$p 2>/dev/null | sed 's/^/  \/tmp use of the point: /' | tee -a $S; df -h /tmp | tail -1 | sed 's/^/  /' | tee -a $S
  python3 $TB/e18_point.py $RES0/$p > $RES0/$p/point.txt 2>&1
  python3 $TB/e18_locks.py $RES0/$p > $RES0/$p/locks.txt 2>&1
  python3 $TB/e18_steady.py $RES0/$p 2>&1 | sed 's/^/   /' >> $RES0/$p/point.txt
  python3 $PO/e21_mem.py $RES0/$p >> $RES0/$p/point.txt 2>&1
  [ $FORM != a ] && [ -f $RES0/$p/mem.json ] && LASTMEM=$RES0/$p/mem.json && LASTMEMS=$US
  cut -c1-400 $RES0/$p/point.txt | tee -a $S
  # identity of every pool run (exit 0) against its references
  if [ $FORM != a ] && [ "$IDENT" = 1 ]; then
    while IFS=$'\t' read -r i w s t0 t1 rc; do
      if [ "$rc" != 0 ]; then echo "exit $rc: not compared" > $RES0/$p/c$i/verdict.txt; rm -rf $SCR/$p/c$i/out; continue; fi
      while [ $(jobs -rp | wc -l) -ge $IDPAR ]; do sleep 1; done
      ident $p $i $s $T &
    done < $RES0/$p/runs.tsv
    wait
    while IFS=$'\t' read -r i w s t0 t1 rc; do
      [ -f $RES0/$p/c$i/identity.txt ] && { echo "######## c$i ($s)"; cat $RES0/$p/c$i/identity.txt; } >> $RES0/$p/identity.txt
      while read -r ref verdict; do
        echo "VERDICT $p c$i $s: $verdict" | tee -a $S
        printf "%s\tc%s\t%s\t%s\t%s\n" $p $i $s $ref "$verdict" >> $RES0/verdicts.tsv
      done < $RES0/$p/c$i/verdict.txt
    done < $RES0/$p/runs.tsv
  fi
  rm -rf "${SCR:?}/${p:?}"
done
echo "######## verdicts" | tee -a $S
if [ -f $RES0/verdicts.tsv ]; then
  awk -F'\t' '{split($5,a," "); v=a[1]=="IDENTICAL"?"IDENTICAL":($5 ~ /^NOT IDENTICAL/?"NOT-IDENTICAL":"other"); n[$1" vs "$4" "v]++} END{for (k in n) print "  "k": "n[k]}' $RES0/verdicts.tsv | sort | tee -a $S
fi
python3 $TB/e18_point.py --table $RES0 > $RES0/table.md 2>&1; tee -a $S < $RES0/table.md
echo done $(date -Is) | tee -a $S
