#!/bin/bash
#SBATCH --job-name=gbold-tp
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:55:00
#SBATCH --exclude=g050
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU BOLD copy of the batch 1 throughput harness (bench/gpu_batch1/tp.sbatch at 0fdec57f), for STAGE CEILING runs of gpu/bold.
# Differences: results under $W/jobs/gpu-bold, identity off by default (ceiling runs change outputs on purpose), per-run outputs deleted at once,
# point tags may carry a config: k8t7s2r3@alloff (configs: base nodemux nonh nodn alloff maponly, see cfg_set below), nvidia-smi dmon per point.
# Never compare or keep outputs of a ceiling run.
# Original header follows.
# Throughput harness of GPU batch 1 (bench/gpu_batch1/README.md). It started as tp-run.sbatch (E18, docs/gpu-engine/fable/E18-throughput.md):
# several samples in flight on the job's GPU, level 5 + ingest, the engine's device slots.
#
# Parameters (sbatch --export=ALL,NAME=value,...):
#   BINDIR         required. Directory with singlet-gpu and commit-gpu. Never taken from a default.
#   EXPECT_COMMIT  required. The job stops unless commit-gpu in BINDIR starts with this (8 or more hex digits).
#   POINTS         tags in order. Tag = [mix]k<K>t<T>s<S>[m<GB>][r<R>][p][f]. k: runs in flight. t: threads per run. s: device slots.
#                  r: rounds, K*R runs in the queue (default 1). m43: device memory split. p: pin worker to GPU. f: fine sampling.
#                  k8t7s2r5 is 40 runs with 8 in flight.
#   NRUNS          optional. Total runs in the queue of every point. It replaces K*R. NRUNS=40 equals k8 with r5.
#   SAMPLE         sample of the queue (default pbmc1k_v3). MIX: the samples of a mix tag.
#   CPU_REF        directory with the CPU references of this binary, as written by gate.sbatch.
#                  Default $W/jobs/gpu-batch1/ref/<commit8>. Run <CPU_REF>/T<T>/<sample>/out is the reference.
#   SCRIPTS_BRANCH branch of the scripts (default bench/gpu-batch1). EXPECT_SCRIPTS: optional commit prefix that the branch tip must have.
#   DUMP, IDENT, IDPAR: as in E18 (DUMP=1 hashes mapDistinct records of item 0, IDENT=0 skips identity, IDPAR comparisons in parallel).
#   RES0           results directory. Default $W/jobs/gpu-batch1/results/tp-<job>. It must lie under $W/jobs/gpu-batch1.
#
# What it does beyond E18.
#  1. Provenance: the first line of the log and every results.tsv row carry job, node, binary, sha256, build commit, scripts commit, T.
#  2. Windowed rate and the whole-run measures in <point>/results.tsv (tp_point.py). The old span rate and steady number are still printed.
#  3. Fallback runs are counted (fallback.py allow-list, level 5, GPU pileup). The point prints "POINT FAILED: n fallback runs". The job goes on.
#  4. Identity of every run against a run alone at the same T and slots, and against the CPU reference. Each run's outputs are deleted after the comparison.
#  5. Co-tenants: squeue of the node at the start and end of each point and once a minute.
#
# Alone run: before a point the job runs each sample of the point once alone (one run, the point's T and slots) unless it has done so already.
# Its outputs are kept in node-local /tmp, in pruned form, until the job ends.
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,BINDIR=...,EXPECT_COMMIT=...,SAMPLE=pbmc2m,POINTS="k2t4s1r3" tp.sbatch
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:?BINDIR: the directory with singlet-gpu and commit-gpu}; BIN=$BINDIR/singlet-gpu
EXPECT_COMMIT=${EXPECT_COMMIT:?EXPECT_COMMIT: the commit that BINDIR must hold}
[ ${#EXPECT_COMMIT} -ge 8 ] || { echo "EXPECT_COMMIT needs 8 or more hex digits"; exit 2; }
DATA2=$W/jobs/fable/release/data
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-batch1}; EXPECT_SCRIPTS=${EXPECT_SCRIPTS:-0fdec57f}
POINTS=(${POINTS:-k1t28s1 k2t14s1 k3t9s1 k4t7s1}); SAMPLE=${SAMPLE:-pbmc1k_v3}; MIX=(${MIX:-pbmc1k_v3 pbmc5p_v1 neuron1k_v3 nuclei2k_v2})
DUMP=${DUMP:-0}; IDENT=${IDENT:-0}; NRUNS=${NRUNS:-}
RES0=${RES0:-$W/jobs/gpu-bold/results/tp-$SLURM_JOB_ID}
case $RES0 in $W/jobs/gpu-bold/*) ;; *) echo "RES0 must lie under $W/jobs/gpu-bold"; exit 2 ;; esac
mkdir -p $RES0; S=$RES0/summary.txt

# --- provenance of the binary: the build record must match EXPECT_COMMIT -------------------------------------------------------------
[ -f $BIN ] || { echo "no binary $BIN"; exit 2; }
COMMIT=$(cat $BINDIR/commit-gpu 2>/dev/null || true)
BSHA=$(sha256sum $BIN | cut -d' ' -f1)
case $COMMIT in $EXPECT_COMMIT*) ;; *) echo "REFUSED: $BINDIR records commit '${COMMIT:-none}', EXPECT_COMMIT is $EXPECT_COMMIT" | tee $S; exit 2 ;; esac
EMBED=no; grep -aqF "$COMMIT" $BIN && EMBED=yes
[ $EMBED = yes ] || [ "${ALLOW_NO_EMBED:-0}" = 1 ] || { echo "REFUSED: the commit $COMMIT is not inside the binary (ALLOW_NO_EMBED=1 skips this check)" | tee $S; exit 2; }
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists

# --- scripts from the bench/gpu-batch1 branch, one fetch at a time, with a commit check (as build.sbatch does) ------------------------
SCR=/tmp/$USER-gbtp-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data $SCR/ref
SREF=refs/gbold/tp-$SLURM_JOB_ID
trap 'kill $(jobs -p) 2>/dev/null; git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$SBR | cut -f1); [ -n "$WANT" ] || { echo "no scripts branch $SBR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
SFULL=$(git -C $BARE rev-parse $SREF); [ "$SFULL" = "$WANT" ] || { echo "fetched $SFULL, branch tip is $WANT"; exit 3; }
[ -z "${EXPECT_SCRIPTS:-}" ] || [[ $SFULL == $EXPECT_SCRIPTS* ]] || { echo "scripts commit $SFULL, EXPECT_SCRIPTS is $EXPECT_SCRIPTS"; exit 3; }
SCRIPTS=${SFULL:0:8}
git -C $BARE archive $SREF bench/star_accel/out_equiv.py bench/fable/throughput bench/gpu_batch1 | tar -x -C $SCR/repo
GB=$SCR/repo/bench/gpu_batch1; TPD=$SCR/repo/bench/fable/throughput
export PROV_JOB=$SLURM_JOB_ID PROV_NODE=$(hostname -s) PROV_BIN=$BIN PROV_SHA256=$BSHA PROV_COMMIT=$COMMIT PROV_SCRIPTS=$SCRIPTS

CPU_REF=${CPU_REF:-$W/jobs/gpu-batch1/ref/${COMMIT:0:8}}
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
NGPU=$(nvidia-smi -L | wc -l)
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$BSHA commit=$COMMIT expect=$EXPECT_COMMIT embedded=$EMBED scripts=$SCRIPTS T=points:${POINTS[*]} $(date -Is)" | tee $S
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$SLURM_CPUS_PER_TASK gpus=$NGPU cpu_ref=$CPU_REF points=${POINTS[*]} nruns=${NRUNS:-by-tag} sample=$SAMPLE" | tee -a $S
{ nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)|^NUMA node[0-9]"; free -g | head -2
  df -h /tmp /dev/shm | tail -n +1; findmnt -n -o SOURCE,FSTYPE,SIZE /tmp 2>/dev/null; echo "cgroup $CG memory.max $(cat $CG/memory.max 2>/dev/null)"
  cat /sys/kernel/mm/transparent_hugepage/enabled; ls -l /tmp/singlet_engine_gpu_*.lock 2>/dev/null; } 2>&1 | sed 's/^/  /' | tee -a $S

# stage ceiling configs. CFG_ARGS go on the singlet command line, CFG_ENV are exported for the point. Outputs are incomplete or changed on purpose.
start_mps() {   # CUDA MPS daemon for the rest of the job (kernels of different processes can then run at the same time)
  [ -n "${MPS_ON:-}" ] && return 0
  export CUDA_MPS_PIPE_DIRECTORY=/tmp/gbmps-$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=$RES0/mps-log; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  nvidia-cuda-mps-control -d && MPS_ON=1; echo "MPS daemon started: $CUDA_MPS_PIPE_DIRECTORY ($(date -Is))" | tee -a $S
}
stop_mps() {
  [ -z "${MPS_ON:-}" ] && return 0
  echo quit | nvidia-cuda-mps-control; MPS_ON=""; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; sleep 3; echo "MPS daemon stopped ($(date -Is))" | tee -a $S
}
cfg_set() {
  case $1 in *mps*) ;; *) stop_mps ;; esac
  CFG_ARGS=""; CFG_ENV=""
  case $1 in
    base) ;;
    nodemux) CFG_ARGS="--n-donors 1" ;;                                                  # auto-K and VB skipped (K=1 synthesized)
    nonh) CFG_ARGS="--no-viral-screen --no-microbial-screen" ;;                          # non-host screen and EM skipped
    nodn) CFG_ARGS="--n-donors 1 --no-viral-screen --no-microbial-screen" ;;
    alloff) CFG_ARGS="--n-donors 1 --no-viral-screen --no-microbial-screen"; CFG_ENV="SINGLET_BENCH_SKIP=ambient,doublet SINGLET_CR4_PROBE=skip" ;;
    maponly) CFG_ARGS="--no-viral-screen --no-microbial-screen"; CFG_ENV="SINGLET_BENCH_EXIT_AFTER=star SINGLET_CR4_PROBE=skip" ;;  # ends after pileup + STAR
    adm) CFG_ENV="SINGLET_GPU_ADMIT=1" ;;                                                 # gpu/bold-admit build b2: memory-aware admission instead of slot locks
    admmps) CFG_ENV="SINGLET_GPU_ADMIT=1"; start_mps ;;                                  # the same with CUDA MPS (concurrent kernels of several processes)
    admtuned) CFG_ENV="SINGLET_GPU_ADMIT=1 SINGLET_ADMIT_CAP_GB=92.5 SINGLET_ADMIT_IDLE_GB=0 SINGLET_ENGINE_GPU_MEM_GB=36 SINGLET_ADMIT_EST_PREFILL_GB=36 SINGLET_GPU_PILEUP_ARENA_GB=28 SINGLET_ADMIT_EST_PILEUP_GB=28 SINGLET_ADMIT_EST_NONHOST_GB=29 SINGLET_ADMIT_EST_HANDOFF_GB=6 SINGLET_ADMIT_EST_SOLO_GB=2.5" ;;   # estimates from measured peaks (solo run 800349 and E17 line): prefill 34.8 GB normal plan, pileup peak 26.3 GB, non-host index 24.9 GB, kept 1.6 GB
    admmap) CFG_ARGS="--no-viral-screen --no-microbial-screen"; CFG_ENV="SINGLET_GPU_ADMIT=1 SINGLET_BENCH_EXIT_AFTER=star SINGLET_CR4_PROBE=skip" ;;
    mps) start_mps ;;                                                                     # CUDA MPS, slot lock unchanged
    m45) CFG_ENV="SINGLET_ENGINE_GPU_MEM_GB=45" ;;                                        # device budget of a 2-slot node
    m91) CFG_ENV="SINGLET_ENGINE_GPU_MEM_GB=91" ;;
    bsort) CFG_ENV="SINGLET_BOLD_BSORT=1" ;;                                              # gpu/bold-kern (build b5)
    b64) CFG_ENV="SINGLET_BOLD_B64=1" ;;
    bsortb64) CFG_ENV="SINGLET_BOLD_BSORT=1 SINGLET_BOLD_B64=1" ;;
    nohc) CFG_ENV="SINGLET_BENCH_NOHOSTCOPY=1" ;;                                        # prototype: no host copy of lane results and records (needs build b1)
    nohc_alloff) CFG_ARGS="--n-donors 1 --no-viral-screen --no-microbial-screen"; CFG_ENV="SINGLET_BENCH_SKIP=ambient,doublet SINGLET_CR4_PROBE=skip SINGLET_BENCH_NOHOSTCOPY=1" ;;
    nohc_map) CFG_ARGS="--no-viral-screen --no-microbial-screen"; CFG_ENV="SINGLET_BENCH_EXIT_AFTER=star SINGLET_CR4_PROBE=skip SINGLET_BENCH_NOHOSTCOPY=1" ;;
    *) echo "!! unknown config $1" >&2; return 1 ;;
  esac
}
genome() { case $1 in neuron1k_v3|nuclei2k_v2) echo $REF/GRCm39-2024-A ;; hgmm1k_v3) echo $REF/GRCh38-GRCm39-barnyard ;; *) echo $REF/GRCh38-2024-A ;; esac; }
# the device memory split for slots > 1 (pbmc1k_v3 at level 5 + ingest; device marks of a run alone: index 16.4 GB, reads + clip 9.5,
# context 0.5 -> 26.4 GB before the lanes; Work per read in flight 36 / 68 / 550 KB for lanes A / M / B; outputs per sequence of a chunk
# 640 / 2048 / 4096 B): chunk 4 M sequences, each lane's Work within the budget, the pileup arena below it
budget() { case $1 in
  43) echo "SINGLET_ENGINE_GPU_CHUNK=4194304 SINGLET_ENGINE_GPU_NB=274432 SINGLET_ENGINE_GPU_NBM=155648 SINGLET_ENGINE_GPU_NBB=21504 SINGLET_GPU_PILEUP_ARENA_GB=38" ;;
  *) echo "!! budget m$1 undefined" >&2; echo "" ;;
esac; }

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
  [ "$DUMP" = 1 ] && [ $i = 0 ] && XE+=(SINGLET_ENGINE_GPU_DUMP=$SCR/dump-$p-c$i.bin)
  [ -n "$PIN" ] && XE+=(SINGLET_ENGINE_GPU_DEVICE=$((w % NGPU)))
  local CRS=$D/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $D/CellReads.stats ) 2>/dev/null &
  local WATCH=$!
  local NS=""
  case " ${NSYS_ITEMS:-} " in *" $i "*) NS="${NSYS_BIN:-/usr/local/cuda/bin/nsys} profile -o $O/prof --force-overwrite=true --trace=$(echo ${NSYS_TRACE:-cuda+nvtx} | tr + ,) ${NSYS_EXTRA:-} --sample=none --cpuctxsw=none --gpu-metrics-devices=all --gpu-metrics-frequency=${NSYS_GM_HZ:-10000}" ;; esac
  local T0; T0=$(date +%s.%N)
  ( cd $D && env "${XE[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $NS $BIN $SCR/data/$s.1fq --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $D/out/ --threads $T --pipeline --output-format 1pz --engine gpu $EXTRA_ARGS 2>&1 \
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
    [ "$IDENT" = 1 ] || rm -rf $SCR/$p/c$i      # ceiling runs: outputs are never kept
  done
}

# --- identity ------------------------------------------------------------------------------------------------------------------------
# E11 section 2a's run-dependent files once the read files are set aside (as the gate): timings, the file path's clip5p probe count, bytes_public, memory_tier.
# fieldcheck.py (gate.sbatch) looks into them. This harness uses the same by-name list as the gate.
EXPECT_DIFF=" ${EXPECT_DIFF:-pileup_stats.json read_stats.tsv star_Log.std.out summary.json} "
KNOWN_nuclei2k_v2=${KNOWN_nuclei2k_v2-nonhost_reads.fq}   # the known non-host writer bug of the gate (E15 section 5.7)
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
prune() { find $1 \( -name '*.bam' -o -name '*.sam' -o -name '*.cram' -o -name '*.fq*' -o -name '*.fastq*' -o -name '*.1fq' -o -name '.gpu_handoff*' -o -name '*Unmapped*' \
                     -o -name '_STARtmp*' -o -name '*.reads' -o -name '*.reads.q' \) -prune -exec rm -rf {} + 2>/dev/null; }
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=4 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }

# compare_out LABEL A_OUT A_CRS A_RH B_OUT B_CRS B_RH DETAILFILE   -> 0 when identical, 1 when not (same checks as the gate's pair comparison)
compare_out() {
  local lab=$1 a=$2 ca=$3 rha=$4 b=$5 cb=$6 rhb=$7 d=$8 ok=1 f
  echo "=== $lab: $b vs $a" >> $d
  if [ ! -d "$a" ]; then echo "  no reference ($a)" >> $d; return 2; fi
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $d.equiv 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $d.equiv | cut -c1-220 >> $d
  local UNX=""
  for f in $(grep -a '^  DIFF ' $d.equiv | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac; done
  grep -aq '^  DIFF provenance.json: /references/whitelist/name: [^;]*$' $d.equiv && UNX=${UNX/ provenance.json/}
  for f in $(grep -a '^  only in ' $d.equiv | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
  grep -aq '^files:' $d.equiv || UNX="$UNX out_equiv-failed"
  [ -n "$UNX" ] && { echo "  out_equiv UNEXPECTED:$UNX" >> $d; ok=0; }
  for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
           Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
    cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f || { echo "  star_Solo.out/$f DIFFERS" >> $d; ok=0; }
  done
  cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) || { echo "  star_Log.final.out STATISTICS DIFFER" >> $d; ok=0; }
  cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab || { echo "  star_SJ.out.tab DIFFERS" >> $d; ok=0; }
  if [ -f "$rha" ] && [ -f "$rhb" ]; then
    local kv=KNOWN_$s; local KN=" ${!kv:-} "
    for f in $READF; do
      local ha hb; ha=$(awk -v f=$f '$1==f{print $2" "$3}' $rha); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rhb)
      if [ "$ha" = "$hb" ]; then :
      elif [[ "$KN" == *" $f "* ]] && [ -n "$ha" ] && [ "${ha##* }" = "${hb##* }" ]; then echo "  $f: other records, same count (known non-host writer bug)" >> $d
      else echo "  $f: RECORDS DIFFER ($ha vs $hb)" >> $d; ok=0; fi
    done
  else echo "  read files: not compared (no hashes)" >> $d; ok=0; fi
  if [ -f "$ca" ] && [ -f "$cb" ]; then
    { cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); } || { echo "  CellReads.stats ROWS DIFFER" >> $d; ok=0; }
  else echo "  CellReads.stats: not compared" >> $d; ok=0; fi
  [ $ok = 1 ] && { echo "  IDENTICAL" >> $d; return 0; } || { echo "  NOT IDENTICAL" >> $d; return 1; }
}

# identity of run c<i> of point p against the alone run (same sample, T, slots) and the CPU reference; deletes the run's outputs afterwards
ident_run() {   # point item sample T slots
  local p=$1 i=$2 s=$3 T=$4 SL=$5
  local b=$SCR/$p/c$i/out cb=$RES0/$p/c$i/CellReads.stats d=$RES0/$p/c$i/identity.txt
  local al=$SCR/ref/$s-t$T-s$SL cpu=$CPU_REF/T$T/$s r
  : > $d
  readhash $b > $RES0/$p/c$i/readhash.txt; prune $b
  if [ -d $al/out ]; then compare_out alone $al/out $al/CellReads.stats $al/readhash.txt $b $cb $RES0/$p/c$i/readhash.txt $d; r=$?
    printf "%d\talone\t%d\n" $i $([ $r = 0 ] && echo 1 || echo 0) >> $RES0/$p/ident.tsv
  else echo "  no alone run for $s at T=$T slots=$SL" >> $d; fi
  if [ -d $cpu/out ]; then compare_out cpu $cpu/out $cpu/CellReads.stats $cpu/readhash.txt $b $cb $RES0/$p/c$i/readhash.txt $d; r=$?
    printf "%d\tcpu\t%d\n" $i $([ $r = 0 ] && echo 1 || echo 0) >> $RES0/$p/ident.tsv
  else echo "  NO CPU REFERENCE at $cpu (run gate.sbatch with MODES=cpu for this binary and T)" >> $d; fi
  echo "  c$i ($s): $(grep -a 'IDENTICAL' $d | tr '\n' ' ')" >> $RES0/$p/identity.lines
  rm -rf $SCR/$p/c$i
}

collect_dump() {   # point
  local p=$1 f c s
  for f in $SCR/dump-$p-c*.bin; do
    [ -f $f ] || continue
    c=$(basename $f .bin); c=${c##*-}; s=$(awk -F'\t' -v i=${c#c} '$1==i{print $3}' $RES0/$p/runs.tsv)
    printf "%s\t%s\t%s\t%s\t%s\n" $p $c $s $(stat -c %s $f) $(sha256sum $f | cut -c1-64) >> $RES0/dumps.tsv; rm -f $f
    echo "  E6 dump $p $c ($s): $(tail -1 $RES0/dumps.tsv | cut -f4) bytes, sha256 $(tail -1 $RES0/dumps.tsv | cut -f5 | cut -c1-16)" | tee -a $S
  done
}

# a run alone at the same T and slots, kept (pruned) as the reference of the point's runs
ensure_alone() {   # sample T slots
  local s=$1 T=$2 SL=$3 key=$1-t$2-s$3
  [ -d $SCR/ref/$key/out ] && return
  local p=alone-$key keep=$PIN; PIN=""
  stage_input $s; mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv
  echo "######## $p: one run alone, T=$T, slots=$SL; $(date -Is)" | tee -a $S
  run_item $p 0 0 $s $T
  local rc; rc=$(awk -F'\t' '{print $6}' $RES0/$p/runs.tsv | head -1)
  if [ "${rc:-1}" = 0 ]; then
    mkdir -p $SCR/ref/$key; mv $SCR/$p/c0/out $SCR/ref/$key/out; cp $RES0/$p/c0/CellReads.stats $SCR/ref/$key/ 2>/dev/null
    readhash $SCR/ref/$key/out > $SCR/ref/$key/readhash.txt; prune $SCR/ref/$key/out
    echo "  alone run $key: exit 0, wall $(awk -F'\t' '{printf "%.1f", $5-$4}' $RES0/$p/runs.tsv) s, kept as the reference" | tee -a $S
    local fb l5=no; fb=$(python3 $GB/fallback.py $RES0/$p/c0 | awk -F'	' '$1=="SUMMARY"{print $2}')
    grep -aq "E11: Solo on the device" $RES0/$p/c0/star_Log.out 2>/dev/null && l5=yes
    echo "  alone run $key: disallowed fallback lines ${fb:-?}, level 5 $l5$([ "${fb:-1}" != 0 ] || [ $l5 = no ] && echo ' !! THE REFERENCE RUN FELL BACK, identity below proves little')" | tee -a $S
    # the alone run against the CPU reference as well
    local cpu=$CPU_REF/T$T/$s d=$RES0/$p/c0/identity.txt; : > $d
    if [ -d $cpu/out ]; then compare_out cpu $cpu/out $cpu/CellReads.stats $cpu/readhash.txt $SCR/ref/$key/out $SCR/ref/$key/CellReads.stats $SCR/ref/$key/readhash.txt $d
      echo "  alone run $key vs the CPU reference: $(grep -a 'IDENTICAL' $d | tr '\n' ' ')" | tee -a $S
    else echo "  alone run $key: NO CPU REFERENCE at $cpu" | tee -a $S; fi
  else echo "!! alone run $key failed (exit $rc): no reference for it" | tee -a $S; fi
  collect_dump $p; rm -rf $SCR/$p; PIN=$keep
}

for pp in "${POINTS[@]}"; do
  PT=${pp%%@*}; CFG=base; [[ $pp == *@* ]] && CFG=${pp#*@}; CFG=${CFG%%#*}; p=${pp//[@#]/_}
  cfg_set $CFG || continue
  EXTRA_ARGS="$CFG_ARGS"
  if ! [[ $PT =~ ^(mix)?k([0-9]+)t([0-9]+)s([0-9]+)(m([0-9]+))?(r([0-9]+))?(p)?(f)?$ ]]; then echo "!! bad point tag $PT" | tee -a $S; continue; fi
  MIXP=${BASH_REMATCH[1]}; K=${BASH_REMATCH[2]}; T=${BASH_REMATCH[3]}; SL=${BASH_REMATCH[4]}; BG=${BASH_REMATCH[6]}; R=${BASH_REMATCH[8]:-1}; PIN=${BASH_REMATCH[9]}
  SDT=0.5; SGE=2; [ -n "${BASH_REMATCH[10]}" ] && SDT=0.2 && SGE=1
  NI=$((K * R)); [ -n "$NRUNS" ] && NI=$NRUNS
  ITEMS=()
  if [ -n "$MIXP" ]; then   # the mix repeated R times (E18), or NRUNS items cycling through it
    if [ -n "$NRUNS" ]; then for ((j = 0; j < NI; j++)); do ITEMS+=("${MIX[$((j % ${#MIX[@]}))]}"); done; else for ((j = 0; j < R; j++)); do ITEMS+=("${MIX[@]}"); done; NI=${#ITEMS[@]}; fi
  else for ((j = 0; j < NI; j++)); do ITEMS+=($SAMPLE); done; fi
  for s in $(printf "%s\n" "${ITEMS[@]}" | sort -u); do stage_input $s; done
  XENV="SINGLET_ENGINE_GPU_SLOTS=$SL"; [ -n "$BG" ] && XENV+=" $(budget $BG)"
  unset SINGLET_ENGINE_GPU_SLOTS SINGLET_ENGINE_GPU_CHUNK SINGLET_ENGINE_GPU_NB SINGLET_ENGINE_GPU_NBM SINGLET_ENGINE_GPU_NBB SINGLET_GPU_PILEUP_ARENA_GB SINGLET_ENGINE_GPU_DEVICE
  export $XENV
  for kv in $CFG_ENV; do export "$kv"; done
  if [ "$IDENT" = 1 ]; then for s in $(printf "%s\n" "${ITEMS[@]}" | sort -u); do ensure_alone $s $T $SL; done; fi
  rm -f /tmp/singlet_admit_*.tab
  mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv; : > $RES0/$p/ident.tsv; : > $RES0/$p/identity.lines
  printf '{"tag":"%s","k":%d,"T":%d,"slots":%d,"rounds":%d,"runs":%d,"pin":"%s","items":"%s","env":"%s","node":"%s","gpus":%d,"cfg":"%s","cfg_args":"%s","cfg_env":"%s"}\n' $p $K $T $SL $R $NI "$PIN" "${ITEMS[*]}" "$XENV" "$(hostname)" $NGPU "$CFG" "$CFG_ARGS" "$CFG_ENV" > $RES0/$p/meta.json
  echo "######## $p: $K in flight, T=$T, $NI runs, items ${ITEMS[*]:0:3}...; $XENV${PIN:+; pinned}; cfg=$CFG args=[$CFG_ARGS] env=[$CFG_ENV]; $(date -Is)" | tee -a $S
  # co-tenants: the node's jobs at the start, once a minute, and at the end
  CT=$RES0/$p/cotenants.log; : > $CT
  ctsnap() { { echo "## $1 $(date -Is)"; squeue -w $(hostname -s) -o "%i %u %j %C %b" 2>&1; } >> $CT; }
  ctsnap start
  cat $CT | sed 's/^/  /' | tee -a $S
  ( while sleep 60; do ctsnap minute; done ) &
  CTP=$!
  python3 $TPD/e18_sampler.py $RES0/$p/samples.jsonl $CG $SDT $SGE 2>$RES0/$p/sampler.err &
  SMP=$!
  nvidia-smi dmon -s pucvmet -d 1 -o T > $RES0/$p/dmon.log 2>&1 &
  DMP=$!
  nvidia-smi dmon --gpm-metrics 2,3,4,10,12,20,21 -d 1 -o T > $RES0/$p/gpm.log 2>&1 &
  GMP=$!
  sleep 1
  WP=()
  for ((w = 0; w < K; w++)); do worker $p $w $T & WP+=($!); done
  wait "${WP[@]}"
  sleep 1; kill $SMP $CTP $DMP $GMP 2>/dev/null; wait $SMP 2>/dev/null
  ctsnap end
  du -sh $SCR/$p 2>/dev/null | sed 's/^/  \/tmp use of the point before identity: /' | tee -a $S; df -h /tmp | tail -1 | sed 's/^/  /' | tee -a $S
  python3 $TPD/e18_point.py $RES0/$p > $RES0/$p/point.txt 2>&1
  [ -f $TPD/e18_locks.py ] && python3 $TPD/e18_locks.py $RES0/$p >> $RES0/$p/point.txt 2>&1
  [ -f $TPD/e18_steady.py ] && python3 $TPD/e18_steady.py $RES0/$p 2>&1 | sed 's/^/   /' >> $RES0/$p/point.txt
  cut -c1-400 $RES0/$p/point.txt | tee -a $S
  # identity, the dump, references
  while IFS=$'\t' read -r i w s t0 t1 rc; do
    [ "$i" = item ] && continue
    if [ $K = 1 ] && [ ! -d $SCR/ref/$s-t$T-s$SL/out ] && [ $rc = 0 ]; then
      mkdir -p $SCR/ref/$s-t$T-s$SL && mv $SCR/$p/c$i/out $SCR/ref/$s-t$T-s$SL/out && cp $RES0/$p/c$i/CellReads.stats $SCR/ref/$s-t$T-s$SL/ 2>/dev/null
      readhash $SCR/ref/$s-t$T-s$SL/out > $SCR/ref/$s-t$T-s$SL/readhash.txt; prune $SCR/ref/$s-t$T-s$SL/out
      echo "  c$i ($s): kept as the reference for $s at T=$T, slots=$SL" | tee -a $S
    elif [ "$IDENT" = 1 ] && [ $rc = 0 ]; then
      while [ $(jobs -rp | wc -l) -ge ${IDPAR:-4} ]; do sleep 1; done
      ident_run $p $i $s $T $SL &
    fi
  done < <(sort -n $RES0/$p/runs.tsv)
  wait
  for f in $(ls $RES0/$p/c*/identity.txt 2>/dev/null | sort -V); do cat $f >> $RES0/$p/identity.txt; done
  sort -V $RES0/$p/identity.lines 2>/dev/null | tee -a $S
  collect_dump $p
  python3 $GB/tp_point.py $RES0/$p 2>&1 | tee -a $S $RES0/$p/results.txt
  python3 $W/jobs/gpu-bold/jobs/admit_stats.py $RES0/$p > $RES0/$p/admit.txt 2>&1; sed 's/^/  admit: /' $RES0/$p/admit.txt | tee -a $S
  [ -f $RES0/results.tsv ] || head -1 $RES0/$p/results.tsv > $RES0/results.tsv
  tail -n +2 $RES0/$p/results.tsv >> $RES0/results.tsv
  rm -rf "${SCR:?}/${p:?}"
  for kv in $CFG_ENV; do unset "${kv%%=*}"; done
done
echo "=== mapDistinct records (E6 dumps) per sample" | tee -a $S
[ -f $RES0/dumps.tsv ] && for s in $(cut -f3 $RES0/dumps.tsv | sort -u); do
  n=$(awk -F'\t' -v s=$s '$3==s' $RES0/dumps.tsv | wc -l); u=$(awk -F'\t' -v s=$s '$3==s{print $5}' $RES0/dumps.tsv | sort -u | wc -l)
  [ $u = 1 ] && echo "  $s: $n dumps, all IDENTICAL (sha256 $(awk -F'\t' -v s=$s '$3==s{print substr($5,1,16); exit}' $RES0/dumps.tsv))" | tee -a $S \
             || echo "  $s: $n dumps, $u DIFFERENT hashes" | tee -a $S
done
python3 $TPD/e18_point.py --table $RES0 > $RES0/table.md 2>&1; tee -a $S < $RES0/table.md
echo "=== results.tsv" | tee -a $S; column -t -s $'\t' $RES0/results.tsv 2>/dev/null | tee -a $S
if [ -n "${NSYS_ITEMS:-}" ]; then
  echo "=== nsys export and summary" | tee -a $S
  for r in $RES0/*/c*/prof.nsys-rep; do
    [ -f $r ] || continue
    d=$(dirname $r); echo "  $r $(stat -c %s $r) bytes" | tee -a $S
    ${NSYS_BIN:-/usr/local/cuda/bin/nsys} export --type=sqlite --force-overwrite=true -o $d/prof.sqlite $r > $d/export.log 2>&1
    python3 $W/jobs/gpu-bold/jobs/nsys_summary.py $d/prof.sqlite --csv-prefix $d/prof > $d/nsys_summary.txt 2>&1
    echo "  summary: $d/nsys_summary.txt" | tee -a $S
  done
fi
[ -n "${MPS_ON:-}" ] && { echo quit | nvidia-cuda-mps-control; echo "MPS daemon stopped" | tee -a $S; }
echo "PROVENANCE job=$SLURM_JOB_ID node=$(hostname -s) bin=$BIN sha256=$BSHA commit=$COMMIT scripts=$SCRIPTS" | tee -a $S
echo done $(date -Is) | tee -a $S
