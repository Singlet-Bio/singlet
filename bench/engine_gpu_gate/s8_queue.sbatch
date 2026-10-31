#!/bin/bash
#SBATCH --job-name=s8-queue
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: steady-state node throughput. One whole H100 NVL node (2 GPUs, 56 cores). For each config "S:T" (S slots per GPU, T threads per sample): a queue of
# NS samples (the same pbmc1k_v3, half per GPU, CUDA_VISIBLE_DEVICES pins them) runs through 2S slots (xargs -P S per GPU). Steady-state rate = the samples that
# finish after the first wave (2S completions) over the time from that completion to the last one; makespan, mean sample wall, lock waits, core-s/M and
# GPU utilization are reported too. For the config in TL (default 8:12) one mid-queue sample also gets a phase timeline: its process tree's CPU rate every 2 s
# (threads actually busy), with the log's stage markers, so serial sections, lock waits and I/O phases can be read off.
# Env: BIN (required), CONFIGS (default "4:7 4:14 4:24 8:12"), NS (default 32), SAMPLE, EXTRA (default "--genome-shared on"), TL.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?set BIN}; s=${SAMPLE:-pbmc1k_v3}; NS=${NS:-32}; CONFIGS=${CONFIGS:-"4:7 4:14 4:24 8:12"}; TL=${TL:-8:12}
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq; READS=66601887
# SAMPLES="a b c": the queue cycles through these (a read-weighted mix); each sample's read count comes from its own log; neuron1k_v3 is mouse
SAMPLES=${SAMPLES:-$s}; export SAMPLES
RES=$W/jobs/s8/results/queue-$SLURM_JOB_ID; mkdir -p $RES; SCR=/tmp/$USER-s8q-$SLURM_JOB_ID; mkdir -p $SCR
PRE_SHM=$(ipcs -m | awk '/^0x/{print $2}' | sort); PRE_DEV=$(ls /dev/shm 2>/dev/null | sort)
cleanup() { kill $(jobs -p) 2>/dev/null; rm -rf $SCR
  for id in $(comm -13 <(echo "$PRE_SHM") <(ipcs -m | awk '/^0x/{print $2}' | sort)); do ipcrm -m $id 2>/dev/null && echo "removed shm segment $id"; done
  for f in $(comm -13 <(echo "$PRE_DEV") <(ls /dev/shm 2>/dev/null | sort)); do rm -rf /dev/shm/$f && echo "removed /dev/shm/$f"; done; }
cleanup2() { [ -n "${PRELOAD:-}" ] && $BIN genome unload $G/star_2.7.11b > /dev/null 2>&1; [ -n "${TMPDIR_RAM:-}" ] && rm -rf /dev/shm/s8q_$SLURM_JOB_ID; cleanup; }
mps_stop() { [ -n "${MPS:-}" ] && echo quit | CUDA_MPS_PIPE_DIRECTORY=/tmp/s8mps_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/s8mpslog_$SLURM_JOB_ID nvidia-cuda-mps-control > /dev/null 2>&1; }
trap 'mps_stop; cleanup2' EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=7200 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_GPU_PILEUP=on
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
# PRELOAD=1: the genome goes into shared memory once (`singlet genome load`), every sample attaches (N21 auto mode); TMPDIR_RAM=1: the decode temp files go to /dev/shm
# MPS=1: a user-level CUDA MPS daemon for the node (kernels of the samples sharing a GPU run concurrently instead of time-slicing)
if [ -n "${MPS:-}" ]; then export CUDA_MPS_PIPE_DIRECTORY=/tmp/s8mps_$SLURM_JOB_ID CUDA_MPS_LOG_DIRECTORY=/tmp/s8mpslog_$SLURM_JOB_ID; mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY; nvidia-cuda-mps-control -d; sleep 2; echo "MPS daemon started" | tee -a $RES/summary.txt; fi
[ -n "${PRELOAD:-}" ] && { $BIN genome load $G/star_2.7.11b 2>&1 | tee -a $RES/summary.txt; }
[ -n "${TMPDIR_RAM:-}" ] && { mkdir -p /dev/shm/s8q_$SLURM_JOB_ID; export TMPDIR=/dev/shm/s8q_$SLURM_JOB_ID; }
lscpu | grep -E "Model name|NUMA|Socket|L3" | tee $RES/lscpu.txt; echo "node=$(hostname) job=$SLURM_JOB_ID bin=$(cat $(dirname $BIN)/commit) sample=$s NS=$NS configs=$CONFIGS $(date -Is)" | tee $RES/summary.txt

tree() { echo $1; for c in $(pgrep -P $1 2>/dev/null); do tree $c; done; }
export -f tree
run_one() {   # gpu index config-dir slots threads timeline?
  local g=$1 i=$2 D=$3 T=$4 d=$SCR/$(basename $D)/g${1}_$2 TLFLAG=0
  local arr=($SAMPLES) smp FQ G; smp=${arr[$(( (i - 1 + g * 3) % ${#arr[@]} ))]}
  FQ=$PB/data/$smp/$smp.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$smp/sample.1fq; [ -e $FQ ] || FQ=$W/data/w11/$smp/sample.1fq
  G=$REF/GRCh38-2024-A; [ $smp = neuron1k_v3 ] && G=$REF/GRCm39-2024-A
  [ "$(basename $D)" = "cfg_${TL%%:*}x${TL##*:}" ] && [ $g = 0 ] && [ $i = 10 ] && TLFLAG=1
  mkdir -p $d/out
  local s0=$(date +%s.%N)
  local NUMACMD=""
  if [ -n "${NUMA_PIN:-}" ]; then   # NUMA_PIN=1: the sample runs on the socket its GPU hangs off, with its memory there
    local bus nn; bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i $g | tr "A-Z" "a-z" | sed "s/^0000//"); nn=$(cat /sys/bus/pci/devices/${bus}/numa_node 2>/dev/null || echo -1)
    [ "$nn" -ge 0 ] && NUMACMD="numactl --cpunodebind=$nn --membind=$nn"
  fi
  ( cd $d && CUDA_VISIBLE_DEVICES=$g $NUMACMD $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $d/out/ \
      --threads $T --pipeline --output-format 1pz --engine gpu $EXTRA 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $D/g${g}_$i.log ) &
  local wp=$!
  if [ "$TLFLAG" = 1 ]; then
    ( sleep 3; sp=$(pgrep -f "singlet-gpu $FQ.*out-prefix $d/out/" | head -1)
      declare -A last   # per pid, the last CPU ticks seen: an exited process (STAR) keeps its share in the total
      while [ -n "$sp" ] && kill -0 $sp 2>/dev/null; do
        t=$(date +%s.%N); c=0; for p in $(tree $sp); do v=$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null); [ -n "$v" ] && last[$p]=$v; done
        for p in "${!last[@]}"; do c=$((c + ${last[$p]})); done
        echo "$t $c" >> $D/timeline_g${g}_$i.cpu
        # every ~30 s: IPC of the busiest child (STAR) over 4 s, and its memory placement by NUMA node
        n=$((${n:-0}+1))
        if [ $((n % 15)) = 0 ]; then
          kp=$(for p in $(tree $sp); do [ $p != $sp ] && echo "$(awk '{print $14+$15}' /proc/$p/stat 2>/dev/null) $p"; done | sort -rn | head -1 | cut -d' ' -f2)
          if [ -n "$kp" ]; then
            echo "t=$t pid=$kp $(perf stat -x, -e cycles,instructions -p $kp -- sleep 4 2>&1 | tr '
' ' ')" >> $D/perf_g${g}_$i.txt
            numastat -p $kp 2>/dev/null | tail -3 | sed "s/^/t=$t /" >> $D/numa_g${g}_$i.txt
          fi
        fi
        sleep 2
      done ) &
  fi
  wait $wp
  local nr=$(grep -a -m1 -o "1fq: [0-9]* reads" $D/g${g}_$i.log | grep -o "[0-9][0-9][0-9][0-9]*" | tail -1)
  echo "$g $i $s0 $(date +%s.%N) $smp ${nr:-0}" >> $D/done.tsv
  grep -a -h "waited" $d/out/*Log.out 2>/dev/null | head -1 | sed "s/^/$g $i /" >> $D/waits.txt
  rm -rf $d
}
export -f run_one
EXTRA=${EXTRA-"--genome-shared on"}; export BIN FQ REF G SCR TL EXTRA W PB

for cfg in $CONFIGS; do
  S=${cfg%%:*}; T=${cfg##*:}; D=$RES/cfg_${S}x${T}; mkdir -p $D
  ( while true; do nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | tr -d ' ' | sed "s/^/$(date +%s) /"; sleep 2; done ) > $D/gpu.csv 2>/dev/null &
  GM=$!
  # pressure-stall information every 5 s (node and this job's cgroup): time tasks spent stalled on io / memory / cpu
  ( while true; do
      echo "$(date +%s) mem_cur: $(cat $CG/memory.current 2>/dev/null) io: $(head -1 /proc/pressure/io) | mem: $(head -1 /proc/pressure/memory) | cpu: $(head -1 /proc/pressure/cpu) | cg_io: $(head -1 $CG/io.pressure 2>/dev/null) | cg_mem: $(head -1 $CG/memory.pressure 2>/dev/null) | cg_cpu: $(head -1 $CG/cpu.pressure 2>/dev/null)"
      sleep 5
    done ) > $D/psi.log 2>/dev/null &
  PSIPID=$!
  cat $CG/io.stat > $D/io.stat.start 2>/dev/null
  C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T0=$(date +%s.%N)
  U0=$(awk '$1=="user_usec"{print $2}' $CG/cpu.stat); Y0=$(awk '$1=="system_usec"{print $2}' $CG/cpu.stat)
  for g in 0 1; do
    ( seq 1 $((NS/2)) | xargs -P $S -I{} bash -c "run_one $g {} $D $T" ) &
  done
  wait $(jobs -p | grep -v -x -e "$GM" -e "$PSIPID") 2>/dev/null
  T1=$(date +%s.%N); C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); kill $GM $PSIPID 2>/dev/null; cat $CG/io.stat > $D/io.stat.end 2>/dev/null
  U1=$(awk '$1=="user_usec"{print $2}' $CG/cpu.stat); Y1=$(awk '$1=="system_usec"{print $2}' $CG/cpu.stat)
  echo "cgroup CPU split for ${S}x${T}: user $(( (U1-U0)/1000000 )) s, system $(( (Y1-Y0)/1000000 )) s" | tee -a $RES/summary.txt
  echo "cgroup memory: peak $(awk '{printf "%.0f", $1/1073741824}' $CG/memory.peak 2>/dev/null) GB; max sampled current $(grep -o 'mem_cur: [0-9]*' $D/psi.log | awk '{if ($2>m) m=$2} END {printf "%.0f", m/1073741824}') GB" | tee -a $RES/summary.txt
  python3 - "$D" "$S" "$T" "$NS" "$READS" "$T0" "$T1" "$C0" "$C1" <<'PY' | tee -a $RES/summary.txt
import sys, re, glob, collections
D, S, T, NS, READS, T0, T1, C0, C1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), *map(float, sys.argv[6:10])
rows = sorted((float(l.split()[3]), float(l.split()[2]), int(l.split()[5]) if len(l.split()) > 5 and int(l.split()[5]) > 0 else READS) for l in open(D + "/done.tsv"))   # (end, start, reads)
ends = [r[0] for r in rows]; walls = [r[0] - r[1] for r in rows]; rds = [r[2] for r in rows]
first = 2 * S
steady = sum(rds[first:]) / 1e6 / (ends[-1] - ends[first - 1]) if len(ends) > first else float("nan")
TOTAL = sum(rds)
cs = (C1 - C0) / 1e6; wait = []
try: wait = [float(x) for x in re.findall(r"waited ([\d.e+-]+) s", open(D + "/waits.txt").read())]
except Exception: pass
held = [float(x) for x in re.findall(r"held it ([\d.e+-]+) s", open(D + "/waits.txt").read())] if wait else []
pwait, pheld = [], []
for f in glob.glob(D + "/g*_*.log"):
    for m in re.finditer(r"device lock: waited ([\d.e+-]+) s, held ([\d.e+-]+) s", open(f, errors="replace").read()):
        pwait.append(float(m.group(1))); pheld.append(float(m.group(2)))
mean = lambda v: sum(v) / len(v) if v else 0
lockutil = (sum(held) + sum(pheld)) / (2 * (T1 - T0))
util = collections.defaultdict(list)
for l in open(D + "/gpu.csv"):
    p = l.split()
    if len(p) == 2: i, u = p[1].split(","); util[int(i)].append(float(u))
print(f"{S} per GPU x {T} threads, {len(ends)}/{NS} samples: makespan {T1-T0:.0f} s -> {TOTAL/1e6/(T1-T0):.2f} M pairs/s; STEADY {steady:.2f} M pairs/s (after the first {first} completions); "
      f"host {cs/(TOTAL/1e6):.2f} core-s/M ({cs/(T1-T0):.1f} cores busy); sample wall mean/max {sum(walls)/len(walls):.0f}/{max(walls):.0f} s; lock wait mean/max {sum(wait)/max(len(wait),1):.0f}/{max(wait) if wait else 0:.0f} s; prefill lock held mean {mean(held):.0f} s, pileup lock wait/held mean {mean(pwait):.0f}/{mean(pheld):.0f} s, LOCK UTILIZATION {lockutil*100:.0f}% (sum of holds / (2 GPUs x makespan)); "
      f"GPU util {', '.join(f'{sum(v)/len(v):.0f}%' for k, v in sorted(util.items()))}")
for f in sorted(glob.glob(D + "/timeline_*.cpu")):
    g, i = re.search(r"timeline_g(\d+)_(\d+)", f).groups()
    pts = [tuple(map(float, l.split())) for l in open(f)]
    log = [(int(l[:10]), l[11:].rstrip()) for l in open(D + f"/g{g}_{i}.log", errors="replace") if l[:10].isdigit()]
    marks = [("start", None), ("decode", "Decoding "), ("STAR start", "STAR pid="), ("stream", "SNPs to BAM chromosome"), ("hand-off pass", "hand-off pass: entries"), ("GPU pileup", "[gpu_pileup] on:"), ("non-host", "STAR completed"), ("export", "Exporting feature matrices"), ("demux", "[demux] Starting")]
    t0 = pts[0][0]; bounds = [t0]
    for n, pat in marks[1:]:
        b = next((ts for ts, s in log if pat in s and ts >= bounds[-1]), None); bounds.append(b if b else bounds[-1])
    bounds.append(pts[-1][0])
    print(f"   timeline of sample g{g}_{i} (CPU ticks of its process tree; busy threads = CPU s / wall s):")
    def cpu_at(t):
        import bisect
        k = min(max(bisect.bisect_left([p[0] for p in pts], t), 1), len(pts) - 1); a, b = pts[k-1], pts[k]
        return (a[1] + (b[1] - a[1]) * (t - a[0]) / max(b[0] - a[0], 1e-9)) / 100.0
    for k, (n, _) in enumerate(marks):
        a, b = bounds[k], bounds[k + 1]
        if b > a: print(f"      {n:14s} {b-a:6.0f} s wall, {cpu_at(b)-cpu_at(a):7.0f} core-s, {(cpu_at(b)-cpu_at(a))/(b-a):5.1f} threads busy")
    wl = open(D + "/waits.txt").read().split("\n") if False else []
PY
done
echo done $(date -Is) | tee -a $RES/summary.txt
