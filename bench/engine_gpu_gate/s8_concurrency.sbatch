#!/bin/bash
#SBATCH --job-name=s8-conc
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: node throughput with several samples per GPU. One whole H100 NVL node (2 GPUs, 56 cores): for N in 1 2 4 8, run 2N concurrent copies of the same sample
# (N per GPU, CUDA_VISIBLE_DEVICES pins each to one GPU), each with 56/(2N) threads (min 3). Per N: batch wall, node pairs/s, host core-s per M (cgroup), GPU
# utilization (both GPUs, 2 s samples), peak memory, per-sample wall, and the device waits the engine logs.
# Env: BIN (singlet-gpu), SAMPLE (default pbmc1k_v3), NLIST (default "1 2 4 8"), EXTRA (extra singlet args, e.g. --genome-shared on).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?set BIN}; s=${SAMPLE:-pbmc1k_v3}; NLIST=${NLIST:-1 2 4 8}
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
RES=$W/jobs/s8/results/conc-$SLURM_JOB_ID; mkdir -p $RES; SCR=/tmp/$USER-s8conc-$SLURM_JOB_ID; mkdir -p $SCR
PRE_SHM=$(ipcs -m | awk '/^0x/{print $2}' | sort); PRE_DEV=$(ls /dev/shm 2>/dev/null | sort)
cleanup() { kill $(jobs -p) 2>/dev/null; rm -rf $SCR
  for id in $(comm -13 <(echo "$PRE_SHM") <(ipcs -m | awk '/^0x/{print $2}' | sort)); do ipcrm -m $id 2>/dev/null && echo "removed shm segment $id"; done   # the shared genome this job created (exclusive node)
  for f in $(comm -13 <(echo "$PRE_DEV") <(ls /dev/shm 2>/dev/null | sort)); do rm -rf /dev/shm/$f && echo "removed /dev/shm/$f"; done; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=7200 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH SINGLET_GPU_PILEUP=on
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
READS=$(python3 -c "print(66601887)")
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$(cat $(dirname $BIN)/commit) sample=$s $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | tee -a $RES/summary.txt
for N in $NLIST; do
  NP=$((2*N)); T=$((56/NP)); [ $T -lt 3 ] && T=3
  D=$RES/N$N; mkdir -p $D
  ( while true; do nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | sed "s/^/$(date +%s) /"; sleep 2; done ) > $D/gpu.csv 2>/dev/null &
  GM=$!
  ( while true; do echo "$(date +%s) $(( $(cat $CG/memory.current) / 1048576 ))"; sleep 2; done ) > $D/mem.csv 2>/dev/null &
  MM=$!
  C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T0=$(date +%s.%N)
  for ((p=0; p<NP; p++)); do
    dev=$((p%2)); mkdir -p $SCR/N$N/p$p/out
    ( cd $SCR/N$N/p$p && s0=$(date +%s.%N) && CUDA_VISIBLE_DEVICES=$dev $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/N$N/p$p/out/ --threads $T --pipeline --output-format 1pz --engine gpu ${EXTRA---genome-shared on} > $D/p$p.log 2>&1; \
      echo "exit $? wall $(echo "$(date +%s.%N) - $s0" | bc)" >> $D/p$p.log; grep -ah "ENGINE_GPU: device\|cache filled" $SCR/N$N/p$p/out/*Log.out 2>/dev/null | head -3 >> $D/p$p.log; rm -rf $SCR/N$N/p$p ) &
  done
  wait $(jobs -p | grep -v -e "^$GM$" -e "^$MM$") 2>/dev/null
  # the samplers are background jobs too: wait only for the runs
  T1=$(date +%s.%N); C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); kill $GM $MM 2>/dev/null
  python3 - "$D" "$N" "$NP" "$T" "$READS" "$T0" "$T1" "$C0" "$C1" <<'PY' | tee -a $RES/summary.txt
import sys, re, glob, collections
D, N, NP, T, READS, T0, T1, C0, C1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), float(sys.argv[8]), float(sys.argv[9])
wall = T1 - T0; tot = NP * READS / 1e6; cs = (C1 - C0) / 1e6
walls, exits, waits = [], [], []
for f in sorted(glob.glob(D + "/p*.log")):
    t = open(f, errors="replace").read()
    m = re.search(r"exit (\d+) wall ([\d.]+)", t)
    if m: exits.append(int(m.group(1))); walls.append(float(m.group(2)))
    waits += [float(x) for x in re.findall(r"waited ([\d.e+-]+) s", t)]
util = collections.defaultdict(list); mem = collections.defaultdict(list)
for l in open(D + "/gpu.csv"):
    p = l.split()
    if len(p) == 2:
        i, u, m = p[1].split(","); util[int(i)].append(float(u)); mem[int(i)].append(float(m))
cm = [float(l.split()[1]) for l in open(D + "/mem.csv") if len(l.split()) == 2]
print(f"N={N} per GPU ({NP} samples, {T} threads each): batch wall {wall:.0f} s, {tot:.0f} M reads -> {tot/wall:.2f} M pairs/s per node; host {cs:.0f} core-s = {cs/tot:.2f} core-s/M; "
      f"sample wall min/mean/max {min(walls):.0f}/{sum(walls)/len(walls):.0f}/{max(walls):.0f} s; exits {sorted(set(exits))}; node memory peak {max(cm)/1024:.0f} GB")
for i in sorted(util):
    print(f"   GPU {i}: mean util {sum(util[i])/len(util[i]):.1f}% (busy {sum(util[i])/100*2:.0f} s), peak memory {max(mem[i])/1024:.0f} GB; device wait max {max(waits) if waits else 0:.1f} s")
PY
  rm -rf $SCR/N$N
done
echo done $(date -Is) | tee -a $RES/summary.txt
