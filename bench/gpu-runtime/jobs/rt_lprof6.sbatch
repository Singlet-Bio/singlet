#!/bin/bash
#SBATCH -A debruinz -p short-gpu --gres=gpu:nvidia_h100_nvl:1 -c 28 --mem=200G --time=02:30:00 --exclude=g050 -J rt-lprof6 -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-lprof: per-kernel time (nsys) and efficiency (ncu, a window of launches) of the lane kernels on one d300_3p sample, production path (31 GB budget, 2 slots).
# BIN=<binary> [SKIP=118 COUNT=120]. Measurement only; single process, SINGLET_GPU_SHARED_INDEX unset.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
RT=$W/jobs/gpu-runtime; : "${BIN:?}"; GD=$REF/GRCh38-2024-A; DP=$W/data/gpu-deep; NSYS=/usr/local/cuda/bin/nsys; NCU=/usr/local/cuda/bin/ncu
SKIP=${SKIP:-118}; COUNT=${COUNT:-120}
O=$RT/results/rt-lprof-$SLURM_JOB_ID; mkdir -p $O; SCR=/tmp/$USER-rtlprof-$SLURM_JOB_ID; mkdir -p $SCR/nstmp; R=$O/results.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=21600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_MIN_CC=70 CUDA_VISIBLE_DEVICES=0
export SINGLET_ENGINE_GPU_MEM_GB=31 SINGLET_ENGINE_GPU_SLOTS=2 SINGLET_NVTX=1 SINGLET_TAIL_THREADS=16 SINGLET_INGEST_THREADS=16
unset SINGLET_GPU_SHARED_INDEX
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
mypids() { find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u; }
killmine() { local p; for p in $(mypids); do [ "$p" = "$$" ] && continue
    tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -aq -e "$2" && kill -$1 $p 2>/dev/null; done; true; }
cleanup() { killmine TERM "singlet-gpu"; sleep 2; rm -rf "${SCR:?}"; }
trap 'kill $(jobs -p) 2>/dev/null; cleanup' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BIN ($(cat $(dirname $BIN)/commit-gpu 2>/dev/null)) $(date -Is)" | tee $R
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader | tee -a $R; $NSYS --version | head -1 | tee -a $R; $NCU --version | tail -1 | tee -a $R
[ -x $BIN ] || { echo NOBIN | tee -a $R; exit 2; }
S1=$DP/d300_3p/d300_3p.1fq; [ -s $S1 ] || { echo MISSING | tee -a $R; exit 2; }
F=$SCR/d300_3p.1fq; cp $S1 $F
cp $RT/rt_prof_an.py $O/
cat > $SCR/fl.c <<'CEOF'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
typedef int (*fl_t)(unsigned int);
static void fl_flush(void) {
  char line[1024], path[900] = ""; FILE* f = fopen("/proc/self/maps", "r");
  if (f) { while (fgets(line, sizeof line, f)) { char* p = strchr(line, '/'); if (p && strstr(p, "libcupti.so")) { strncpy(path, p, 899); char* n = strchr(path, '\n'); if (n) *n = 0; break; } } fclose(f); }
  if (!path[0]) return;
  void* h = dlopen(path, RTLD_NOW | RTLD_NOLOAD); if (!h) return;
  fl_t fn = (fl_t)dlsym(h, "cuptiActivityFlushAll"); if (fn) { fn(1); usleep(3000000); }
}
void _exit(int s) { static void (*real)(int); fl_flush(); real = (void (*)(int))dlsym(RTLD_NEXT, "_exit"); real(s); for (;;) {} }
void _Exit(int s) { _exit(s); }
CEOF
gcc -shared -fPIC -O1 -o $SCR/fl.so $SCR/fl.c -ldl && echo "shim built" | tee -a $R
ARGS="$F --ref-base $REF --genome-dir $GD/star_2.7.11b --exons $GD/genes/genes.gtf --out-prefix OUTP --threads 8 --pipeline --output-format 1pz --engine gpu --no-hla --no-editing"
arm() {  # tag mode cap_s envx
  local tag=$1 mode=$2 cap=$3 ex="$4" A=$O/$1 SC=$SCR/$1 pre="" xe="" rc st en t
  mkdir -p $A $SC/out; local AR=${ARGS//OUTP/$SC/out/}
  if [ $mode = nsys ]; then
    pre="$NSYS profile --trace=cuda,nvtx,osrt --sample=none --cpuctxsw=none --trace-fork-before-exec=true --cuda-flush-interval=500 --force-overwrite=true --stats=false -o $SC/prof"; xe="env LD_PRELOAD=$SCR/fl.so"
  elif [ $mode = ncu ]; then
    pre="$NCU --target-processes all -k regex:k_p1s|k_p1w|k_p2|k_p3|k_wplan --launch-skip $SKIP --launch-count $COUNT --cache-control all --import-source yes --force-overwrite \
      --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section MemoryWorkloadAnalysis --section InstructionStats --section SourceCounters \
      --metrics smsp__thread_inst_executed_per_inst_executed.ratio,smsp__sass_average_branch_targets_threads_uniform.pct,l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,launch__registers_per_thread,launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum -o $SC/ncu"
  fi
  echo "######## $tag mode=$mode $(date -Is)" | tee -a $R; st=$(date +%s)
  ( cd $SC && TMPDIR=$SCR/nstmp timeout -k 30 $cap env $ex $pre $xe $BIN $AR ) 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $A/run.log
  rc=${PIPESTATUS[0]}; en=$(date +%s)
  cp $SC/out/star_Log.out $A/ 2>/dev/null; grep -a -i -E "engine_gpu|P3-MEM-03|E12 |E17 |budget" $SC/out/star_Log.out > $A/engine.log 2>/dev/null
  echo "ARM $tag rc=$rc wall=$(( en - st )) s" | tee -a $R
  if [ $mode = nsys ]; then
    $NSYS export --type sqlite --force-overwrite=true -o $SC/prof.sqlite $SC/prof.nsys-rep > $A/export.log 2>&1; cp $SC/prof.sqlite $A/
    python3 $O/rt_prof_an.py $A/prof.sqlite > $A/an.txt 2>&1; echo "  analysis rc=$? lines $(wc -l < $A/an.txt)" | tee -a $R; cp $SC/prof.nsys-rep $A/ 2>/dev/null
  elif [ $mode = ncu ]; then
    ls -la $SC/ncu.ncu-rep 2>&1 | tee -a $R; cp $SC/ncu.ncu-rep $A/ 2>/dev/null
    $NCU --import $A/ncu.ncu-rep --page details --print-summary per-kernel > $A/details.txt 2>&1
    $NCU --import $A/ncu.ncu-rep --page raw --csv > $A/raw.csv 2>&1
    echo "  ncu text lines $(wc -l < $A/details.txt)" | tee -a $R
  fi
  rm -rf $SC; return 0
}
arm BASE nsys 2400 "SINGLET_LANE_STREAMS=1 SINGLET_LANE_SORTA=0 SINGLET_LANE_SORTB=0"
arm SORT nsys 2400 "SINGLET_LANE_STREAMS=1"
arm STR2 nsys 2400 "SINGLET_LANE_STREAMS=2 SINGLET_LANE_SORTA=0 SINGLET_LANE_SORTB=0"
arm DEF nsys 2400 ""
arm STR3 nsys 2400 "SINGLET_LANE_STREAMS=3"
echo "done $(date -Is)" | tee -a $R
