#!/bin/bash
#SBATCH --job-name=e1-ncu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1 profile (docs/gpu-engine/fable/E1-profile.md): Nsight Compute 2026.2 (CUDA 13.3 image) on k_p1, k_p2, k_p3 of each staged lane,
# --set full with source counters (m5 built with -lineinfo), on the standard 2 M pbmc1k_v3 dump set; then the one-kernel lane A
# with its Work in global memory (array of structs) and in local memory (interleaved per thread). Kernels are selected by their
# mangled names (Caps template arguments): A = Caps<160,4,32,16,16,256,...>, M = <160,4,32,64,8,1024,...>, B = <160,8,128,128,50,4096,...>.
# The staged lanes are profiled on a full batch: A and M the 2nd batch (--launch-skip 3), B the 1st (its 2nd is a partial batch).
# Read sequences stay in job /tmp; reports (.ncu-rep: metrics and source, no read data) go to $OUT.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/GRCh38-2024-A/star_2.7.11b; SD=$W/jobs/w11/bin/stardump; E=$W/jobs/fable/profile
IMG=$W/images/cuda-13-devel.sif; PY=$W/images/python-3.12-bookworm.sif; NCU=/usr/local/cuda/bin/ncu; NSYS=/usr/local/bin/nsys
QI=/opt/nvidia/nsight-compute/2026.2.0/host/linux-desktop-glibc_2_11_3-x64/QdstrmImporter   # /usr/local/bin/nsys cannot find it itself
OUT=$E/results/ncu-$SLURM_JOB_ID; mkdir -p $OUT
SCR=/tmp/$USER-e1n-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export LC_ALL=C
MAXD=${MAXD:-2000000}; STEPS=,${STEPS:-nsys,tail,A,B,M,monog,monol},; NBA=${NBA:-262144}; SET=${SET:-full}; PTO=${PTO:-2700}
echo "job=$SLURM_JOB_ID node=$(hostname) $(date -Is) MAXD=$MAXD steps=$STEPS NBA=$NBA set=$SET m5=$(cat $E/bin/m5.commit)"
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
apptainer exec $IMG $NCU --version | tail -1
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
$PB/bin/singlet-w3 decode $PB/data/pbmc1k_v3/pbmc1k_v3.1fq --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' \
  | sort -S 60G --parallel=16 -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
echo "distinct=$ND every_k=$K selected=$(( $(wc -l < $SCR/distinct.fq) / 4 ))"; rm -f $SCR/distinct.txt
mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > $SCR/star.out 2>&1
echo "stardump exit=$?"; rm -f $SCR/distinct.fq; echo "[t] truth $(( $(date +%s) - T0 )) s"
DEF="--batch-a $NBA --batch-m 262144 --batch-b 65536 --budget-a 1024 --budget-m 2048 --budget-b 2048"
CA='4CapsILi160ELi4ELi32ELi16ELi16ELi256E'; CM='4CapsILi160ELi4ELi32ELi64ELi8ELi1024E'; CB='4CapsILi160ELi8ELi128ELi128ELi50ELi4096E'
has() { [[ $STEPS == *,$1,* ]]; }
m5run() {   # label binary args... ($PRE: profiler command in front of the binary, inside the container)
  local L=$1 B=$2; shift 2; local t=$(date +%s)
  echo "=== $L: $B $*"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 ${PRE:-} $E/bin/$B --genome-dir $SCR/idx --truth $SCR/m5/t     --json $OUT/$L.json "$@" 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $OUT/$L.log
  echo "exit=${PIPESTATUS[0]} wall=$(( $(date +%s) - t )) s"
  grep -E "^M5:|GPU lane A|TAILSUM|TAILEVSUM|error" $OUT/$L.log | head -20
}
if has nsys; then   # Nsight Systems timeline of one full run (3 reps after a warm-up); the .qdstrm is imported with QdstrmImporter
  PRE="$NSYS profile -t cuda --cuda-memory-usage=false --stats=false -o $SCR/nsys_base -f true" m5run nsys m5 --reps 3 $DEF
  ls -la $SCR | grep nsys_base
  if [ ! -f $SCR/nsys_base.nsys-rep ] && [ -f $SCR/nsys_base.qdstrm ]; then
    apptainer exec -B $W -B $SCR $IMG $QI --help 2>&1 | head -15
    apptainer exec -B $W -B $SCR $IMG $QI -i $SCR/nsys_base.qdstrm 2>&1 | tail -5
    [ -f $SCR/nsys_base.nsys-rep ] || apptainer exec -B $W -B $SCR $IMG $QI --input-file $SCR/nsys_base.qdstrm 2>&1 | tail -5
    cp $SCR/nsys_base.qdstrm $OUT/
  fi
  if [ -f $SCR/nsys_base.nsys-rep ]; then
    cp $SCR/nsys_base.nsys-rep $OUT/
    apptainer exec -B $W -B $SCR $IMG $NSYS stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum,cuda_gpu_trace -f csv       -o $OUT/nsys_base $OUT/nsys_base.nsys-rep > $OUT/nsys_stats.log 2>&1; echo "nsys stats exit=$?"; tail -3 $OUT/nsys_stats.log
  fi
  ls -la $OUT | grep nsys
fi
has tail && m5run tail m5t --reps 3 $DEF
prof() {   # label regex(mangled) skip count sections... -- m5 args
  local L=$1 RX=$2 SKIP=$3 CNT=$4; shift 4; local SEC=(); while [ "$1" != "--" ]; do SEC+=("$1"); shift; done; shift
  local t=$(date +%s)
  echo "=== $L: regex $RX skip $SKIP count $CNT ${SEC[*]} -- $*"
  timeout $PTO apptainer exec --nv -B $W -B $SCR $IMG $NCU "${SEC[@]}" --import-source yes --clock-control none \
    --kernel-name-base mangled --kernel-name "regex:$RX" --launch-skip $SKIP --launch-count $CNT -o $OUT/ncu_$L -f \
    env OMP_NUM_THREADS=24 $E/bin/m5 --genome-dir $SCR/idx --truth $SCR/m5/t --reps 0 "$@" 2>&1 \
    | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $OUT/ncu_$L.log
  echo "exit=${PIPESTATUS[0]} wall=$(( $(date +%s) - t )) s"
  grep -E "==PROF== Profiling|==ERROR==|==WARNING==|^M5:|GPU lane A|B/read" $OUT/ncu_$L.log | cut -c1-220 | head -20
  [ -f $OUT/ncu_$L.ncu-rep ] || return
  apptainer exec -B $W $IMG $NCU --import $OUT/ncu_$L.ncu-rep --page raw --csv > $OUT/ncu_${L}_raw.csv 2>&1
  apptainer exec -B $W $IMG $NCU --import $OUT/ncu_$L.ncu-rep --page details --csv > $OUT/ncu_${L}_details.csv 2>&1
  for k in k_p1 k_p2 k_p3 k_lane; do
    grep -q "$k" $OUT/ncu_${L}_raw.csv || continue
    apptainer exec -B $W $IMG $NCU --import $OUT/ncu_$L.ncu-rep --kernel-name-base function --kernel-name regex:"$k" --page source --csv \
      --print-source cuda > $OUT/ncu_${L}_${k}_src.csv 2>&1
  done
  ls -la $OUT | grep "ncu_$L" | awk '{print $5, $9}'
}
has A && prof A "_Z4k_p[123]IN2m5${CA}" 3 3 --set $SET -- $DEF
has B && prof B "_Z4k_p[123]IN2m5${CB}" 0 3 --set $SET -- $DEF
has M && prof M "_Z4k_p[123]IN2m5${CM}" 3 3 --set $SET -- $DEF
# one-kernel lane A (grid 132 x 4 blocks of 128, grid-stride loop over all reads, step budget 1024): global vs local Work
MSEC=(--section LaunchStats --section Occupancy --section SpeedOfLight --section MemoryWorkloadAnalysis --section MemoryWorkloadAnalysis_Tables
      --section MemoryWorkloadAnalysis_Chart --section WarpStateStats --section SchedulerStats --section ComputeWorkloadAnalysis --section SourceCounters)
has monog && prof monog "_Z6k_laneIN2m5${CA}" 0 1 "${MSEC[@]}" -- --mono --mono-budget 1024 $DEF
has monol && prof monol "_Z6k_laneIN2m5${CA}" 0 1 "${MSEC[@]}" -- --mono --local --mono-budget 1024 $DEF
BARE=$W/repos/singlet-py.git; mkdir -p $SCR/py
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/profile:refs/fable/profile && git -C $BARE archive refs/fable/profile bench/fable/profile | tar -x -C $SCR/py
[ -f $SCR/py/bench/fable/profile/ncu_summary.py ] && apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/ncu_summary.py $OUT > $OUT/summary.txt 2>&1
[ -f $SCR/py/bench/fable/profile/e1_summarize.py ] && apptainer exec -B $W -B $SCR $PY python3 $SCR/py/bench/fable/profile/e1_summarize.py $OUT > $OUT/summary_time.txt 2>&1
tail -80 $OUT/summary.txt 2>/dev/null; tail -40 $OUT/summary_time.txt 2>/dev/null
echo "[t] total $(( $(date +%s) - T0 )) s"
