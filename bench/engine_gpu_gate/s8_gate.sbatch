#!/bin/bash
#SBATCH --job-name=s8-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --array=0-1
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# S8 phase 1: host-stage profile of `--engine gpu` on prod/campaign (stream path, K5 on by the CR4 probe). One sample per array task.
# S8 gate on d9834b8a (GPU pileup under the default intron rules). Modes: cpu = CPU engine; gpuD = BINDIR (baseline) --engine gpu + SINGLET_GPU_PILEUP=on; gpuD_n = BINDIR_N (new).
# (old header) Modes: cpu (default engine), gpuA (--engine gpu, default intron rules: CPU pileup fallback), gpuB (--engine gpu + SINGLET_GPU_PILEUP=on
# + SINGLET_INTRON_RULES=0: everything that can be on the GPU is), prof (gpuB with the -DSTAR_PROFILE binary).
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU (utilization.gpu, memory.used), /usr/bin/time.
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/s8/bin_base2}; BINDIR_N=${BINDIR_N:?set BINDIR_N to the new build}; BINDIR_M=${BINDIR_M:-$BINDIR_N}; T=${T:-18}
TASKS=("pbmc1k_v3:cpu gpuD gpuD_m" "neuron1k_v3:cpu gpuD gpuD_m")
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}; s=${SPEC%%:*}; MODES=(${SPEC#*:})
BARE=$W/repos/singlet-py.git
# singlet resolves the barcode whitelist at <binary dir>/../whitelists; without it the run silently takes the sorted-BAM path
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES=$W/jobs/s8/results/gate-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $RES
SCR=/tmp/$USER-s8p-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE archive refs/s8/campaign bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(cat $BINDIR/commit) T=$T $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/GRCh38-2024-A; [ $s = neuron1k_v3 ] && G=$REF/GRCm39-2024-A
FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || { D=$W/data/w11/$s; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$s; FQ=$D/sample.1fq; }
ls -l $FQ | tee -a $RES/summary.txt
for m in "${MODES[@]}"; do
  mkdir -p $SCR/$s/$m/out; O=$RES/$m; mkdir -p $O
  unset SINGLET_GPU_PILEUP SINGLET_INTRON_RULES; BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
  case $m in cpu) ENGARG="" ;; gpuD|gpuD_n|gpuD_m) export SINGLET_GPU_PILEUP=on ;; esac
  case $m in *_n) BIN=$BINDIR_N/singlet-gpu ;; *_m) BIN=$BINDIR_M/singlet-gpu ;; esac
  ( echo -e "t\tcpu_s\tgpu_util\tgpu_mem_mb"
    while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')
      printf "%s\t%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
    done ) > $O/monitor.tsv 2>/dev/null &
  MON=$!
  T0=$(date +%s.%N); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/$s/$m && /usr/bin/time -v -o $O/time.txt timeout -k 30 7000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON 2>/dev/null
  echo "$s $m: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), start $T0 end $T1, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $O/time.txt)" | tee -a $RES/summary.txt
  grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: STREAM PATH NOT USED ($m)" | tee -a $RES/summary.txt; FAIL=1; }
  case $m in gpuD*) grep -aq "\[gpu_pileup\] hand-off pass:" $O/run.log.ts && echo "GPU PILEUP HAND-OFF OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: no GPU-pileup hand-off pass ($m)" | tee -a $RES/summary.txt; FAIL=1; } ;; esac
  grep -a -e "unsorted STAR stream" -e "stream CB" -e "streaming from STAR pipe" -e "sorted path" -e "gpu_pileup\] on" -e "gpu_pileup\] hand-off" -e "ignor" -e "CR4 probe" -e "whitelist" $O/run.log.ts | cut -c1-230 | tee -a $RES/summary.txt
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "STAR_PROFILE\|PROF" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-230 | head -80 > $O/starlog_engine.txt
  cp $SCR/$s/$m/out/*Log.final.out $O/ 2>/dev/null
  cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
done
cmp2() { [ -d $SCR/$s/$1/out ] && [ -d $SCR/$s/$2/out ] || { echo "!! GATE FAIL: missing $1 or $2" | tee -a $RES/summary.txt; FAIL=1; return; }
  echo "=== $s $1 vs $2" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$s/$1/out $SCR/$s/$2/out > $RES/equiv-$1-$2.txt 2>&1
  grep -a '^  DIFF\|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -30 | tee -a $RES/summary.txt; }
cmp2 cpu gpuD_m; cmp2 gpuD gpuD_m
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES/summary.txt || echo "GATE FAILED" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
