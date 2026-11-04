#!/bin/bash
#SBATCH --job-name=e8-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E8 gate (a) on pbmc1k_v3 (docs/gpu-engine/fable/E8-e2e.md), the binary of bench/fable/e2e/jobs/build_singlet.sbatch. Modes (MODES, in order):
#   cpu   = the CPU engine
#   gpuE  = --engine gpu + SINGLET_GPU_PILEUP=on + SINGLET_ENGINE_GPU_E2E=1 (records and the hand-off pass on the device, no hand-off file)
#   gpuEc = gpuE + SINGLET_ENGINE_GPU_E2E_CHECK=1 (the device-gathered records checked against the host-assembled ones; slower)
#   gpuD  = --engine gpu + SINGLET_GPU_PILEUP=on, E2E unset (the default path of the same binary: must be unchanged)
# Every run: timestamped log, 1-s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time; then out_equiv.py on every pair with cpu
# and gpuD vs gpuE. Read sequences stay in job /tmp (trap); the outputs without sequence-bearing files are kept in $RES/<mode>/out, and a
# mode run in an earlier job is compared through OUT_<mode>=<kept dir>. T (threads, default 18) and the sbatch resources are overridable on
# the command line (debug-gpu: 8 CPUs, 29 min; a cpu-only run needs no GPU).
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/e2e/bin}; T=${T:-18}; s=${SAMPLE:-pbmc1k_v3}; MODES=(${MODES:-gpuE gpuD cpu})
BARE=$W/repos/singlet-py.git
# singlet resolves the barcode whitelist at <binary dir>/../whitelists; without it the run silently takes the sorted-BAM path
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES=$W/jobs/fable/e2e/results/gate-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e8g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE archive refs/fable/e2e bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(cat $BINDIR/commit) T=$T modes=${MODES[*]} $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
ls -l $FQ | tee -a $RES/summary.txt
for m in "${MODES[@]}"; do
  mkdir -p $SCR/$s/$m/out; O=$RES/$m; mkdir -p $O
  unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_ENGINE_GPU_E2E_CHECK; BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
  case $m in
    cpu) ENGARG="" ;;
    gpuD) export SINGLET_GPU_PILEUP=on ;;
    gpuE) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1 ;;
    gpuEc) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1 SINGLET_ENGINE_GPU_E2E_CHECK=1 ;;
  esac
  ( echo -e "t\tcpu_s\tgpu_util\tgpu_mem_mb"
    while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')
      printf "%s\t%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
    done ) > $O/monitor.tsv 2>/dev/null &
  MON=$!
  T0=$(date +%s.%N); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/$s/$m && /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON 2>/dev/null
  echo "$s $m: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), start $T0 end $T1, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $O/time.txt)" | tee -a $RES/summary.txt
  grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: STREAM PATH NOT USED ($m)" | tee -a $RES/summary.txt; FAIL=1; }
  case $m in gpu*) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP OK" | tee -a $RES/summary.txt; else echo "!! GATE FAIL: GPU pileup did not run cleanly ($m)" | tee -a $RES/summary.txt; FAIL=1; fi
                 grep -aq "\[gpu_pileup\] hand-off pass:" $O/run.log.ts && echo "GPU PILEUP HAND-OFF OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: no GPU-pileup hand-off pass ($m)" | tee -a $RES/summary.txt; FAIL=1; } ;; esac
  case $m in gpuE*) grep -aq "E8: on the device" $O/run.log.ts && echo "E8 DEVICE PASS OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: E8 path not taken ($m)" | tee -a $RES/summary.txt; FAIL=1; } ;; esac
  case $m in gpuD) ! grep -aq "E8" $O/run.log.ts && echo "DEFAULT PATH: no E8 line" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: E8 active without SINGLET_ENGINE_GPU_E2E ($m)" | tee -a $RES/summary.txt; FAIL=1; } ;; esac
  grep -a -e "unsorted STAR stream" -e "stream CB" -e "streaming from STAR pipe" -e "sorted path" -e "gpu_pileup\]" -e "ignor" -e "CR4 probe" -e "whitelist" -e "E8" $O/run.log.ts | cut -c1-400 | tee -a $RES/summary.txt
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-400 | head -80 > $O/starlog_engine.txt
  cat $O/starlog_engine.txt | tee -a $RES/summary.txt
  ls -la $SCR/$s/$m/out/ | grep -a "handoff" | tee -a $RES/summary.txt
  cp $SCR/$s/$m/out/*Log.final.out $O/ 2>/dev/null
  cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
  # the outputs for comparisons across jobs (OUT_<mode>=<dir> in a later job): everything but files that can hold read sequences
  [ "${KEEPOUT:-1}" = 1 ] && rsync -a --exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq'       --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' $SCR/$s/$m/out/ $O/out/
done
# a mode's output: this job's run, else OUT_<mode> (a kept output of an earlier job, e.g. OUT_cpu=$W/jobs/fable/e2e/results/gate-<id>/cpu/out)
odir() { if [ -d $SCR/$s/$1/out ]; then echo $SCR/$s/$1/out; else v=OUT_$1; echo "${!v:-}"; fi; }
cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $1 vs $2 (not run)" | tee -a $RES/summary.txt; return; }
  echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -30 | tee -a $RES/summary.txt
  # the read-order files (STAR's threads write them in completion order): same records? (sorted 4-line records; only in this job's /tmp)
  for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    [ -f $a/$f ] && [ -f $b/$f ] || continue
    ha=$(paste - - - - < $a/$f | LC_ALL=C sort -S 2G | sha256sum | cut -c1-16); hb=$(paste - - - - < $b/$f | LC_ALL=C sort -S 2G | sha256sum | cut -c1-16)
    [ "$ha" = "$hb" ] && echo "  $f: same records in another order (sorted sha256 $ha)" | tee -a $RES/summary.txt || echo "  $f: RECORDS DIFFER (sorted sha256 $ha vs $hb)" | tee -a $RES/summary.txt
  done; }
cmp2 cpu gpuE; cmp2 cpu gpuD; cmp2 gpuD gpuE; cmp2 cpu gpuEc
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES/summary.txt || echo "GATE FAILED" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
