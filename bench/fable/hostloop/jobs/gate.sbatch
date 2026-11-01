#!/bin/bash
#SBATCH --job-name=e10-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E10 (docs/gpu-engine/fable/E10-hostloop.md): profile and gate (a) on pbmc1k_v3, from E8's bench/fable/e2e/jobs/gate.sbatch. Binaries of
# bench/fable/hostloop/jobs/build.sbatch in $BINDIR (singlet-gpu; singlet-prof = -DSTAR_PROFILE). Modes (MODES, in order):
#   cpu    the CPU engine
#   gpuD   --engine gpu + SINGLET_GPU_PILEUP=on                         (the default GPU path)
#   gpuE   gpuD + SINGLET_ENGINE_GPU_E2E=1                              (E8)
#   prof   gpuE with the STAR_PROFILE binary                            (STAR's per-phase counters in Log.out)
#   gpuE2  gpuD + SINGLET_ENGINE_GPU_E2E=2   (E10 steps 2-3: per-read entries, Log.final.out counters, SJ.out.tab and the unmapped reads from the device)
#   gpuE2c gpuE2 + SINGLET_ENGINE_GPU_E10_CHECK=1 (STAR also writes the entries; compared byte for byte at the end of mapping)
#   gpuE3  gpuD + SINGLET_ENGINE_GPU_E2E=3   (step 4a: STAR's loop keeps only readLoad + barcode + Solo for reads with a record)
#   gpuE4  gpuD + SINGLET_ENGINE_GPU_E2E=4   (step 4b: reads with a record are not given to STAR; Solo outputs incomplete)
#   prof3 / prof4: gpuE3 / gpuE4 with the STAR_PROFILE binary
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time. Then out_equiv.py on the pairs (PAIRS=a:b ...),
# the mapping statistics of star_Log.final.out without its 4 timing lines, SJ.out.tab, and the read-order files by sorted 4-line records.
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/hostloop/bin}; T=${T:-18}; s=${SAMPLE:-pbmc1k_v3}; MODES=(${MODES:-gpuE prof})
BARE=$W/repos/singlet-py.git
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES=$W/jobs/fable/hostloop/results/gate-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e10g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE archive refs/fable/hostloop-src bench/star_accel/out_equiv.py bench/engine_gpu_gate/s8_stages.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID commit=$(cat $BINDIR/commit-gpu 2>/dev/null) prof=$(cat $BINDIR/commit-prof 2>/dev/null) T=$T modes=${MODES[*]} $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
ls -l $FQ | tee -a $RES/summary.txt
for m in "${MODES[@]}"; do
  mkdir -p $SCR/$s/$m/out; O=$RES/$m; mkdir -p $O
  unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_ENGINE_GPU_E2E_CHECK SINGLET_ENGINE_GPU_E10_CHECK SINGLET_ENGINE_GPU_E10_R11; BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
  case $m in
    cpu) ENGARG="" ;;
    gpuD) export SINGLET_GPU_PILEUP=on ;;
    gpuE) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1 ;;
    prof) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1; BIN=$BINDIR/singlet-prof ;;
    gpuE2) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=2 ;;
    gpuE2c) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=2 SINGLET_ENGINE_GPU_E10_CHECK=1 SINGLET_ENGINE_GPU_E10_R11=$O/r11_device.bin ;;
    gpuE3) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=3 ;;
    gpuE4) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=4 ;;
    prof3) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=3; BIN=$BINDIR/singlet-prof ;;
    prof4) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=4; BIN=$BINDIR/singlet-prof ;;
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
  case $m in cpu) ;; *) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP OK" | tee -a $RES/summary.txt; else echo "!! GATE FAIL: GPU pileup did not run cleanly ($m)" | tee -a $RES/summary.txt; FAIL=1; fi ;; esac
  grep -a -e "unsorted STAR stream" -e "streaming from STAR pipe" -e "sorted path" -e "gpu_pileup\]" -e "ignor" -e "E8" -e "E10" $O/run.log.ts | cut -c1-400 | tee -a $RES/summary.txt
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "STAR_PROFILE" -e "E10" -e "loading the genome" -e "started mapping" -e "finished mapping" -e "Loading SA" -e "Loading Genome" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-6000 | head -120 > $O/starlog_engine.txt
  cut -c1-600 $O/starlog_engine.txt | tee -a $RES/summary.txt
  cp $SCR/$s/$m/out/*Log.final.out $SCR/$s/$m/out/*Log.out $O/ 2>/dev/null
  if [ -f $O/r11_device.bin ]; then   # E10: the device's R11 exact CB histogram against r11_hist.cpp's (E6, job 795356 era; same .1fq and whitelist)
    R11REF=${R11REF:-$W/jobs/fable/records/results/r11_exact_hist.bin}
    if cmp -s $O/r11_device.bin $R11REF; then echo "  R11 exact CB histogram: device == r11_hist.cpp byte for byte ($(stat -c %s $R11REF) bytes)" | tee -a $RES/summary.txt
    else echo "  R11 exact CB histogram: DIFFERS from $R11REF ($(stat -c %s $O/r11_device.bin) vs $(stat -c %s $R11REF) bytes)" | tee -a $RES/summary.txt
         python3 -c "import struct,sys;a=open(sys.argv[1],'rb').read(32);b=open(sys.argv[2],'rb').read(32);print('  headers device',struct.unpack('<3Q',a[8:]),'ref',struct.unpack('<3Q',b[8:]))" $O/r11_device.bin $R11REF | tee -a $RES/summary.txt; fi
  fi
  cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
  python3 $SCR/repo/bench/engine_gpu_gate/s8_stages.py $O 66.601887 > $O/stages.txt 2>&1; tee -a $RES/summary.txt < $O/stages.txt
  [ "${KEEPOUT:-1}" = 1 ] && rsync -a --exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' \
      --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' $SCR/$s/$m/out/ $O/out/
done
odir() { if [ -d $SCR/$s/$1/out ]; then echo $SCR/$s/$1/out; else v=OUT_$1; echo "${!v:-}"; fi; }
cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $1 vs $2 (not run)" | tee -a $RES/summary.txt; return; }
  echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -40 | tee -a $RES/summary.txt
  if [ -f $a/star_Log.final.out ] && [ -f $b/star_Log.final.out ]; then
    if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then echo "  star_Log.final.out: mapping statistics identical (timing lines excluded)" | tee -a $RES/summary.txt
    else echo "  star_Log.final.out: MAPPING STATISTICS DIFFER" | tee -a $RES/summary.txt; diff <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) | head -20 | tee -a $RES/summary.txt; fi
  fi
  if [ -f $a/star_SJ.out.tab ] && [ -f $b/star_SJ.out.tab ]; then
    if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then echo "  star_SJ.out.tab: identical ($(wc -l < $a/star_SJ.out.tab) junctions)" | tee -a $RES/summary.txt; else echo "  star_SJ.out.tab: DIFFERS" | tee -a $RES/summary.txt; fi
  fi
  for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    [ -f $a/$f ] && [ -f $b/$f ] || continue
    ha=$(paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=8 | sha256sum | cut -c1-16); hb=$(paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=8 | sha256sum | cut -c1-16)
    if [ "$ha" = "$hb" ]; then echo "  $f: same records (sorted sha256 $ha; $(( $(wc -l < $a/$f) / 4 )) reads)" | tee -a $RES/summary.txt; else echo "  $f: RECORDS DIFFER (sorted sha256 $ha vs $hb)" | tee -a $RES/summary.txt; fi
  done; }
for p in ${PAIRS:-cpu:gpuE}; do cmp2 ${p%%:*} ${p##*:}; done
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES/summary.txt || echo "GATE FAILED" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
