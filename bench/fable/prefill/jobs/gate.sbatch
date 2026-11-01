#!/bin/bash
#SBATCH --job-name=e12-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E12 (docs/gpu-engine/fable/E12-prefill.md): gate (a) and timings on pbmc1k_v3, from E11's bench/fable/solowire/jobs/gate.sbatch. Binaries of
# bench/fable/prefill/jobs/build.sbatch in $BINDIR. A mode is KIND[_TAG]; KIND sets the switches:
#   cpu    the CPU engine
#   gpuD   --engine gpu + SINGLET_GPU_PILEUP=on        (the default GPU path)
#   gpuE   gpuD + SINGLET_ENGINE_GPU_E2E=1             (E8)
#   gpuE3  gpuD + SINGLET_ENGINE_GPU_E2E=3             (E10 level 3)
#   gpuE4  gpuD + SINGLET_ENGINE_GPU_E2E=4             (E10 level 4)
#   gpuE5  gpuD + SINGLET_ENGINE_GPU_E2E=5             (E11: level 5)
# and per mode BIN_<mode> names the binary in $BINDIR (default singlet-gpu) and ENV_<mode> adds "K=V ..." to its environment, e.g.
#   MODES="gpuE5_base gpuE5" BIN_gpuE5_base=singlet-base ENV_gpuE5="SINGLET_ENGINE_GPU_CFG=old"
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time; Gene/CellReads.stats hard-linked the moment it
# appears (the pipeline deletes it after parsing). Then per pair (PAIRS=a:b ...; a mode not run here is taken from OUT_<mode>, a kept out/ dir):
# out_equiv.py, the mapping statistics of star_Log.final.out, SJ.out.tab, the read-order files by sorted 4-line records, and CellReads.stats
# by sorted rows (CRS_<mode>: a kept copy for a mode not run here). DUMP=1: every GPU run also writes mapDistinct's records (E6 dump,
# SINGLET_ENGINE_GPU_DUMP) to the node's /tmp, and the dumps of all GPU runs are compared byte for byte (sha256) at the end.
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/prefill/bin}; T=${T:-18}; s=${SAMPLE:-pbmc1k_v3}; MODES=(${MODES:-cpu gpuE5})
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/prefill}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES=$W/jobs/fable/prefill/results/gate-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e12g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/prefill-src
git -C $BARE archive refs/fable/prefill-src bench/star_accel/out_equiv.py bench/fable/solowire/s8_stages_e11.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID T=$T modes=${MODES[*]} $(date -Is)" | tee $RES/summary.txt
for f in $BINDIR/commit-*; do echo "  $(basename $f): $(cat $f)"; done | tee -a $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/GRCh38-2024-A; FQ=$PB/data/$s/$s.1fq
ls -l $FQ | tee -a $RES/summary.txt
for m in "${MODES[@]}"; do
  mkdir -p $SCR/$s/$m/out; O=$RES/$m; mkdir -p $O
  unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_ENGINE_GPU_E2E_CHECK SINGLET_ENGINE_GPU_E10_CHECK SINGLET_ENGINE_GPU_E10_R11 SINGLET_ENGINE_GPU_DUMP
  ENGARG="--engine gpu"; kind=${m%%_*}
  bv=BIN_$m; BIN=$BINDIR/${!bv:-singlet-gpu}
  case $kind in
    cpu) ENGARG="" ;;
    gpuD) export SINGLET_GPU_PILEUP=on ;;
    gpuE) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=1 ;;
    gpuE3) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=3 ;;
    gpuE4) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=4 ;;
    gpuE5) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 ;;
    *) echo "!! unknown mode $m" | tee -a $RES/summary.txt; continue ;;
  esac
  ev=ENV_$m; XENV=(${!ev:-})
  [ "${DUMP:-0}" = 1 ] && [ $kind != cpu ] && XENV+=(SINGLET_ENGINE_GPU_DUMP=$SCR/dump-$m.bin)
  echo "--- $m: binary $BIN ($(sha256sum $BIN | cut -c1-16)), extra env: ${XENV[*]:-none}" | tee -a $RES/summary.txt
  ( echo -e "t\tcpu_s\tgpu_util\tgpu_mem_mb"
    while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')
      printf "%s\t%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
    done ) > $O/monitor.tsv 2>/dev/null &
  MON=$!
  CRS=$SCR/$s/$m/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $SCR/$s/$m/CellReads.stats ) &
  WATCH=$!
  T0=$(date +%s.%N); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/$s/$m && env "${XENV[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON $WATCH 2>/dev/null
  echo "$s $m: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), start $T0 end $T1, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $O/time.txt)" | tee -a $RES/summary.txt
  grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: STREAM PATH NOT USED ($m)" | tee -a $RES/summary.txt; FAIL=1; }
  case $kind in cpu) ;; *) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP OK" | tee -a $RES/summary.txt; else echo "!! GATE FAIL: GPU pileup did not run cleanly ($m)" | tee -a $RES/summary.txt; FAIL=1; fi ;; esac
  grep -a -e "unsorted STAR stream" -e "streaming from STAR pipe" -e "sorted path" -e "gpu_pileup\]" -e "ignor" -e "E8" -e "E10" -e "E11" -e "cell_qc\]" $O/run.log.ts | cut -c1-400 | tee -a $RES/summary.txt
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "E10" -e "E11" -e "soloRun" -e "Solo counting" -e "loading the genome" -e "started mapping" -e "finished mapping" -e "Loading SA" -e "Loading Genome" -e "ALL DONE" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-6000 | head -200 > $O/starlog_engine.txt
  cut -c1-1500 $O/starlog_engine.txt | tee -a $RES/summary.txt
  cp $SCR/$s/$m/out/*Log.final.out $SCR/$s/$m/out/*Log.out $O/ 2>/dev/null
  [ -f $SCR/$s/$m/CellReads.stats ] && cp $SCR/$s/$m/CellReads.stats $O/CellReads.stats && echo "  CellReads.stats kept: $(wc -l < $O/CellReads.stats) lines" | tee -a $RES/summary.txt
  cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
  python3 $SCR/repo/bench/fable/solowire/s8_stages_e11.py $O 66.601887 > $O/stages.txt 2>&1; tee -a $RES/summary.txt < $O/stages.txt
  [ "${KEEPOUT:-1}" = 1 ] && rsync -a --exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' \
      --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' $SCR/$s/$m/out/ $O/out/
  [ -f $SCR/dump-$m.bin ] && echo "  E6 dump $m: $(stat -c %s $SCR/dump-$m.bin) bytes, sha256 $(sha256sum $SCR/dump-$m.bin | cut -c1-16)" | tee -a $RES/summary.txt
done
if [ "${DUMP:-0}" = 1 ]; then
  echo "=== mapDistinct records (E6 dumps) of the GPU runs" | tee -a $RES/summary.txt
  ref=""; for m in "${MODES[@]}"; do f=$SCR/dump-$m.bin; [ -f $f ] || continue
    if [ -z "$ref" ]; then ref=$m; echo "  reference $m" | tee -a $RES/summary.txt
    elif cmp -s $SCR/dump-$ref.bin $f; then echo "  $m: records IDENTICAL to $ref (byte for byte)" | tee -a $RES/summary.txt
    else echo "  $m: records DIFFER from $ref: $(cmp $SCR/dump-$ref.bin $f | head -1)" | tee -a $RES/summary.txt; FAIL=1; fi
  done
fi
odir() { if [ -d $SCR/$s/$1/out ]; then echo $SCR/$s/$1/out; else v=OUT_$1; echo "${!v:-}"; fi; }
crs() { if [ -f $RES/$1/CellReads.stats ]; then echo $RES/$1/CellReads.stats; else v=CRS_$1; echo "${!v:-}"; fi; }
cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $1 vs $2 (not run)" | tee -a $RES/summary.txt; return; }
  echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -60 | tee -a $RES/summary.txt
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
  done
  ca=$(crs $1); cb=$(crs $2)
  if [ -n "$ca" ] && [ -f "$ca" ] && [ -n "$cb" ] && [ -f "$cb" ]; then
    if cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then
      echo "  Gene/CellReads.stats: same rows (sorted; $(( $(wc -l < $ca) - 1 )) rows; byte-identical in file order: $(cmp -s $ca $cb && echo yes || echo no))" | tee -a $RES/summary.txt
    else echo "  Gene/CellReads.stats: ROWS DIFFER ($(wc -l < $ca) vs $(wc -l < $cb) lines)" | tee -a $RES/summary.txt
         diff <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort) | head -8 | cut -c1-200 | tee -a $RES/summary.txt; fi
  else echo "  Gene/CellReads.stats: not compared (no copy for $1 or $2)" | tee -a $RES/summary.txt; fi; }
for p in ${PAIRS:-}; do cmp2 ${p%%:*} ${p##*:}; done
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES/summary.txt || echo "GATE FAILED" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
