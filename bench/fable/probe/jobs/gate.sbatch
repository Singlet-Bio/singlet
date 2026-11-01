#!/bin/bash
#SBATCH --job-name=e14-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E14 (docs/gpu-engine/fable/E14-probe.md): gate (a) and timings, from E11's bench/fable/solowire/jobs/gate.sbatch. Two binaries of
# bench/fable/probe/jobs/build.sbatch: the new one in $BINDIR, the baseline (8e54be40, the E11 head) in $BASEDIR. Modes (MODES, in order):
#   cpu    the CPU engine                                       cpuB    the same with the baseline binary
#   gpuE5  --engine gpu + SINGLET_GPU_PILEUP=on + SINGLET_ENGINE_GPU_E2E=5 (level 5)    gpuE5B  the same with the baseline binary
#   gpuD   --engine gpu + SINGLET_GPU_PILEUP=on                  gpuDB
# A mode name may end in "+ENV=VALUE" pairs separated by "," after a colon, e.g. "cpu:SINGLET_CR4_PROBE=serial" (results dir cpu-1).
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time; Gene/CellReads.stats hard-linked the
# moment it appears. Per run: s8_stages_e11.py (E11's stage table) and s8_probe_e14.py (the probe stage in detail). Then per pair
# (PAIRS=a:b ...; a mode not run here is taken from OUT_<mode>, a kept out/ dir): out_equiv.py, the mapping statistics of
# star_Log.final.out, SJ.out.tab, the read-order files by sorted 4-line records, CellReads.stats by sorted rows, and cr4_probe.json.
# SAMPLE=neuron1k_v3 GENOME=GRCm39-2024-A READSM=92.902231 for the second sample.
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/probe/bin}; BASEDIR=${BASEDIR:-$W/jobs/fable/probe/bin-base}; T=${T:-18}; s=${SAMPLE:-pbmc1k_v3}
GEN=${GENOME:-GRCh38-2024-A}; READSM=${READSM:-66.601887}; MODES=(${MODES:-cpuB cpu gpuE5B gpuE5})
BARE=$W/repos/singlet-py.git
for d in $BINDIR $BASEDIR; do [ -e $d/../whitelists ] || ln -s $PB/whitelists $d/../whitelists; done
RES=$W/jobs/fable/probe/results/gate-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e14g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/probe:refs/fable/probe-src
git -C $BARE archive refs/fable/probe-src bench/star_accel/out_equiv.py bench/fable/solowire/s8_stages_e11.py bench/fable/probe/s8_probe_e14.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID new=$(cat $BINDIR/commit-gpu 2>/dev/null) base=$(cat $BASEDIR/commit-gpu 2>/dev/null) T=$T sample=$s modes=${MODES[*]} $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
G=$REF/$GEN; FQ=$PB/data/$s/$s.1fq
ls -l $FQ | tee -a $RES/summary.txt
t0=$(date +%s.%N); cat $G/star_2.7.11b/Genome $G/star_2.7.11b/SA $G/star_2.7.11b/SAindex > /dev/null   # same page cache for every run
echo "index into the page cache: $(echo "$(date +%s.%N) - $t0" | bc) s" | tee -a $RES/summary.txt
for spec in "${MODES[@]}"; do
  m=${spec%%:*}; envs=""; [ "$spec" != "$m" ] && envs=${spec#*:}
  name=$m; k=1; while [ -d $RES/$name ]; do name=$m-$k; k=$((k+1)); done
  mkdir -p $SCR/$s/$name/out; O=$RES/$name; mkdir -p $O
  unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_CR4_PROBE SINGLET_GENOME_PRELOAD; BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
  case $m in *B) BIN=$BASEDIR/singlet-gpu ;; esac
  case ${m%B} in
    cpu) ENGARG="" ;;
    gpuD) export SINGLET_GPU_PILEUP=on ;;
    gpuE5) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 ;;
  esac
  for e in ${envs//,/ }; do export "$e"; done
  echo "== $name: $BIN $ENGARG ${envs}" | tee -a $RES/summary.txt
  ( echo -e "t\tcpu_s\tgpu_util\tgpu_mem_mb"
    while true; do
      g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')
      printf "%s\t%s\t%s\n" $(date +%s.%N) $(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
    done ) > $O/monitor.tsv 2>/dev/null &
  MON=$!
  CRS=$SCR/$s/$name/out/star_Solo.out/Gene/CellReads.stats
  ( while [ ! -e $CRS ]; do sleep 0.2; done; ln $CRS $SCR/$s/$name/CellReads.stats ) &
  WATCH=$!
  T0=$(date +%s.%N); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/$s/$name && /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$s/$name/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON $WATCH 2>/dev/null
  for e in ${envs//,/ }; do unset "${e%%=*}"; done
  echo "$s $name: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), start $T0 end $T1, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $O/time.txt)" | tee -a $RES/summary.txt
  grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH OK" | tee -a $RES/summary.txt || { echo "!! GATE FAIL: STREAM PATH NOT USED ($name)" | tee -a $RES/summary.txt; FAIL=1; }
  case ${m%B} in cpu) ;; *) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP OK" | tee -a $RES/summary.txt; else echo "!! GATE FAIL: GPU pileup did not run cleanly ($name)" | tee -a $RES/summary.txt; FAIL=1; fi ;; esac
  grep -a -e "Genome preload" -e "Genome prefetch" -e "STAR pid" -e "CR4 probe" -e "Beta aligner" -e "E11" $O/run.log.ts | cut -c1-400 | tee -a $RES/summary.txt
  grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "E10" -e "E11" -e "soloRun" -e "Solo counting" -e "loading the genome" -e "Genome preload" -e "started mapping" -e "finished mapping" -e "Loading SA" -e "Loading Genome" -e "ALL DONE" $SCR/$s/$name/out/*Log.out 2>/dev/null | cut -c1-6000 | head -160 > $O/starlog_engine.txt
  cp $SCR/$s/$name/out/*Log.final.out $SCR/$s/$name/out/*Log.out $O/ 2>/dev/null
  [ -f $SCR/$s/$name/CellReads.stats ] && cp $SCR/$s/$name/CellReads.stats $O/CellReads.stats && echo "  CellReads.stats kept: $(wc -l < $O/CellReads.stats) lines" | tee -a $RES/summary.txt
  cp $SCR/$s/$name/out/read_stats.tsv $SCR/$s/$name/out/summary.json $SCR/$s/$name/out/cr4_probe.json $O/ 2>/dev/null
  echo "  cr4_probe.json: $(cat $O/cr4_probe.json 2>/dev/null)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/fable/solowire/s8_stages_e11.py $O $READSM > $O/stages.txt 2>&1; tee -a $RES/summary.txt < $O/stages.txt
  python3 $SCR/repo/bench/fable/probe/s8_probe_e14.py $O > $O/probe_stage.txt 2>&1; tee -a $RES/summary.txt < $O/probe_stage.txt
  [ "${KEEPOUT:-1}" = 1 ] && rsync -a --exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' \
      --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' $SCR/$s/$name/out/ $O/out/
done
odir() { if [ -d $SCR/$s/$1/out ]; then echo $SCR/$s/$1/out; else v=OUT_$1; echo "${!v:-}"; fi; }
crs() { if [ -f $RES/$1/CellReads.stats ]; then echo $RES/$1/CellReads.stats; else v=CRS_$1; echo "${!v:-}"; fi; }
cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $1 vs $2 (not run)" | tee -a $RES/summary.txt; return; }
  echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES/summary.txt
  python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
  grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -60 | tee -a $RES/summary.txt
  if cmp -s $a/cr4_probe.json $b/cr4_probe.json; then echo "  cr4_probe.json: byte-identical ($(cat $a/cr4_probe.json))" | tee -a $RES/summary.txt
  else echo "  cr4_probe.json: DIFFERS" | tee -a $RES/summary.txt; cat $a/cr4_probe.json $b/cr4_probe.json | tee -a $RES/summary.txt; fi
  for t in plain cr4; do
    [ -f $a/cr4_probe/${t}_SJ.out.tab ] && [ -f $b/cr4_probe/${t}_SJ.out.tab ] && { cmp -s $a/cr4_probe/${t}_SJ.out.tab $b/cr4_probe/${t}_SJ.out.tab && echo "  cr4_probe/${t}_SJ.out.tab: identical" || echo "  cr4_probe/${t}_SJ.out.tab: DIFFERS"; } | tee -a $RES/summary.txt
    [ -f $a/cr4_probe/${t}_Log.final.out ] && [ -f $b/cr4_probe/${t}_Log.final.out ] && { cmp -s <(tail -n +5 $a/cr4_probe/${t}_Log.final.out) <(tail -n +5 $b/cr4_probe/${t}_Log.final.out) && echo "  cr4_probe/${t}_Log.final.out: mapping statistics identical" || { echo "  cr4_probe/${t}_Log.final.out: MAPPING STATISTICS DIFFER"; diff <(tail -n +5 $a/cr4_probe/${t}_Log.final.out) <(tail -n +5 $b/cr4_probe/${t}_Log.final.out) | head -10; }; } | tee -a $RES/summary.txt
  done
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
for p in ${PAIRS:-cpuK:cpu cpu:gpuE5 cpuB:cpu gpuE5B:gpuE5}; do cmp2 ${p%%:*} ${p##*:}; done
echo "ipcs -m at the end: $(ipcs -m | grep -c '^0x') segments" | tee -a $RES/summary.txt
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES/summary.txt || echo "GATE FAILED" | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
