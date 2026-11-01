#!/bin/bash
#SBATCH --job-name=e15-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=02:45:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E15 (docs/gpu-engine/fable/E15-posterior.md): gates and timings of level 5 on several samples, from E11's bench/fable/solowire/jobs/gate.sbatch.
# Binary of bench/fable/posterior/jobs/build.sbatch in $BINDIR (singlet-gpu). Per sample in SAMPLES (genome by sample: GRCm39 for the mouse
# samples, the barnyard for hgmm1k_v3, else GRCh38), modes (MODES, in order):
#   cpu     the CPU engine
#   gpuD    --engine gpu + SINGLET_GPU_PILEUP=on        (the default GPU path)
#   gpuE3   gpuD + SINGLET_ENGINE_GPU_E2E=3             (E10 level 3)
#   gpuE5   gpuD + SINGLET_ENGINE_GPU_E2E=5             (E11/E15 level 5; kept inputs in page-locked host memory during STAR's loop)
#   gpuE5p  gpuE5 + SINGLET_E15_PERREAD=1               (the per-read posterior path even when one quality factor would do)
#   gpuE5k  gpuE5 + SINGLET_E15_KEEP_ON_DEVICE=1        (E11's behaviour: kept inputs stay on the device)
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time; Gene/CellReads.stats hard-linked the moment it
# appears (the pipeline deletes it after parsing). Then per pair (PAIRS=a:b ...; a mode not run here is taken from OUT_<mode>, a kept out/ dir;
# for pbmc1k_v3 OUT_cpu / CRS_cpu default to E11's gate 796377): out_equiv.py, the 10 STAR Solo files by cmp, the mapping statistics of
# star_Log.final.out, SJ.out.tab, the read-order files by sorted 4-line records, and CellReads.stats by sorted rows. Stages: s8_stages_e11.py
# with the run's input reads.
# Per sample: MODES_<sample> and PAIRS_<sample> override MODES and PAIRS.
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,T=8,SAMPLES=pbmc2m,... gate.sbatch
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/posterior/bin}; T=${T:-18}; SAMPLES=(${SAMPLES:-pbmc1k_v3}); MODES=(${MODES:-cpu gpuE5})
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/posterior}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES0=$W/jobs/fable/posterior/results/gate-$SLURM_JOB_ID; mkdir -p $RES0
SCR=/tmp/$USER-e15g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/posterior-gate
git -C $BARE archive refs/fable/posterior-gate bench/star_accel/out_equiv.py bench/fable/solowire/s8_stages_e11.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BINDIR commit=$(cat $BINDIR/commit-gpu 2>/dev/null) scripts=$(git -C $BARE rev-parse --short=8 refs/fable/posterior-gate) T=$T samples=${SAMPLES[*]} modes=${MODES[*]} $(date -Is)" | tee $RES0/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES0/summary.txt
for s in "${SAMPLES[@]}"; do
  RES=$RES0/$s; mkdir -p $RES
  G=$REF/GRCh38-2024-A
  case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
  FQ=$PB/data/$s/$s.1fq
  echo "######## $s ($(ls -l $FQ | awk '{print $5}') bytes; genome $G)" | tee -a $RES0/summary.txt
  mv=MODES_$s; MS=(${!mv:-${MODES[*]}}); pv=PAIRS_$s; PS="${!pv:-${PAIRS:-cpu:gpuE5}}"
  for m in "${MS[@]}"; do
    mkdir -p $SCR/$s/$m/out; O=$RES/$m; mkdir -p $O
    unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_E15_PERREAD SINGLET_E15_KEEP_ON_DEVICE; BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
    case $m in
      cpu) ENGARG="" ;;
      gpuD) export SINGLET_GPU_PILEUP=on ;;
      gpuE3) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=3 ;;
      gpuE5|gpuE5b) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 ;;
      gpuE5p) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_E15_PERREAD=1 ;;
      gpuE5k) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_E15_KEEP_ON_DEVICE=1 ;;
    esac
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
    ( cd $SCR/$s/$m && /usr/bin/time -v -o $O/time.txt timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $O/run.log.ts )
    C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON $WATCH 2>/dev/null
    echo "$s $m: wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), start $T0 end $T1, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $O/time.txt)" | tee -a $RES0/summary.txt
    grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH OK" | tee -a $RES0/summary.txt || { echo "!! GATE FAIL: STREAM PATH NOT USED ($s $m)" | tee -a $RES0/summary.txt; FAIL=1; }
    case $m in cpu) ;; *) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP OK" | tee -a $RES0/summary.txt; else echo "!! GATE FAIL: GPU pileup did not run cleanly ($s $m)" | tee -a $RES0/summary.txt; FAIL=1; fi ;; esac
    case $m in gpuE5*) grep -aq "E11: Solo on the device" $SCR/$s/$m/out/star_Log.out && echo "LEVEL 5 OK" | tee -a $RES0/summary.txt || { echo "!! GATE FAIL: level 5 not used ($s $m)" | tee -a $RES0/summary.txt; FAIL=1; } ;; esac
    grep -a -e "unsorted STAR stream" -e "streaming from STAR pipe" -e "sorted path" -e "gpu_pileup\]" -e "ignor" -e "E8" -e "E10" -e "E11" -e "cell_qc\]" $O/run.log.ts | cut -c1-400 | tee -a $RES0/summary.txt
    grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "E10" -e "E11" -e "E15" -e "soloRun" -e "Solo counting" -e "loading the genome" -e "started mapping" -e "finished mapping" -e "Loading SA" -e "Loading Genome" -e "ALL DONE" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-6000 | head -160 > $O/starlog_engine.txt
    cut -c1-700 $O/starlog_engine.txt | tee -a $RES0/summary.txt
    cp $SCR/$s/$m/out/*Log.final.out $SCR/$s/$m/out/*Log.out $O/ 2>/dev/null
    [ -f $SCR/$s/$m/CellReads.stats ] && cp $SCR/$s/$m/CellReads.stats $O/CellReads.stats && echo "  CellReads.stats kept: $(wc -l < $O/CellReads.stats) lines" | tee -a $RES0/summary.txt
    cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
    RM=$(awk -F'|' '/Number of input reads/{gsub(/[ \t]/,"",$2); printf "%.6f", $2/1e6}' $O/star_Log.final.out 2>/dev/null); RM=${RM:-66.601887}
    python3 $SCR/repo/bench/fable/solowire/s8_stages_e11.py $O $RM > $O/stages.txt 2>&1; tee -a $RES0/summary.txt < $O/stages.txt
    [ "${KEEPOUT:-0}" = 1 ] && rsync -a --exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' \
        --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q' $SCR/$s/$m/out/ $O/out/
  done
  E11C=$W/jobs/fable/solowire/results/gate-796377/cpu
  odir() { if [ -d $SCR/$s/$1/out ]; then echo $SCR/$s/$1/out; else v=OUT_$1; if [ $s = pbmc1k_v3 ] && [ $1 = cpu ]; then echo "${!v:-$E11C/out}"; else echo "${!v:-}"; fi; fi; }
  crs() { if [ -f $RES/$1/CellReads.stats ]; then echo $RES/$1/CellReads.stats; else v=CRS_$1; if [ $s = pbmc1k_v3 ] && [ $1 = cpu ]; then echo "${!v:-$E11C/CellReads.stats}"; else echo "${!v:-}"; fi; fi; }
  cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $1 vs $2 (not run)" | tee -a $RES0/summary.txt; return; }
    echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES0/summary.txt
    python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
    grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -60 | tee -a $RES0/summary.txt
    for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
             Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
      if cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f; then r=identical; else r=DIFFERS; FAIL=1; fi
      echo "  star_Solo.out/$f: $r" | tee -a $RES0/summary.txt
    done
    if [ -f $a/star_Log.final.out ] && [ -f $b/star_Log.final.out ]; then
      if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then echo "  star_Log.final.out: mapping statistics identical (timing lines excluded)" | tee -a $RES0/summary.txt
      else echo "  star_Log.final.out: MAPPING STATISTICS DIFFER" | tee -a $RES0/summary.txt; diff <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) | head -20 | tee -a $RES0/summary.txt; fi
    fi
    if [ -f $a/star_SJ.out.tab ] && [ -f $b/star_SJ.out.tab ]; then
      if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then echo "  star_SJ.out.tab: identical ($(wc -l < $a/star_SJ.out.tab) junctions)" | tee -a $RES0/summary.txt; else echo "  star_SJ.out.tab: DIFFERS" | tee -a $RES0/summary.txt; fi
    fi
    for f in nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
      [ -f $a/$f ] && [ -f $b/$f ] || continue
      ha=$(paste - - - - < $a/$f | LC_ALL=C sort -S 4G --parallel=8 | sha256sum | cut -c1-16); hb=$(paste - - - - < $b/$f | LC_ALL=C sort -S 4G --parallel=8 | sha256sum | cut -c1-16)
      if [ "$ha" = "$hb" ]; then echo "  $f: same records (sorted sha256 $ha; $(( $(wc -l < $a/$f) / 4 )) reads)" | tee -a $RES0/summary.txt; else echo "  $f: RECORDS DIFFER (sorted sha256 $ha vs $hb)" | tee -a $RES0/summary.txt; fi
    done
    ca=$(crs $1); cb=$(crs $2)
    if [ -n "$ca" ] && [ -f "$ca" ] && [ -n "$cb" ] && [ -f "$cb" ]; then
      if cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then
        echo "  Gene/CellReads.stats: same rows (sorted; $(( $(wc -l < $ca) - 1 )) rows; byte-identical in file order: $(cmp -s $ca $cb && echo yes || echo no))" | tee -a $RES0/summary.txt
      else echo "  Gene/CellReads.stats: ROWS DIFFER ($(wc -l < $ca) vs $(wc -l < $cb) lines)" | tee -a $RES0/summary.txt; FAIL=1
           diff <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort) | head -8 | cut -c1-200 | tee -a $RES0/summary.txt; fi
    else echo "  Gene/CellReads.stats: not compared (no copy for $1 or $2)" | tee -a $RES0/summary.txt; fi; }
  for p in $PS; do cmp2 ${p%%:*} ${p##*:}; done
  rm -rf "${SCR:?}/${s:?}"
done
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES0/summary.txt || echo "GATE FAILED" | tee -a $RES0/summary.txt
echo done $(date -Is) | tee -a $RES0/summary.txt
