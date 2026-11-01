#!/bin/bash
#SBATCH --job-name=e19-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=200G
#SBATCH --time=04:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E19 (docs/gpu-engine/fable/E19-release.md): the all-sample gate of the GPU path's defaults, from E15's bench/fable/posterior/jobs/gate.sbatch
# and E16's bench/fable/tail/jobs/gate.sbatch. Binary of bench/fable/release/jobs/build.sbatch in $BINDIR (singlet-gpu). Per sample in SAMPLES
# (default: every sample under pipeline-benchmark/data; the .1fq from there, else from $W/jobs/fable/release/data/<s>/ written by encode.sbatch;
# genome by sample: GRCm39 for the mouse samples, the barnyard for hgmm1k_v3, else GRCh38), modes (MODES, in order; MODES_<sample> overrides):
#   cpu     the CPU engine. Its outputs are the sample's reference, kept under $REFD/T<T>/<sample>/ (out/ without the read files, the sorted
#           sha256 of the read files, CellReads.stats, logs, timings) and reused by later gates: when the reference exists the run is skipped
#           (REFRESH_REF=1 runs it again and replaces it)
#   gpu     --engine gpu + SINGLET_GPU_PILEUP=on: the defaults (level 5 + E13 ingest + E16's tail)
#   gpuT    gpu + SINGLET_E16_TRACE=1 (E16's "[e16]" marks: bench/fable/tail/e16_steps.py's step table)
#   gpuO0   gpu + SINGLET_ENGINE_GPU_E2E=0 (the opt-out: the path before E8)
#   gpuI0   gpu + SINGLET_ENGINE_GPU_INGEST=0 (level 5 with the FASTQ temp files)
#   gpuL<N> gpu + SINGLET_ENGINE_GPU_E2E=N
#   gpuX<N> gpu + the variables of XENV<N> ("A=1 B=2")
# Every run: timestamped log, 1 s sampler of the job's cgroup CPU time and the GPU, /usr/bin/time; Gene/CellReads.stats hard-linked the moment
# it appears (the pipeline deletes it after parsing); its exit status; the defaults line, the level reached and every fallback line
# ("[engine_gpu] fallback:", "[1fq-ingest] not used", "SINGLET_GPU_PILEUP ignored", ENGINE_GPU ... not used in star_Log.out) in fallbacks.txt and
# one row of runs.tsv (wall, core-s, peak RSS, device peak, level, ingest, fallbacks). The read files (nonhost_reads.fq, star_Unmapped.out.mate1/2:
# record order is run-dependent, E11 §2a) are compared by the sha256 of their sorted 4-line records and set aside before out_equiv.py.
# Then per pair (PAIRS=a:b ..., PAIRS_<sample> overrides; "cpu" is the kept reference; "e11gpuD" is E11's kept gpuD run of pbmc1k_v3,
# gate 796377; OUT_<mode>/CRS_<mode>: a kept out/ dir and CellReads.stats for any other mode not run here): out_equiv.py (its DIFF files must
# be E11's run-dependent ones: EXPECT_DIFF), the 10 STAR Solo files by cmp, the mapping statistics of star_Log.final.out, SJ.out.tab, the read
# files, and CellReads.stats by sorted rows. Stages: E13's s8_stages_e13.py with the run's input reads.
# debug-gpu: sbatch -p debug-gpu -c 8 --mem=120G --time=00:29:00 --export=ALL,T=8,SAMPLES=pbmc2m,MODES="cpu gpu" gate.sbatch
set -uo pipefail
FAIL=0
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/release/bin}; T=${T:-18}
SAMPLES=(${SAMPLES:-pbmc2m pbmc10m pbmc1k_v3 pbmc1k_v2 pbmc1k_cite_v3 pbmc5p_v1 neuron1k_v3 hgmm1k_v3 nuclei2k_v2}); MODES=(${MODES:-cpu gpu})
REFD=${REFD:-$W/jobs/fable/release/ref}/T$T; DATA2=$W/jobs/fable/release/data; E11=$W/jobs/fable/solowire/results/gate-796377
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/release}
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
RES0=$W/jobs/fable/release/results/gate-$SLURM_JOB_ID; mkdir -p $RES0 $REFD
SCR=/tmp/$USER-e19g-$SLURM_JOB_ID; mkdir -p $SCR/repo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
for i in 1 2 3; do git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/release-gate && break; sleep 5; done
git -C $BARE archive refs/fable/release-gate bench/star_accel/out_equiv.py bench/fable/ingest/s8_stages_e13.py bench/fable/tail/e16_steps.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
COMMIT=$(cat $BINDIR/commit-gpu 2>/dev/null)
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BINDIR commit=$COMMIT scripts=$(git -C $BARE rev-parse --short=8 refs/fable/release-gate) T=$T samples=${SAMPLES[*]} modes=${MODES[*]} refs=$REFD $(date -Is)" | tee $RES0/summary.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $RES0/summary.txt
# E11 §2a's run-dependent files once the read files are set aside: timings (read_stats.tsv, star_Log.std.out), the file path's clip5p probe count
# and bytes_public (pileup_stats.json), memory_tier (summary.json); provenance.json and ambient_contamination.json differ in timings/paths only
EXPECT_DIFF=" ${EXPECT_DIFF:-pileup_stats.json read_stats.tsv star_Log.std.out summary.json} "
EXCL=(--exclude='*.bam' --exclude='*.sam' --exclude='*.cram' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' --exclude='.gpu_handoff*'
      --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q')
READF="nonhost_reads.fq star_Unmapped.out.mate1 star_Unmapped.out.mate2"
readhash() { for f in $READF; do
    if [ -f $1/$f ]; then echo "$f $(paste - - - - < $1/$f | LC_ALL=C sort -S 4G --parallel=8 -T $SCR | sha256sum | cut -c1-16) $(( $(wc -l < $1/$f) / 4 ))"
    else echo "$f absent 0"; fi; done; }
[ -f $RES0/runs.tsv ] || printf "sample\tmode\texit\twall_s\tcore_s\tmaxrss_gb\tdev_peak_gb\tlevel\tingest\tfallback_lines\tnode\n" > $RES0/runs.tsv
for s in "${SAMPLES[@]}"; do
  RES=$RES0/$s; mkdir -p $RES; RS=$REFD/$s
  G=$REF/GRCh38-2024-A
  case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
  FQ=$PB/data/$s/$s.1fq; [ -f $FQ ] || FQ=$DATA2/$s/$s.1fq
  if [ ! -f $FQ ]; then echo "!! GATE FAIL: $s has no .1fq ($PB/data/$s, $DATA2/$s)" | tee -a $RES0/summary.txt; FAIL=1; continue; fi
  echo "######## $s ($FQ, $(ls -l $FQ | awk '{print $5}') bytes; genome $G)" | tee -a $RES0/summary.txt
  mv=MODES_$s; MS=(${!mv:-${MODES[*]}}); pv=PAIRS_$s; PS="${!pv:-${PAIRS:-cpu:gpu}}"
  for m in "${MS[@]}"; do
    O=$RES/$m
    if [ $m = cpu ] && [ -f $RS/DONE ] && [ "${REFRESH_REF:-0}" != 1 ]; then
      echo "$s cpu: the kept reference $RS ($(head -1 $RS/DONE))" | tee -a $RES0/summary.txt; cat $RS/run.txt | tee -a $RES0/summary.txt
      grep -a "^$s	cpu	" $RS/runs.tsv | sed 's/	cpu	/	cpu(kept)	/' >> $RES0/runs.tsv; continue
    fi
    mkdir -p $SCR/$s/$m/out $O
    unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_ENGINE_GPU_INGEST SINGLET_E16 SINGLET_E16_OFF SINGLET_E16_TRACE SINGLET_GPU_PILEUP_FORCE_DECLINE
    for kv in ${XENV1:-} ${XENV2:-} ${XENV3:-} ${XENV4:-}; do unset ${kv%%=*}; done
    BIN=$BINDIR/singlet-gpu; ENGARG="--engine gpu"
    case $m in
      cpu) ENGARG="" ;;
      gpu) export SINGLET_GPU_PILEUP=on ;;
      gpuT) export SINGLET_GPU_PILEUP=on SINGLET_E16_TRACE=1 ;;
      gpuO0) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=0 ;;
      gpuI0) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_INGEST=0 ;;
      gpuL[0-5]) export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=${m#gpuL} ;;
      gpuX[1-4]) export SINGLET_GPU_PILEUP=on; xv=XENV${m#gpuX}; for kv in ${!xv:-}; do export "$kv"; done ;;
      *) echo "!! GATE FAIL: unknown mode $m" | tee -a $RES0/summary.txt; FAIL=1; continue ;;
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
    WALL=$(echo "$T1 - $T0" | bc | cut -c1-7); CORES=$(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-7)
    RSS=$(awk -F': ' '/Maximum resident/{printf "%.1f", $2/1048576}' $O/time.txt); DEV=$(awk -F'\t' 'NR>1 && $4>m{m=$4} END{printf "%.1f", m/1024}' $O/monitor.tsv)
    ES=$(awk -F': ' '/Exit status/{print $2}' $O/time.txt 2>/dev/null)
    echo "$s $m: env $(env | grep -E '^SINGLET_(GPU_PILEUP|ENGINE_GPU|E16)' | sort | tr '\n' ' ')" | tee -a $RES0/summary.txt
    echo "$s $m: wall $WALL s, core-s $CORES, start $T0 end $T1, maxRSS $RSS GB, device peak $DEV GB, exit $ES" | tee -a $RES0/summary.txt $O/run.txt
    [ "$ES" = 0 ] && echo "EXIT STATUS 0" | tee -a $RES0/summary.txt || { echo "!! GATE FAIL: exit status ${ES:-?} ($s $m)" | tee -a $RES0/summary.txt; FAIL=1; tail -5 $O/run.log.ts | tee -a $RES0/summary.txt; }
    grep -aq "unsorted STAR stream" $O/run.log.ts && grep -aq "Pileup: streaming from STAR pipe" $O/run.log.ts && echo "STREAM PATH: yes" | tee -a $RES0/summary.txt || echo "STREAM PATH: NO (sorted path)" | tee -a $RES0/summary.txt
    case $m in cpu) LV=cpu; ING=- ;;
      *) if grep -aq "\[gpu_pileup\] on:" $O/run.log.ts && ! grep -aE "\[gpu_pileup\] (not run|failed|ERROR)|GPU_PILEUP ignored" $O/run.log.ts | grep -q .; then echo "GPU PILEUP: on" | tee -a $RES0/summary.txt; else echo "GPU PILEUP: NOT ON" | tee -a $RES0/summary.txt; fi
         LV=$(grep -ao "\[engine_gpu\] level [0-9] (requested [0-9])" $O/run.log.ts | tail -1 | sed -E 's/.*level ([0-9]) \(requested ([0-9])\)/\1 of \2/')
         [ -z "$LV" ] && grep -aq "fallback: ENGINE_GPU: not used" $O/run.log.ts && LV="0 (GPU mapping not used)"
         if grep -aq "\.1fq scanned: STAR's prefill decodes" $O/run.log.ts; then ING=yes
         else ING=$(grep -ao "\[1fq-ingest\] not used ([^)]*)" $O/run.log.ts | tail -1 | sed -E 's/.*not used \((.*)\)/declined: \1/'); ING=${ING:-no}; fi
         echo "LEVEL ${LV:-?}; INGEST $ING" | tee -a $RES0/summary.txt ;;
    esac
    grep -a -e "\[engine_gpu\] fallback" -e "\[1fq-ingest\] not used" -e "SINGLET_GPU_PILEUP ignored" -e "E8: no shared memory" -e "prefill wrote none" -e "FATAL" -e "FELL BACK" \
         -e "\[gpu_pileup\] ERROR" -e "\[gpu_pileup\] not run" -e "\[gpu_pileup\] failed" -e "screening on the CPU" -e "built without CUDA" -e "no usable GPU" $O/run.log.ts | cut -c1-600 > $O/fallbacks.txt
    NFB=$(grep -ac . $O/fallbacks.txt); echo "FALLBACK LINES: $NFB" | tee -a $RES0/summary.txt; cut -c1-400 $O/fallbacks.txt | tee -a $RES0/summary.txt
    grep -a -e "\[singlet\] --engine gpu" -e "GPU probe" -e "\[engine_gpu\]" -e "\[1fq-ingest\]" -e "\.1fq scanned" -e "\.1fq decoded" -e "\[singlet\] \.1fq" -e "unsorted STAR stream" \
         -e "sorted path" -e "gpu_pileup\]" -e "ignor" -e "E8" -e "E10" -e "E11" -e "E13" -e "E16" -e "cell_qc\]" $O/run.log.ts | grep -av "^[0-9]* \[e16\] " | cut -c1-400 > $O/engine_lines.txt
    tee -a $RES0/summary.txt < $O/engine_lines.txt
    grep -ah -e "ENGINE_GPU" -e "engine_gpu:" -e "E10" -e "E11" -e "E13" -e "E15" -e "E16" -e "E19" -e "soloRun" -e "Solo counting" -e "loading the genome" -e "started mapping" \
         -e "finished mapping" -e "Loading SA" -e "Loading Genome" -e "ALL DONE" $SCR/$s/$m/out/*Log.out 2>/dev/null | cut -c1-6000 | head -160 > $O/starlog_engine.txt
    grep -a -e "not used" -e "not built" -e "FATAL" $O/starlog_engine.txt | grep -av "Shared memory is not used" | cut -c1-400 | sed 's/^/  star_Log.out: /' | tee -a $RES0/summary.txt $O/fallbacks.txt
    cp $SCR/$s/$m/out/*Log.final.out $SCR/$s/$m/out/*Log.out $O/ 2>/dev/null
    [ -f $SCR/$s/$m/CellReads.stats ] && cp $SCR/$s/$m/CellReads.stats $O/CellReads.stats && echo "  CellReads.stats kept: $(wc -l < $O/CellReads.stats) lines" | tee -a $RES0/summary.txt
    cp $SCR/$s/$m/out/read_stats.tsv $SCR/$s/$m/out/summary.json $O/ 2>/dev/null
    RM=$(awk -F'|' '/Number of input reads/{gsub(/[ \t]/,"",$2); printf "%.6f", $2/1e6}' $O/star_Log.final.out 2>/dev/null); RM=${RM:-1}
    python3 $SCR/repo/bench/fable/ingest/s8_stages_e13.py $O $RM > $O/stages.txt 2>&1; tee -a $RES0/summary.txt < $O/stages.txt
    [ $m = gpuT ] && { python3 $SCR/repo/bench/fable/tail/e16_steps.py $O > $O/e16.txt 2>&1; tee -a $RES0/summary.txt < $O/e16.txt; }
    readhash $SCR/$s/$m/out > $O/readhash.txt
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$s" "$m" "${ES:-?}" "$WALL" "$CORES" "$RSS" "$DEV" "${LV:-?}" "$ING" "$NFB" "$(hostname -s)" >> $RES0/runs.tsv
    if [ $m = cpu ] && [ "${ES:-1}" = 0 ]; then   # the sample's reference, for this gate's pairs and the later ones
      N=$RS.new-$SLURM_JOB_ID; rm -rf $N; mkdir -p $N
      rsync -a "${EXCL[@]}" $SCR/$s/$m/out/ $N/out/
      cp $O/readhash.txt $O/run.txt $O/time.txt $O/run.log.ts $O/monitor.tsv $O/stages.txt $O/star_Log.out $N/ 2>/dev/null
      [ -f $O/CellReads.stats ] && cp $O/CellReads.stats $N/
      head -1 $RES0/runs.tsv > $N/runs.tsv; grep -a "^$s	cpu	" $RES0/runs.tsv | tail -1 >> $N/runs.tsv
      echo "job $SLURM_JOB_ID node $(hostname -s) commit $COMMIT T=$T $(date -Is)" > $N/DONE
      [ -d $RS ] && mv $RS $RS.old-$SLURM_JOB_ID; mv $N $RS; rm -rf $RS.old-$SLURM_JOB_ID
      echo "  reference kept: $RS" | tee -a $RES0/summary.txt
    elif [ $m != cpu ]; then
      rsync -a "${EXCL[@]}" $SCR/$s/$m/out/ $SCR/$s/$m/cmp/
      [ "${KEEPOUT:-0}" = 1 ] && rsync -a $SCR/$s/$m/cmp/ $O/out/
    fi
    rm -rf $SCR/$s/$m/out
  done
  odir() { case $1 in cpu) [ -f $RS/DONE ] && echo $RS/out ;; e11gpuD) echo "${OUT_e11gpuD:-$E11/gpuD/out}" ;;
             *) if [ -d $SCR/$s/$1/cmp ]; then echo $SCR/$s/$1/cmp; else v=OUT_$1; echo "${!v:-}"; fi ;; esac; }
  crs() { case $1 in cpu) echo $RS/CellReads.stats ;; e11gpuD) echo "${CRS_e11gpuD:-$E11/gpuD/CellReads.stats}" ;;
            *) if [ -f $RES/$1/CellReads.stats ]; then echo $RES/$1/CellReads.stats; else v=CRS_$1; echo "${!v:-}"; fi ;; esac; }
  rhf() { case $1 in cpu) echo $RS/readhash.txt ;; *) echo $RES/$1/readhash.txt ;; esac; }
  cmp2() { a=$(odir $1); b=$(odir $2); [ -n "$a" ] && [ -d "$a" ] && [ -n "$b" ] && [ -d "$b" ] || { echo "-- skip $s $1 vs $2 (not run)" | tee -a $RES0/summary.txt; return; }
    local ok=1
    echo "=== $s $1 vs $2 ($a vs $b)" | tee -a $RES0/summary.txt
    python3 $SCR/repo/bench/star_accel/out_equiv.py $a $b > $RES/equiv-$1-$2.txt 2>&1
    grep -a '^  DIFF\|^  only in\|^    \|^files:\|^RESULT' $RES/equiv-$1-$2.txt | cut -c1-200 | head -60 | tee -a $RES0/summary.txt
    FL=$(grep -a '^files:' $RES/equiv-$1-$2.txt | sed -E 's/files: ([0-9]+) identical, ([0-9]+) differ only in timings\/paths, ([0-9]+) differ/\1\/\2\/\3/')
    UNX=""; for f in $(grep -a '^  DIFF ' $RES/equiv-$1-$2.txt | sed -E 's/^  DIFF ([^ :]+).*/\1/'); do case "$EXPECT_DIFF" in *" $f "*) ;; *) UNX="$UNX $f" ;; esac; done
    for f in $(grep -a '^  only in ' $RES/equiv-$1-$2.txt | sed -E 's/^  only in [ab]: //'); do UNX="$UNX only:$f"; done
    [ -n "$FL" ] || UNX="$UNX out_equiv-failed"
    if [ -n "$UNX" ]; then echo "  out_equiv: UNEXPECTED:$UNX" | tee -a $RES0/summary.txt; ok=0; else echo "  out_equiv: $FL (identical/timing-path/differ), every DIFF in E11's run-dependent set" | tee -a $RES0/summary.txt; fi
    local ns=0
    for f in Barcodes.stats Gene/Features.stats Gene/Summary.csv Gene/UMIperCellSorted.txt Gene/raw/barcodes.tsv Gene/raw/features.tsv Gene/raw/matrix.mtx \
             Gene/filtered/barcodes.tsv Gene/filtered/features.tsv Gene/filtered/matrix.mtx; do
      if cmp -s $a/star_Solo.out/$f $b/star_Solo.out/$f; then ns=$((ns+1)); else echo "  star_Solo.out/$f: DIFFERS" | tee -a $RES0/summary.txt; ok=0; fi
    done
    echo "  star_Solo.out: $ns of 10 files identical" | tee -a $RES0/summary.txt
    if cmp -s <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out); then echo "  star_Log.final.out: mapping statistics identical (timing lines excluded)" | tee -a $RES0/summary.txt
    else echo "  star_Log.final.out: MAPPING STATISTICS DIFFER" | tee -a $RES0/summary.txt; ok=0; diff <(tail -n +5 $a/star_Log.final.out) <(tail -n +5 $b/star_Log.final.out) | head -20 | tee -a $RES0/summary.txt; fi
    if cmp -s $a/star_SJ.out.tab $b/star_SJ.out.tab; then echo "  star_SJ.out.tab: identical ($(wc -l < $a/star_SJ.out.tab) junctions)" | tee -a $RES0/summary.txt; else echo "  star_SJ.out.tab: DIFFERS" | tee -a $RES0/summary.txt; ok=0; fi
    ra=$(rhf $1); rb=$(rhf $2)
    if [ -f "$ra" ] && [ -f "$rb" ]; then
      for f in $READF; do ha=$(awk -v f=$f '$1==f{print $2" "$3}' $ra); hb=$(awk -v f=$f '$1==f{print $2" "$3}' $rb)
        if [ "$ha" = "$hb" ]; then echo "  $f: same records (sorted sha256 and reads: $ha)" | tee -a $RES0/summary.txt; else echo "  $f: RECORDS DIFFER ($ha vs $hb)" | tee -a $RES0/summary.txt; ok=0; fi; done
    else echo "  read files: not compared (no sorted hashes for $1 or $2)" | tee -a $RES0/summary.txt; fi
    ca=$(crs $1); cb=$(crs $2)
    if [ -n "$ca" ] && [ -f "$ca" ] && [ -n "$cb" ] && [ -f "$cb" ]; then
      if cmp -s <(head -1 $ca) <(head -1 $cb) && cmp -s <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort); then
        echo "  Gene/CellReads.stats: same rows (sorted; $(( $(wc -l < $ca) - 1 )) rows; byte-identical in file order: $(cmp -s $ca $cb && echo yes || echo no))" | tee -a $RES0/summary.txt
      else echo "  Gene/CellReads.stats: ROWS DIFFER ($(wc -l < $ca) vs $(wc -l < $cb) lines)" | tee -a $RES0/summary.txt; ok=0
           diff <(tail -n +2 $ca | LC_ALL=C sort) <(tail -n +2 $cb | LC_ALL=C sort) | head -8 | cut -c1-200 | tee -a $RES0/summary.txt; fi
    else echo "  Gene/CellReads.stats: not compared (no copy for $1 or $2)" | tee -a $RES0/summary.txt; fi
    if [ $ok = 1 ]; then echo "VERDICT $s $1:$2 IDENTICAL (except E11's run-dependent files; out_equiv $FL)" | tee -a $RES0/summary.txt
    else echo "VERDICT $s $1:$2 NOT IDENTICAL" | tee -a $RES0/summary.txt; FAIL=1; fi
    printf "%s\t%s:%s\t%s\t%s\n" $s $1 $2 "$([ $ok = 1 ] && echo IDENTICAL || echo DIFFERENT)" "$FL" >> $RES0/verdicts.tsv; }
  for p in $PS; do cmp2 ${p%%:*} ${p##*:}; done
  rm -rf "${SCR:?}/${s:?}"
done
echo "######## runs" | tee -a $RES0/summary.txt; column -t -s $'\t' $RES0/runs.tsv | tee -a $RES0/summary.txt
echo "######## verdicts" | tee -a $RES0/summary.txt; [ -f $RES0/verdicts.tsv ] && column -t -s $'\t' $RES0/verdicts.tsv | tee -a $RES0/summary.txt
[ $FAIL = 0 ] && echo "GATE RUN OK (identity verdicts above)" | tee -a $RES0/summary.txt || echo "GATE FAILED" | tee -a $RES0/summary.txt
echo done $(date -Is) | tee -a $RES0/summary.txt
