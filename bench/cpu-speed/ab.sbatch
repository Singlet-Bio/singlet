#!/bin/bash
#SBATCH --job-name=cs-ab
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Paired same-node A/B of CPU-engine binaries on one real sample, runs in the order ORDER (e.g. "A B B A A B"), one at a time, CPU nodes only.
# Env: A, B (and optionally C, D): binary dirs under $W/jobs/cpu-speed/bin; SAMPLE; T (threads, default the allocation); ORDER;
#      PROF=1 adds the samprof sampler (LD_PRELOAD, 5 ms of thread CPU per sample) to every run; XENV_<arm>="K=V ..." extra env per arm;
#      KEEP=<arm> keeps that arm's first out/ (without read files) under the result dir.
# Per run: timestamped log, /usr/bin/time -v, the job cgroup's CPU counter every second, sha256 of every output file except the read files.
# Result dir: $W/jobs/cpu-speed/results/ab-<SAMPLE>-<jobid>/ with runs.tsv (arm, wall, core-s, stage marks) and identity.txt (per-file hash
# agreement of each run with the first run of arm A; timing/path files are listed as such, not hidden).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
S=${SAMPLE:?}; T=${T:-$SLURM_CPUS_PER_TASK}; ORDER=(${ORDER:-A B B A})
RES=$W/jobs/cpu-speed/results/ab-$S-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-csab-$SLURM_JOB_ID; mkdir -p $SCR
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
G=$REF/GRCh38-2024-A
case $S in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
FQ=$PB/data/$S/$S.1fq
[ -f $FQ ] || FQ=$W/jobs/fable/release/data/$S/$S.1fq
[ -f $FQ ] || FQ=$W/data/atlas-rerun/perf/$S/sample.1fq
[ $S = gsm7064234s ] && FQ=$W/pilot/GSM7064234/cap1fq_0ea48783/SRR23617320.1fq
[ -f $FQ ] || { echo "no .1fq for $S"; exit 2; }
export SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID sample=$S fq=$FQ T=$T order=${ORDER[*]} A=${A:-} B=${B:-} C=${C:-} D=${D:-} prof=${PROF:-0} $(lscpu | grep 'Model name' | sed 's/  */ /g') $(date -Is)" | tee $RES/summary.txt
printf "run\tarm\tbin\texit\twall_s\tcore_s\tmaxrss_gb\tstar_fork_s\tpileup_done_s\texport_start_s\tend_s\n" > $RES/runs.tsv
i=0
for arm in "${ORDER[@]}"; do
  i=$((i+1)); BD=${!arm:?arm $arm has no binary dir}; BIN=$W/jobs/cpu-speed/bin/$BD/singlet-cpu
  O=$RES/r$i-$arm; mkdir -p $O $SCR/r$i/out
  xv=XENV_$arm; ENVS=(${!xv:-})
  PRE=(); if [ "${PROF:-0}" = 1 ]; then mkdir -p $O/samprof; PRE=(env LD_PRELOAD=$W/jobs/cpu-speed/bin/$BD/libsamprof.so SAMPROF_OUT=$O/samprof SAMPROF_MS=5); fi
  ( while true; do echo "$(date +%s.%N) $(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)"; sleep 1; done ) > $O/cpu.txt & MON=$!
  T0=$(date +%s.%N); C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat)
  ( cd $SCR/r$i && env "${ENVS[@]}" "${PRE[@]}" /usr/bin/time -v -o $O/time.txt timeout -k 30 ${RUN_TIMEOUT:-12000} $BIN $FQ --ref-base $REF \
      --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/r$i/out/ --threads $T --pipeline --output-format 1pz --engine cpu 2>&1 \
      | while IFS= read -r l; do printf '%s %s\n' "$(date +%s.%N)" "$l"; done > $O/log.txt )
  C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N); kill $MON 2>/dev/null
  ES=$(awk -F': ' '/Exit status/{print $2}' $O/time.txt)
  WALL=$(echo "$T1 - $T0" | bc); CORES=$(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-9)
  RSS=$(awk -F': ' '/Maximum resident/{printf "%.1f", $2/1048576}' $O/time.txt)
  mark() { awk -v t0=$T0 -v p="$1" 'index($0,p){printf "%.1f", $1-t0; exit}' $O/log.txt; }
  printf "%s\t%s\t%s\t%s\t%.1f\t%s\t%s\t%s\t%s\t%s\t%.1f\n" $i $arm $BD "${ES:-?}" $WALL $CORES $RSS "$(mark 'STAR pid=')" "$(mark '[singlet] Pileup: ')" \
      "$(mark 'Exporting feature matrices')" $WALL >> $RES/runs.tsv
  ( cd $SCR/r$i/out && find . -type f ! -name '*.fq' ! -name '*Unmapped*' ! -name '*.bam' ! -name '*.1fq' -print0 | LC_ALL=C sort -z | xargs -0 sha256sum ) > $O/hashes.txt
  cp $SCR/r$i/out/star_Log.final.out $O/ 2>/dev/null
  if [ "${KEEP:-}" = $arm ] && [ ! -d $RES/keep-$arm ]; then rsync -a --exclude='*.fq' --exclude='*Unmapped*' --exclude='*.bam' $SCR/r$i/out/ $RES/keep-$arm/; fi
  echo "run $i $arm ($BD): exit ${ES:-?}, wall $WALL s, core-s $CORES, maxRSS $RSS GB" | tee -a $RES/summary.txt
  rm -rf $SCR/r$i
done
# identity: every run against the first run of arm A
REFH=$(ls -d $RES/r*-A | head -1)/hashes.txt
TIMING="pileup_stats.json read_stats.tsv star_Log.std.out summary.json provenance.json ambient_contamination.json star_Log.out star_Log.progress.out star_Log.final.out"
for d in $RES/r*-*; do
  [ "$d/hashes.txt" = "$REFH" ] && continue
  diffs=$(join -j 2 <(sort -k2 $REFH) <(sort -k2 $d/hashes.txt) | awk '$2!=$3{print $1}' | tr '\n' ' ')
  only=$(diff <(awk '{print $2}' $REFH | sort) <(awk '{print $2}' $d/hashes.txt | sort) | grep '^[<>]' | tr '\n' ' ')
  unexp=""; for f in $diffs; do case " $TIMING " in *" ${f#./} "*) ;; *) unexp="$unexp $f" ;; esac; done
  echo "$(basename $d) vs A: differ [$diffs] only [$only] unexpected [${unexp# }]" | tee -a $RES/identity.txt
done
column -t -s $'\t' $RES/runs.tsv | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
