#!/bin/bash
#SBATCH --job-name=eng-cpu-same
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=none
#SBATCH --cpus-per-task=54
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The CPU engine is unchanged by this branch: the same samples through (ref) a CPU-only build of prod/campaign, (cpu) this branch's
# CPU-only build and (gpubin) this branch's GPU build with --engine cpu, three samples at a time; every output file compared with ref
# (ref2 = ref again: the run-to-run baseline) byte for byte, then order-independently (sorted lines / sorted FASTQ records) for files that differ. Outputs in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
REFBIN=${REFBIN:-$W/jobs/w11/engine/bin9/singlet-cpu}; CPUBIN=${CPUBIN:-$W/jobs/w11/engine/bin8/singlet-cpu}; GPUBIN=${GPUBIN:-$W/jobs/w11/engine/bin8/singlet-gpu}
SETS=(${SETS:-GSM6996272 GSM6996274 pbmc1k_v3}); T=${T:-18}
RES=$W/jobs/w11/engine/results/cpu_unchanged-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-cpusame-$SLURM_JOB_ID; mkdir -p $SCR
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 2; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF
G=$REF/GRCh38-2024-A
echo "node=$(hostname) job=$SLURM_JOB_ID ref=$(sha256sum $REFBIN | cut -c1-16) cpu=$(sha256sum $CPUBIN | cut -c1-16) gpubin=$(sha256sum $GPUBIN | cut -c1-16) $(date -Is)" | tee $RES/summary.txt
for r in ref ref2 cpu gpubin; do
  case $r in ref|ref2) B=$REFBIN;; cpu) B=$CPUBIN;; gpubin) B=$GPUBIN;; esac
  for s in "${SETS[@]}"; do
    FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || FQ=$W/data/atlas-rerun/perf/$s/sample.1fq
    SCREEN=(); [ -f $(dirname $FQ)/screen.json ] && SCREEN=(--run-screen $(dirname $FQ)/screen.json)
    mkdir -p $SCR/$r/$s/out
    ( cd $SCR/$r/$s && timeout -k 30 3000 $B $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$r/$s/out/ --threads $T --pipeline --output-format 1pz --engine cpu "${SCREEN[@]}" > run.log 2>&1; echo "exit $?" >> run.log ) &
  done
  wait
  for s in "${SETS[@]}"; do echo "$r $s: $(tail -1 $SCR/$r/$s/run.log)" | tee -a $RES/summary.txt; done
done
for r in ref2 cpu gpubin; do
  for s in "${SETS[@]}"; do
    n=0; d=0; od=0
    for f in $(cd $SCR/ref/$s/out && find . -type f | sort); do
      n=$((n+1)); cmp -s $SCR/ref/$s/out/$f $SCR/$r/$s/out/$f && continue
      d=$((d+1))
      case $f in *.fq|*mate1|*mate2) a=$(paste - - - - < $SCR/ref/$s/out/$f | LC_ALL=C sort | md5sum); b=$(paste - - - - < $SCR/$r/$s/out/$f | LC_ALL=C sort | md5sum);;
                 *) a=$(LC_ALL=C sort $SCR/ref/$s/out/$f | md5sum); b=$(LC_ALL=C sort $SCR/$r/$s/out/$f 2>/dev/null | md5sum);; esac
      if [ "$a" = "$b" ]; then echo "    $r $s same content, other order: $f" >> $RES/summary.txt
      else od=$((od+1)); echo "    $r $s differs: $f" | tee -a $RES/summary.txt; fi
    done
    extra=$(comm -13 <(cd $SCR/ref/$s/out && find . -type f | sort) <(cd $SCR/$r/$s/out && find . -type f | sort) | tr '\n' ' ')
    echo "  $r vs ref, $s: $n files, $d differ byte-wise, $od differ order-independently; files only in $r: ${extra:-none}" | tee -a $RES/summary.txt
  done
done
echo done $(date -Is)
