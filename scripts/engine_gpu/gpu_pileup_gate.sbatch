#!/bin/bash
#SBATCH --job-name=eng-gpu-pugate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Gate for SINGLET_GPU_PILEUP=on: per sample, --engine gpu with the CPU pileup ("cpu") and with the GPU pileup ("on"), every output
# compared (bench/star_accel/out_equiv.py: byte, then content; JSON fields listed), plus each run's CPU (cgroup) and wall. A second
# "cpu" run (cpu2) gives the run-to-run baseline. Outputs in job /tmp; summaries to results/pugate-<job>/.
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:?set BIN=$W/builds/<commit8>/bin/<binary> (an immutable build from scripts/engine_gpu/build_clipper.sbatch)}; pin "$BIN"; SETS=(${SETS:-pbmc1k_v3}); T=${T:-18}; MODES=(${MODES:-cpu on cpu2})
BARE=$W/repos/singlet-py.git
RES=$W/jobs/w11/engine/results/pugate-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-pugate-$SLURM_JOB_ID; mkdir -p $SCR/repo
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 3; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
git -C $BARE archive refs/w11/engine-stream bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit) $(date -Is)" | tee $RES/summary.txt
for s in "${SETS[@]}"; do
  G=$REF/GRCh38-2024-A; [ "$s" = GSM5411066 ] && G=$REF/GRCm39-2024-A
  FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || { D=$W/data/bam-stream/$s; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$s; FQ=$D/sample.1fq; }
  for m in "${MODES[@]}"; do
    mkdir -p $SCR/$s/$m/out
    if [ $m = on ]; then export SINGLET_GPU_PILEUP=on; else unset SINGLET_GPU_PILEUP; fi
    C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T0=$(date +%s.%N)
    ( cd $SCR/$s/$m && timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz --engine gpu > $RES/$s-$m.log 2>&1; echo "exit $?" >> $RES/$s-$m.log )
    C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N)
    echo "$s $m: $(tail -1 $RES/$s-$m.log), wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8)" | tee -a $RES/summary.txt
    grep -a "\[gpu_pileup\]\|Pileup: [0-9]" $RES/$s-$m.log | tee -a $RES/summary.txt
  done
  unset SINGLET_GPU_PILEUP
  for m in on cpu2; do
    [ -d $SCR/$s/$m/out ] || continue
    echo "=== $s cpu vs $m" | tee -a $RES/summary.txt
    python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$s/cpu/out $SCR/$s/$m/out 2>&1 | tail -40 | tee -a $RES/summary.txt
  done
  rm -rf $SCR/$s
done
echo done $(date -Is)
