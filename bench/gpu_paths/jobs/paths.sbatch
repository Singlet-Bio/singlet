#!/bin/bash
#SBATCH --job-name=gpu-paths
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Code-path corpus run (bench/gpu_paths/README.md). One H100, debug-gpu (30 min limit, 8 CPUs).
# Parameters (environment):
#   BIN      directory with singlet-gpu and commit-gpu (required)
#   COMMIT   the commit that BIN must record (8 or more hex digits, required)
#   T        threads, default 8 (the debug-gpu QOS allows 8 CPUs)
#   ONLY     comma list of corpus ids, default all of corpus/manifest.tsv
#   MODES    "cpu gpu" (default). "cpu" only builds the kept CPU references (run it on short-cpu with --gres=none).
#   REFRESH_REF=1  runs the CPU mode again and replaces the reference
#   EXPECT_SCRIPTS  commit prefix of the bench/gpu-paths branch that the job must fetch
# Per input: the CPU run of BIN is the oracle, kept under $W/jobs/gpu-paths/ref/<commit8>/T<T>/<id>. The GPU run (--engine gpu, SINGLET_GPU_PILEUP=on)
# is compared with it and checked for its target path and the invariants (check_paths.py). One verdict table: results/paths-<job>/verdicts.tsv.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export W
REF=/mnt/projects/debruinz_project/cellarium/reference; PB=$W/pipeline-benchmark
BINDIR=${BIN:?BIN: the directory with singlet-gpu}; T=${T:-8}
EXPECT_COMMIT=${COMMIT:?COMMIT: the commit that BIN must hold}
[ ${#EXPECT_COMMIT} -ge 8 ] || { echo "COMMIT needs 8 or more hex digits"; exit 2; }
MODES=(${MODES:-cpu gpu}); CORPUS=$W/jobs/gpu-paths/corpus
RES0=$W/jobs/gpu-paths/results/paths-${SLURM_JOB_ID:-local}; mkdir -p $RES0
[ -f $BINDIR/singlet-gpu ] || { echo "no binary $BINDIR/singlet-gpu"; exit 2; }
COMMITREC=$(cat $BINDIR/commit-gpu 2>/dev/null || true)
case $COMMITREC in $EXPECT_COMMIT*) ;; *) echo "REFUSED: $BINDIR records commit '${COMMITREC:-none}', COMMIT is $EXPECT_COMMIT" | tee $RES0/summary.txt; exit 2 ;; esac
BSHA=$(sha256sum $BINDIR/singlet-gpu | cut -d' ' -f1); EMBED=no; grep -aqF "$COMMITREC" $BINDIR/singlet-gpu && EMBED=yes
[ $EMBED = yes ] || [ "${ALLOW_NO_EMBED:-0}" = 1 ] || { echo "REFUSED: the commit $COMMITREC is not inside the binary" | tee $RES0/summary.txt; exit 2; }
C8=${COMMITREC:0:8}; REFROOT=$W/jobs/gpu-paths/ref/$C8/T$T; mkdir -p $REFROOT
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists 2>/dev/null
SCR=/tmp/$USER-gpup-${SLURM_JOB_ID:-local}; mkdir -p $SCR/repo; SREF=refs/gpup/paths-${SLURM_JOB_ID:-local}; BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-paths}
trap 'kill $(jobs -p) 2>/dev/null; git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$SBR | cut -f1); [ -n "$WANT" ] || { echo "no scripts branch $SBR"; exit 3; }
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
SFULL=$(git -C $BARE rev-parse $SREF); [ "$SFULL" = "$WANT" ] || { echo "fetched $SFULL, branch tip is $WANT"; exit 3; }
[ -z "${EXPECT_SCRIPTS:-}" ] || [[ $SFULL == $EXPECT_SCRIPTS* ]] || { echo "scripts commit $SFULL, EXPECT_SCRIPTS is $EXPECT_SCRIPTS"; exit 3; }
SCRIPTS=${SFULL:0:8}
git -C $BARE archive $SREF bench/star_accel/out_equiv.py bench/gpu_batch1 bench/gpu_paths | tar -x -C $SCR/repo
GP=$SCR/repo/bench/gpu_paths; GB=$SCR/repo/bench/gpu_batch1; OE=$SCR/repo/bench/star_accel/out_equiv.py; export OE_DIR=$SCR/repo/bench/star_accel
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=900 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CP() { python3 $GP/check_paths.py "$@"; }
# a job array (the reference builds of submit_paths.sh) takes the n-th corpus id
if [ -n "${SLURM_ARRAY_TASK_ID:-}" ] && [ -z "${ONLY:-}" ]; then ONLY=$(CP ids | tr ' ' '
' | sed -n "$((SLURM_ARRAY_TASK_ID + 1))p"); fi
echo "PROVENANCE job=${SLURM_JOB_ID:-local} node=$(hostname -s) bin=$BINDIR/singlet-gpu sha256=$BSHA commit=$COMMITREC expect=$EXPECT_COMMIT embedded=$EMBED scripts=$SCRIPTS T=$T modes=${MODES[*]} $(date -Is)" | tee $RES0/summary.txt
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | tee -a $RES0/summary.txt
T00=$(date +%s)
# one engine run: run_one ID MODE -> sets RUN_EXIT RUN_WALL RUN_RSS, leaves $SCR/$id/$mode/out and the log in $OD
run_one() {
  local id=$1 m=$2 FQ=$CORPUS/$1/$1.1fq G
  G=$REF/$(CP get $id genome)
  mkdir -p $SCR/$id/$m/out; OD=$3; mkdir -p $OD
  unset SINGLET_GPU_PILEUP SINGLET_ENGINE_GPU_E2E SINGLET_ENGINE_GPU_INGEST SINGLET_E16 SINGLET_E16_OFF SINGLET_E16_TRACE SINGLET_GPU_PILEUP_FORCE_DECLINE
  local ENGARG=""
  [ $m = gpu ] && { export SINGLET_GPU_PILEUP=on; ENGARG="--engine gpu"; }
  ( while [ ! -e $SCR/$id/$m/out/star_Solo.out/Gene/CellReads.stats ]; do sleep 0.2; done; ln $SCR/$id/$m/out/star_Solo.out/Gene/CellReads.stats $SCR/$id/$m/CellReads.stats ) 2>/dev/null &
  local WATCH=$!
  local T0=$(date +%s.%N)
  ( cd $SCR/$id/$m && /usr/bin/time -v -o $OD/time.txt timeout -k 30 ${RUN_TIMEOUT:-900} $BINDIR/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$id/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $OD/run.log.ts )
  kill $WATCH 2>/dev/null
  RUN_WALL=$(echo "$(date +%s.%N) - $T0" | bc | cut -c1-7)
  RUN_RSS=$(awk -F': ' '/Maximum resident/{printf "%.1f", $2/1048576}' $OD/time.txt 2>/dev/null); RUN_RSS=${RUN_RSS:-0}
  RUN_EXIT=$(awk -F': ' '/Exit status/{print $2}' $OD/time.txt 2>/dev/null); RUN_EXIT=${RUN_EXIT:-99}
  cp $SCR/$id/$m/out/*Log.final.out $SCR/$id/$m/out/*Log.out $OD/ 2>/dev/null
  CRS=-; [ -f $SCR/$id/$m/CellReads.stats ] && CRS=$SCR/$id/$m/CellReads.stats
}
for id in $(CP ids "${ONLY:-}"); do
  T1=$(date +%s)
  echo "######## $id ($(CP get $id target)) [$(CP get $id kind)]" | tee -a $RES0/summary.txt
  FQ=$CORPUS/$id/$id.1fq
  if [ ! -f $FQ ]; then echo "!! $id: input missing ($FQ): run jobs/build_corpus.sbatch" | tee -a $RES0/summary.txt; continue; fi
  REFD=$REFROOT/$id
  for m in "${MODES[@]}"; do
    if [ $m = cpu ]; then
      if [ -f $REFD/meta.json ] && [ "${REFRESH_REF:-0}" != 1 ]; then echo "$id cpu: kept reference $REFD" | tee -a $RES0/summary.txt; continue; fi
      run_one $id cpu $SCR/$id/cpulog
      echo "$id cpu: exit $RUN_EXIT wall $RUN_WALL s maxRSS $RUN_RSS GB" | tee -a $RES0/summary.txt
      CP keepref $id $SCR/$id/cpu/out $REFD $RUN_EXIT $RUN_WALL $RUN_RSS $COMMITREC $CRS | tee -a $RES0/summary.txt
      cp $SCR/$id/cpulog/run.log.ts $REFD/ 2>/dev/null
      rm -rf $SCR/$id/cpu
    elif [ $m = gpu ]; then
      run_one $id gpu $RES0/$id
      echo "$id gpu: exit $RUN_EXIT wall $RUN_WALL s maxRSS $RUN_RSS GB" | tee -a $RES0/summary.txt
      CP verify $id $SCR/$id/gpu/out $REFD $RES0/$id/run.log.ts $RUN_EXIT $RUN_WALL $RUN_RSS $RES0 $GB $OE $CRS | tee -a $RES0/summary.txt
      rm -rf $SCR/$id
    fi
  done
  echo "$id: $(( $(date +%s) - T1 )) s" | tee -a $RES0/summary.txt
done
RC=0
case " ${MODES[*]} " in *" gpu "*) CP table $RES0 | tee -a $RES0/summary.txt; RC=${PIPESTATUS[0]}; column -t -s $'\t' $RES0/verdicts.tsv | cut -c1-250 | tee -a $RES0/summary.txt ;; esac
echo "TOTAL $(( $(date +%s) - T00 )) s" | tee -a $RES0/summary.txt
echo "PROVENANCE job=${SLURM_JOB_ID:-local} bin=$BINDIR/singlet-gpu sha256=$BSHA commit=$COMMITREC scripts=$SCRIPTS T=$T" | tee -a $RES0/summary.txt
[ $RC = 0 ] && echo "PATHS CORPUS OK" | tee -a $RES0/summary.txt || echo "PATHS CORPUS FAILED" | tee -a $RES0/summary.txt
exit $RC
