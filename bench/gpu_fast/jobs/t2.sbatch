#!/bin/bash
#SBATCH --job-name=gf-t2
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=02:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fast lane, tier 2 (bench/gpu_fast/README.md): a short load of K runs in flight on one H100, arms alternated on the same node (ORDER, default ABBA).
# Derived from bench/gpu_batch1/jobs/tp.sbatch at 0fdec57f. Started by ab.py. The node is not chosen here. ab.py passes --exclude and -w.
#
# Parameter (sbatch --export=ALL,SPEC=<file>): SPEC is a shell file written by ab.py. It sets
#   NAME RUN SAMPLE T SLOTS K NRUNS ORDER REFDIR IDENT_N INTENDED SBATCH_SHA SCRIPTS_BRANCH EXPECT_SCRIPTS
#   SEP_RATE STOP_ON_FALLBACK
#   BIN_A COMMIT_A ENV_A ARGS_A and BIN_B COMMIT_B ENV_B ARGS_B (ENV and ARGS are bash arrays)
#
# Differences from tp.sbatch.
#   1. Two arms. Each leg is one point of K runs in flight and NRUNS runs in the queue, one binary and one set of environment variables.
#   2. No run alone. Identity is checked for a sample of IDENT_N runs per arm against REFDIR, which holds the CPU reference of one sample.
#      tier 3 (tp.sbatch) checks every run.
#   3. Fallback is checked on every run, during the leg and after it. A leg stops taking new runs at the first fallback or failed run.
#      The job stops after that leg when STOP_ON_FALLBACK is 1.
#   4. After each leg, from the third on, the job stops when B is clearly separated from A on the progress rate (abcore.separation, SEP_RATE).
#   5. Results go under $RUN/res. The scripts branch is bench/gpu-fast and the commit must equal EXPECT_SCRIPTS.
# The tier 2 numbers cannot carry a merge. Run tier 3 for that.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
DATA2=$W/jobs/fable/release/data
SPEC=${SPEC:?SPEC: the spec file written by ab.py}; [ -f "$SPEC" ] || { echo "no spec file $SPEC"; exit 2; }
# shellcheck disable=SC1090
. "$SPEC"
case $RUN in $W/jobs/gpu-fast/*) ;; *) echo "RUN must lie under $W/jobs/gpu-fast"; exit 2 ;; esac
RES0=$RUN/res; mkdir -p $RES0; S=$RES0/summary.txt; : > $S
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-fast}
echo "tier 2 rule values come from abcore.TIER2_RULE on the scripts branch. Early stop uses SEP_RATE=${SEP_RATE:-default}." | tee -a $S

arm_load() {   # A or B -> BIN BINDIR COMMIT_WANT ARM_ENV ARM_ARGS
  case $1 in
    A) BIN=$BIN_A/singlet-gpu; BINDIR=$BIN_A; COMMIT_WANT=$COMMIT_A; ARM_ENV=(${ENV_A[@]+"${ENV_A[@]}"}); ARM_ARGS=(${ARGS_A[@]+"${ARGS_A[@]}"}) ;;
    B) BIN=$BIN_B/singlet-gpu; BINDIR=$BIN_B; COMMIT_WANT=$COMMIT_B; ARM_ENV=(${ENV_B[@]+"${ENV_B[@]}"}); ARM_ARGS=(${ARGS_B[@]+"${ARGS_B[@]}"}) ;;
  esac
}
refuse() { echo "REFUSED: $*" | tee -a $S; echo "$*" > $RES0/REFUSED; exit 2; }

# --- provenance of both binaries, before anything runs ----------------------------------------------------------------------------------
: > $RES0/arms.tsv; printf "arm\tbin\tsha256\tcommit\tembedded\tenv\targs\n" >> $RES0/arms.tsv
for arm in A B; do
  arm_load $arm
  [ -f $BIN ] || refuse "no binary $BIN"
  COMMIT=$(cat $BINDIR/commit-gpu 2>/dev/null || true)
  [ -n "$COMMIT" ] || refuse "$BINDIR has no commit-gpu record"
  if [ -n "$COMMIT_WANT" ]; then case $COMMIT in $COMMIT_WANT*) ;; *) refuse "$BINDIR records commit '$COMMIT', arm $arm expects $COMMIT_WANT" ;; esac; fi
  EMBED=no; grep -aqF "$COMMIT" $BIN && EMBED=yes
  [ $EMBED = yes ] || [ "${ALLOW_NO_EMBED:-0}" = 1 ] || refuse "the commit $COMMIT recorded for arm $arm is not inside the binary $BIN"
  BSHA=$(sha256sum $BIN | cut -d' ' -f1)
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $arm $BINDIR $BSHA $COMMIT $EMBED "${ARM_ENV[*]:-}" "${ARM_ARGS[*]:-}" >> $RES0/arms.tsv
  [ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists 2>/dev/null || true
done
[ -d $REFDIR/out ] || refuse "no reference at $REFDIR/out. Run bench/gpu_batch1/jobs/gate.sbatch with MODES=cpu for this sample and T."

# --- scripts from the branch, one fetch at a time, with a commit check ---------------------------------------------------------------------
SCR=/tmp/$USER-gft2-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data
SREF=refs/gf/t2-$SLURM_JOB_ID
trap 'kill $(jobs -p) 2>/dev/null; git -C $BARE update-ref -d $SREF 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
URL="$(git -C $BARE config remote.origin.url)"
WANT=$(git ls-remote "$URL" refs/heads/$SBR | cut -f1); [ -n "$WANT" ] || refuse "no scripts branch $SBR"
( flock 9; git -C $BARE fetch -q "$URL" +refs/heads/$SBR:$SREF ) 9>$BARE/.gn-fetch.lock
SFULL=$(git -C $BARE rev-parse $SREF); [ "$SFULL" = "$WANT" ] || refuse "fetched $SFULL, branch tip is $WANT"
[ -z "${EXPECT_SCRIPTS:-}" ] || [[ $SFULL == $EXPECT_SCRIPTS* ]] || refuse "scripts commit $SFULL, EXPECT_SCRIPTS is $EXPECT_SCRIPTS"
SCRIPTS=${SFULL:0:8}
git -C $BARE archive $SREF bench/star_accel/out_equiv.py bench/fable/throughput bench/gpu_batch1 bench/gpu_fast | tar -x -C $SCR/repo
GB=$SCR/repo/bench/gpu_batch1; TPD=$SCR/repo/bench/fable/throughput; GF=$SCR/repo/bench/gpu_fast
if [ -n "${SBATCH_SHA:-}" ]; then
  have=$(sha256sum $GF/jobs/t2.sbatch | cut -d' ' -f1)
  [ "$have" = "$SBATCH_SHA" ] || refuse "t2.sbatch on $SBR ($have) differs from the copy that was submitted ($SBATCH_SHA)"
fi
# shellcheck disable=SC1091
. $GF/jobs/lib_ident.sh
# shellcheck disable=SC1091
. $GF/jobs/lib_run.sh

export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1
unset SINGLET_ENGINE_GPU_SLOTS SINGLET_ENGINE_GPU_CHUNK SINGLET_ENGINE_GPU_NB SINGLET_ENGINE_GPU_NBM SINGLET_ENGINE_GPU_NBB SINGLET_GPU_PILEUP_ARENA_GB SINGLET_ENGINE_GPU_DEVICE
export SINGLET_ENGINE_GPU_SLOTS=$SLOTS
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
NGPU=$(nvidia-smi -L | wc -l); NODE=$(hostname -s)
echo "PROVENANCE job=$SLURM_JOB_ID node=$NODE tier=2 name=$NAME scripts=$SCRIPTS sbatch_sha=${SBATCH_SHA:-none} sample=$SAMPLE T=$T slots=$SLOTS k=$K runs=$NRUNS order=$ORDER ident_n=$IDENT_N ref=$REFDIR $(date -Is)" | tee -a $S
column -t -s $'\t' $RES0/arms.tsv | sed 's/^/  /' | tee -a $S
printf '{"tier":2,"name":"%s","job":"%s","node":"%s","scripts":"%s","ref":"%s","T":%d,"slots":%d,"sample":"%s","k":%d,"runs":%d,"order":"%s","ident_n":%d,"intended_change":%d,"sep_rate":"%s"}\n' \
  "$NAME" $SLURM_JOB_ID $NODE $SCRIPTS "$REFDIR" $T $SLOTS $SAMPLE $K $NRUNS $ORDER $IDENT_N ${INTENDED:-0} "${SEP_RATE:-}" > $RES0/params.json
{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)"; free -g | head -2; df -h /tmp | tail -1; } 2>&1 | sed 's/^/  /' | tee -a $S

stage_input $SAMPLE
ITEMS=(); for ((j = 0; j < NRUNS; j++)); do ITEMS+=($SAMPLE); done
ctsnap() { { echo "## $2 $(date -Is)"; squeue -w $NODE -o "%i %u %j %C %b" 2>&1; } >> $1; }
if [ "${WARM:-1}" = 1 ]; then   # as in t1.sbatch: the first leg must not read the genome index from the shared disk alone
  t0=$SECONDS; find $(genome $SAMPLE)/star_2.7.11b -maxdepth 1 -type f -size +100M -exec cat {} + > /dev/null 2>&1
  echo "  genome index files read into the page cache in $((SECONDS - t0)) s" | tee -a $S
fi

write_results() {   # tp_point.py for each leg that has run, with the provenance of its arm, then collect.py
  local i=0 a pp
  for a in $(echo $ORDER | fold -w1); do
    i=$((i + 1)); pp=leg$i$a
    [ -f $RES0/$pp/point.json ] || continue
    arm_load $a
    PROV_JOB=$SLURM_JOB_ID PROV_NODE=$NODE PROV_BIN=$BIN PROV_SHA256=$(awk -F'\t' -v a=$a '$1==a{print $3}' $RES0/arms.tsv) \
      PROV_COMMIT=$(awk -F'\t' -v a=$a '$1==a{print $4}' $RES0/arms.tsv) PROV_SCRIPTS=$SCRIPTS python3 $GB/tp_point.py $RES0/$pp > $RES0/$pp/results.txt 2>&1
  done
  python3 $GF/collect.py $RES0 > /dev/null
}

# --- the legs ---------------------------------------------------------------------------------------------------------------------------------
idx=0; STOPPED_AT=""
for arm in $(echo $ORDER | fold -w1); do
  idx=$((idx + 1)); arm_load $arm; p=leg$idx$arm
  mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv; : > $RES0/$p/ident.tsv; : > $RES0/$p/identity.lines
  XENV="SINGLET_ENGINE_GPU_SLOTS=$SLOTS ${ARM_ENV[*]:-}"
  printf '{"tag":"%s","k":%d,"T":%d,"slots":%d,"rounds":1,"runs":%d,"pin":"","items":"%s","env":"%s","node":"%s","gpus":%d}\n' $p $K $T $SLOTS $NRUNS "$SAMPLE" "$XENV" "$(hostname)" $NGPU > $RES0/$p/meta.json
  echo "######## leg $idx arm $arm: $K in flight, T=$T, $NRUNS runs, slots=$SLOTS env=[${ARM_ENV[*]:-}] args=[${ARM_ARGS[*]:-}] $(date -Is)" | tee -a $S
  CT=$RES0/$p/cotenants.log; : > $CT; ctsnap $CT start
  ( while sleep 60; do ctsnap $CT minute; done ) &
  CTP=$!
  python3 $TPD/e18_sampler.py $RES0/$p/samples.jsonl $CG 0.5 2 2>$RES0/$p/sampler.err &
  SMP=$!
  ( while sleep 15; do python3 $GF/watch.py $RES0/$p $SCR/$p >> $RES0/$p/watch.log 2>&1; done ) &
  WCH=$!
  sleep 1
  WP=()
  for ((w = 0; w < K; w++)); do worker $p $w $T & WP+=($!); done
  wait "${WP[@]}"
  sleep 6; python3 $GF/watch.py $RES0/$p $SCR/$p >> $RES0/$p/watch.log 2>&1   # a last look after the grace period
  sleep 1; kill $SMP $CTP $WCH 2>/dev/null; wait $SMP 2>/dev/null
  ctsnap $CT end
  python3 $TPD/e18_point.py $RES0/$p > $RES0/$p/point.txt 2>&1
  [ -f $TPD/e18_locks.py ] && python3 $TPD/e18_locks.py $RES0/$p >> $RES0/$p/point.txt 2>&1
  # identity of a sample of the runs of this arm, spread over the queue, then the outputs of the leg are deleted
  cnt=$(echo $ORDER | tr -cd $arm | wc -c); n=$(( (IDENT_N + cnt - 1) / cnt ))
  for ((j = 0; j < n; j++)); do
    i=$(( (2 * j + 1) * NRUNS / (2 * n) ))
    rcI=$(awk -F'\t' -v i=$i '$1==i{print $6}' $RES0/$p/runs.tsv)
    [ "${rcI:-1}" = 0 ] || continue
    while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 1; done
    s=$SAMPLE; ident_ref $p $i $SAMPLE $REFDIR &
  done
  wait
  sort -V $RES0/$p/identity.lines 2>/dev/null | tee -a $S
  rm -rf $SCR/$p
  write_results
  tail -n 12 $RES0/$p/results.txt | cut -c1-300 | tee -a $S
  D=$(python3 $GF/abcore.py decide $RES0/result.json "${SEP_RATE:-}")
  echo "  after leg $idx: $D" | tee -a $S
  case $D in STOP*) if [ "$D" != "${D#STOP fallback}" ] && [ "${STOP_ON_FALLBACK:-1}" != 1 ]; then :; else STOPPED_AT=$D; break; fi ;; esac
done
[ -z "$STOPPED_AT" ] || { echo "$STOPPED_AT after leg $idx of ${#ORDER}" > $RES0/notes.txt; echo "$STOPPED_AT" | tee -a $S; }
write_results
echo "RESULT $RES0/result.json" | tee -a $S
echo done $(date -Is) | tee -a $S
