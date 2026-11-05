#!/bin/bash
#SBATCH --job-name=gf-t1
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Fast lane, tier 1 (bench/gpu_fast/README.md): one sample alone, arms in the order of ORDER (default ABBA), then identity against a kept reference.
# Started by ab.py. Do not start it by hand unless you wrote the spec file yourself.
#
# Parameter (sbatch --export=ALL,SPEC=<file>): SPEC is a shell file written by ab.py. It sets
#   NAME RUN SAMPLE T SLOTS ORDER REFDIR INTENDED SBATCH_SHA SCRIPTS_BRANCH EXPECT_SCRIPTS
#   BIN_A COMMIT_A ENV_A ARGS_A and BIN_B COMMIT_B ENV_B ARGS_B (ENV and ARGS are bash arrays)
# Results: $RUN/res/ (result.json, one directory per leg, summary.txt). Nothing outside $W/jobs/gpu-fast is written.
#
# What it checks. Each arm's directory must record a commit (commit-gpu) that starts with COMMIT_x when given, and that commit string must be
# inside the binary. The sha256 of the binary is recorded. The scripts come from the branch under flock and the tip must equal EXPECT_SCRIPTS.
# This file must equal the copy on that branch (SBATCH_SHA). Timing runs come first and nothing else runs between them. Identity comes last.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
DATA2=$W/jobs/fable/release/data
SPEC=${SPEC:?SPEC: the spec file written by ab.py}; [ -f "$SPEC" ] || { echo "no spec file $SPEC"; exit 2; }
# shellcheck disable=SC1090
. <(tr -d '\r' < "$SPEC")   # the spec is written with LF. Dropping a CR costs nothing and a stray one breaks every path
case $RUN in $W/jobs/gpu-fast/*) ;; *) echo "RUN must lie under $W/jobs/gpu-fast"; exit 2 ;; esac
RES0=$RUN/res; mkdir -p $RES0; S=$RES0/summary.txt; : > $S
BARE=$W/repos/singlet-py.git; SBR=${SCRIPTS_BRANCH:-bench/gpu-fast}

arm_load() {   # A or B -> BIN COMMIT ARM_ENV ARM_ARGS
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
SCR=/tmp/$USER-gft1-$SLURM_JOB_ID; mkdir -p $SCR/repo $SCR/data
SREF=refs/gf/t1-$SLURM_JOB_ID
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
  have=$(sha256sum $GF/jobs/t1.sbatch | cut -d' ' -f1)
  [ "$have" = "$SBATCH_SHA" ] || refuse "t1.sbatch on $SBR ($have) differs from the copy that was submitted ($SBATCH_SHA)"
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
echo "PROVENANCE job=$SLURM_JOB_ID node=$NODE tier=1 name=$NAME scripts=$SCRIPTS sbatch_sha=${SBATCH_SHA:-none} sample=$SAMPLE T=$T slots=$SLOTS order=$ORDER ref=$REFDIR $(date -Is)" | tee -a $S
column -t -s $'\t' $RES0/arms.tsv | sed 's/^/  /' | tee -a $S
printf '{"tier":1,"name":"%s","job":"%s","node":"%s","scripts":"%s","ref":"%s","T":%d,"slots":%d,"sample":"%s","order":"%s","intended_change":%d}\n' \
  "$NAME" $SLURM_JOB_ID $NODE $SCRIPTS "$REFDIR" $T $SLOTS $SAMPLE $ORDER ${INTENDED:-0} > $RES0/params.json
{ nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader; lscpu | grep -E "^Model name|^CPU\(s\)"; free -g | head -2; } 2>&1 | sed 's/^/  /' | tee -a $S

stage_input $SAMPLE
ITEMS=($SAMPLE)
# The first leg would otherwise read the genome index from the shared disk while the last leg finds it cached. The order ABBA cancels a linear
# drift but not a cold first leg. Read the index files into the page cache once, before the first leg. WARM=0 skips it.
if [ "${WARM:-1}" = 1 ]; then
  t0=$SECONDS; find $(genome $SAMPLE)/star_2.7.11b -maxdepth 1 -type f -size +100M -exec cat {} + > /dev/null 2>&1
  echo "  genome index files read into the page cache in $((SECONDS - t0)) s" | tee -a $S
fi
ctsnap() { { echo "## $2 $(date -Is)"; squeue -w $NODE -o "%i %u %j %C %b" 2>&1; } >> $1; }

# --- the timing legs. Nothing else runs on the node between them ----------------------------------------------------------------------------
idx=0
for arm in $(echo $ORDER | fold -w1); do
  idx=$((idx + 1)); arm_load $arm; p=leg$idx$arm
  mkdir -p $SCR/$p $RES0/$p; echo 0 > $SCR/$p/queue; : > $RES0/$p/runs.tsv; : > $RES0/$p/ident.tsv; : > $RES0/$p/identity.lines
  XENV="SINGLET_ENGINE_GPU_SLOTS=$SLOTS ${ARM_ENV[*]:-}"
  printf '{"tag":"%s","k":1,"T":%d,"slots":%d,"rounds":1,"runs":1,"pin":"","items":"%s","env":"%s","node":"%s","gpus":%d}\n' $p $T $SLOTS $SAMPLE "$XENV" "$(hostname)" $NGPU > $RES0/$p/meta.json
  echo "######## leg $idx arm $arm: one run alone, T=$T slots=$SLOTS env=[${ARM_ENV[*]:-}] args=[${ARM_ARGS[*]:-}] $(date -Is)" | tee -a $S
  CT=$RES0/$p/cotenants.log; : > $CT; ctsnap $CT start
  python3 $TPD/e18_sampler.py $RES0/$p/samples.jsonl $CG 0.5 2 2>$RES0/$p/sampler.err &
  SMP=$!
  sleep 1
  run_item $p 0 0 $SAMPLE $T
  sleep 1; kill $SMP 2>/dev/null; wait $SMP 2>/dev/null
  ctsnap $CT end
  rc=$(awk -F'\t' '{print $6}' $RES0/$p/runs.tsv | head -1)
  echo "  leg $idx ($arm): exit ${rc:-?}, wall $(awk -F'\t' '{printf "%.1f", $5-$4}' $RES0/$p/runs.tsv) s" | tee -a $S
done
TIMING_DONE=$SECONDS

# --- analysis, once all timing runs are over --------------------------------------------------------------------------------------------------
idx=0
for arm in $(echo $ORDER | fold -w1); do
  idx=$((idx + 1)); p=leg$idx$arm
  python3 $TPD/e18_point.py $RES0/$p > $RES0/$p/point.txt 2>&1
  [ -f $TPD/e18_locks.py ] && python3 $TPD/e18_locks.py $RES0/$p >> $RES0/$p/point.txt 2>&1
done
write_results() {   # runs tp_point.py for each leg with the provenance of its arm, then collect.py
  local i=0 a pp
  for a in $(echo $ORDER | fold -w1); do
    i=$((i + 1)); pp=leg$i$a
    arm_load $a
    PROV_JOB=$SLURM_JOB_ID PROV_NODE=$NODE PROV_BIN=$BIN PROV_SHA256=$(awk -F'\t' -v a=$a '$1==a{print $3}' $RES0/arms.tsv) \
      PROV_COMMIT=$(awk -F'\t' -v a=$a '$1==a{print $4}' $RES0/arms.tsv) PROV_SCRIPTS=$SCRIPTS python3 $GB/tp_point.py $RES0/$pp 2>&1 | tee -a $S $RES0/$pp/results.txt
  done
  python3 $GF/collect.py $RES0 | tee -a $S
}
write_results   # the timing numbers are on disk even if identity runs out of time

if [ $SECONDS -gt 1500 ]; then
  echo "IDENTITY SKIPPED: $SECONDS s used of the 29 min limit" | tee -a $S
else
  idx=0
  for arm in $(echo $ORDER | fold -w1); do
    idx=$((idx + 1)); p=leg$idx$arm; s=$SAMPLE
    while [ $(jobs -rp | wc -l) -ge 4 ]; do sleep 1; done
    ident_ref $p 0 $SAMPLE $REFDIR &
  done
  wait
  for f in $(ls $RES0/leg*/identity.lines 2>/dev/null | sort -V); do cat $f | tee -a $S; done
  write_results
fi
echo "RESULT $RES0/result.json" | tee -a $S
echo done $(date -Is) | tee -a $S
