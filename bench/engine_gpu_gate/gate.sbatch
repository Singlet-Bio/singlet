#!/bin/bash
#SBATCH --job-name=eng-ir-gate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=03:59:00
#SBATCH --array=0-2
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Phase 1b gates: (c) partition invariance (forced K = 1, 1, 2, 7: digest of every GPU pileup output, and every file compared),
# (f) the largest sample with the automatic K (no unified-memory retry), (a) CPU pileup vs GPU pileup on the rest (R9 + R10).
# Mode names: cpu (CPU pileup), on (GPU pileup, automatic K), cmp (compare), onN / onNb (GPU pileup, K forced to N). A trailing 0: SINGLET_INTRON_RULES=0
# (the pre-#80 classification); without it the default rules (389: kUnique, kNoSplicedIntron, kUmiGeneCR, kMtPrefer) run. RULES=default: gate (a) with them.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/w11/engine/bin_ir/singlet-gpu}; T=${T:-18}
if [ "${RULES:-0}" = default ]; then TASKS=("pbmc1k_v3:cpu cmp on" "GSM6470208:cpu cmp on" "GSM4006843:cpu on"); else
TASKS=("pbmc1k_v3:cpu0 cmp0 on0" "GSM6470208:cpu0 cmp0 on0 cpuD onD" "GSM4006843:cpu0 on0"); fi
SPEC=${TASKS[$SLURM_ARRAY_TASK_ID]}
BARE=$W/repos/singlet-py.git
RES=$W/jobs/w11/engine/results/gate-ir-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $RES
SCR=/tmp/$USER-gate-ir-$SLURM_JOB_ID; mkdir -p $SCR/repo
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 3; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
git -C $BARE archive refs/w11/intron-rules bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit) $(date -Is)" | tee $RES/summary.txt
PAIRS=(); cur=""; for tok in $SPEC; do if [[ $tok == *:* ]]; then [ -n "$cur" ] && PAIRS+=("$cur"); cur="$tok"; else cur="$cur,$tok"; fi; done; PAIRS+=("$cur")
for pr in "${PAIRS[@]}"; do
  s=${pr%%:*}; IFS=, read -ra MODES <<< "${pr#*:}"
  G=$REF/GRCh38-2024-A
  FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || { D=$W/data/w11/$s; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$s; FQ=$D/sample.1fq; }
  for m in "${MODES[@]}"; do
    mkdir -p $SCR/$s/$m/out
    unset SINGLET_GPU_PILEUP SINGLET_GPU_PILEUP_PARTS SINGLET_HANDOFF_LEAN SINGLET_HANDOFF_CLIPCHECK SINGLET_INTRON_RULES
    case $m in *0) export SINGLET_INTRON_RULES=0 ;; esac
    ENGARG="--engine gpu"; case $m in cpu|cpu0|cpuD) ENGARG="" ;; esac   # cpu: the default CPU engine and CPU stream pileup (the baseline)
    case $m in cpu|cpu0|cpuD) ;; cmp) export SINGLET_GPU_PILEUP=compare ;; cmp0) export SINGLET_GPU_PILEUP=compare ;; on0|onD) export SINGLET_GPU_PILEUP=on ;; on) export SINGLET_GPU_PILEUP=on ;; onv) export SINGLET_GPU_PILEUP=on SINGLET_HANDOFF_LEAN=verify ;;
      onc) export SINGLET_GPU_PILEUP=on SINGLET_HANDOFF_CLIPCHECK=1 ;; on*) export SINGLET_GPU_PILEUP=on SINGLET_GPU_PILEUP_PARTS=$(echo $m | tr -dc 0-9) ;; esac
    C0=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T0=$(date +%s.%N)
    ( cd $SCR/$s/$m && /usr/bin/time -v -o $RES/$s-$m.time timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$s/$m/out/ --threads $T --pipeline --output-format 1pz $ENGARG > $RES/$s-$m.log 2>&1; echo "exit $?" >> $RES/$s-$m.log )
    C1=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat); T1=$(date +%s.%N)
    echo "$s $m: $(tail -1 $RES/$s-$m.log), wall $(echo "$T1 - $T0" | bc) s, core-s $(echo "($C1 - $C0)/1000000" | bc -l | cut -c1-8), maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $RES/$s-$m.time)" | tee -a $RES/summary.txt
    grep -a -e "gpu_pileup" -e "hand-off" $RES/$s-$m.log | tee -a $RES/summary.txt
    grep -ah "ENGINE_GPU: cache filled\|engine_gpu: clipping\|lean alignment fields" $SCR/$s/$m/out/*Log.out 2>/dev/null | head -3 | tee -a $RES/summary.txt
    grep -ac "SINGLET_HANDOFF_CLIPCHECK" $RES/$s-$m.log | sed "s/^/clipcheck lines: /" | tee -a $RES/summary.txt
  done
  unset SINGLET_GPU_PILEUP SINGLET_GPU_PILEUP_PARTS SINGLET_HANDOFF_LEAN SINGLET_HANDOFF_CLIPCHECK
  for pr2 in "cpu0 on0" "cmp0 on0" "cpuD onD" "cpu on" "cmp on"; do set -- $pr2; { [ -d $SCR/$s/$1/out ] && [ -d $SCR/$s/$2/out ]; } || continue;
    echo "=== $s $1 vs $2" | tee -a $RES/summary.txt
    python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$s/$1/out $SCR/$s/$2/out > $RES/equiv-$s-$1-$2.txt 2>&1
    grep -a '^  DIFF\|^files:\|^RESULT' $RES/equiv-$s-$1-$2.txt | cut -c1-230 | head -70 | tee -a $RES/summary.txt
  done
  rm -rf $SCR/$s
done
echo done $(date -Is)
