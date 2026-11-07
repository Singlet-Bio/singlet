#!/bin/bash
#SBATCH --job-name=e17-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E17 (docs/gpu-engine/fable/E17-budget.md): the prefill's device memory, run by run, stopped once the prefill is over. From E15's probe.sbatch.
# RUNS="label|sample|K=V,K=V ..." (space-separated runs): each is level 5 + ingest (SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5
# SINGLET_ENGINE_GPU_INGEST=1, --engine gpu) at T threads with the binary BINDIR/singlet-gpu (BIN_<label> overrides), plus the run's K=V pairs.
# A run stops LIMIT s after its start or 5 s after star_Log.out says the prefill is over ("cache filled", "not used", a FATAL ERROR). Per run: the
# engine lines of star_Log.out, the device memory marks and the E17 budget lines, and the peak of a 1 s nvidia-smi sampler (MiB) over the run.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/budget/bin}; T=${T:-8}; LIMIT=${LIMIT:-600}
RUNS=(${RUNS:-"hgmm|hgmm1k_v3|"})
RES=$W/jobs/fable/budget/results/probe-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e17p-$SLURM_JOB_ID; mkdir -p $SCR
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BINDIR commit=$(cat $BINDIR/commit-gpu 2>/dev/null) T=$T runs=${RUNS[*]} $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee -a $RES/summary.txt
for r in "${RUNS[@]}"; do
  IFS='|' read -r lab s envs <<< "$r"
  G=$REF/GRCh38-2024-A; case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
  bv=BIN_$lab; BIN=${!bv:-$BINDIR/singlet-gpu}
  FQ=$PB/data/$s/$s.1fq; O=$SCR/$lab; mkdir -p $O/out
  echo "=== $lab: $s (genome $G) env: ${envs//,/ } bin $BIN" | tee -a $RES/summary.txt
  ( echo -e "t\tgpu_util\tgpu_mem_mb"
    while true; do printf "%s\t%s\n" $(date +%s.%N) "$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ' | tr , '\t')"; sleep 1; done
  ) > $RES/$lab.monitor.tsv 2>/dev/null &
  MON=$!
  ( cd $O && export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 && for kv in ${envs//,/ }; do export "$kv"; done \
    && timeout -k 10 $LIMIT $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/out/ --threads $T \
       --pipeline --output-format 1pz --engine gpu > $O/run.log 2>&1 ) &
  P=$!; T0=$(date +%s)
  while kill -0 $P 2>/dev/null; do
    if grep -aq "cache filled\|ENGINE_GPU: not used\|FATAL ERROR" $O/out/star_Log.out 2>/dev/null; then sleep 5; break; fi
    [ $(( $(date +%s) - T0 )) -gt $LIMIT ] && break; sleep 2
  done
  pkill -TERM -f "out-prefix $O/out/" 2>/dev/null; sleep 5; pkill -KILL -f "out-prefix $O/out/" 2>/dev/null; wait $P 2>/dev/null
  sleep 2; kill $MON 2>/dev/null
  echo "  $lab: $(( $(date +%s) - T0 )) s; nvidia-smi peak $(awk -F'\t' 'NR>1 && $3+0>m{m=$3+0} END{print m+0}' $RES/$lab.monitor.tsv) MiB" | tee -a $RES/summary.txt
  grep -ah "ENGINE_GPU\|engine_gpu\|FATAL" $O/out/star_Log.out 2>/dev/null | cut -c1-3000 > $RES/$lab.engine.txt
  grep -ah "device memory\|E17\|budget\|not used\|out of memory\|FATAL\|E12 lanes\|cache filled\|steps (s)\|engine_gpu: [0-9]" $RES/$lab.engine.txt | cut -c1-1500 | tee -a $RES/summary.txt
  cp $O/out/star_Log.out $RES/$lab.star_Log.out 2>/dev/null; cp $O/run.log $RES/$lab.run.log 2>/dev/null
  rm -rf $O
done
echo done $(date -Is) | tee -a $RES/summary.txt
