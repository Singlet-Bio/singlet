#!/bin/bash
#SBATCH --job-name=e15-probe
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E15 (docs/gpu-engine/fable/E15-posterior.md): which samples run E11's level 5 and why not (the "E11 (level 5) not used" line), and which
# CB quality characters STAR sees (the "E11 decode" line), with the binary in BINDIR (default: E11's bin-s6). Per sample in SAMPLES a level-5
# run at T threads is stopped once the prefill is over ("cache filled") or after LIMIT s. Also: the quality characters of the first
# 250,000 records of the sample's original R1 FASTQ (gex_fastqs.txt), which the .1fq does not store.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/solowire/bin-s6}; T=${T:-8}; LIMIT=${LIMIT:-300}
SAMPLES=(${SAMPLES:-pbmc2m pbmc5p_v1 neuron1k_v3 hgmm1k_v3 nuclei2k_v2})
RES=$W/jobs/fable/posterior/results/probe-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e15p-$SLURM_JOB_ID; mkdir -p $SCR
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BINDIR commit=$(cat $BINDIR/commit-gpu 2>/dev/null) T=$T $(date -Is)" | tee $RES/summary.txt
for s in "${SAMPLES[@]}"; do
  G=$REF/GRCh38-2024-A; case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; esac; [ $s = hgmm1k_v3 ] && G=$REF/GRCh38-GRCm39-barnyard
  FQ=$PB/data/$s/$s.1fq; O=$SCR/$s; mkdir -p $O/out
  echo "=== $s ($(ls -l $FQ | awk '{print $5}') bytes, genome $G)" | tee -a $RES/summary.txt
  r1=$(grep -m1 "_R1_" $PB/data/$s/gex_fastqs.txt 2>/dev/null)
  [ -z "$r1" ] && [ $s = pbmc2m ] && r1=$(grep -m1 "_R1_" $PB/data/pbmc1k_v3/gex_fastqs.txt)
  if [ -n "$r1" ] && [ -f "$r1" ]; then
    echo "  original R1 $r1: quality characters of the first 250000 records (char:count)" | tee -a $RES/summary.txt
    zcat $r1 2>/dev/null | head -n 1000000 | awk 'NR%4==0{for(i=1;i<=length($0);i++)c[substr($0,i,1)]++} END{for(k in c) printf "%s:%d ", k, c[k]; print ""}' | tee -a $RES/summary.txt
  fi
  ( cd $O && SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 timeout -k 10 $LIMIT $BINDIR/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b \
      --exons $G/genes/genes.gtf --out-prefix $O/out/ --threads $T --pipeline --output-format 1pz --engine gpu > $O/run.log 2>&1 ) &
  P=$!; T0=$(date +%s)
  while kill -0 $P 2>/dev/null; do
    if grep -aq "cache filled\|E11 (level 5) not used\|E10 (level 5) not used" $O/out/star_Log.out 2>/dev/null; then sleep 5; break; fi
    [ $(( $(date +%s) - T0 )) -gt $LIMIT ] && break; sleep 2
  done
  pkill -TERM -f "out-prefix $O/out/" 2>/dev/null; sleep 5; pkill -KILL -f "out-prefix $O/out/" 2>/dev/null; wait $P 2>/dev/null
  grep -a -A1 "^##### Command Line" $O/out/star_Log.out 2>/dev/null | tail -1 | tr ' ' '\n' | grep -a -A1 "^--solo\|^--clip" | grep -v "^--$" | paste -sd' ' | cut -c1-1500 | tee -a $RES/summary.txt
  grep -ah "ENGINE_GPU: E10\|ENGINE_GPU: E11\|E11 (level\|E10 (level\|ENGINE_GPU: cache filled\|ENGINE_GPU: not used\|ENGINE_GPU:.*skip" $O/out/star_Log.out 2>/dev/null | cut -c1-500 | tee -a $RES/summary.txt
  grep -a "engine_gpu\|SINGLET_GPU_PILEUP\|stream\|E8\|gpu_pileup" $O/run.log | head -12 | cut -c1-300 | tee -a $RES/summary.txt
  cp $O/out/star_Log.out $RES/$s.star_Log.out 2>/dev/null; cp $O/run.log $RES/$s.run.log 2>/dev/null
  rm -rf $O
done
echo done $(date -Is) | tee -a $RES/summary.txt
