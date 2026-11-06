#!/bin/bash
#SBATCH --job-name=gshidx-pre
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Preflight of the shared index: the holder started by hand (resident, listening), then one pbmc1k_v3 run with SINGLET_GPU_SHARED_INDEX=1
# that must log "shared index: imported". Exit 1 otherwise. Nothing is compared here (identity.sbatch does that).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SH=${SH:-$W/jobs/gpu-shidx/bin/shidx}; G=$REF/GRCh38-2024-A; s=pbmc1k_v3
SCR=/tmp/$USER-pre-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'kill $(jobs -p) 2>/dev/null; $SH/singlet-gpu gpu-holder --stop --genome-dir $G/star_2.7.11b --device 0; rm -rf $SCR' EXIT
rm -f /tmp/singlet_gpu_holder.* 2>/dev/null
echo "commit $(cat $SH/commit-gpu); short hash in binary: $(grep -ac $(cut -c1-8 $SH/commit-gpu) $SH/singlet-gpu) full hash: $(grep -acF $(cat $SH/commit-gpu) $SH/singlet-gpu)"
$SH/singlet-gpu gpu-holder --genome-dir $G/star_2.7.11b --device 0 > $SCR/holder.log 2>&1 &
for i in $(seq 1 120); do ls /tmp/singlet_gpu_holder.*.info >/dev/null 2>&1 && break; sleep 2; done
echo "holder log:"; cat $SCR/holder.log | cut -c1-300; nvidia-smi --query-gpu=memory.used --format=csv,noheader
ls /tmp/singlet_gpu_holder.*.info >/dev/null 2>&1 || { echo "PREFLIGHT FAIL: the holder is not resident"; exit 1; }
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
export SINGLET_GPU_PILEUP=on SINGLET_ENGINE_GPU_E2E=5 SINGLET_ENGINE_GPU_INGEST=1 SINGLET_ENGINE_GPU_SLOTS=1 SINGLET_GPU_SHARED_INDEX=1
src=$PB/data/$s/$s.1fq; [ -f $src ] || src=$W/jobs/fable/release/data/$s/$s.1fq
timeout 1200 $SH/singlet-gpu $src --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads 8 --pipeline --output-format 1pz --engine gpu > $SCR/run.log 2>&1
echo "run exit $?"
grep -a "shared index\|E17 budget" $SCR/out/star_Log.out | cut -c1-330
if grep -aq "shared index: imported" $SCR/out/star_Log.out; then echo "PREFLIGHT PASS"; else echo "PREFLIGHT FAIL: imported=0"; cat $SCR/holder.log | tail -3; exit 1; fi
