#!/bin/bash
#SBATCH --job-name=eng-gpu-pucmp
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# The GPU pileup inside singlet vs the CPU pileup of the same pass (SINGLET_GPU_PILEUP=compare): one sample per SET, stream path,
# both engines' mapping (ENGINES). Prints the [gpu_pileup] lines (every covered output compared entry by entry). Outputs in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/w11/engine/bin15/singlet-gpu}; SETS=(${SETS:-pbmc1k_v3}); T=${T:-20}; ENGINES=(${ENGINES:-gpu})
RES=$W/jobs/w11/engine/results/pucmp-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-pucmp-$SLURM_JOB_ID; mkdir -p $SCR
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 3; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF SINGLET_GPU_PILEUP=compare
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit) $(date -Is)"
for s in "${SETS[@]}"; do
  G=$REF/GRCh38-2024-A; [ "$s" = GSM5411066 ] && G=$REF/GRCm39-2024-A
  FQ=$PB/data/$s/$s.1fq; [ -e $FQ ] || { D=$W/data/bam-stream/$s; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$s; FQ=$D/sample.1fq; }
  for E in "${ENGINES[@]}"; do
    mkdir -p $SCR/$s-$E/out
    ( cd $SCR/$s-$E && timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $SCR/$s-$E/out/ --threads $T --pipeline --output-format 1pz --engine $E > $RES/$s-$E.log 2>&1; echo "exit $?" >> $RES/$s-$E.log )
    echo "== $s --engine $E: $(tail -1 $RES/$s-$E.log)"; grep -a "BAM:\|\[gpu_pileup\]" $RES/$s-$E.log
  done
  rm -rf $SCR/$s-*
done
echo done $(date -Is)
