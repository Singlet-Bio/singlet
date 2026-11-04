#!/bin/bash
#SBATCH --job-name=e23-prof
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E23 (docs/gpu-engine/fable/E23-mtpileup.md): the GPU pileup's profile per sample. Runs the pipeline (--engine gpu, SINGLET_GPU_PILEUP=on, the
# defaults) on each of SAMPLES with SINGLET_GPU_PILEUP_PROF=1 (PROF=0: without) and keeps the run log; the pileup's lines ("[gpu_pileup] on:" with the
# per-stage GPU times, "stages (ms)", and the "prof" group statistics) go to $RES/<sample>-<bin>.pileup.txt. Outputs stay in the job's /tmp.
# BINS: binaries of $BINDIR to run (singlet-base singlet-gpu ...), each on every sample, in order.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/mtpileup/bin}; T=${T:-8}; SAMPLES=(${SAMPLES:-neuron1k_v3 pbmc1k_v3}); BINS=(${BINS:-singlet-base}); PROF=${PROF:-1}
RES=$W/jobs/fable/mtpileup/results/prof-$SLURM_JOB_ID; mkdir -p $RES
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
SCR=/tmp/$USER-e23p-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf "${SCR:?}"' EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID T=$T samples=${SAMPLES[*]} bins=${BINS[*]} prof=$PROF $(date -Is)" | tee $RES/summary.txt
for b in "${BINS[@]}"; do echo "$b $(cat $BINDIR/commit-${b#singlet-} 2>/dev/null) $(sha256sum $BINDIR/$b | cut -c1-16)" | tee -a $RES/summary.txt; done
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | tee -a $RES/summary.txt
for s in "${SAMPLES[@]}"; do
  G=$REF/GRCh38-2024-A
  case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
  FQ=$PB/data/$s/$s.1fq; [ -f $FQ ] || FQ=$W/jobs/fable/release/data/$s/$s.1fq
  for b in "${BINS[@]}"; do
    O=$SCR/$s-$b; mkdir -p $O/out
    export SINGLET_GPU_PILEUP=on; if [ "$PROF" = 1 ]; then export SINGLET_GPU_PILEUP_PROF=1; else unset SINGLET_GPU_PILEUP_PROF; fi
    T0=$(date +%s.%N)
    ( cd $O && /usr/bin/time -v -o $RES/$s-$b.time.txt timeout -k 30 1500 $BINDIR/$b $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $O/out/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $RES/$s-$b.run.log.ts )
    T1=$(date +%s.%N)
    echo "$s $b: wall $(echo "$T1 - $T0" | bc | cut -c1-7) s, exit $(awk -F': ' '/Exit status/{print $2}' $RES/$s-$b.time.txt)" | tee -a $RES/summary.txt
    grep -a -e "gpu_pileup\] on:" -e "device lock" -e "apply:" -e "gpu_pileup\] [0-9]* reads" -e "per-base quality" $RES/$s-$b.run.log.ts | cut -c1-200000 | sed 's/; /;\n    /g' > $RES/$s-$b.pileup.txt
    grep -a -e "device lock" -e "stages (ms)" -e "^    prof" -e "GPU [0-9.e+]* ms" -e "cell partitions" -e "part 0" $RES/$s-$b.pileup.txt | cut -c1-3000 | tee -a $RES/summary.txt
    cp $O/out/*Log.final.out $RES/$s-$b.Log.final.out 2>/dev/null
    rm -rf $O
  done
done
echo done $(date -Is) | tee -a $RES/summary.txt
