#!/bin/bash
#SBATCH --job-name=e23-ab
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E23 (docs/gpu-engine/fable/E23-mtpileup.md): A/B of the fix in one binary on the debug partition. Per sample of SAMPLES the pipeline runs
# (--engine gpu, SINGLET_GPU_PILEUP=on, the defaults, SINGLET_E16_OFF=digest so the pileup logs the digest of everything it returns) with
# SINGLET_GPU_PILEUP_E23=0 (k_r9, k_cr_roots: before E23) and =1 (E23), on $BINDIR/$BIN. Compares the two runs: the pileup digests, and
# out_equiv.py over the outputs (read files set aside, as the gate). The pileup's log lines go to $RES/<sample>-<mode>.pileup.txt.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BINDIR=${BINDIR:-$W/jobs/fable/mtpileup/bin}; BIN=${BIN:-singlet-gpu}; T=${T:-8}; SAMPLES=(${SAMPLES:-neuron1k_v3 pbmc1k_v3}); MODES=(${MODES:-off on})
RES=$W/jobs/fable/mtpileup/results/ab-$SLURM_JOB_ID; mkdir -p $RES
BARE=$W/repos/singlet-py.git
[ -e $BINDIR/../whitelists ] || ln -s $PB/whitelists $BINDIR/../whitelists
SCR=/tmp/$USER-e23ab-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf "${SCR:?}"' EXIT INT TERM
git -C $BARE archive $(cat $BINDIR/commit-${BIN#singlet-}) bench/star_accel/out_equiv.py | tar -x -C $SCR/repo
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID T=$T samples=${SAMPLES[*]} bin=$BIN $(cat $BINDIR/commit-${BIN#singlet-}) modes=${MODES[*]} $(date -Is)" | tee $RES/summary.txt
EXCL=(--exclude='*.bam' --exclude='*.sam' --exclude='*.fq*' --exclude='*.fastq*' --exclude='*.1fq' --exclude='.gpu_handoff*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.reads' --exclude='*.reads.q')
for s in "${SAMPLES[@]}"; do
  G=$REF/GRCh38-2024-A
  case $s in neuron1k_v3|nuclei2k_v2) G=$REF/GRCm39-2024-A ;; hgmm1k_v3) G=$REF/GRCh38-GRCm39-barnyard ;; esac
  FQ=$PB/data/$s/$s.1fq; [ -f $FQ ] || FQ=$W/jobs/fable/release/data/$s/$s.1fq
  for md in "${MODES[@]}"; do
    O=$SCR/$s-$md; mkdir -p $O/out
    export SINGLET_GPU_PILEUP=on SINGLET_E16_OFF=digest; [ $md = off ] && export SINGLET_GPU_PILEUP_E23=0 || export SINGLET_GPU_PILEUP_E23=1
    T0=$(date +%s.%N)
    ( cd $O && /usr/bin/time -v -o $RES/$s-$md.time.txt timeout -k 30 1500 $BINDIR/$BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
        --out-prefix $O/out/ --threads $T --pipeline --output-format 1pz --engine gpu 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $RES/$s-$md.run.log.ts )
    T1=$(date +%s.%N)
    echo "$s $md: wall $(echo "$T1 - $T0" | bc | cut -c1-7) s, exit $(awk -F': ' '/Exit status/{print $2}' $RES/$s-$md.time.txt)" | tee -a $RES/summary.txt
    grep -a -e "gpu_pileup\] on:" -e "device lock" -e "apply:" $RES/$s-$md.run.log.ts | cut -c1-200000 | sed 's/; /;\n    /g' > $RES/$s-$md.pileup.txt
    grep -a -e "device lock" -e "stages (ms)" -e "digest" -e "GPU [0-9.e+]* ms" $RES/$s-$md.pileup.txt | cut -c1-2000 | tee -a $RES/summary.txt
    rsync -a "${EXCL[@]}" $O/out/ $O/cmp/; rm -rf $O/out
  done
  if [ ${#MODES[@]} -ge 2 ]; then
    a=${MODES[0]}; b=${MODES[1]}
    da=$(grep -ao "digest [0-9a-f]*" $RES/$s-$a.pileup.txt | head -1); db=$(grep -ao "digest [0-9a-f]*" $RES/$s-$b.pileup.txt | head -1)
    [ -n "$da" ] && [ "$da" = "$db" ] && echo "$s pileup digest: SAME ($da)" | tee -a $RES/summary.txt || echo "$s pileup digest: DIFFERENT ($da vs $db)" | tee -a $RES/summary.txt
    python3 $SCR/repo/bench/star_accel/out_equiv.py $SCR/$s-$a/cmp $SCR/$s-$b/cmp > $RES/$s-equiv.txt 2>&1
    grep -a '^  DIFF\|^  only in\|^files:\|^RESULT' $RES/$s-equiv.txt | cut -c1-300 | tee -a $RES/summary.txt
  fi
  rm -rf $SCR/$s-*
done
echo done $(date -Is) | tee -a $RES/summary.txt
