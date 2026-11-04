#!/bin/bash
#SBATCH --job-name=e9-run
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E9 (docs/gpu-engine/fable/E9-solo.md): the Solo driver on the truth run's dumps. TRUTH=<truth job id> (bench/fable/solo/jobs/truth.sbatch).
# MODE=gpu|host|both, REPS. The dumps (read-level data) are read from $W/jobs/fable/solo/scratch/truth-<id>/; our Solo.out goes to job /tmp; the
# log (comparison, timings) and our small stats files go to results/run-<job>/.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; REF=/mnt/projects/debruinz_project/cellarium/reference
E9=$W/jobs/fable/solo; TRUTH=${TRUTH:?TRUTH=<truth job id>}; MODE=${MODE:-both}; REPS=${REPS:-3}
SC=$E9/scratch/truth-$TRUTH; TR=$E9/results/truth-$TRUTH/dump/star_Solo.out
RES=$E9/results/run-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e9r-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT INT TERM
echo "node=$(hostname) job=$SLURM_JOB_ID truth=$TRUTH commit=$(cat $E9/bin/solo_test.commit) mode=$MODE reps=$REPS $(date -Is)" | tee $RES/summary.txt
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
/usr/bin/time -v -o $RES/time.txt apptainer exec --nv -B $W -B /mnt/projects/debruinz_project/cellarium -B $SCR $IMG $E9/bin/solo_test --dump $SC/engine.dump --solo-dir $SC/solo \
  --genome-dir $REF/GRCh38-2024-A/star_2.7.11b --whitelist $W/pipeline-benchmark/whitelists/3M-february-2018.txt --truth $TR --out $SCR/out \
  --mode $MODE --reps $REPS 2>&1 | tee -a $RES/summary.txt
for m in host gpu; do [ -d $SCR/out/$m ] && { mkdir -p $RES/$m; cp $SCR/out/$m/star_Solo.out/Barcodes.stats $SCR/out/$m/star_Solo.out/Gene/Features.stats $SCR/out/$m/star_Solo.out/Gene/UMIperCellSorted.txt $RES/$m/ 2>/dev/null; }; done
grep -e "Maximum resident" -e "Elapsed" $RES/time.txt | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
