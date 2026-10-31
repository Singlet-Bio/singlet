#!/bin/bash
#SBATCH --job-name=eng-ir-polya
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# poly(A) records of the CPU engine vs --engine gpu + GPU pileup (stream path, default rules) on pbmc1k_v3: SINGLET_POLYA_DUMP + polya_diff.py.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/w11/engine/bin_ir/singlet-gpu}; S=${S:-pbmc1k_v3}; T=18
BARE=$W/repos/singlet-py.git
RES=$W/jobs/w11/engine/results/polya-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-polya-$SLURM_JOB_ID; mkdir -p $SCR/repo; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/w11/intron-rules bench/engine_gpu_gate/polya_diff.py | tar -x -C $SCR/repo
export SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
G=$REF/GRCh38-2024-A; FQ=$PB/data/$S/$S.1fq
echo "bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit)" | tee $RES/summary.txt
for m in cpu gpu; do
  mkdir -p $SCR/$m/out; if [ $m = gpu ]; then export SINGLET_GPU_PILEUP=on; else unset SINGLET_GPU_PILEUP; fi
  ( cd $SCR/$m && SINGLET_POLYA_DUMP=$RES/$S-$m.polya.bin timeout -k 30 3000 $BIN $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
      --out-prefix $SCR/$m/out/ --threads $T --pipeline --output-format 1pz --engine $m > $RES/$S-$m.log 2>&1; echo "exit $?" >> $RES/$S-$m.log )
  grep -a "\[polya\]\|^exit" $RES/$S-$m.log | tee -a $RES/summary.txt
done
python3 $SCR/repo/bench/engine_gpu_gate/polya_diff.py $RES/$S-cpu.polya.bin $RES/$S-gpu.polya.bin 2>&1 | tee -a $RES/summary.txt
