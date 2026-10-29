#!/bin/bash
#SBATCH --job-name=pe1d-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=160G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Phase 1d harness on one H100. STEP=unit: compile + `--unit` only. STEP=atac: compile, unit, then 200k pairs of atac_pbmc_5k_nextgem through
# STAR and the kernel, and the comparison. Code arrives by git (BRANCH, fetched by URL into a private ref); the harness is built with the CUDA
# image's gcc 11 (-ccbin /usr/bin/g++), never the conda compiler. Scratch is node-local /tmp and removed on exit. Results: $RES.
set -euo pipefail
STEP=${STEP:-unit}; N=${N:-200000}
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
BARE=$W/repos/singlet-py.git; BR=${BRANCH:-claude/gpu-paired-prep-f54059}
GDIR=${GDIR:-$W/data/w11/GRCh38-2024-A-nosjdb}
RES=$W/results/pe1d/$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/w11/paired-1d
COMMIT=$(git -C $BARE rev-parse refs/w11/paired-1d); echo "COMMIT=$COMMIT STEP=$STEP"; echo $COMMIT > $RES/commit
git -C $BARE archive refs/w11/paired-1d | tar -x -C $SCR/src
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
cat > $SCR/inner.sh <<EOS
set -euo pipefail
export PATH=/usr/local/cuda/bin:\$PATH      # system gcc 11 first; conda only by explicit path below
cd $SCR/src
nvcc --version | tail -1; g++ --version | head -1
CCBIN=/usr/bin/g++ CUDA_ARCH=sm_90 timeout 20m bash bench/gpu_paired/build.sh 2>&1 | tee $RES/build_unit.txt
[ "$STEP" = unit ] && exit 0
B=bench/gpu_paired
A=$W/data/atac-spatial/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fastqs/atac_pbmc_5k_nextgem_S1_L001
# the two genomic mates of 10x ATAC are R1 and R3; first N pairs of lane 1 (head closes the pipe: ignore SIGPIPE from zcat)
(zcat \${A}_R1_001.fastq.gz | head -n \$((4*$N)) > $SCR/r1.fq) || true
(zcat \${A}_R3_001.fastq.gz | head -n \$((4*$N)) > $SCR/r2.fq) || true
wc -l $SCR/r1.fq $SCR/r2.fq; head -2 $SCR/r1.fq; head -2 $SCR/r2.fq
head -1 $SCR/r1.fq | cut -d' ' -f1 > $RES/first_names.txt; head -1 $SCR/r2.fq | cut -d' ' -f1 >> $RES/first_names.txt
# STAR on the same pairs (conda STAR 2.7.11b), ATAC parameters; times itself
cd $SCR
T0=\$(date +%s.%N)
env LD_LIBRARY_PATH=$ENV/lib timeout 40m $ENV/bin/STAR --runThreadN 28 --genomeDir $GDIR --readFilesIn r1.fq r2.fq --outSAMtype SAM \
  --outSAMunmapped Within --outSAMattributes NH HI AS nM --outSAMmultNmax -1 --alignIntronMax 1 --alignMatesGapMax 2000 --outFileNamePrefix ref_ > $RES/star_stdout.txt 2>&1
echo "STAR wall seconds: \$(echo \$(date +%s.%N) - \$T0 | bc)" | tee $RES/star_time.txt
cp ref_Log.final.out $RES/; grep -E "Mapping speed|Number of input|Uniquely mapped reads n|multiple loci" ref_Log.final.out || true
# kernel
cd $SCR/src
timeout 40m \$B/paired_microbench --genome $GDIR --r1 $SCR/r1.fq --r2 $SCR/r2.fq --n $N --mode atac --batch 4096 --repeat 3 \
  --out $SCR/kernel.tsv --dump $SCR/win.bin 2>&1 | tee $RES/kernel_stderr.txt
timeout 30m env LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -I \$B/compare_star_sam.py $SCR/kernel.tsv $SCR/ref_Aligned.out.sam $GDIR --show 40 2>&1 | tee $RES/compare.txt
timeout 60m env LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -P \$B/replay_model.py $GDIR $SCR/r1.fq $SCR/r2.fq $SCR/win.bin $SCR/kernel.tsv --mode atac 2>&1 | tee $RES/replay.txt || echo "replay_model exit \$?"
gzip -c $SCR/kernel.tsv > $RES/kernel.tsv.gz
EOS
apptainer exec --nv -B $W -B $SCR $IMG bash $SCR/inner.sh
echo "[pe1d] ok $STEP"
