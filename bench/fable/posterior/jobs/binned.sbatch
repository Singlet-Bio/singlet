#!/bin/bash
#SBATCH --job-name=e15-binned
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E15 (docs/gpu-engine/fable/E15-posterior.md): the per-read CB posterior against STAR on binned NovaSeq barcode qualities.
# The pipeline never gives STAR the barcode read's own qualities: its .1fq path writes 'F' for every R1 base, and its legacy FASTQ path
# (`--reads`) runs CB_samTagOut (no Gene counting). So STAR runs here as itself: `e15star` (jobs/starbuild.sbatch, this branch's vendored
# STAR) on the sample's original L001 FASTQs (gzipped, read with zcat; or their first NREADS reads), with the CPU engine's STAR Solo options
# and SINGLET_SOLO_DUMP. Then bench/fable/posterior/post_test.cu (compiled here from BRANCH) runs E9's library on STAR's per-read Solo inputs
# with the reads' CB qualities (perread) and with one 'F' quality (const, the control) and compares per read and per file with STAR's
# star_Solo.out. Read-level data (FASTQs, dump) stay in /tmp; results/binned-<job>/ keeps STAR's star_Solo.out, its logs, the driver's log.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
BINDIR=${BINDIR:-$W/jobs/fable/posterior/bin}; T=${T:-8}; NREADS=${NREADS:-0}; BR=${BRANCH:-gpu/fable/posterior}
s=${SAMPLE:-pbmc1k_v3}; WL=${WL:-3M-february-2018.txt}; UMIL=${UMIL:-12}; G=${G:-$REF/GRCh38-2024-A}
RES=$W/jobs/fable/posterior/results/binned-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e15d-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/hts $SCR/solo
trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
export PATH=$ENV/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID e15star=$(cat $BINDIR/e15star.commit) sample=$s NREADS=$NREADS T=$T $(date -Is)" | tee $RES/summary.txt
# ---- the driver
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/posterior-drv
echo "driver commit $(git -C $BARE rev-parse --short=8 refs/fable/posterior-drv)" | tee -a $RES/summary.txt
git -C $BARE archive refs/fable/posterior-drv src/engine_gpu include/singlet bench/fable/posterior | tar -x -C $SCR/src
ln -s $ENV/include/htslib $SCR/hts/htslib
S=$SCR/src; T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -Xcompiler -fopenmp -Xcompiler -Wno-format -I$S/src/engine_gpu -I$S/include \
  -I$SCR/hts --extended-lambda -o $SCR/post_test $S/bench/fable/posterior/post_test.cu $S/src/engine_gpu/solo_gpu.cu -lgomp 2>&1 | grep -v "^$" | head -60 | tee -a $RES/summary.txt
[ -x $SCR/post_test ] || { echo "compile FAILED" | tee -a $RES/summary.txt; exit 1; }
echo "compiled in $(( $(date +%s) - T0 )) s" | tee -a $RES/summary.txt
# ---- the reads (original qualities)
R1=$(grep -m1 "_L001_R1_" $PB/data/$s/gex_fastqs.txt); R2=${R1/_R1_/_R2_}
T0=$(date +%s)
if [ $NREADS -gt 0 ]; then   # the first NREADS reads, gzipped again
  zcat $R1 | head -n $((4 * NREADS)) | tee $SCR/r1.fastq | gzip -1 > $SCR/r1.fastq.gz & zcat $R2 | head -n $((4 * NREADS)) | gzip -1 > $SCR/r2.fastq.gz & wait
  GZ1=$SCR/r1.fastq.gz; GZ2=$SCR/r2.fastq.gz
else   # the whole lane, as it is; the barcode reads uncompressed for the driver
  zcat $R1 > $SCR/r1.fastq; GZ1=$R1; GZ2=$R2
fi
echo "reads: STAR reads $GZ2 (cDNA) + $GZ1 (barcode); $(( $(wc -l < $SCR/r1.fastq) / 4 )) barcode records for the driver ($(( $(date +%s) - T0 )) s)" | tee -a $RES/summary.txt
# ---- STAR with the per-read Solo dump: the CPU engine's STAR Solo, clip and filter options (gate 796377's command line), FASTQ input, no BAM
O=$SCR/run; mkdir -p $O
T0=$(date +%s)
SINGLET_SOLO_DUMP=$SCR/solo timeout -k 30 1300 $BINDIR/e15star --runMode alignReads --genomeDir $G/star_2.7.11b --readFilesIn $GZ2 $GZ1 --readFilesCommand zcat \
  --soloType CB_UMI_Simple --soloCBwhitelist $PB/whitelists/$WL --soloCBlen 16 --soloCBstart 1 --soloUMIstart 17 --soloBarcodeReadLength 0 \
  --soloCellReadStats Standard --soloUMIlen $UMIL --outSAMtype None --clipAdapterType CellRanger4 --alignSJoverhangMin 8 \
  --outFilterIntronMotifs RemoveNoncanonicalUnannotated --outFilterMismatchNoverLmax 0.05 --outFilterScoreMin 10 \
  --soloCBmatchWLtype 1MM_multi_Nbase_pseudocounts --runThreadN $T --genomeLoad NoSharedMemory --outFileNamePrefix $O/star_ > $RES/star.stdout 2>&1
echo "STAR: rc $? in $(( $(date +%s) - T0 )) s; $(grep -c "ALL DONE" $O/star_Log.out) ALL DONE" | tee -a $RES/summary.txt
grep -a -A1 "^##### Command Line" $O/star_Log.out | tail -1 | cut -c1-1500 | tee -a $RES/summary.txt
grep -a "Number of input reads\|Uniquely mapped reads %" $O/star_Log.final.out | tee -a $RES/summary.txt
cp -r $O/star_Solo.out $RES/; cp $O/star_Log.out $O/star_Log.final.out $RES/ 2>/dev/null
ls -la $SCR/solo | tee -a $RES/summary.txt
# ---- the driver
T0=$(date +%s)
OMP_NUM_THREADS=$T $SCR/post_test --solo-dir $SCR/solo --r1 $SCR/r1.fastq --genome-dir $G/star_2.7.11b --whitelist $PB/whitelists/$WL \
  --truth $RES/star_Solo.out --out $SCR/drv --mode both --umi-len $UMIL > $RES/post_test.log 2>&1
echo "post_test: rc $? in $(( $(date +%s) - T0 )) s" | tee -a $RES/summary.txt
grep -v "^  \[read" $RES/post_test.log | cut -c1-400 | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
