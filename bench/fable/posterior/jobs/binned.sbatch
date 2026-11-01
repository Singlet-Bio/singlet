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
# E15 (docs/gpu-engine/fable/E15-posterior.md): the per-read CB posterior against STAR on binned NovaSeq barcode qualities. The pipeline's
# .1fq never carries the barcode read's qualities (STAR sees 'F' for every base), so the binned case is built here: the first NREADS reads of
# the sample's original L001 FASTQs (uncompressed, in /tmp) through `singlet --reads R1 R2 --engine cpu` (STAR reads them as they are) with
# SINGLET_SOLO_DUMP, then bench/fable/posterior/post_test.cu (compiled here from BRANCH) runs E9's library on STAR's per-read Solo inputs with
# the reads' CB qualities (perread) and with one 'F' quality (const, the control) and compares per read and per file with STAR's star_Solo.out.
# Read-level data (FASTQs, dump) stay in /tmp; results/binned-<job>/ keeps STAR's star_Solo.out, its logs, the driver's log.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
BINDIR=${BINDIR:-$W/jobs/fable/posterior/bin}; T=${T:-8}; NREADS=${NREADS:-6000000}; BR=${BRANCH:-gpu/fable/posterior}
s=${SAMPLE:-pbmc1k_v3}; WL=${WL:-3M-february-2018.txt}; UMIL=${UMIL:-12}; G=${G:-$REF/GRCh38-2024-A}
# the legacy FASTQ path wants a cell list for the pileup (--barcodes; STAR's Solo does not read it): E11's CPU-engine cells of pbmc1k_v3
BC=${BC:-$W/jobs/fable/solowire/results/gate-796377/cpu/out/star_Solo.out/Gene/filtered/barcodes.tsv}
RES=$W/jobs/fable/posterior/results/binned-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e15d-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/hts $SCR/solo $SCR/keep
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$ENV/bin:$PATH
echo "node=$(hostname) job=$SLURM_JOB_ID bin=$BINDIR commit=$(cat $BINDIR/commit-gpu) sample=$s NREADS=$NREADS T=$T $(date -Is)" | tee $RES/summary.txt
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
zcat $R1 | head -n $((4 * NREADS)) > $SCR/r1.fastq & zcat $R2 | head -n $((4 * NREADS)) > $SCR/r2.fastq & wait
echo "reads: $R1 + R2, $(( $(wc -l < $SCR/r1.fastq) / 4 )) / $(( $(wc -l < $SCR/r2.fastq) / 4 )) records in $(( $(date +%s) - T0 )) s" | tee -a $RES/summary.txt
# ---- STAR (CPU engine) with the per-read Solo dump
O=$SCR/run/out; mkdir -p $O
( while [ ! -e $O/star_Solo.out/Gene/CellReads.stats ]; do sleep 0.2; done; ln $O/star_Solo.out/Gene/CellReads.stats $SCR/keep/CellReads.stats ) &
WATCH=$!
T0=$(date +%s)
( cd $SCR/run && SINGLET_SOLO_DUMP=$SCR/solo timeout -k 30 1200 $BINDIR/singlet-gpu --reads $SCR/r1.fastq $SCR/r2.fastq --whitelist $PB/whitelists/$WL --barcodes $BC \
    --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $O/ --threads $T --pipeline --output-format 1pz \
    --engine cpu > $RES/run.log 2>&1 ) &
P=$!
while kill -0 $P 2>/dev/null; do grep -q "ALL DONE" $O/star_Log.out 2>/dev/null && break; sleep 2; done
sleep 10; pkill -TERM -f "out-prefix $O/" 2>/dev/null; sleep 5; pkill -KILL -f "out-prefix $O/" 2>/dev/null; wait $P 2>/dev/null; kill $WATCH 2>/dev/null
echo "STAR: $(( $(date +%s) - T0 )) s; $(grep -c "ALL DONE" $O/star_Log.out) ALL DONE" | tee -a $RES/summary.txt
grep -a -A1 "^##### Command Line" $O/star_Log.out | tail -1 | tr ' ' '\n' | grep -a -A2 "^--solo\|^--clip\|^--readFilesIn\|^--readFilesCommand" | grep -v "^--$" | paste -sd' ' | cut -c1-1500 | tee -a $RES/summary.txt
cp -r $O/star_Solo.out $RES/; cp $SCR/keep/CellReads.stats $RES/star_Solo.out/Gene/ 2>/dev/null; cp $O/star_Log.out $O/star_Log.final.out $RES/ 2>/dev/null
ls -la $SCR/solo | tee -a $RES/summary.txt
# ---- the driver
T0=$(date +%s)
OMP_NUM_THREADS=$T $SCR/post_test --solo-dir $SCR/solo --r1 $SCR/r1.fastq --genome-dir $G/star_2.7.11b --whitelist $PB/whitelists/$WL \
  --truth $RES/star_Solo.out --out $SCR/drv --mode both --umi-len $UMIL > $RES/post_test.log 2>&1
echo "post_test: rc $? in $(( $(date +%s) - T0 )) s" | tee -a $RES/summary.txt
grep -v "^  \[read" $RES/post_test.log | cut -c1-400 | tee -a $RES/summary.txt
echo done $(date -Is) | tee -a $RES/summary.txt
