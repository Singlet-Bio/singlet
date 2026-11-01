#!/bin/bash
#SBATCH --job-name=e6-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E6 run: one `singlet --engine gpu` run of pbmc1k_v3 (SINGLET_GPU_PILEUP=on, the stream path) with SINGLET_ENGINE_GPU_DUMP (mapDistinct's
# records, engine_gpu.cu) and the hand-off it writes kept by hard links (the pileup unlinks it), then the records driver against them.
# Everything with read sequences stays in job /tmp. After the first driver run the job serves further driver runs on the same data: write a
# commit (or "stop") into $W/jobs/fable/records/trigger and it fetches gpu/fable/records, compiles and runs the driver again (idle limit IDLE s).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
E6=$W/jobs/fable/records; BIN=$E6/bin/singlet-gpu; IMG=$W/images/cuda-13-devel.sif; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
T=${T:-18}; IDLE=${IDLE:-1200}; DARGS=${DARGS:-}
RES=$E6/results/run-$SLURM_JOB_ID; mkdir -p $RES
SCR=/tmp/$USER-e6r-$SLURM_JOB_ID; mkdir -p $SCR/out $SCR/keep $SCR/hts
trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT INT TERM
ln -s $ENV/include/htslib $SCR/hts/htslib
# singlet resolves the barcode whitelist at <binary dir>/../whitelists; without it the run silently takes the sorted-BAM path
[ -e $E6/whitelists ] || ln -s $PB/whitelists $E6/whitelists
echo "node=$(hostname) job=$SLURM_JOB_ID engine commit=$(cat $E6/bin/commit) T=$T $(date -Is)" | tee $RES/summary.txt
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee -a $RES/summary.txt
df -h /tmp | tail -1 | tee -a $RES/summary.txt
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$ENV/bin:$PATH
FQ=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq
# the hand-off files, hard-linked as soon as STAR's prefill creates them (same inode: the later writes land in the link too)
( for f in .gpu_handoff.bin .gpu_handoff.bin.reads .gpu_handoff.bin.reads.q; do
    p=""
    for k in $(seq 1 30000); do p=$(find $SCR/out -maxdepth 4 -name "$f" 2>/dev/null | head -1); [ -n "$p" ] && break; sleep 0.2; done
    [ -n "$p" ] && ln "$p" $SCR/keep/$f && echo "[keep] $f <- $p at $(date +%T)" | tee -a $RES/summary.txt
  done ) &
T0=$(date +%s)
( cd $SCR/out && SINGLET_ENGINE_GPU_DUMP=$SCR/keep/e6dump.bin SINGLET_GPU_PILEUP=on /usr/bin/time -v -o $RES/time.txt timeout -k 30 3600 $BIN $FQ \
    --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz \
    --engine gpu > $RES/engine.log 2>&1 )
echo "engine exit=$? wall $(( $(date +%s) - T0 )) s, maxRSS $(awk -F': ' '/Maximum resident/{printf "%.1f GB", $2/1048576}' $RES/time.txt)" | tee -a $RES/summary.txt
grep -a -e "unsorted STAR stream" -e "streaming from STAR pipe" -e "gpu_pileup\] on" -e "gpu_pileup\] hand-off" -e "stream CB" $RES/engine.log | cut -c1-200 | tee -a $RES/summary.txt
LOGOUT=$(find $SCR/out -name "*Log.out" | head -1)
grep -ah -e "ENGINE_GPU" -e "engine_gpu:" $LOGOUT | cut -c1-260 > $RES/starlog_engine.txt; cat $RES/starlog_engine.txt >> $RES/summary.txt
grep -aE "^(outSAMmapqUnique|outSAMflagAND|outSAMflagOR|outSAMmultNmax|outFilterMultimapNmax|outFilterScoreMin|outSAMprimaryFlag|clip3pAdapterSeq|soloType) " $LOGOUT | sort -u | tee -a $RES/summary.txt
grep -a -A1 "Final effective command line" $LOGOUT | cut -c1-600 | tee -a $RES/summary.txt
BC=$(find $SCR/out -name auto_barcodes.tsv | head -1); [ -n "$BC" ] && cp $BC $SCR/keep/auto_barcodes.tsv
echo "barcodes: $BC ($(wc -l < $SCR/keep/auto_barcodes.tsv 2>/dev/null) lines)" | tee -a $RES/summary.txt
ls -la $SCR/keep | tee -a $RES/summary.txt
find $SCR/out -maxdepth 2 -name "*barcode*" | head -20 >> $RES/summary.txt
run_driver() {
  git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/records:refs/fable/records
  local SHA=$(git -C $BARE rev-parse --short=8 refs/fable/records)
  rm -rf $SCR/src; mkdir -p $SCR/src
  git -C $BARE archive refs/fable/records src/engine_gpu include/singlet bench/fable/records | tar -x -C $SCR/src
  local S=$SCR/src
  apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -I$S/src/engine_gpu -I$S/include -I$SCR/hts \
    -o $SCR/records_test $S/bench/fable/records/records_test.cu $S/src/engine_gpu/records_gpu.cu > $RES/compile-$SHA.log 2>&1 \
    || { echo "driver $SHA: compile FAILED" | tee -a $RES/summary.txt; tail -30 $RES/compile-$SHA.log; LAST=$SHA; return; }
  local Q=""; [ -e $SCR/keep/.gpu_handoff.bin.reads.q ] && Q="--q $SCR/keep/.gpu_handoff.bin.reads.q"
  local T1=$(date +%s)
  apptainer exec --nv -B $W -B $SCR $IMG $SCR/records_test --dump $SCR/keep/e6dump.bin --handoff $SCR/keep/.gpu_handoff.bin \
    --reads $SCR/keep/.gpu_handoff.bin.reads $Q --barcodes $SCR/keep/auto_barcodes.tsv --reps 5 --host-threads $T --json $RES/e6-$SHA.json $DARGS \
    > $RES/driver-$SHA.log 2>&1
  echo "driver $SHA exit=$? in $(( $(date +%s) - T1 )) s" | tee -a $RES/summary.txt
  grep -a -E "HOP|RESULT|GPU|HOST|subset|first|differ|readRecord|dump:|hand-off:|params" $RES/driver-$SHA.log | cut -c1-400 | head -40 | tee -a $RES/summary.txt
  LAST=$SHA
}
LAST=""
run_driver
TRIG=$E6/trigger; IDLE0=$(date +%s)
while [ $(( $(date +%s) - IDLE0 )) -lt $IDLE ] && [ $(( $(date +%s) - T0 )) -lt 6600 ]; do
  sleep 20
  t=$(cat $TRIG 2>/dev/null | tr -d '[:space:]')
  [ -z "$t" ] && continue
  [ "$t" = stop ] && { echo "stop" | tee -a $RES/summary.txt; : > $TRIG; break; }
  case $LAST in $t*) continue ;; esac
  echo "trigger $t" | tee -a $RES/summary.txt; : > $TRIG
  run_driver; IDLE0=$(date +%s)
done
echo "done $(date -Is)" | tee -a $RES/summary.txt
