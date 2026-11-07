#!/bin/bash
#SBATCH --job-name=gn-large
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=370G
#SBATCH --time=03:55:00
#SBATCH --exclude=g050
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One run of the engine on the GSM7064234 reads (1.198 G pairs, staged .1fq of the pilot) or on its 100 M-read slice.
#   sbatch --export=ALL,BIN=<dir with singlet-gpu>,MODE=gpu|cpu,INPUT=full|slice,TAG=<name> bench/gpu_next/jobs/large.sbatch
# The 370 G memory limit is the drain worker's H100 class. The cgroup's memory and the GPU's utilization and memory are sampled every second.
# Outputs go to node-local scratch. A summary and the small result files are kept in $W/jobs/gpu-next/results/large-<jobid>. No reads are kept.
set -uo pipefail
: "${BIN:?}" "${MODE:?}" "${INPUT:?}"; TAG=${TAG:-run}; T=${T:-16}
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
P=$W/pilot/GSM7064234; G=$REF/GRCh38-2024-A
case $INPUT in full) FQ=$P/stage_0ea48783/SRR23617320.1fq ;; slice) FQ=$P/cap1fq_0ea48783/SRR23617320.1fq ;; esac
RES=$W/jobs/gpu-next/results/large-$SLURM_JOB_ID-$TAG; mkdir -p $RES
SCR=/tmp/$USER-gnl-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'kill $(jobs -p) 2>/dev/null; rm -rf "${SCR:?}"' EXIT INT TERM
[ -e $BIN/../whitelists ] || ln -s $PB/whitelists $BIN/../whitelists
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
ENG=""; [ $MODE = gpu ] && { ENG="--engine gpu"; export SINGLET_GPU_PILEUP=on; }
echo "node=$(hostname) bin=$BIN commit=$(cat $BIN/commit-gpu 2>/dev/null) mode=$MODE input=$INPUT ($(stat -c %s $FQ) bytes) T=$T cgroup=$CG limit=$(cat $CG/memory.max 2>/dev/null) $(date -Is)" | tee $RES/summary.txt
( echo -e "t\tcg_mem_gb\tcpu_s\tgpu_util\tgpu_mem_mb"
  while true; do
    g=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits 2>/dev/null | tr -d ' ' | tr , '\t')
    printf "%s\t%s\t%s\t%s\n" $(date +%s) $(awk '{printf "%.1f", $1/1073741824}' $CG/memory.current) $(awk '$1=="usage_usec"{printf "%.1f", $2/1e6}' $CG/cpu.stat) "$g"; sleep 1
  done ) > $RES/monitor.tsv 2>/dev/null &
T0=$(date +%s.%N)
( cd $SCR && /usr/bin/time -v -o $RES/time.txt timeout -k 30 13500 $BIN/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz $ENG 2>&1 | while IFS= read -r l; do printf '%(%s)T %s\n' -1 "$l"; done > $RES/run.log.ts )
T1=$(date +%s.%N)
ES=$(awk -F': ' '/Exit status/{print $2}' $RES/time.txt)
RSS=$(awk -F': ' '/Maximum resident/{printf "%.1f", $2/1048576}' $RES/time.txt)
CGMAX=$(awk -F'\t' 'NR>1 && $2>m{m=$2} END{print m}' $RES/monitor.tsv)
GMAX=$(awk -F'\t' 'NR>1 && $5>m{m=$5} END{printf "%.1f", m/1024}' $RES/monitor.tsv)
GUT=$(awk -F'\t' 'NR>1 && $4!=""{s+=$4;n++} END{printf "%.1f", s/n}' $RES/monitor.tsv)
echo "RESULT $TAG: exit $ES wall $(echo "$T1 - $T0" | bc | cut -c1-8) s, maxRSS(time) $RSS GB, cgroup peak $CGMAX GB, GPU mem peak $GMAX GB, GPU util mean $GUT %" | tee -a $RES/summary.txt
grep -a -e "\[engine_gpu\]" -e "ENGINE_GPU: device" -e "\[gpu_pileup\]" -e "1fq-ingest" -e "fallback" -e "oom\|OOM\|killed" $RES/run.log.ts | cut -c1-300 | head -60 > $RES/engine_lines.txt
grep -a "ALL DONE\|Number of input reads\|Uniquely mapped reads %" $SCR/out/*Log.final.out 2>/dev/null | head -3 >> $RES/summary.txt
cp $SCR/out/*Log.final.out $SCR/out/summary.json $SCR/out/read_stats.tsv $SCR/out/pileup_stats.json $RES/ 2>/dev/null
if [ "${KEEPOUT:-0}" = 1 ]; then rsync -a --exclude='*.bam' --exclude='*.fq*' --exclude='*Unmapped*' --exclude='_STARtmp*' --exclude='*.1fq' $SCR/out/ $W/jobs/gpu-next/results/out-$SLURM_JOB_ID-$TAG/; echo "outputs: $W/jobs/gpu-next/results/out-$SLURM_JOB_ID-$TAG" | tee -a $RES/summary.txt; fi
