#!/bin/bash
#SBATCH --job-name=w11-pack
#SBATCH --account=debruinz
#SBATCH --nice=10000
#SBATCH --partition=short-cpu
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 CPU baseline, step 3: several samples concurrently on ONE exclusive node (campaign command line, singlet-w3).
#   sbatch --export=ALL,TAG=mix6_t4,T=4,SETS="GSM4006843 GSM5411066 ..." run_pack.sbatch
# SETS may repeat a sample. Every instance has its own /tmp dir, log, /usr/bin/time and wall stamps; the job cgroup is the
# whole node so monitor.tsv is node CPU (core-s) and memory.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
T=${T:-4}; BIN=${BIN:-$PB/bin/singlet-w3}; TAG=${TAG:-pack}; A=($SETS)
REF=/mnt/projects/debruinz_project/cellarium/reference
O=$W/jobs/w11/cost/runs/$TAG; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-w11p-$SLURM_JOB_ID; mkdir -p $SCR
trap 'kill $MON 2>/dev/null || true; rm -rf $SCR; exit 0' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
{ echo "node=$(hostname) set_count=${#A[@]} T=$T tag=$TAG jobid=$SLURM_JOB_ID $(date -Is)"; sha256sum $BIN
  echo "cotenant_cpus=$(squeue -w $(hostname -s) -h -t R -o '%C' | awk -v j=$SLURM_JOB_ID '{s+=$1}END{print s-0}')"
  lscpu | grep -E '^(Model name|CPU\(s\)|Thread|Core|Socket)'; nproc; free -g | head -2; } | tee $O/node.txt
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
echo "memory.max=$(cat $CG/memory.max) memory.high=$(cat $CG/memory.high 2>/dev/null) shm_size=$(df -m /dev/shm | awk 'NR==2{print $2}')MB tmp_free=$(df -m /tmp | awk 'NR==2{print $4}')MB" | tee -a $O/node.txt
( echo -e "t\tmem_mb\tanon_mb\tcpu_s\tfile_mb\tshmem_mb\tdirty_mb\twriteback_mb\tshm_used_mb\tmem_psi_full\tio_psi_full"
  last=0; t00=$(date +%s)
  while true; do
    now=$(date +%s.%N); cpu=$(awk '$1=="usage_usec"{printf "%.2f", $2/1e6}' $CG/cpu.stat)
    st=$(awk '{m[$1]=$2}END{printf "%d\t%d\t%d\t%d", m["file"]/1048576, m["shmem"]/1048576, m["file_dirty"]/1048576, m["file_writeback"]/1048576}' $CG/memory.stat)
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" $now $(( $(cat $CG/memory.current) / 1048576 )) \
      $(( $(awk '$1=="anon"{print $2}' $CG/memory.stat) / 1048576 )) "$cpu" "$st" $(df -m /dev/shm | awk 'NR==2{print $3}') \
      "$(awk '/^full/{print $2}' $CG/memory.pressure 2>/dev/null)" "$(awk '/^full/{print $2}' $CG/io.pressure 2>/dev/null)"
    # watchdog (cost guard): after 10 min, < 100 core-s of progress in 4 min = stalled; cancel the job
    e=$(( $(date +%s) - t00 )); if [ $((e % 240)) -lt 5 ] && [ $e -gt 600 ]; then
      if awk -v a=$cpu -v b=$last 'BEGIN{exit !(a-b<100)}'; then echo "STALL at $(date -Is) cpu=$cpu last=$last" >> $O/stall.txt; scancel $SLURM_JOB_ID; fi
      last=$cpu
    fi
    sleep 5
  done ) > $O/monitor.tsv 2>/dev/null &
MON=$!
run_one() {  # k set
  local k=$1 SET=$2 G=$REF/GRCh38-2024-A
  case $SET in GSM5411066|nuclei*|neuron*) G=$REF/GRCm39-2024-A;; esac
  local D=$W/data/atlas-rerun/perf/$SET; F=$D/sample.1fq; case $SET in GSM*) ;; *) D=$PB/data/$SET; F=$D/$SET.1fq;; esac; [ -e $F ] || { echo "MISSING INPUT $F"; exit 3; }
  local I=$O/i${k}_$SET S=$SCR/i$k; mkdir -p $I $S/out
  local SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
  cd $S; local T0=$(date +%s.%N)
  /usr/bin/time -v -o $I/time.txt $BIN $F --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix $S/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off \
    "${SCREEN[@]}" > $I/run.log 2>&1 || echo "exit $?" >> $I/run.log
  echo "t0=$T0 t1=$(date +%s.%N)" > $I/wall.txt
  grep -b -E 'Loading screening indices|Loading host Bloom filter|EM screening' $I/run.log > $I/nonhost_offsets.txt || true
  $W/envs/pbench/bin/python $W/bam-stream/parse_run.py $I > $I/stages.json 2>/dev/null || true
  rm -rf $S
}
echo "t0=$(date +%s.%N)" > $O/pack_wall.txt
k=0; for s in "${A[@]}"; do run_one $k $s & k=$((k+1)); done
wait
echo "t1=$(date +%s.%N)" >> $O/pack_wall.txt
echo "memory_peak_mb=$(( $(cat $CG/memory.peak 2>/dev/null || echo 0) / 1048576 ))" > $O/mem.txt
cat $O/pack_wall.txt $O/mem.txt; exit 0
