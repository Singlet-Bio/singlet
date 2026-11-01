#!/bin/bash
#SBATCH --job-name=e3-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E3 (warp-window): baseline lanes and the warp-stitched lanes M/B in the same job, on the standard 2 M distinct-read set of
# pbmc1k_v3 (every 16th distinct cDNA sequence), byte for byte against STAR (stardump M5 truth regenerated here, in job /tmp).
#   RUNS="label:extra m5 args|label:args|..."   (default: base, warp, warp+sort, warp+sort+cfg1; host emulation on "warp")
#   BIN=path to m5 (default $W/jobs/fable/warp-window/bin/m5e3); NCUS: optional Nsight Compute profiles (see the end)
# Results: $W/jobs/fable/warp-window/results/e3_<jobid>_<label>.json and .json.e3.json; the log keeps no read sequence.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump
E=$W/jobs/fable/warp-window; BIN=${BIN:-$E/bin/m5e3}; IMG=$W/images/cuda-13-devel.sif; OUT=$E/results; mkdir -p $OUT
G=$REF/GRCh38-2024-A/star_2.7.11b; F=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq
MAXD=${MAXD:-2000000}
BASE=${BASE:-"--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536"}
RUNS=${RUNS:-"base:|warp:--warp-m --warp-b --host-emul|warpsort:--warp-m --warp-b --wsort|warpsort1:--warp-m --warp-b --wsort --wb-cfg 1"}
SCR=/tmp/$USER-e3-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID $(date -Is) MAXD=$MAXD"; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "m5 commit $(cat $BIN.commit 2>/dev/null) sha256 $(sha256sum $BIN | cut -c1-16); stardump commit $(cat $SD.commit)"
cp $BIN $SCR/m5e3   # (not $SCR/m5: that is the truth directory)
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
echo "[t] index copy $(( $(date +%s) - T0 )) s"
export LC_ALL=C
$DEC decode "$F" --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' | sort -S 60G --parallel=16 -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
NS=$(( $(wc -l < $SCR/distinct.fq) / 4 ))
echo "reads=$NR distinct=$ND every_k=$K selected=$NS"; rm -f $SCR/distinct.txt
echo "[t] distinct $(( $(date +%s) - T0 )) s"
mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -2 $SCR/star.out; rm -f $SCR/distinct.fq
[ -f $SCR/m5/t.params.txt ] || { echo "no STAR truth: stop"; exit 1; }
echo "[t] stardump $(( $(date +%s) - T0 )) s"
IFS='|' read -ra VARS <<< "$RUNS"
for v in "${VARS[@]}"; do
  LABEL=${v%%:*}; ARGS=${v#*:}
  echo "=== $LABEL: $BASE $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 $SCR/m5e3 --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3 \
    --json $OUT/e3_${SLURM_JOB_ID}_$LABEL.json --tag $LABEL $BASE $ARGS > $SCR/m5_$LABEL.log 2>&1
  echo "m5 exit=$?"
  # no read sequences in the log: drop the "  seq" lines of DIFF records (blob values and indices are kept, at most 12 records)
  grep -v "^  seq" $SCR/m5_$LABEL.log | awk '/^DIFF/ {nd++} nd <= 12 || !/^(DIFF|  star|  ours)/' | grep -v "^  lane .* run [0-9]" | tail -60
  grep "run [0-9]" $SCR/m5_$LABEL.log | tr '\n' ';' | cut -c1-900; echo
done
# optional Nsight Compute: NCUS="label~mangled-name regex~m5 args|..." (one launch each, --set full; e.g. lane M's warp kernel:
# "wm~k_p3wI4CapsILi160ELi4ELi32ELi64E~--warp-m"; baseline lane M: "bm~k_p3I4CapsILi160ELi4ELi32ELi64E~"). Reports: $OUT/ncu-<job>/
NCUS=${NCUS:-}; NCU=/usr/local/cuda/bin/ncu
if [ -n "$NCUS" ]; then
  ND=$OUT/ncu-$SLURM_JOB_ID; mkdir -p $ND
  IFS='|' read -ra NV <<< "$NCUS"
  for v in "${NV[@]}"; do
    L=${v%%~*}; r=${v#*~}; RX=${r%%~*}; ARGS=${r#*~}; t=$(date +%s)
    echo "=== ncu $L: regex $RX -- $BASE $ARGS"
    timeout 1200 apptainer exec --nv -B $W -B $SCR $IMG $NCU --set full --import-source yes --clock-control none --kernel-name-base mangled \
      --kernel-name "regex:$RX" --launch-count 1 -o $ND/ncu_$L -f env OMP_NUM_THREADS=24 $SCR/m5e3 --genome-dir $SCR/idx --truth $SCR/m5/t \
      --reps 0 $BASE $ARGS 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $ND/ncu_$L.log
    echo "exit=${PIPESTATUS[0]} wall=$(( $(date +%s) - t )) s"; grep -E "==PROF==|==ERROR==|==WARNING==" $ND/ncu_$L.log | cut -c1-200 | head -5
    [ -f $ND/ncu_$L.ncu-rep ] || continue
    apptainer exec -B $W $IMG $NCU --import $ND/ncu_$L.ncu-rep --page details --csv > $ND/ncu_${L}_details.csv 2>&1
    grep -E '"(Duration|Registers Per Thread|Achieved Occupancy|Theoretical Occupancy|Achieved Active Warps Per SM|Block Limit Registers|Block Limit Shared Mem|Block Limit Warps|Local Memory Spilling Requests|Compute \(SM\) Throughput|Memory Throughput|DRAM Throughput|L1/TEX Hit Rate|L2 Hit Rate|Warp Cycles Per Issued Instruction|Issued Warp Per Scheduler|No Eligible|Avg\. Active Threads Per Warp|Avg\. Not Predicated Off Threads Per Warp|Branch Efficiency|Dynamic Shared Memory Per Block|Threads|Grid Size|Block Size)"' \
      $ND/ncu_${L}_details.csv | awk -F'","' '{printf "  %-45s %-10s %s\n", $(NF-2), $(NF-1), $NF}' | tr -d '"' | head -40
  done
fi
echo "[t] total $(( $(date +%s) - T0 )) s"
