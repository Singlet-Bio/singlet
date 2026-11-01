#!/bin/bash
#SBATCH --job-name=e5-run
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E5 (compact): M5 on pbmc1k_v3, the standard 2 M distinct-sequence set (MAXD=2000000: every 16th distinct cDNA sequence,
# 1,991,926 sequences), truth regenerated in the job (decode, distinct, stardump), then every variant in this one job:
#   M5RUNS="binary:args|binary:args"  (binaries in $OUT/bin, built by e5_compile.sbatch)
# Each variant: --reps 3 (median after a warm-up) --json, identity vs STAR byte for byte. Read sequences stay in job /tmp.
# Host threads follow --cpus-per-task; on debug-gpu (8 CPUs, 29 min) submit with
#   --partition=debug-gpu --cpus-per-task=8 --mem=100G --time=00:29:00 --export=ALL,SORTMEM=30G,...
# Optional NCU="binary:args|..." profiles k_p3 of lane A (2nd launch) with Nsight Compute for each entry: NCUSET (default
# "--set full"); NCUMET adds metrics (default: L2 read/write sectors, DRAM bytes, warp/thread instructions, local sectors).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump
IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/compact; RES=$OUT/results; mkdir -p $RES
MAXD=${MAXD:-2000000}; TAG=${TAG:-e5}; NCPU=${SLURM_CPUS_PER_TASK:-24}; SORTMEM=${SORTMEM:-60G}
STD="--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536"
F=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq; G=$REF/GRCh38-2024-A/star_2.7.11b
SCR=/tmp/$USER-e5-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "tag=$TAG node=$(hostname) job=$SLURM_JOB_ID partition=$SLURM_JOB_PARTITION cpus=$NCPU $(date -Is) MAXD=$MAXD"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
echo "[t] index copy $(( $(date +%s) - T0 )) s"
export LC_ALL=C
$DEC decode "$F" --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' | sort -S $SORTMEM --parallel=$NCPU -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
NS=$(( $(wc -l < $SCR/distinct.fq) / 4 ))
echo "reads=$NR distinct=$ND every_k=$K selected=$NS"; rm -f $SCR/distinct.txt
echo "[t] distinct $(( $(date +%s) - T0 )) s"
mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN $NCPU --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -1 $SCR/star.out; rm -f $SCR/distinct.fq
echo "[t] stardump $(( $(date +%s) - T0 )) s"
IFS='|' read -ra VARS <<< "${M5RUNS:-m5:}"
for vi in "${!VARS[@]}"; do
  BIN=${VARS[$vi]%%:*}; ARGS=${VARS[$vi]#*:}; ARGS=${ARGS//@STD/$STD}
  L=$RES/${TAG}_v${vi}_$BIN
  echo "=== variant $vi: $BIN $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=$NCPU $OUT/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3 \
    --json $L.json $ARGS 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $L.log
  grep -E "sizeof Work|^lane A|lane A (run|thread|inside|cop)|^M5:|GPU lane A|lane A overflow|tr32" $L.log | grep -v "run 0:"
done
# Nsight Compute on lane A's phase-3 kernel (second k_p3 launch = lane A, batch 1 of the warm-up pass)
NCUSET=${NCUSET:---set full}
NCUMET=${NCUMET:-lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,gpu__time_duration.sum}
IFS='|' read -ra NV <<< "${NCU:-}"
for ni in "${!NV[@]}"; do
  BIN=${NV[$ni]%%:*}; ARGS=${NV[$ni]#*:}; ARGS=${ARGS//@STD/$STD}
  echo "=== ncu $ni: $BIN $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG /usr/local/cuda/bin/ncu $NCUSET --metrics $NCUMET --kernel-name regex:k_p3 --launch-skip 1 --launch-count 1 \
    -o $SCR/ncu_$ni -f env OMP_NUM_THREADS=$NCPU $OUT/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 0 $ARGS > $SCR/ncu_$ni.log 2>&1
  echo "ncu exit=$?"; tail -2 $SCR/ncu_$ni.log
  apptainer exec -B $SCR $IMG /usr/local/cuda/bin/ncu --import $SCR/ncu_$ni.ncu-rep --page details --print-units base > $RES/${TAG}_ncu${ni}_$BIN.txt 2>&1
  apptainer exec -B $SCR $IMG /usr/local/cuda/bin/ncu --import $SCR/ncu_$ni.ncu-rep --page raw --csv --print-units base > $RES/${TAG}_ncu${ni}_${BIN}_raw.csv 2>&1
  cp $SCR/ncu_$ni.ncu-rep $RES/${TAG}_ncu${ni}_$BIN.ncu-rep 2>/dev/null
  grep -E "Duration|Registers Per|Achieved Occupancy|Theoretical Occupancy|Memory Throughput|L1/TEX Hit|L2 Hit|DRAM Throughput|lts__t_sectors|dram__bytes|inst_executed|lsu_mem|gpu__time" $RES/${TAG}_ncu${ni}_$BIN.txt | head -40
done
echo "[t] total $(( $(date +%s) - T0 )) s"
