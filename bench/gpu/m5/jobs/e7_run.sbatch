#!/bin/bash
#SBATCH --job-name=e7-run
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E7 (seeds): M5 on pbmc1k_v3, the standard 2 M distinct-sequence set (MAXD=2000000: every 16th distinct cDNA sequence,
# 1,991,926 sequences), STAR truth regenerated in the job (decode, distinct, stardump), then baseline and every variant in
# this one job, each byte for byte against STAR (--reps 3: median of 3 after a warm-up), then Nsight Compute on phase 1.
#   M5RUNS="bin:args|bin:args"     binaries in $OUT/bin (e7_compile.sbatch); @STD = the standard budgets and batches
#   NCU="bin;skip;count|..."       ncu --set full on the phase-1 kernels (regex k_p1): skip/count matching launches
#                                  (k_p1: 1;1 = lane A batch 1; split variants k_p1s+k_p1w: 2;2 = lane A batch 1)
# Read sequences stay in job /tmp. Results: $OUT/results/<TAG>_<job>_v<i>_<bin>.{log,json}, ncu in $OUT/results/ncu-<job>/.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump; IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/seeds; RES=$OUT/results; mkdir -p $RES
MAXD=${MAXD:-2000000}; TAG=${TAG:-e7}; NCPU=${SLURM_CPUS_PER_TASK:-8}; SORTMEM=${SORTMEM:-30G}
STD="--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536"
F=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq; G=$REF/GRCh38-2024-A/star_2.7.11b
SCR=/tmp/$USER-e7-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
TLIM=${TLIM:-1380}   # seconds after start: no new variant or ncu run starts later than this
echo "tag=$TAG node=$(hostname) job=$SLURM_JOB_ID cpus=$NCPU $(date -Is) commit=$(cat $OUT/bin/commit 2>/dev/null)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
for b in $OUT/bin/*; do [ -L $b ] && echo "  bin $(basename $b) -> $(readlink $b)"; done
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
echo "[t] index copy $(( $(date +%s) - T0 )) s"; export LC_ALL=C
$DEC decode "$F" --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' | sort -S $SORTMEM --parallel=$NCPU -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt); K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
echo "reads=$NR distinct=$ND every_k=$K selected=$(( $(wc -l < $SCR/distinct.fq) / 4 ))"; rm -f $SCR/distinct.txt
echo "[t] distinct $(( $(date +%s) - T0 )) s"; mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN $NCPU --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > $SCR/star.out 2>&1
echo "stardump exit=$?"; rm -f $SCR/distinct.fq; echo "[t] stardump $(( $(date +%s) - T0 )) s"
IFS='|' read -ra VARS <<< "${M5RUNS:-m5:@STD|flatq:@STD|flatqs8:@STD|m5p:@STD|m5p2:@STD|flatqp:@STD|flatqp2:@STD|flatqs8p:@STD|flatqs8p2:@STD|flatqs:@STD|flatqg7:@STD|flatqsg8:@STD|flatqw:@STD|flatqs8w:@STD|m5:@STD --p1probe|flatqs8p:@STD --p1probe|m5:@STD}"
for vi in "${!VARS[@]}"; do
  [ $(( $(date +%s) - T0 )) -gt $TLIM ] && { echo "[t] time limit: skipping the remaining variants"; break; }
  BIN=${VARS[$vi]%%:*}; ARGS=${VARS[$vi]#*:}; ARGS=${ARGS//@STD/$STD}; L=$RES/${TAG}_${SLURM_JOB_ID}_v${vi}_$BIN
  echo "=== variant $vi: $BIN $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=$NCPU $OUT/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3 --json $L.json $ARGS 2>&1 \
    > $L.log
  grep -E "sizeof Work|k_p1|p1probe|2-bit|^lane [AMB] |lane [AMB] run [1-3]|phases|thread cycles|^M5:|GPU lane|unmapped by|overflow why|heavy lane" $L.log | head -40
  grep -m 3 "^DIFF" $L.log
  echo "[t] variant $vi done $(( $(date +%s) - T0 )) s"
done
# Nsight Compute on the phase-1 kernels of lane A
NCUSET=${NCUSET:---set full}
NCUMET=${NCUMET:-lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,dram__bytes_read.sum,dram__bytes_write.sum,smsp__inst_executed.sum,smsp__thread_inst_executed.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,gpu__time_duration.sum}
ND=$RES/ncu-$SLURM_JOB_ID; mkdir -p $ND
IFS='|' read -ra NV <<< "${NCU:-flatq;2;2|flatqs8p;2;2|m5p;1;1}"
for ni in "${!NV[@]}"; do
  [ $(( $(date +%s) - T0 )) -gt $TLIM ] && { echo "[t] time limit: skipping the remaining ncu runs"; break; }
  IFS=';' read -r BIN SKIP CNT <<< "${NV[$ni]}"
  echo "=== ncu $ni: $BIN skip $SKIP count $CNT"
  apptainer exec --nv -B $W -B $SCR $IMG /usr/local/cuda/bin/ncu $NCUSET --metrics $NCUMET --kernel-name regex:k_p1 --launch-skip $SKIP --launch-count $CNT \
    -o $SCR/ncu_$ni -f env OMP_NUM_THREADS=$NCPU $OUT/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 0 $STD > $SCR/ncu_$ni.log 2>&1
  echo "ncu exit=$?"; tail -2 $SCR/ncu_$ni.log
  apptainer exec -B $SCR $IMG /usr/local/cuda/bin/ncu --import $SCR/ncu_$ni.ncu-rep --page details --print-units base > $ND/ncu_${ni}_$BIN.txt 2>&1
  apptainer exec -B $SCR $IMG /usr/local/cuda/bin/ncu --import $SCR/ncu_$ni.ncu-rep --page raw --csv --print-units base > $ND/ncu_${ni}_${BIN}_raw.csv 2>&1
  cp $SCR/ncu_$ni.ncu-rep $ND/ncu_${ni}_$BIN.ncu-rep 2>/dev/null
  grep -E "^  k_p1|Duration|Registers Per|Achieved Occupancy|Theoretical Occupancy|Memory Throughput|L1/TEX Hit|L2 Hit|DRAM Throughput|Active Threads|Warp Cycles Per Issued|Issue Slots Busy" $ND/ncu_${ni}_$BIN.txt | head -40
done
# stall mix and SIMT per kernel (E1's summariser, stdlib python)
git -C $W/repos/singlet-py.git archive refs/fable/seeds bench/fable/profile/ncu_summary.py 2>/dev/null | tar -x -C $SCR 2>/dev/null
if [ -f $SCR/bench/fable/profile/ncu_summary.py ] && ls $ND/*_raw.csv >/dev/null 2>&1; then
  python3 $SCR/bench/fable/profile/ncu_summary.py $ND 2>&1 | head -60
fi
echo "[t] total $(( $(date +%s) - T0 )) s"
