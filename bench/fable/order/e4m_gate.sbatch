#!/bin/bash
#SBATCH --job-name=e4m-gate
#SBATCH --account=debruinz
#SBATCH --partition=debug-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E4 after the merge with gpu/fable-engine (docs/gpu-engine/fable/E4-order.md, "Merge and M5_NOCLK"); template:
# bench/fable/engine/fable_gate.sbatch. The merged m5 builds of e4m_cc.sbatch ($W/jobs/fable/order/merge/bin) on the standard
# 2 M distinct set of pbmc1k_v3 (truth regenerated here): identity byte for byte and lane timings (median of 3) per variant.
# M5RUNS: "binary:args|..." (@STD = the gate's standard budgets and batches).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump; IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/order/merge; RES=$OUT/results/$SLURM_JOB_ID; mkdir -p $RES
MAXD=${MAXD:-2000000}; NCPU=${SLURM_CPUS_PER_TASK:-8}; SORTMEM=${SORTMEM:-30G}
STD="--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 524288 --batch-m 393216 --batch-b 65536"
BEST="--best-order --p3-maxreg-a 64"
DEFRUNS="m5:@STD|m5n:@STD|m5n:@STD $BEST|m5n:@STD $BEST --batch-b 131072|m5:@STD $BEST --batch-b 131072|m5n:@STD --e4-best|m5:@STD|m5n:@STD"
F=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq; G=$REF/GRCh38-2024-A/star_2.7.11b
SCR=/tmp/$USER-e4mgate-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID cpus=$NCPU $(date -Is) commit=$(cat $OUT/bin/commit 2>/dev/null)"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
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
IFS='|' read -ra VARS <<< "${M5RUNS:-$DEFRUNS}"
for vi in "${!VARS[@]}"; do
  BIN=${VARS[$vi]%%:*}; ARGS=${VARS[$vi]#*:}; ARGS=${ARGS//@STD/$STD}; L=$RES/v${vi}_$BIN
  echo "=== variant $vi: $BIN $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=$NCPU $OUT/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3 --json $L.json \
    --label "v${vi}_$BIN" $ARGS 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" > $L.log
  grep -E "E4-order build|identical|^lane [AMB] |lane [AMB] run [1-3]|thread cycles|^M5:|total|distinct/s|GPU" $L.log | grep -v "run 0:" | head -30
  echo "[t] v$vi done $(( $(date +%s) - T0 )) s"
done
echo "[t] total $(( $(date +%s) - T0 )) s"
