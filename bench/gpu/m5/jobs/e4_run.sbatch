#!/bin/bash
#SBATCH --job-name=e4-order
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=01:50:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E4 (docs/gpu-engine/fable/E4-order.md): phase-3 read order and batch scheduling variants of M5, every one checked byte for byte
# against STAR in the same job. Recipe of m5_run.sbatch: index copy, decode, distinct (MAXD = 2 M: every 16th distinct cDNA
# sequence of pbmc1k_v3 = 1,991,926), stardump truth, then each variant with --reps 3 --json. Sequences only in job /tmp.
# M5BIN: the binary (default: the latest build of e4_compile.sbatch). E4RUNS: "label:args;label:args;..." (default below).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump
O=$W/jobs/fable/order; M5=${M5BIN:-$O/bin/m5-$(cat $O/bin/latest.sha)}
IMG=$W/images/cuda-13-devel.sif; OUT=$O/results/$SLURM_JOB_ID; mkdir -p $OUT
MAXD=${MAXD:-2000000}
G=$REF/GRCh38-2024-A/star_2.7.11b; F=$PB/data/pbmc1k_v3/pbmc1k_v3.1fq
SCR=/tmp/$USER-e4-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) job=$SLURM_JOB_ID $(date -Is) MAXD=$MAXD"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
echo "m5=$M5 sha256 $(sha256sum $M5 | cut -c1-16); stardump commit $(cat $SD.commit)"
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
echo "[t] stardump $(( $(date +%s) - T0 )) s"
BASE="--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 262144 --batch-m 262144 --batch-b 65536"
DEFAULT="v0raw:;v0:--presort;v0t:--presort --tail;v1bin:--presort --bin --tail;v1desc:--presort --p3order cost-desc --tail"
DEFAULT="$DEFAULT;v1nw:--presort --p3order nw-desc --tail;v2locus:--presort --p3order locus --tail;v2anchor:--presort --p3order anchor --tail"
DEFAULT="$DEFAULT;v2nwlocus:--presort --p3order nwlocus --tail;v3q:--presort --queue --tail;v3qdesc:--presort --queue --p3order cost-desc --tail"
DEFAULT="$DEFAULT;v3qnwlocus:--presort --queue --p3order nwlocus --tail;v4fA:--presort --fused A --tail;v4fAMB:--presort --fused AMB --tail"
DEFAULT="$DEFAULT;v4fAMB1:--presort --fused AMB --fused-bps-a 1 --fused-bps-m 1 --fused-bps-b 1 --tail;v4big:--presort --batch-a 524288 --batch-m 393216"
# round 2 (ROUND=2): reference builds (base source cf92cdb9 with CUDA 13.3; the W11 binary of 2026-10-28, CUDA 12.4.1), per-lane
# order for lane M, bigger batches, lane B in one launch, the fused kernel at 4-5 resident blocks per SM (--fused-lb)
R2="v0raw:;base13:@m5-cf92cdb987;w11:@$W/jobs/w11/bin/m5;v0:--presort;v5m:--presort --p3order-m nw-desc"
R2="$R2;v4big:--presort --batch-a 524288 --batch-m 393216;v4bigB1:--presort --batch-a 524288 --batch-m 393216 --batch-b 131072"
R2="$R2;v5best:--presort --batch-a 524288 --batch-m 393216 --batch-b 131072 --p3order-m nw-desc;v5bestq:--presort --batch-a 524288 --batch-m 393216 --batch-b 131072 --p3order-m nw-desc --queue"
R2="$R2;v4fA5:--presort --fused A --fused-lb 5 --tail;v4fA4:--presort --fused A --fused-lb 4 --tail;v4fAMB5:--presort --fused AMB --fused-lb 5 --tail"
R2="$R2;v5mt:--presort --p3order-m nw-desc --tail;v0b:--presort"
[ "${ROUND:-1}" = 2 ] && DEFAULT=$R2
IFS=';' read -ra VARS <<< "${E4RUNS:-$DEFAULT}"
for v in "${VARS[@]}"; do
  LABEL=${v%%:*}; ARGS=${v#*:}; BIN=$M5
  if [[ "$ARGS" == @* ]]; then   # "@binary [args]": another build (relative names are in $O/bin)
    BIN=${ARGS%% *}; BIN=${BIN#@}; [[ "$BIN" != /* ]] && BIN=$O/bin/$BIN
    if [[ "$ARGS" == *" "* ]]; then ARGS=${ARGS#* }; else ARGS=""; fi
  fi
  echo "=== $LABEL: $ARGS (binary $BIN sha256 $(sha256sum $BIN | cut -c1-16))"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 $BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3 $BASE $ARGS \
    --label $LABEL --json $OUT/e4_$LABEL.json > $SCR/$LABEL.log 2>&1
  echo "m5 exit=$?"
  grep -E "^m5 E4-order build|^lane [AMB] \(|schedule:|run 3:|^M5:|unmapped by STAR|GPU lane A|heavy lane|assigned every|registers|DIFF" $SCR/$LABEL.log | grep -v "^DIFF" | head -30
  echo "DIFF lines: $(grep -c '^DIFF' $SCR/$LABEL.log)"
  grep -v "^DIFF\|^  seq\|^  star\|^  ours" $SCR/$LABEL.log > $OUT/e4_$LABEL.log
  echo "[t] $LABEL done $(( $(date +%s) - T0 )) s"
done
echo "[t] total $(( $(date +%s) - T0 )) s"
