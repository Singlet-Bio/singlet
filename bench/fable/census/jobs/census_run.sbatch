#!/bin/bash
#SBATCH --job-name=fable-census
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# E2 census (docs/gpu-engine/fable/E2-census.md) on one sample, the m5 harness recipe (bench/gpu/m5/jobs/m5_run.sbatch):
#  1. distinct cDNA sequences of the sample (every k-th of the sorted distinct list, MAXD = 2 M: the standard dump set)
#  2. stardump maps them with the campaign's STAR parameters and writes the M5 truth
#  3. m5 (default build) and m5c (-DM5_CENSUS, --census) with the engine's lane settings, both compared byte for byte
#  4. census_agg.py -> $OUT/results/census_<label>.{json,md}; raw records (no sequences) -> $OUT/results/raw/
# Read sequences stay in job /tmp.
# Single task outside an array (debug-gpu, 8 CPUs): sbatch --export=ALL,TASK=2 with the partition options on the command line.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump
IMG=$W/images/cuda-13-devel.sif; OUT=$W/jobs/fable/census; BIN=$OUT/bin; mkdir -p $OUT/results/raw
PY=$W/venvs/grantfig/bin/python3
MAXD=${MAXD:-2000000}; TID=${SLURM_ARRAY_TASK_ID:-${TASK:-0}}; JID=${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}
NT=${SLURM_CPUS_PER_TASK:-24}; SORTMEM=$(( NT >= 24 ? 60 : 30 ))G
LANEARGS="--budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 262144 --batch-m 262144 --batch-b 65536"   # engine_gpu.cu
TASKS=(
"pbmc1k_v3|$PB/data/pbmc1k_v3/pbmc1k_v3.1fq|GRCh38-2024-A|"
"pbmc5p_v1|$PB/data/pbmc5p_v1/pbmc5p_v1.1fq|GRCh38-2024-A|--clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1"
"nuclei2k_v2|$PB/data/nuclei2k_v2/nuclei2k_v2.1fq|GRCm39-2024-A|"
)
IFS='|' read -r LABEL F GEN EXTRA <<< "${TASKS[$TID]}"
LABEL=${LABEL}${TAG:-}
G=$REF/$GEN/star_2.7.11b
SCR=/tmp/$USER-census-$SLURM_JOB_ID-$TID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "label=$LABEL node=$(hostname) job=${JID}_$TID threads=$NT $(date -Is) MAXD=$MAXD"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "stardump commit $(cat $SD.commit); m5/m5c commit $(cat $BIN/commit)"
# the aggregation script of the branch tip
BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/census:refs/fable/census
echo "aggregation from $(git -C $BARE rev-parse refs/fable/census)"
git -C $BARE archive refs/fable/census bench/fable/census | tar -x -C $SCR
T0=$(date +%s)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
echo "[t] index copy $(( $(date +%s) - T0 )) s"
export LC_ALL=C
$DEC decode "$F" --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' | sort -S $SORTMEM --parallel=$(( NT < 16 ? NT : 16 )) -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
NS=$(( $(wc -l < $SCR/distinct.fq) / 4 ))
echo "reads=$NR distinct=$ND every_k=$K selected=$NS"; rm -f $SCR/distinct.txt
echo "[t] distinct $(( $(date +%s) - T0 )) s"
mkdir -p $SCR/m5 $SCR/star $SCR/cen
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN $NT --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ $EXTRA > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -3 $SCR/star.out
rm -f $SCR/distinct.fq
echo "[t] stardump $(( $(date +%s) - T0 )) s"
for B in m5 m5c; do
  CEN=""; [ $B = m5c ] && CEN="--census $SCR/cen/t"
  echo "=== $B $LANEARGS $CEN"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=$NT $BIN/$B --genome-dir $SCR/idx --truth $SCR/m5/t --reps 1 \
    --json $OUT/results/m5_${LABEL}_$B.json $LANEARGS $CEN 2>&1 | grep -v "^  seq\|^  star\|^  ours\|^DIFF" | tail -40
  echo "$B exit=${PIPESTATUS[0]}"
  echo "[t] $B $(( $(date +%s) - T0 )) s"
done
ls -la $SCR/cen
cp $SCR/cen/t.meta.json $OUT/results/raw/census_${LABEL}.meta.json
$PY $SCR/bench/fable/census/census_agg.py $SCR/cen/t $OUT/results/census_${LABEL}.json --label $LABEL > $OUT/results/census_${LABEL}.md 2> $OUT/results/census_${LABEL}.err
echo "agg exit=$?"; tail -5 $OUT/results/census_${LABEL}.err
echo "[t] agg $(( $(date +%s) - T0 )) s"
gzip -1 -c $SCR/cen/t.reads.bin > $OUT/results/raw/census_${LABEL}.reads.bin.gz; gzip -1 -c $SCR/cen/t.wins.bin > $OUT/results/raw/census_${LABEL}.wins.bin.gz
ls -la $OUT/results/raw | grep $LABEL
echo "[t] total $(( $(date +%s) - T0 )) s"
