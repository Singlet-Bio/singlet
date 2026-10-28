#!/bin/bash
#SBATCH --job-name=w11-m5
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# M5 on one real sample (docs/gpu-engine/exploration.md §12.13). Everything with read sequences stays in job /tmp:
#  1. distinct cDNA sequences of the sample with their multiplicities (every k-th of the sorted distinct list, ~MAXD)
#  2. stardump maps them with the campaign's STAR parameters for the sample and writes the M5 truth (STARDUMP_M5)
#  3. m5: GPU port of STAR's mapping after the seed search vs the truth, byte for byte
# Results (small JSON + log) -> $W/jobs/w11/results/m5_<label>.json
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump; M5=$W/jobs/w11/bin/m5
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; OUT=$W/jobs/w11/results; mkdir -p $OUT
MAXD=${MAXD:-10000000}
# label | .1fq | genome | extra STAR args (the campaign's for this sample)
TASKS=(
"pbmc1k_v3|$PB/data/pbmc1k_v3/pbmc1k_v3.1fq|GRCh38-2024-A|"
"pbmc5p_v1|$PB/data/pbmc5p_v1/pbmc5p_v1.1fq|GRCh38-2024-A|--clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1"
"nuclei2k_v2|$PB/data/nuclei2k_v2/nuclei2k_v2.1fq|GRCm39-2024-A|"
)
IFS='|' read -r LABEL F GEN EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
LABEL=${LABEL}${TAG:-}
G=$REF/$GEN/star_2.7.11b
SCR=/tmp/$USER-m5-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "label=$LABEL node=$(hostname) job=$SLURM_JOB_ID $(date -Is) MAXD=$MAXD"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "stardump commit $(cat $SD.commit) m5 sha256 $(sha256sum $M5 | cut -c1-16)"
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
  --outSAMtype None --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ $EXTRA > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -3 $SCR/star.out; cp $SCR/star/Log.final.out $OUT/m5_${LABEL}_star_Log.final.out 2>/dev/null
du -sh $SCR/m5; cp $SCR/m5/t.params.txt $OUT/m5_${LABEL}_params.txt
echo "[t] stardump $(( $(date +%s) - T0 )) s"
# variants: "binary:args|binary:args" (default: m5 with no extra args); the first variant writes m5_<label>.json
IFS='|' read -ra VARS <<< "${M5RUNS:-m5:}"
for vi in "${!VARS[@]}"; do
  BIN=${VARS[$vi]%%:*}; ARGS=${VARS[$vi]#*:}; SUF=""; [ $vi -gt 0 ] && SUF="_v$vi"
  echo "=== variant $vi: $BIN $ARGS"
  apptainer exec --nv -B $W -B $SCR $IMG env OMP_NUM_THREADS=24 $W/jobs/w11/bin/$BIN --genome-dir $SCR/idx --truth $SCR/m5/t --reps 3     --json $OUT/m5_$LABEL$SUF.json $ARGS ${M5ARGS:-} 2>&1 | grep -v "^  seq\|^  star\|^  ours" | grep -v "^DIFF" | tail -14
  echo "m5 exit=${PIPESTATUS[0]}"
done
echo "[t] total $(( $(date +%s) - T0 )) s"
