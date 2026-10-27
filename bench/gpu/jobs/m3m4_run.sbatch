#!/bin/bash
#SBATCH --job-name=w11-m34
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=02:30:00
#SBATCH --array=0-2%2
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# M3 + M4 on one real sample (docs/gpu-engine/exploration.md §9). Everything with read sequences stays in job /tmp:
#  1. distinct cDNA sequences of the sample with their multiplicities (every k-th of the sorted distinct list, ~MAXD)
#  2. stardump (vendored STAR + seed dump) maps them with the sample's STAR alignment parameters
#  3. seedsearch: GPU port of STAR's seed search vs the dump (M3), U-array build + certificates vs the dump (M4)
# Results (small JSON + logs) -> $W/jobs/w11/results/m3m4_<label>.json
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
REF=/mnt/projects/debruinz_project/cellarium/reference
DEC=$PB/bin/singlet-w3; SD=$W/jobs/w11/bin/stardump; SS=$W/jobs/w11/bin/seedsearch
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; OUT=$W/jobs/w11/results; mkdir -p $OUT
MAXD=${MAXD:-10000000}
# label | .1fq | genome | extra STAR args (as the campaign run logs show for this sample)
TASKS=(
"pbmc1k_v3|$PB/data/pbmc1k_v3/pbmc1k_v3.1fq|GRCh38-2024-A|"
"pbmc5p_v1|$PB/data/pbmc5p_v1/pbmc5p_v1.1fq|GRCh38-2024-A|--clip3pAdapterSeq AAGCAGTGGTATCAACGCAGAGTACATGGG --clip3pAdapterMMp 0.1"
"nuclei2k_v2|$PB/data/nuclei2k_v2/nuclei2k_v2.1fq|GRCm39-2024-A|"
)
IFS='|' read -r LABEL F GEN EXTRA <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
G=$REF/$GEN/star_2.7.11b
SCR=/tmp/$USER-m34-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
echo "label=$LABEL node=$(hostname) job=$SLURM_JOB_ID $(date -Is)"; nvidia-smi --query-gpu=name,driver_version,clocks.max.sm --format=csv,noheader
echo "stardump sha256 $(sha256sum $SD | cut -c1-16) commit $(cat $SD.commit)"
T0=$(date +%s)
# index to local NVMe (both STAR and seedsearch read it)
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null; ls $SCR/idx | tr '\n' ' '; echo
echo "[t] index copy $(( $(date +%s) - T0 )) s"
# 1. distinct sequences with counts (sorted), then every k-th
export LC_ALL=C
$DEC decode "$F" --r2 /dev/stdout --threads 4 2>$SCR/decode.err | awk 'NR%4==2' | sort -S 60G --parallel=16 -T $SCR | uniq -c > $SCR/distinct.txt
ND=$(wc -l < $SCR/distinct.txt); NR=$(awk '{s+=$1} END {print s}' $SCR/distinct.txt)
K=$(( (ND + MAXD - 1) / MAXD )); [ $K -lt 1 ] && K=1
awk -v k=$K '(NR-1)%k==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' $SCR/distinct.txt > $SCR/distinct.fq
NS=$(( $(wc -l < $SCR/distinct.fq) / 4 ))
echo "reads=$NR distinct=$ND every_k=$K selected=$NS"; rm -f $SCR/distinct.txt; echo "DEC=$DEC F=[$F]"; head -4 $SCR/decode.err; grep -m1 "model name" /proc/cpuinfo
echo "[t] distinct $(( $(date +%s) - T0 )) s"
# 2. STAR dump (sequence cache off so every read is mapped)
mkdir -p $SCR/dump $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_PREFIX=$SCR/dump/seeds $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/distinct.fq \
  --outSAMtype None --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ $EXTRA > $SCR/star.out 2>&1
echo "stardump exit=$?"; tail -3 $SCR/star.out; cp $SCR/star/Log.final.out $OUT/m3m4_${LABEL}_star_Log.final.out 2>/dev/null
du -sh $SCR/dump; ls $SCR/dump | wc -l
echo "[t] stardump $(( $(date +%s) - T0 )) s"
# 3. GPU
apptainer exec --nv -B $W -B $SCR $IMG $SS --genome-dir $SCR/idx --dump $SCR/dump/seeds --reps 3 --json $OUT/m3m4_$LABEL.json
echo "seedsearch exit=$?"
echo "[t] total $(( $(date +%s) - T0 )) s"
