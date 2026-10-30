#!/bin/bash
#SBATCH --job-name=w11-m5ncu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Nsight Compute source-level hot spots of M5's phase-3 kernel (lanes A, M, B) on the 2 M pbmc1k_v3 truth set (§12.12).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
SD=$W/jobs/w11/bin/stardump; M5=$W/jobs/w11/bin/m5; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; OUT=$W/jobs/w11/results
G=$REF/GRCh38-2024-A/star_2.7.11b; SCR=/tmp/$USER-ncu-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT; export LC_ALL=C
mkdir -p $SCR/idx && cp $G/Genome $G/SA $G/SAindex $G/*.txt $G/*.tab $SCR/idx/ 2>/dev/null
$PB/bin/singlet-w3 decode $PB/data/pbmc1k_v3/pbmc1k_v3.1fq --r2 /dev/stdout --threads 4 2>/dev/null | awk 'NR%4==2' | sort -S 60G --parallel=16 -T $SCR | uniq -c \
  | awk '(NR-1)%16==0 {printf "@d%d_%d\n%s\n+\n", NR, $1, $2; q=$2; gsub(/./,"I",q); print q}' > $SCR/d.fq
mkdir -p $SCR/m5 $SCR/star
SINGLET_STAR_SEQCACHE_MB=0 STARDUMP_M5=$SCR/m5/t $SD --runMode alignReads --genomeDir $SCR/idx --readFilesIn $SCR/d.fq --outSAMtype None \
  --outFilterScoreMin 10 --runThreadN 24 --genomeLoad NoSharedMemory --outFileNamePrefix $SCR/star/ > /dev/null 2>&1
NCU=$(apptainer exec $IMG bash -c 'ls /usr/local/cuda/bin/ncu /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | head -1'); echo "ncu=$NCU"
for L in A M B; do
  case $L in A) K=0;; M) K=1;; B) K=2;; esac
  # the k-th distinct k_p3 instantiation launched is lane L; profile its 2nd launch (first batch is warm-up of caches)
  apptainer exec --nv -B $W -B $SCR $IMG $NCU --target-processes all --kernel-name regex:k_p3 --launch-skip $(( K * 0 )) --launch-count 1 \
    --kernel-id ::regex:k_p3:$(( 1 + K * 0 )) --section SourceCounters --section WarpStateStats --section LaunchStats --section Occupancy \
    --import-source no -o $SCR/p3_$L -f env OMP_NUM_THREADS=24 $M5 --genome-dir $SCR/idx --truth $SCR/m5/t --reps 0 \
    --budget-a 1024 --budget-m 2048 --budget-b 2048 --batch-a 262144 --batch-m 262144 --batch-b 65536 > $SCR/ncu_$L.log 2>&1
  tail -3 $SCR/ncu_$L.log
  break
done
apptainer exec -B $SCR $IMG $NCU --import $SCR/p3_A.ncu-rep --page details --print-summary none 2>&1 | head -80 > $OUT/m5_ncu_details.txt
apptainer exec -B $SCR $IMG $NCU --import $SCR/p3_A.ncu-rep --page source --csv --print-source sass 2>/dev/null > $SCR/src.csv
wc -l $SCR/src.csv; head -3 $SCR/src.csv | cut -c1-300
cp $SCR/src.csv $OUT/m5_ncu_source_laneA.csv 2>/dev/null
cat $OUT/m5_ncu_details.txt | head -80
