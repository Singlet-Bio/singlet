#!/bin/bash
#SBATCH --job-name=w11-pu-run
#SBATCH --account=debruinz
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# W11 pileup, one sample end to end inside ONE job (reads, BAM and records live only in this job's /tmp):
#   1. dev-build CPU engine (c4a1be38 + pileup_dump.patch) on the real .1fq with SINGLET_KEEP_STAR_BAM=1 -> truth + annotation dump
#   2. extract.cpp: BAM -> records (stand-in for M5)     3. pileup (GPU): annotation + pileup, compared with the CPU truth
#   sbatch --partition=debug-gpu|short-gpu --time=HH:MM:SS --export=ALL,SET=<pbmc2m|GSM4006843|GSM5411066|GSM6470208>[,T=24][,REPS=3] pu_run.sbatch
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; M=$W/jobs/w11/pileup; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; SET=${SET:-pbmc2m}; T=${T:-24}; REPS=${REPS:-3}
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
if [ "$SET" = pbmc2m ]; then F=$PB/data/pbmc2m/pbmc2m.1fq; else D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET; F=$D/sample.1fq; fi
R=$M/results; mkdir -p $R $M/bin
SCR=/tmp/$USER-pu-$SLURM_JOB_ID; mkdir -p $SCR/out $SCR/dump $SCR/rec; trap 'kill $MON 2>/dev/null; rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_PILEUP_DUMP=$SCR/dump SINGLET_KEEP_STAR_BAM=1
echo "node=$(hostname) job=$SLURM_JOB_ID set=$SET file=$F T=$T $(date -Is)"; sha256sum $M/bin/singlet-pudump | cut -c1-16; cat $M/bin/singlet-pudump.commit
nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,memory.total --format=csv
( pk=0; while true; do u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "${u:-0}" -gt $pk ] && { pk=$u; echo "gpu_mem_peak_mib=$pk" > $R/$SET.gpumem.txt; }; sleep 1; done ) &
MON=$!
cd $SCR; S0=$(date +%s)
/usr/bin/time -v -o $R/$SET.engine.time.txt $M/bin/singlet-pudump $F --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --bam-sort-mode disk --genome-shared off --parallel-pileup > $R/$SET.engine.log 2>&1 || echo "engine exit $?"
echo "engine wall $(( $(date +%s) - S0 )) s"; grep -E "pudump|Pileup:|Indexing|CB-tag|Strand|cross-worker|Directional|Molecule|barcodes|Loaded" $R/$SET.engine.log | head -30
BAM=$(ls $SCR/out/*Aligned.sortedByCoord.out.bam 2>/dev/null | head -1); ls -la $BAM $SCR/dump
[ -n "$BAM" ] || { echo "no BAM"; exit 1; }
S1=$(date +%s); $M/bin/extract $BAM $SCR/dump $SCR/rec --threads 8 2> $R/$SET.extract.log; echo "extract exit $? wall $(( $(date +%s) - S1 )) s"; cat $R/$SET.extract.log; ls -la $SCR/rec
rm -f $BAM ${BAM}.bai
apptainer exec --nv -B $W -B /tmp $IMG $M/bin/pileup --rec $SCR/rec --dump $SCR/dump --label $SET --json $R/$SET.json --reps $REPS 2> $R/$SET.gpu.log; echo "pileup exit $?"
cat $R/$SET.gpu.log | tail -30; cat $R/$SET.gpumem.txt 2>/dev/null
