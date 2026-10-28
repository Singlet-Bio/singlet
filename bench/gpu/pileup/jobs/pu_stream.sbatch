#!/bin/bash
#SBATCH --job-name=w11-pu-stream
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=240G
#SBATCH --time=01:30:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU pileup vs the CPU engine's unsorted-stream path under docs/research-beta/order-free-rules.md, one sample, everything in job /tmp:
#   1. engine (branch gpu/engine-gpu-stream: PR #66 + perf/bam-stream + the W11 debug hooks) on the real .1fq, stream mode, with
#      SINGLET_PILEUP_DUMP (what the callers receive + the annotation) and SINGLET_STREAM_TEE (a copy of STAR's unsorted record stream)
#   2. extract --cb-from-cr: the stream's records -> GPU records      3. pileup (GPU, R1 SNP consensus) compared with the CPU dump
#   sbatch --export=ALL,SET=pbmc1k_v3[,ENGINE=gpu][,T=24] pu_stream.sbatch
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; M=$W/jobs/w11/pileup; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A; SET=${SET:-pbmc1k_v3}; T=${T:-24}; REPS=${REPS:-1}; ENGINE=${ENGINE:-cpu}
BIN=${BIN:-$W/jobs/w11/engine/bin14/singlet-gpu}
[ "$SET" = GSM5411066 ] && G=$REF/GRCm39-2024-A
F=$PB/data/$SET/$SET.1fq; [ -e $F ] || { D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET; F=$D/sample.1fq; }
R=$M/results/stream; mkdir -p $R
SCR=/tmp/$USER-pus-$SLURM_JOB_ID; mkdir -p $SCR/out $SCR/dump $SCR/rec
tree() { local c; for c in $(pgrep -P $1); do tree $c; echo $c; done; }   # every descendant, deepest first
cleanup() { kill -TERM $(tree $$) 2>/dev/null; sleep 3; kill -KILL $(tree $$) 2>/dev/null; rm -rf $SCR; }
trap cleanup EXIT INT TERM
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF SINGLET_PILEUP_DUMP=$SCR/dump SINGLET_STREAM_TEE=$SCR/stream.bam
echo "node=$(hostname) job=$SLURM_JOB_ID set=$SET engine=$ENGINE file=$F T=$T bin=$(sha256sum $BIN | cut -c1-16) commit=$(cat $(dirname $BIN)/commit) $(date -Is)"
cd $SCR; S0=$(date +%s)
timeout -k 30 3000 /usr/bin/time -v -o $R/$SET.engine.time.txt $BIN $F --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
  --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz --engine $ENGINE > $R/$SET.engine.log 2>&1 || echo "engine exit $?"
echo "engine wall $(( $(date +%s) - S0 )) s"; grep -aE "gpudump|BAM:|Pileup:|snp_molecules|strand|barcodes" $R/$SET.engine.log | head -20
grep -o '"snp_molecules[a-z_]*": *[0-9]*' $SCR/out/pileup_stats.json 2>/dev/null
ls -la $SCR/stream.bam $SCR/dump
[ -s $SCR/stream.bam ] || { echo "no stream copy"; exit 1; }
S1=$(date +%s); timeout -k 10 1800 $M/bin/extract $SCR/stream.bam $SCR/dump $SCR/rec --threads 8 --cb-from-cr 2> $R/$SET.extract.log; echo "extract exit $? wall $(( $(date +%s) - S1 )) s"; cat $R/$SET.extract.log
rm -f $SCR/stream.bam
timeout -k 10 1200 apptainer exec --nv -B $W -B /tmp $IMG $M/bin/pileup --rec $SCR/rec --dump $SCR/dump --label stream_$SET --json $R/$SET.json --reps $REPS > $R/$SET.gpu.out 2> $R/$SET.gpu.log
echo "pileup exit $?"; cat $R/$SET.gpu.out; tail -25 $R/$SET.gpu.log
echo done $(date -Is)
