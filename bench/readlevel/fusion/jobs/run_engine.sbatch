#!/bin/bash
#SBATCH --job-name=rl-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# One engine run for the read-level benchmarks (CPU engine; ENGARG="--engine gpu" plus SINGLET_GPU_PILEUP=on on a GPU node).
# BIN: directory with singlet-gpu. SAMPLE: name. Input: FQ (a .1fq) or FQDIR (a FASTQ folder, encoded once to $W/data/readlevel/1fq/).
# GENOME (default GRCh38-2024-A). EXTRA: engine arguments (e.g. --chimeric). TAG: output group. Output: $W/jobs/readlevel/runs/$TAG/$SAMPLE/out
set -uo pipefail
: "${BIN:?}" "${SAMPLE:?}" "${TAG:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
G=$REF/${GENOME:-GRCh38-2024-A}; T=${SLURM_CPUS_PER_TASK:-24}
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=3600 SINGLET_REF_BASE=$REF PATH=$W/envs/pbench/bin:$PATH
O=$W/jobs/readlevel/runs/$TAG/$SAMPLE; [ -e $O/out ] && { echo "refusing to overwrite $O"; exit 2; }; mkdir -p $O
if [ -z "${FQ:-}" ]; then
  : "${FQDIR:?FQ or FQDIR}"; mkdir -p $W/data/readlevel/1fq; FQ=$W/data/readlevel/1fq/$SAMPLE.1fq
  if [ ! -s $FQ ]; then
    ( cd /tmp && $BIN/singlet-gpu download --fastq-dir $FQDIR -o $FQ.part ${DL_EXTRA:-} ) > $O/encode.log 2>&1 && mv $FQ.part $FQ || { tail -20 $O/encode.log; exit 1; }
  fi
fi
SCR=/tmp/$USER-rl-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
echo "node=$(hostname) bin=$BIN commit=$(cat $BIN/commit-gpu 2>/dev/null) sample=$SAMPLE fq=$FQ genome=$G extra=${EXTRA:-} engarg=${ENGARG:-} $(date -Is)" | tee $O/run.txt
( cd $SCR && /usr/bin/time -v -o $O/time.txt $BIN/singlet-gpu $FQ --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf \
    --out-prefix $SCR/out/ --threads $T --pipeline --output-format 1pz ${ENGARG:-} ${EXTRA:-} > $O/run.log 2>&1 )
echo "exit $(awk -F': ' '/Exit status/{print $2}' $O/time.txt) wall $(awk -F': ' '/Elapsed/{print $2}' $O/time.txt) rss_kb $(awk -F': ' '/Maximum resident/{print $2}' $O/time.txt)" | tee -a $O/run.txt
grep -a "chimeric\|\[te\]" $O/run.log | tail -20
# keep the outputs, not the reads
rsync -a --exclude='*.bam' --exclude='*Unmapped*' --exclude='*.fq' --exclude='_STARtmp*' --exclude='.gpu_handoff*' $SCR/out/ $O/out/
ls $O/out | head -100
