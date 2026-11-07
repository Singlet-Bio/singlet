#!/bin/bash
#SBATCH --job-name=cp-run
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Count-parity run: one Singlet binary on one dataset (CPU engine unless ENGINE=gpu), same command as bench/exact_vs_pseudo.
#   sbatch --export=ALL,BIN=<singlet>,DS=hgmm1k_v3,TAG=<name>[,RULES=<intron_rules>][,DUMP=1][,TEE=1][,ENGINE=gpu][,GPUPILEUP=1][,ARGS="--no-cr4"] run_singlet.sbatch
#   STAR_EXTRA="--outFilterScoreMin 30" is passed as --star-extra (export it before sbatch --export=ALL,...)
#   GPUPILEUP=1 runs the CPU aligner with the GPU pileup (SINGLET_GPU_PILEUP=on): submit with -p short-gpu --gres=gpu:nvidia_h100_nvl:1
# Output: $W/count-parity/runs/<DS>/<TAG>/ (raw matrices, pileup_stats.json, molecules.tsv.gz with DUMP=1, stream.bam with TEE=1)
set -euo pipefail
: "${BIN:?}" "${DS:?}" "${TAG:?}"
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; REF=/mnt/projects/debruinz_project/cellarium/reference
case $DS in
  hgmm*) SPECIES=GRCh38-GRCm39-barnyard ;;
  *) SPECIES=GRCh38-2024-A ;;
esac
GENOME=$REF/$SPECIES/star_2.7.11b; GTF=$REF/$SPECIES/genes/genes.gtf
O=$W/count-parity/runs/$DS/$TAG; mkdir -p $O
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
ONEFQ=$PB/data/$DS/$DS.1fq
if [ ! -s $ONEFQ ]; then
  cat $(grep "_R1_" $PB/data/$DS/gex_fastqs.txt) > $SCR/R1.fastq.gz; cat $(grep "_R2_" $PB/data/$DS/gex_fastqs.txt) > $SCR/R2.fastq.gz
  $BIN encode --reads $SCR/R1.fastq.gz $SCR/R2.fastq.gz -o $SCR/$DS.1fq > $O/encode.log 2>&1; rm -f $SCR/R*.fastq.gz
  cp $SCR/$DS.1fq $ONEFQ.tmp && mv $ONEFQ.tmp $ONEFQ
fi
echo "bin $BIN sha256 $(sha256sum $BIN | cut -c1-64) commit $(cat $(dirname $BIN)/commit* 2>/dev/null | head -1) node $(hostname)" | tee $O/bin.txt
export SINGLET_STALL_TIMEOUT=7200
[ -n "${RULES:-}" ] && export SINGLET_INTRON_RULES=$RULES
[ "${DUMP:-0}" = 1 ] && export SINGLET_MOLECULE_DUMP=$O/molecules.tsv.gz
[ "${TEE:-0}" = 1 ] && export SINGLET_TEE_STREAM_BAM=$O/stream.bam
[ "${GPUPILEUP:-0}" = 1 ] && export SINGLET_GPU_PILEUP=on
EXTRA="${ARGS:-}"; [ "${ENGINE:-cpu}" = gpu ] && EXTRA="$EXTRA --engine gpu"
XARGS=(); [ -n "${STAR_EXTRA:-}" ] && XARGS=(--star-extra "$STAR_EXTRA")
cd $SCR
/usr/bin/time -v -o $O/time.txt $BIN $ONEFQ --ref-base $REF --genome-dir $GENOME --exons $GTF \
    --out-prefix $O/ --threads $SLURM_CPUS_PER_TASK --pipeline --output-format mtx --raw-matrix $EXTRA ${XARGS[@]+"${XARGS[@]}"} > $O/run.log 2>&1
grep -E "Elapsed \(wall|Maximum resident" $O/time.txt
tail -5 $O/run.log
