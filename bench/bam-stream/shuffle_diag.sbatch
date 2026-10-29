#!/bin/bash
#SBATCH --job-name=bs-shufdx
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<id>,BIN=<name> shuffle_diag.sbatch      (~1.3 node-h at 4 CPUs)
# Diagnoses the exon_counts difference found by shuffle_gate.sbatch: same stream as there, replayed as written
# (twice), collated and coordinate-sorted, with mtx output; the exon/intron matrices (small) are kept in
# bam-stream/shuffle_diag/<SET>/keep/<variant>/ and exon_order_diff.py reports which entries move.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-r8}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; O=$W/bam-stream/shuffle_diag/$SET; rm -rf $O; mkdir -p $O/keep
SCR=/tmp/$USER-bsshufdx-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
ST=$W/envs/pbench/bin/samtools
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
COMMON=(--ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 4 --pipeline "${SCREEN[@]}")
cd $SCR
SINGLET_TEE_STREAM_BAM=$SCR/stream.bam $BIN $D/sample.1fq "${COMMON[@]}" --output-format 1pz --out-prefix $SCR/o0/ \
  --genome-shared off > $O/run0.log 2>&1 || echo "exit $?" >> $O/run0.log
$ST collate -@ 4 -o $SCR/collate.bam $SCR/stream.bam $SCR/coll_tmp
$ST sort -@ 4 -m 2G -o $SCR/coord.bam $SCR/stream.bam
for v in stream stream2 collate coord; do
  b=${v%2}
  $BIN --bam-file $SCR/$b.bam "${COMMON[@]}" --output-format mtx --stream-unsorted --barcodes $SCR/o0/auto_barcodes.tsv \
    --cell-calling --out-prefix $SCR/o_$v/ > $O/replay_$v.log 2>&1 || echo "exit $?" >> $O/replay_$v.log
  mkdir -p $O/keep/$v
  cp $SCR/o_$v/{exon,intron}_counts{.mtx.gz,_features.tsv.gz,_barcodes.tsv.gz} $SCR/o_$v/pileup_stats.json $O/keep/$v/ 2>/dev/null || true
  rm -rf $SCR/o_$v
done
in_runtime python -I $W/jobs/bam-stream/exon_order_diff.py $O/keep stream stream2 collate coord > $O/diff.json 2>&1
head -c 4000 $O/diff.json
