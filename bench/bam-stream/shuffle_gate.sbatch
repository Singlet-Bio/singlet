#!/bin/bash
#SBATCH --job-name=bs-shuf
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --exclusive
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<id>,BIN=<name> shuffle_gate.sbatch      (one exclusive node, ~1.2 node-h)
# Shuffled-order gate (docs/research-beta/order-free-rules.md): one default stream run that saves the unsorted STAR
# stream (SINGLET_TEE_STREAM_BAM), then three replays of the same alignments in different record orders through
# --bam-file --stream-unsorted with the stream run's barcode list and cell calling on: as written, samtools collate
# (reads grouped, groups shuffled), and coordinate-sorted (alignments of a read scattered). out_equiv must find the
# three replays identical. Everything stays in job /tmp; only logs and the comparisons are kept.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-rules}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; O=$W/bam-stream/shuffle/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bsshuf-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
ST=$W/envs/pbench/bin/samtools
sha256sum $BIN > $O/node.txt; echo "node=$(hostname) $(date -Is)" >> $O/node.txt
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
COMMON=(--ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 4 --pipeline --output-format 1pz "${SCREEN[@]}")
cd $SCR
SINGLET_TEE_STREAM_BAM=$SCR/stream.bam $BIN $D/sample.1fq "${COMMON[@]}" --out-prefix $SCR/o0/ --genome-shared off > $O/run0.log 2>&1 || echo "exit $?" >> $O/run0.log
ls -l $SCR/stream.bam | tee $O/bams.txt
$ST collate -@ 4 -o $SCR/collate.bam $SCR/stream.bam $SCR/coll_tmp
$ST sort -@ 4 -m 2G -o $SCR/coord.bam $SCR/stream.bam
ls -l $SCR/*.bam >> $O/bams.txt
for v in stream collate coord; do
  $BIN --bam-file $SCR/$v.bam "${COMMON[@]}" --stream-unsorted --barcodes $SCR/o0/auto_barcodes.tsv --cell-calling \
    --out-prefix $SCR/o_$v/ > $O/replay_$v.log 2>&1 || echo "exit $?" >> $O/replay_$v.log
  grep -E "BAM:|cell_calling\] n_cells|status=|ERROR" $O/replay_$v.log | head -5
done
for p in "stream collate" "stream coord" "collate coord"; do
  set -- $p; echo "######## $1 vs $2" | tee -a $O/equiv.txt
  in_runtime python -I $W/jobs/deep-perf/out_equiv.py $SCR/o_$1 $SCR/o_$2 2>&1 | cut -c1-260 | tee -a $O/equiv.txt | grep -E "DIFF|RESULT|files:"
done
echo "######## stream run vs its own replay (as written)" | tee -a $O/equiv.txt
in_runtime python -I $W/jobs/deep-perf/out_equiv.py $SCR/o0 $SCR/o_stream 2>&1 | cut -c1-260 | tee -a $O/equiv.txt | grep -E "DIFF|RESULT|files:"
