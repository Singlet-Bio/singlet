#!/bin/bash
#SBATCH --job-name=bs-p2
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=<id> phase2_stream.sbatch     (~4 core-h per 67 M, ~8 per 111 M)
# Phase 2 streaming-pileup cost. Step 1: normal pipeline; a hard link keeps STAR's coordinate-sorted BAM alive in job /tmp.
# Step 2: the pileup alone on that BAM with --no-parallel-pileup (the existing streaming run()), then step 3 with the
# parallel pileup at the same thread count. Nothing is kept: the BAM and all outputs but logs stay in /tmp and are deleted.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/singlet-campaign
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; [ -e $D/sample.1fq ] || D=$W/data/atlas-rerun/perf/$SET
O=$W/bam-stream/p2/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bsp2-$SLURM_JOB_ID; mkdir -p $SCR/o1 $SCR/o2 $SCR/o3; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
sha256sum $BIN > $O/node.txt; echo "node=$(hostname) $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> $O/node.txt
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
COMMON=(--ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 4 --pipeline --output-format 1pz "${SCREEN[@]}")
cd $SCR
/usr/bin/time -v -o $O/time1.txt $BIN $D/sample.1fq "${COMMON[@]}" --out-prefix $SCR/o1/ --bam-sort-mode disk --genome-shared off \
  --tagged-bam $SCR/tagged.bam > $O/run1.log 2>&1 || echo "exit $?" >> $O/run1.log
ls -l $SCR/tagged.bam | tee $O/bams.txt
BAM=$SCR/tagged.bam   # barcoded reads only (~82% of reads): scale the cost by read count
# --bam-file needs a barcode argument: try auto-discovery first, then the 10x whitelist as the filtered list
for mode in serial parallel; do
  X=(); [ $mode = serial ] && X=(--no-parallel-pileup)
  for V in "--whitelist None" "--barcodes $PB/whitelists/737K-august-2016.txt"; do
    /usr/bin/time -v -o $O/time_$mode.txt $BIN --bam-file $BAM "${COMMON[@]}" $V --out-prefix $SCR/o2/ "${X[@]}"       > $O/run_$mode.log 2>&1 || echo "exit $?" >> $O/run_$mode.log
    grep -q "^ERROR" $O/run_$mode.log || { echo "$mode ok with: $V" >> $O/variant.txt; break; }
  done
done
python3 $W/bam-stream/parse_run.py $O > /dev/null 2>&1 || true
grep -h -E "Elapsed|Maximum res" $O/time*.txt; grep -h -E "Pileup:|\[parallel\]|Done:" $O/run_*.log | cut -c1-200
