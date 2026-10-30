#!/bin/bash
#SBATCH --job-name=bs-pathab
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=GSM7901264,BIN=singlet-bs-r9,CELL=CGATGTAAGGTGCTAG,JX=chrX:154400626-154400701 path_ab.sbatch  (~1.3 node-h)
# Serial (stream) vs region-worker (sorted) pileup on the SAME alignments: one default stream run saves the STAR
# stream; it is replayed as written through the serial pass and, coordinate-sorted and indexed, through the region
# workers (--parallel-pileup). Then, for one cell and one junction, the molecules are counted straight from the BAM.
# Kept: logs, pileup_stats, sj per-junction totals, the read table of that cell x junction.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-r9}
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; O=$W/bam-stream/path_ab/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bspathab-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF
ST=$W/envs/pbench/bin/samtools
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
COMMON=(--ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 4 --pipeline --output-format 1pz "${SCREEN[@]}")
cd $SCR
SINGLET_TEE_STREAM_BAM=$SCR/stream.bam $BIN $D/sample.1fq "${COMMON[@]}" --out-prefix $SCR/o0/ --genome-shared off > $O/run0.log 2>&1 || echo "exit $?" >> $O/run0.log
$ST sort -@ 4 -m 2G -o $SCR/coord.bam $SCR/stream.bam && $ST index -@ 4 $SCR/coord.bam
BC=(--barcodes $SCR/o0/auto_barcodes.tsv --cell-calling)
$BIN --bam-file $SCR/stream.bam "${COMMON[@]}" --stream-unsorted "${BC[@]}" --out-prefix $SCR/serial/ > $O/serial.log 2>&1 || echo "exit $?" >> $O/serial.log
$BIN --bam-file $SCR/coord.bam "${COMMON[@]}" --parallel-pileup "${BC[@]}" --out-prefix $SCR/workers/ > $O/workers.log 2>&1 || echo "exit $?" >> $O/workers.log
for v in serial workers; do
  cp $SCR/$v/pileup_stats.json $O/${v}_pileup_stats.json 2>/dev/null || true
  grep -E "BAM:|Done:|mode" $O/$v.log | head -4
done
# One cell x one junction straight from the BAM: reads whose CIGAR has that N gap
CHR=${JX%%:*}; R=${JX#*:}; DON=${R%-*}; ACC=${R#*-}
$ST view $SCR/coord.bam "$CHR:$DON-$ACC" | awk -v cell="$CELL" -v don="$DON" -v acc="$ACC" -F'\t' '
  { cr=""; ur=""; nh="";
    for (i=12;i<=NF;i++){ if($i~/^CR:Z:/)cr=substr($i,6); if($i~/^UR:Z:/)ur=substr($i,6); if($i~/^NH:i:/)nh=substr($i,6) }
    if (cr!=cell) next
    pos=$4-1; c=$6; hit=0
    while (match(c,/^[0-9]+[MIDNSHP=X]/)) { n=substr(c,1,RLENGTH-1)+0; op=substr(c,RLENGTH,1); c=substr(c,RLENGTH+1)
      if (op=="N" && pos==don && pos+n==acc) hit=1
      if (op~/[MDN=X]/) pos+=n }
    if (hit) print $1"\t"$2"\t"nh"\t"ur"\t"$6 }' > $O/cell_jx_reads.tsv
{
  echo "reads=$(wc -l < $O/cell_jx_reads.tsv)"
  echo "distinct_qname=$(cut -f1 $O/cell_jx_reads.tsv | sort -u | wc -l)"
  echo "distinct_umi=$(cut -f4 $O/cell_jx_reads.tsv | sort -u | wc -l)"
  echo "distinct_umi_primary=$(awk -F'\t' 'and($2,256)==0' $O/cell_jx_reads.tsv | cut -f4 | sort -u | wc -l)"
  echo "qname_seen_twice=$(cut -f1 $O/cell_jx_reads.tsv | sort | uniq -d | wc -l)"
  echo "umi_lengths=$(cut -f4 $O/cell_jx_reads.tsv | awk '{print length($0)}' | sort | uniq -c | tr '\n' ' ')"
  echo "flags=$(cut -f2 $O/cell_jx_reads.tsv | sort | uniq -c | tr '\n' ' ')"
} | tee $O/cell_jx_summary.txt
# The two paths' count for that cell x junction (decoded from sj_counts.1pz)
mkdir -p $SCR/inc; cp -r $W/src/singlet-py/perf__bam-stream/include/singlet $SCR/inc/
sed -i 's/throw PZReadError("rowname count does not match the number of rows");/rownames.resize(hdr.m);/' $SCR/inc/singlet/pileup/pz_reader.h
$W/envs/pbench/bin/x86_64-conda-linux-gnu-g++ -O2 -std=c++17 -I$SCR/inc -I$W/envs/pbench/include $W/jobs/bam-stream/pz2mtx.cpp \
  -L$W/envs/pbench/lib -Wl,-rpath,$W/envs/pbench/lib -lzstd -o $SCR/pz2mtx
for v in serial workers; do
  $SCR/pz2mtx $SCR/$v/sj_counts.1pz $SCR/$v.sj
  r=$(grep -nx "$JX" $SCR/$v.sj.rows.tsv | cut -d: -f1); c=$(grep -nE "^$CELL(-1)?\$" $SCR/$v.sj.cols.tsv | cut -d: -f1)
  tot=$(awk 'NR>2{s+=$3}END{print s}' $SCR/$v.sj.mtx)
  val=$(awk -v r="$r" -v c="$c" 'NR>2 && $1==r && $2==c {print $3}' $SCR/$v.sj.mtx)
  echo "$v: sj_total=$tot cell_jx=${val:-0} (row $r col $c)" | tee -a $O/cell_jx_summary.txt
done
