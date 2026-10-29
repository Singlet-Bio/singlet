#!/bin/bash
#SBATCH --job-name=bs-cbprobe
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,short-bigmem
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,SET=GSM7901264,CELL=CGATGTAAGGTGCTAG,JX=chrX:154400626-154400701 prod_cb_probe.sbatch  (~0.5 node-h)
# Why does prod/campaign's sorted path count 243 molecules where the BAM holds 271 distinct UMIs (stream: 271)?
# Runs singlet-campaign (sorted path) keeping STAR's sorted BAM in job /tmp, then tabulates, for the reads of CELL
# (by CR) that splice JX: their CB tag, UB if any, UR, NH, flag. Kept: the read table and a summary only.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; BIN=$PB/bin/${BIN:-singlet-bs-r9}  # singlet-campaign predates SINGLET_KEEP_STAR_BAM; --sorted-bam forces its path
REF=/mnt/projects/debruinz_project/cellarium/reference; G=$REF/GRCh38-2024-A
D=$W/data/bam-stream/$SET; O=$W/bam-stream/cb_probe/$SET; rm -rf $O; mkdir -p $O
SCR=/tmp/$USER-bscb-$SLURM_JOB_ID; mkdir -p $SCR/out; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=10800 SINGLET_REF_BASE=$REF SINGLET_KEEP_STAR_BAM=1
ST=$W/envs/pbench/bin/samtools
SCREEN=(); [ -f $D/screen.json ] && SCREEN=(--run-screen $D/screen.json)
cd $SCR
$BIN $D/sample.1fq --ref-base $REF --genome-dir $G/star_2.7.11b --exons $G/genes/genes.gtf --threads 4 --pipeline \
  --output-format 1pz --sorted-bam --bam-sort-mode disk --genome-shared off "${SCREEN[@]}" --out-prefix $SCR/out/ > $O/run.log 2>&1 || echo "exit $?" >> $O/run.log
BAM=$(ls $SCR/out/*sortedByCoord*.bam | head -1); echo "bam=$BAM" | tee $O/summary.txt
$ST view -H $BAM | grep -m3 "^@PG" | cut -c1-400 >> $O/summary.txt
CHR=${JX%%:*}; R=${JX#*:}; DON=${R%-*}; ACC=${R#*-}
$ST view $BAM "$CHR:$DON-$ACC" | awk -v cell="$CELL" -v don="$DON" -v acc="$ACC" -F'\t' '
  { cr=""; cb=""; ub=""; ur=""; nh="";
    for (i=12;i<=NF;i++){ t=substr($i,1,5); v=substr($i,6)
      if(t=="CR:Z:")cr=v; else if(t=="CB:Z:")cb=v; else if(t=="UB:Z:")ub=v; else if(t=="UR:Z:")ur=v; else if(t=="NH:i:")nh=v }
    if (cr!=cell && cb!=cell) next
    pos=$4-1; c=$6; hit=0
    while (match(c,/^[0-9]+[MIDNSHP=X]/)) { n=substr(c,1,RLENGTH-1)+0; op=substr(c,RLENGTH,1); c=substr(c,RLENGTH+1)
      if (op=="N" && pos==don && pos+n==acc) hit=1
      if (op~/[MDN=X]/) pos+=n }
    if (hit) print $1"\t"$2"\t"nh"\t"cr"\t"cb"\t"ub"\t"ur }' > $O/cell_jx_reads.tsv
{
  echo "reads=$(wc -l < $O/cell_jx_reads.tsv)"
  echo "cr_is_cell=$(awk -F'\t' -v c=$CELL '$4==c' $O/cell_jx_reads.tsv | wc -l)  cb_is_cell=$(awk -F'\t' -v c=$CELL '$5==c' $O/cell_jx_reads.tsv | wc -l)"
  echo "cb_values=$(cut -f5 $O/cell_jx_reads.tsv | sort | uniq -c | sort -rn | head -5 | tr '\n' ' ')"
  echo "has_ub=$(awk -F'\t' '$6!=""' $O/cell_jx_reads.tsv | wc -l)"
  echo "distinct_ur_all=$(cut -f7 $O/cell_jx_reads.tsv | sort -u | wc -l)"
  echo "distinct_ur_cb_is_cell=$(awk -F'\t' -v c=$CELL '$5==c' $O/cell_jx_reads.tsv | cut -f7 | sort -u | wc -l)"
  echo "distinct_ub_cb_is_cell=$(awk -F'\t' -v c=$CELL '$5==c && $6!=""' $O/cell_jx_reads.tsv | cut -f6 | sort -u | wc -l)"
  echo "ur_only_without_cb=$(comm -23 <(cut -f7 $O/cell_jx_reads.tsv | sort -u) <(awk -F'\t' -v c=$CELL '$5==c' $O/cell_jx_reads.tsv | cut -f7 | sort -u) | wc -l)"
} | tee -a $O/summary.txt
grep -E "Done:|status=" $O/run.log | tail -2 >> $O/summary.txt
