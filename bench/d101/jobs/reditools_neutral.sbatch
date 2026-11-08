#!/bin/bash
#SBATCH --job-name=d101-redin
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# B4 editing comparison on a site set chosen independently of both single-cell tools (row SLURM_ARRAY_TASK_ID of libs.tsv).
# Sites: every REDIportal panel site (the engine's own panel file) where the donor's bulk STAR BAM has >= MINBULK reads (samtools depth).
# On those sites the called-cell reads of the engine BAM get a raw samtools depth too; score_editing.py --sites scores both tools on the
# neutral set (bulk >= MINBULK and raw single-cell depth >= MINSC), reports each tool's coverage of it, and keeps the Singlet-selected
# set only as a secondary line. REDItools3 3.7 (default filters, -me 0, -s 0) runs on the bulk-qualified sites for sc and bulk.
# Output (owner-only): $W/jobs/d101/editing_neutral/<sc_run>/
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; LIBS=${LIBS:?}; T=${SLURM_CPUS_PER_TASK:-16}
REF=/mnt/projects/debruinz_project/cellarium/reference; PANEL=$REF/pileup_indices/rna_editing_GRCh38.tsv.gz
MINBULK=${MINBULK:-10}; MINSC=${MINSC:-5}
read -r DONOR SC SCG BULK BULKG <<< "$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS)"
[ "$BULK" != "-" ] || { echo "donor $DONOR has no bulk library"; exit 0; }
R=$W/jobs/d101/runs/$SC/out; O=$W/jobs/d101/editing_neutral/$SC; mkdir -p $W/jobs/d101/editing_neutral $O; chmod 700 $W/jobs/d101/editing_neutral $O
SCR=/tmp/$USER-d101redin-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
IR=$W/src/singlet-agents/main/hpc/clipper/in_runtime.sh; [ -f $IR ] || IR=$(ls -d $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
BAM=$W/jobs/d101/bam/$SC.bam; BB=$W/jobs/d101/bulkbam/$BULK.bam
FA=$REF/GRCh38-2024-A/fasta/genome.fa
echo "donor $DONOR sc $SC bulk $BULK panel $PANEL $(date -Is)"
# Panel -> BED with strand (A>G is +, T>C is -), skipping comment/header lines
zcat $PANEL | awk -F'\t' '$1!~/^#/ && $2~/^[0-9]+$/{s=($3=="A"?"+":($3=="T"?"-":"")); if(s!="") print $1"\t"$2-1"\t"$2"\t"s}' \
  | sort -k1,1 -k2,2n > $SCR/panel.bed
# Neutral depth 1: bulk STAR reads at each panel site
samtools depth -@ $T -b $SCR/panel.bed $BB | awk -v m=$MINBULK '$3>=m{print $1"\t"$2-1"\t"$2"\t"$3}' > $SCR/bulkdp.txt
awk 'NR==FNR{d[$1":"$3]=$4; next} ($1":"$3) in d{print $1"\t"$2"\t"$3"\t"$4"\t"d[$1":"$3]}' $SCR/bulkdp.txt $SCR/panel.bed > $SCR/bq.bed
echo "panel $(wc -l < $SCR/panel.bed) bulk>=$MINBULK $(wc -l < $SCR/bq.bed) $(date -Is)"
# Called-cell reads of the engine BAM at those sites (barcode form from the BAM), MD tags from the same genome for REDItools3
CB1=$( { samtools view $BAM 2>/dev/null || true; } | grep -m1 -o 'CB:Z:[^[:space:]]*' | cut -c6- || true)
awk -F'\t' 'NR>1{print $1}' $R/editing_cells.tsv | sed 's/-1$//' > $SCR/cells.txt
case "$CB1" in *-1) sed -i 's/$/-1/' $SCR/cells.txt;; esac
cut -f1-3 $SCR/bq.bed > $SCR/bq3.bed
samtools view -@ $T -b -L $SCR/bq3.bed -D CB:$SCR/cells.txt -o $SCR/sc0.bam $BAM
samtools calmd -@ $T -b $SCR/sc0.bam $FA > $SCR/sc.bam 2> $SCR/calmd.log; rm -f $SCR/sc0.bam; samtools index $SCR/sc.bam
# Neutral depth 2: raw single-cell read depth (no base or mapping-quality filter) at the bulk-qualified sites
samtools depth -a -@ $T -b $SCR/bq3.bed $SCR/sc.bam | awk '{print $1":"$2"\t"$3}' > $SCR/scdp.txt
awk 'NR==FNR{d[$1]=$2; next}{k=$1":"$3; print $1"\t"$3"\t"$4"\t"$5"\t"(k in d?d[k]:0)}' $SCR/scdp.txt $SCR/bq.bed \
  | gzip -1 > $O/sites.tsv.gz   # chrom pos strand bulk_depth sc_depth
echo "sites with sc depth>=$MINSC $(zcat $O/sites.tsv.gz | awk -v m=$MINSC '$5>=m' | wc -l) $(date -Is)"
RT="env PYTHONPATH=$W/envs/reditools3 python3 -m reditools analyze"
zcat $O/sites.tsv.gz | awk '$5>0{print $1"\t"$2-1"\t"$2}' > $SCR/scsites.bed
echo "reditools sc $(date -Is)"; t0=$(date +%s)
in_runtime $RT $SCR/sc.bam -B $SCR/scsites.bed -me 0 -s 0 -t $T -o $O/reditools_sc.tsv
echo "reditools bulk $(date -Is)"; t1=$(date +%s)
in_runtime $RT $BB -B $SCR/bq3.bed -me 0 -s 0 -t $T -o $O/reditools_bulk.tsv
t2=$(date +%s)
echo "{\"reditools_sc_s\": $((t1-t0)), \"reditools_bulk_s\": $((t2-t1))}" > $O/times.json
SRC=$(dirname $(readlink -f $0)); [ -f $SRC/score_editing.py ] || SRC=$W/jobs/d101
in_runtime python3 $SRC/score_editing.py --singlet $R/private/editing/editing_sites.tsv.gz --sc $O/reditools_sc.tsv \
  --bulk $O/reditools_bulk.tsv --sites $O/sites.tsv.gz --label $DONOR --min-bulk $MINBULK --min-sc $MINSC --out $O/score.json
cat $O/score.json; echo "end $(date -Is)"
