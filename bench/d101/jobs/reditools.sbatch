#!/bin/bash
#SBATCH --job-name=d101-redi
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Editing comparison for one donor (row SLURM_ARRAY_TASK_ID of libs.tsv). Sites: every REDIportal site Singlet covered in called cells
# (private/editing/editing_sites.tsv.gz). REDItools3 3.7 (default quality filters, -me 0 so unedited sites are reported, -s 0) runs on
#   sc   : the engine's tagged BAM restricted to called-cell reads (samtools view -D CB:), i.e. the same alignments Singlet counted
#   bulk : STAR alignment of the donor's bulk RNA-seq (the reference levels)
# then score.py compares Singlet and REDItools3-sc site levels with bulk. Output: $W/jobs/d101/editing/<sc_run>/ (owner-only).
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; LIBS=${LIBS:?}; T=${SLURM_CPUS_PER_TASK:-16}
read -r DONOR SC SCG BULK BULKG <<< "$(awk -v n=$SLURM_ARRAY_TASK_ID 'NR==n+1' $LIBS)"
R=$W/jobs/d101/runs/$SC/out; O=$W/jobs/d101/editing/$SC; mkdir -p $O; chmod 700 $O
SCR=/tmp/$USER-d101redi-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
IR=$W/src/singlet-agents/main/hpc/clipper/in_runtime.sh; [ -f $IR ] || IR=$(ls -d $W/src/singlet-agents/*/hpc/clipper/in_runtime.sh | head -1); source $IR
# Sites (BED, 0-based half-open) and called cells; the BAM's CB form (with or without -1) decides the barcode list's form
zcat $R/private/editing/editing_sites.tsv.gz | awk -F'\t' 'NR>1{split($1,a,":"); print a[1]"\t"a[2]-1"\t"a[2]}' | sort -k1,1 -k2,2n > $SCR/sites.bed
BAM=$W/jobs/d101/bam/$SC.bam
# grep -m1 closes the pipe early; without the || true, pipefail turns samtools' SIGPIPE into exit 141
CB1=$( { samtools view $BAM 2>/dev/null || true; } | grep -m1 -o 'CB:Z:[^[:space:]]*' | cut -c6- || true)
awk -F'\t' 'NR>1{print $1}' $R/editing_cells.tsv | sed 's/-1$//' > $SCR/cells.txt
case "$CB1" in *-1) sed -i 's/$/-1/' $SCR/cells.txt;; esac
echo "donor $DONOR sc $SC bulk $BULK sites $(wc -l < $SCR/sites.bed) cells $(wc -l < $SCR/cells.txt) CB example $CB1"
# REDItools3 reads mismatches from the MD tag, which the engine's BAM does not carry; calmd adds it from the same genome
FA=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/fasta/genome.fa
samtools view -@ $T -b -L $SCR/sites.bed -D CB:$SCR/cells.txt -o $SCR/sc0.bam $BAM
samtools calmd -@ $T -b $SCR/sc0.bam $FA > $SCR/sc.bam 2> $SCR/calmd.log; rm -f $SCR/sc0.bam; samtools index $SCR/sc.bam
RT="env PYTHONPATH=$W/envs/reditools3 python3 -m reditools analyze"
t0=$(date +%s)
in_runtime $RT $SCR/sc.bam -B $SCR/sites.bed -me 0 -s 0 -t $T -o $O/reditools_sc.tsv
t1=$(date +%s)
[ "$BULK" != "-" ] && in_runtime $RT $W/jobs/d101/bulkbam/$BULK.bam -B $SCR/sites.bed -me 0 -s 0 -t $T -o $O/reditools_bulk.tsv
t2=$(date +%s)
echo "{\"reditools_sc_s\": $((t1-t0)), \"reditools_bulk_s\": $((t2-t1))}" > $O/times.json
SRC=$(dirname $(readlink -f $0)); [ -f $SRC/score_editing.py ] || SRC=$W/jobs/d101
in_runtime python3 $SRC/score_editing.py --singlet $R/private/editing/editing_sites.tsv.gz --sc $O/reditools_sc.tsv \
  --bulk $O/reditools_bulk.tsv --label $DONOR --out $O/score.json
cat $O/score.json
