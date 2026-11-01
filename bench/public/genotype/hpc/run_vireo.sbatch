#!/bin/bash
#SBATCH --job-name=bpg-vireo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_vireo.sbatch <pool> <K_true>
# Competitor: cellsnp-lite 1.2.3 mode 1a + vireoSNP 0.5.9 on the SAME tagged BAM and the SAME called cells as Singlet
# (singlet_auto run), same 1000G panel. Documented settings (cellSNP/vireo manuals): --minMAF 0.1 --minCOUNT 20, vireo
# defaults (nInit 50, doublet detection on). Runs: told K (3 seeds) and N=12 (vireo has no K search: -N can exceed the
# true number, surplus donors end up nearly empty; K-hat = donors holding >= 1% of cells, see vireo_khat.py).
set -u
POOL=$1; KT=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; E=$W/envs/gpool; CODE=$W/bench-public/code/public/genotype/lib
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
S=$W/bench-public/runs/$POOL/singlet_auto; O=$W/bench-public/runs/$POOL/vireo; mkdir -p $O; cd $O
export APPTAINER_BIND=/mnt/projects:/mnt/projects
BAM=$S/possorted.bam
# barcodes: Singlet's called cells, written the way the BAM's CB tag spells them
TAGFMT=$($W/envs/pbench/bin/samtools view $BAM chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1)
echo "CB example: $TAGFMT"
SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $S/cell_calls.tsv > barcodes.txt; wc -l barcodes.txt
/usr/bin/time -v -o cellsnp.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $BAM -b barcodes.txt \
  -O cellsnp -R $PANEL -p 16 --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB > cellsnp.log 2>&1
grep -E "Elapsed|Maximum res" cellsnp.time.txt
for SEED in 1 2 3; do
  /usr/bin/time -v -o vireo_k$KT.s$SEED.time.txt $E/venv/bin/vireo -c cellsnp -o told_s$SEED -N $KT -p 16 --randSeed $SEED --noPlot > told_s$SEED.log 2>&1
  grep -E "Elapsed|Maximum res" vireo_k$KT.s$SEED.time.txt
done
/usr/bin/time -v -o vireo_n12.time.txt $E/venv/bin/vireo -c cellsnp -o n12 -N 12 -p 16 --randSeed 1 --noPlot > n12.log 2>&1
grep -E "Elapsed|Maximum res" vireo_n12.time.txt; tail -4 n12.log
ls $O
