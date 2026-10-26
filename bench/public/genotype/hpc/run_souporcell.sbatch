#!/bin/bash
#SBATCH --job-name=bpg-soup
#SBATCH --account=debruinz
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch run_souporcell.sbatch <pool> <K_true>
# Competitor: souporcell 2.5 (bioconda container) at its documented defaults for human (remap with minimap2, freebayes
# calls, vartrix counts, clustering; all defaults), k = true K (souporcell cannot estimate K), --common_variants =
# the same 1000G panel the other tools use. Input: Singlet's tagged BAM restricted to the called cells (the same cells).
set -u
POOL=$1; KT=$2
W=/mnt/projects/debruinz_project/debruinz/singlet; SIF=$W/envs/souporcell/souporcell.sif
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
PANELVCF=$W/bench-public/ref/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf   # souporcell reads plain text
FA=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/fasta/genome.fa
S=$W/bench-public/runs/$POOL/singlet_auto; O=$W/bench-public/runs/$POOL/souporcell; mkdir -p $O; cd $O
export APPTAINER_BIND=/mnt/projects:/mnt/projects
mkdir -p $W/bench-public/ref; [ -s $PANELVCF ] || { zcat $PANEL > $PANELVCF.tmp && mv $PANELVCF.tmp $PANELVCF; }
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
SAM=$W/envs/pbench/bin/samtools
TAGFMT=$($SAM view $S/possorted.bam chr1 | head -2000 | grep -o -m1 "CB:Z:[ACGT]*[^[:space:]]*" | head -1)
SUF=""; case "$TAGFMT" in *-1) SUF="-1";; esac
awk -F"\t" -v s="$SUF" 'NR>1 && $2=="TRUE"{print $1 s}' $S/cell_calls.tsv > barcodes.txt; wc -l barcodes.txt
$SAM view -@ 8 -b -D CB:barcodes.txt $S/possorted.bam > $SCR/cells.bam && $SAM index -@ 8 $SCR/cells.bam
/usr/bin/time -v -o soup.time.txt apptainer exec $SIF souporcell_pipeline.py -i $SCR/cells.bam -b barcodes.txt -f $FA \
  -t 16 -o $O/out -k $KT --common_variants $PANELVCF > soup.log 2>&1
grep -E "Elapsed|Maximum res" soup.time.txt; tail -5 soup.log; ls $O/out
