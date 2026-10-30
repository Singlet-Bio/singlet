#!/bin/bash
#SBATCH --job-name=gp-vireo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch ref_demux.sbatch <singlet-tag>
# Reference donor demux on the same BAM and the same cells Singlet called: cellsnp-lite mode 1a pileup at the
# same 1000G panel, then vireo (K=8 given, and K auto-by-ELBO is not offered by vireo; N=8 is the stated truth).
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; E=$W/envs/gpool; TAG=$1
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
S=$D/singlet_$TAG; O=$D/ref_$TAG; mkdir -p $O; cd $O
export APPTAINER_BIND=/mnt/projects:/mnt/projects
export PATH=$W/envs/pbench/bin:$PATH; samtools view $S/possorted.bam chr1 | head -1 | cut -f1,12-; awk -F"	" 'NR>1 && $2=="TRUE"{print $1"-1"}' $S/cell_calls.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt
/usr/bin/time -v -o cellsnp.time.txt apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $S/possorted.bam -b $O/barcodes.txt \
  -O $O/cellsnp -R $PANEL -p 16 --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB > cellsnp.log 2>&1
tail -5 cellsnp.log; grep -E "Elapsed|Maximum res" cellsnp.time.txt
/usr/bin/time -v -o vireo.time.txt $E/venv/bin/vireo -c $O/cellsnp -o $O/vireo -N 8 -p 16 --randSeed 42 > vireo.log 2>&1
tail -5 vireo.log; grep -E "Elapsed|Maximum res" vireo.time.txt; ls $O/vireo
