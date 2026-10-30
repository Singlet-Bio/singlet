#!/bin/bash
#SBATCH --job-name=gp-mtref
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch ref_mt.sbatch <singlet-tag>
# Reference mtDNA calls from the same BAM and cells: (1) pysam read-level chrM pileup with Singlet's filters,
# (2) mgatk tenx. Then mt_eval.py scores Singlet against them.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; E=$W/envs/gpool; TAG=$1
export PATH=$E/venv/bin:$W/envs/pbench/bin:$PATH
S=$D/singlet_$TAG; O=$D/ref_$TAG; mkdir -p $O; cd $O
awk -F"	" 'NR>1 && $2=="TRUE"{print $1"-1"}' $S/cell_calls.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt
samtools index -@ 8 $S/possorted.bam 2>&1 | tail -1
samtools view -c $S/possorted.bam chrM
samtools view $S/possorted.bam chrM | head -2 | cut -c1-400
/usr/bin/time -v -o pileup.time.txt python $W/jobs/modalities/genotype_pool/mt_pileup.py $S/possorted.bam barcodes.txt mt_pileup.npz 2>&1 | tail -3
grep -E "Elapsed|Maximum res" pileup.time.txt
mgatk tenx --help 2>&1 | head -40
/usr/bin/time -v -o mgatk.time.txt mgatk tenx -i $S/possorted.bam -n kang -o mgatk -bt CB -b barcodes.txt -c 8 -g rCRS --keep-temp-files > mgatk.log 2>&1
tail -15 mgatk.log; grep -E "Elapsed|Maximum res" mgatk.time.txt; ls mgatk mgatk/final 2>&1 | head -30
