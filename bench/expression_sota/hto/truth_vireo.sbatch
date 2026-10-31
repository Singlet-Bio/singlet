#!/bin/bash
#SBATCH --job-name=hto-vireo
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# I1 truth, step 2: genotype donors. cellSNP-lite mode 1a on Singlet's tagged BAM at the 1000G AF>=5% panel, then vireo -N 8.
# Never touches the hashtag counts. Same commands as bench/modalities/genotype_pool/ref_demux.sbatch.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/expression-sota/hto/data/stoeckius2018; E=$W/envs/gpool
PANEL=/mnt/projects/debruinz_project/cellarium/reference/pileup_indices/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.sorted.chrprefix.vcf.gz
S=$D/singlet; O=$D/vireo; mkdir -p $O; cd $O
export APPTAINER_BIND=/mnt/projects:/mnt/projects; export PATH=$W/envs/pbench/bin:$PATH
samtools view $S/possorted.bam chr1 | head -1 | cut -f1,12-
awk -F"\t" 'NR>1 && $2=="TRUE"{print $1"-1"}' $S/cell_calls.tsv > $O/barcodes.txt; wc -l $O/barcodes.txt
apptainer exec $E/cellsnp-lite.sif cellsnp-lite -s $S/possorted.bam -b $O/barcodes.txt -O $O/cellsnp -R $PANEL -p 16 \
  --minMAF 0.1 --minCOUNT 20 --gzip --cellTAG CB --UMItag UB > cellsnp.log 2>&1
tail -5 cellsnp.log
$E/venv/bin/vireo -c $O/cellsnp -o $O/vireo -N 8 -p 16 --randSeed 42 > vireo.log 2>&1
tail -5 vireo.log; ls $O/vireo; head -3 $O/vireo/donor_ids.tsv; cut -f2 $O/vireo/donor_ids.tsv | sort | uniq -c
