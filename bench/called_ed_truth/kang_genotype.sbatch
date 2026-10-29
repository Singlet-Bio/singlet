#!/bin/bash
# Kang genotype check of the called_ed extras (kang_genotype.py): cellsnp-lite on the reference SNP sites + vireo with known
# donor genotypes. Needs $WORK/state/called_ed_truth/datasets/kang.barcodes.tsv.gz (run.sbatch MODE=datasets task 0).
#SBATCH --job-name=cv2t-geno
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
J=$W/jobs/called_ed_truth
G=$W/modality-bench/genotype_pool
O=$W/state/called_ed_truth/kang_genotype
E=$W/envs/gpool
PY=$E/venv/bin/python
mkdir -p $O
$PY -P $J/kang_genotype.py prep $G/singlet_modfix $W/state/called_ed_truth/datasets/kang.barcodes.tsv.gz $O
rm -rf $O/cellsnp $O/cellsnp_aug $O/vireo
apptainer exec --bind $W $E/cellsnp-lite.sif cellsnp-lite -s $G/singlet_k8/possorted.bam -b $O/barcodes.txt -O $O/cellsnp \
  -R $G/ref_k8/cellsnp/cellSNP.base.vcf.gz -p 16 --minMAF 0 --minCOUNT 0 --gzip --cellTAG CB --UMItag UB > $O/cellsnp.log 2>&1
tail -2 $O/cellsnp.log
# cellsnp --gzip compresses the mtx files too; decompress for scipy/vireo symmetry
for f in $O/cellsnp/*.mtx.gz $O/cellsnp/cellSNP.samples.tsv.gz; do [ -e "$f" ] && gunzip -f "$f"; done
ls $O/cellsnp
$PY -P $J/kang_genotype.py synth $O
$E/venv/bin/vireo -c $O/cellsnp_aug -d $G/ref_k8/vireo/GT_donors.vireo.vcf.gz -t GT -o $O/vireo -p 16 --randSeed 42 > $O/vireo.log 2>&1
tail -3 $O/vireo.log
$PY -P $J/kang_genotype.py summ $O $G/singlet_modfix/donor_assignments.tsv
