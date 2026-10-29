#!/bin/bash
#SBATCH --job-name=mtfix-pileup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=100G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Per-cell chrM allele counts with strand (MAPQ>=20 and any-MAPQ variants) from the tagged BAM of the Kang lane-2.1 run.
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modfix-mt; O=$D/mt_fix; S=$D/singlet_k8
mkdir -p $O
tail -n +2 $S/donor/donor_assignments.tsv | cut -f1 > $O/cells_all.txt
wc -l $O/cells_all.txt
/usr/bin/time -v -o $O/pileup.time.txt $W/envs/gpool/venv/bin/python $J/mt_pileup_strand.py $S/possorted.bam $O/cells_all.txt $O/pile $S/donor/donor_assignments.tsv
grep -E "Elapsed|Maximum res" $O/pileup.time.txt
ls -la $O
