#!/bin/bash
#SBATCH --account=debruinz
#SBATCH --job-name=vn-indpa
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/sratools/bin:$W/envs/pbench/bin:$PATH
J=$W/jobs/validation-engine/non10x; R=$W/results/validation-engine/non10x/ab2/indrop_500k/res; D=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $D; trap 'rm -rf $D' EXIT
echo "== unmapped reads (500k run)"; python3 -I $J/indrop_polya.py $R/star_Unmapped.out.mate1
fastq-dump -X 300000 --split-files -O $D SRR3879601 2>&1 | tail -1
echo "== all cDNA reads (R2 of the first 300k spots)"; python3 -I $J/indrop_polya.py $D/SRR3879601_2.fastq
