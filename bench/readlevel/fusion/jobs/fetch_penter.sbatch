#!/bin/bash
#SBATCH --job-name=fus-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Penter et al. 2024 (nanoranger), GSE243227 / PRJNA1017440: K562 and Kasumi-1 mixed at known ratios, 10x 5' (R1 28, R2 90).
# Four lanes per mix. fasterq-dump --include-technical keeps the barcode read (ENA serves R2 only): _1 = R1, _2 = R2, _3/_4 = I1/I2.
# Output: $D/<mix>/<mix>_S1_L00<lane>_{R1,R2}_001.fastq.gz (10x names, for singlet --fastq-dir and the comparators).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/sratools/bin:$PATH
D=$W/data/readlevel/penter2024; mkdir -p $D; cd $D
MIXES=${MIXES:-"mix1to1p5:SRR26063321,SRR26063322,SRR26063323,SRR26063324 mix1to15:SRR26063325,SRR26063326,SRR26063327,SRR26063328"}
for m in $MIXES; do
  name=${m%%:*}; runs=${m#*:}; mkdir -p $name; lane=0
  for r in ${runs//,/ }; do
    lane=$((lane+1))
    if [ ! -f $r/DONE ]; then
      mkdir -p $r; fasterq-dump --split-files --include-technical -e 8 -t /tmp/$USER-fq-$SLURM_JOB_ID -O $r $r || { echo "fasterq-dump $r failed"; exit 1; }
      for f in $r/*.fastq; do gzip -1 $f; done; ls -l $r > $r/DONE
    fi
    ln -sfn ../$r/${r}_1.fastq.gz $name/${name}_S1_L00${lane}_R1_001.fastq.gz
    ln -sfn ../$r/${r}_2.fastq.gz $name/${name}_S1_L00${lane}_R2_001.fastq.gz
  done
  ls -lL $name
done
