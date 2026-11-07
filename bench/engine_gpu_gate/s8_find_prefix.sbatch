#!/bin/bash
#SBATCH --job-name=s8-findpfx
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S8: does SRR34789664 carry a constant 5' prefix on the cDNA read anywhere? 20k spots at several offsets: most common 20-mer at read start and its share.
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; export HOME=$W/tools/home PATH=$W/envs/sratools/bin:$PATH
T=$W/modality-bench/sj_nonhost/ncbi_cache/tmp; mkdir -p $T; cd $T; SCR=/tmp/$USER-fp-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
for off in 1 5000000 15000000 30000000 60000000; do
  rm -rf $SCR/r; fastq-dump -N $off -X $((off+19999)) --split-files -O $SCR/r SRR34789664 > /dev/null 2>&1
  for f in $SCR/r/*.fastq; do L=$(awk 'NR==2{print length($0)}' $f); echo "off=$off $(basename $f) len=$L n=$(awk 'END{print NR/4}' $f)"; done
  f=$(ls -S $SCR/r/*.fastq | head -1)   # cDNA read = largest
  awk 'NR%4==2{print substr($0,1,20)}' $f | sort | uniq -c | sort -rn | head -3
done
