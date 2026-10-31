#!/bin/bash
#SBATCH -J nh-fetch
#SBATCH -A debruinz
#SBATCH -p short-bigmem
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Gaublomme 2019 (Nat Commun 10:2907) mouse nucleus-hashing library: RNA SRR8703773 + hashtag SRR8703774 (10x 3'). Fetch inside the job.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/expression-sota/data/nucleus_hash; mkdir -p $D; cd $D
export PATH=$W/envs/sratools/bin:$PATH HOME=$W/cache/condahome
for r in SRR8703773 SRR8703774; do
  prefetch -q --max-size 200G -O $D $r
  fasterq-dump -q --split-files --include-technical -e 8 -O $D $D/$r/$r.sra 2>&1 | tail -2
  ls -la $D | grep $r; head -c 400 $D/${r}_1.fastq | head -2; for f in $D/${r}_*.fastq; do echo $f $(awk 'NR==2{print length($0); exit}' $f); done
  for f in $D/${r}_*.fastq; do gzip -f $f; done
done
ls -la $D
