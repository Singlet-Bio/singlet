#!/bin/bash
#SBATCH --job-name=mf-tso
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# TSO orientation in the cDNA read: count the 10x TSO (forward / reverse complement), the 5' R1 TSO tail and
# poly(A)/poly(T) runs in the first 2M reads of each read file of the 5' and a 3' benchmark sample.
PB=/mnt/projects/debruinz_project/debruinz/singlet/pipeline-benchmark
for ds in pbmc5p_v1 pbmc1k_v3; do
  for rd in _R1_ _R2_; do
    f=$(grep "$rd" $PB/data/$ds/gex_fastqs.txt | head -1)
    zcat $f | head -8000000 | awk -v ds=$ds -v rd=$rd 'NR%4==2{n++; len+=length($0);
      if(index($0,"AAGCAGTGGTATCAACGCAG"))f++; if(index($0,"CTGCGTTGATACCACTGCTT"))r++;
      if(index($0,"TTTCTTATATGGG"))t5++; if(index($0,"CCCATATAAGAAA"))t5r++;
      if(index($0,"AAAAAAAAAAAAAAAAAAAA"))a++; if(index($0,"TTTTTTTTTTTTTTTTTTTT"))t++}
      END{printf "%s %s reads=%d meanlen=%.1f TSOfwd20=%d TSOrc20=%d TSO13=%d TSO13rc=%d polyA20=%d polyT20=%d\n",ds,rd,n,len/n,f,r,t5,t5r,a,t}'
  done
done
