#!/bin/bash
#SBATCH --job-name=mf-r2comp
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Per-position dominant-base share of the cDNA read (first 200k reads and 200k reads from the file middle) for the clip
# positive control, and the share of reads that start with its constant 20-mer.
D=/mnt/projects/debruinz_project/debruinz/singlet/modality-bench/clip_positive
for part in head mid; do
  if [ $part = head ]; then zcat $D/R2.fastq.gz | head -800000; else zcat $D/R2.fastq.gz | sed -n '8000001,8800000p'; fi | \
  awk -v part=$part 'NR%4==2{n++; if(substr($0,1,20)=="AGCTGCTGAAAACCACGCAC")p++; for(i=1;i<=60;i++){c[i,substr($0,i,1)]++}}
    END{printf "%s reads=%d prefix20=%.3f\n",part,n,p/n; for(i=1;i<=60;i+=1){m=0;b="";split("A C G T N",B," ");for(k=1;k<=5;k++){if(c[i,B[k]]>m){m=c[i,B[k]];b=B[k]}} printf "%d:%s%.2f ",i,b,m/n} print ""}'
done
zcat $D/R2.fastq.gz | head -400000 | awk 'NR%4==2' | cut -c1-12 | sort | uniq -c | sort -rn | head -8
