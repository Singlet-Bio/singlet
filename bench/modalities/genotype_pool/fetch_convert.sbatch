#!/bin/bash
#SBATCH --job-name=gp-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Kang 2018 (GSE96583) batch 2 control lane "2.1" (SRR5398238): download the submitted Cell Ranger 1.0 BAM,
# rebuild raw R1 (CR+UR) and R2 (read sequence) FASTQs from primary alignments, encode to .1fq.
# Also fetch the truth table, the 1000G SNP list (AF>=5%, hg38) used by cellsnp-lite and Singlet.
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
D=$W/modality-bench/genotype_pool; mkdir -p $D/truth $D/fastq $D/ref; cd $D
U=https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398238/bam/2.1.bam
[ -s ref/x ] || curl -sL --retry 5 -o ref/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz \
  "https://sourceforge.net/projects/cellsnp/files/SNPlist/genome1K.phase3.SNP_AF5e2.chr1toX.hg38.vcf.gz/download"
ls -la ref
if [ ! -e 2.1.bam.done ]; then  # single EBI connection gives ~2 MB/s; fetch 24 byte ranges in parallel
  SZ=$(curl -sIL $U | awk 'tolower($1)=="content-length:"{s=$2+0} END{print s}'); N=24; CH=$(( (SZ+N-1)/N ))
  echo "size=$SZ chunk=$CH"; mkdir -p parts
  for i in $(seq 0 $((N-1))); do
    a=$((i*CH)); b=$((a+CH-1)); [ $b -ge $SZ ] && b=$((SZ-1))
    ( until [ "$(stat -c %s parts/p$i 2>/dev/null || echo 0)" = "$((b-a+1))" ]; do rm -f parts/p$i; curl -sL --retry 3 -r $a-$b -o parts/p$i $U || sleep 10; done ) &
  done; wait
  cat $(for i in $(seq 0 $((N-1))); do echo parts/p$i; done) > 2.1.bam && rm -rf parts && touch 2.1.bam.done
fi
ls -la 2.1.bam
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
# samtools fastq restores the original read orientation and prints the requested tags on the name line
samtools fastq -F 0x900 -n -T CR,CQ,UR,UQ -0 /dev/stdout -s /dev/null -@ 4 2.1.bam | awk -F'\t' -v r1=$SCR/R1.fq -v r2=$SCR/R2.fq '
 NR%4==1 { ok=0; cr=cq=ur=uq=""; for(i=2;i<=NF;i++){t=substr($i,1,2); v=substr($i,6); if(t=="CR")cr=v; else if(t=="CQ")cq=v; else if(t=="UR")ur=v; else if(t=="UQ")uq=v}
           ok=(cr!=""&&ur!=""); name=$1 }
 NR%4==2 { sq=$0 }
 NR%4==0 { if(ok){ print name"\n"cr ur"\n+\n"cq uq > r1; print name"\n"sq"\n+\n"$0 > r2 } }'
wc -l $SCR/R1.fq | head -1
head -4 $SCR/R1.fq
pigz -p 8 -c $SCR/R1.fq > fastq/R1.fastq.gz; pigz -p 8 -c $SCR/R2.fq > fastq/R2.fastq.gz
ls -la fastq
/usr/bin/time -v -o encode.time.txt $W/pipeline-benchmark/bin/singlet-fixed-det encode --reads fastq/R1.fastq.gz fastq/R2.fastq.gz -o kang21.1fq -v > encode.log 2>&1 || true
tail -30 encode.log; ls -la kang21.1fq; grep -E "Elapsed|Maximum" encode.time.txt
