#!/bin/bash
#SBATCH --job-name=bpg-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:55:00
#SBATCH --array=0-3
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%A_%a.out
# Kang 2018 (GSE96583) lanes from the submitted Cell Ranger BAMs (EBI): rebuild raw R1 (CR+UR) / R2 FASTQs from primary
# alignments (same method as bench/modalities/genotype_pool/fetch_convert.sbatch), encode to .1fq.
# Lanes: batch1 A (4 donors), B (4), C (8, the donors of A and B), batch2 stim 2.2 (same 8 donors as the dev lane 2.1).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
LANES=(kang_b1A kang_b1B kang_b1C kang_b2stim)
URLS=(https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398235/bam/A.merged.bam
      https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398236/bam/B.merged.bam
      https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398237/C.merged.bam
      https://ftp.sra.ebi.ac.uk/vol1/run/SRR539/SRR5398239/bam/2.2.bam)
L=${LANES[$SLURM_ARRAY_TASK_ID]}; U=${URLS[$SLURM_ARRAY_TASK_ID]}
D=$W/bench-public/data/$L; mkdir -p $D/fastq; cd $D
if [ ! -e bam.done ]; then
  SZ=$(curl -sIL $U | awk 'tolower($1)=="content-length:"{s=$2+0} END{print s}'); N=24; CH=$(( (SZ+N-1)/N ))
  echo "size=$SZ chunk=$CH"; mkdir -p parts
  for i in $(seq 0 $((N-1))); do
    a=$((i*CH)); b=$((a+CH-1)); [ $b -ge $SZ ] && b=$((SZ-1))
    ( until [ "$(stat -c %s parts/p$i 2>/dev/null || echo 0)" = "$((b-a+1))" ]; do rm -f parts/p$i; curl -sL --retry 3 -r $a-$b -o parts/p$i $U || sleep 10; done ) &
  done; wait
  cat $(for i in $(seq 0 $((N-1))); do echo parts/p$i; done) > lane.bam && rm -rf parts && touch bam.done
fi
ls -la lane.bam
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
samtools fastq -F 0x900 -n -T CR,CQ,UR,UQ -0 /dev/stdout -s /dev/null -@ 4 lane.bam | awk -F'\t' -v r1=$SCR/R1.fq -v r2=$SCR/R2.fq '
 NR%4==1 { ok=0; cr=cq=ur=uq=""; for(i=2;i<=NF;i++){t=substr($i,1,2); v=substr($i,6); if(t=="CR")cr=v; else if(t=="CQ")cq=v; else if(t=="UR")ur=v; else if(t=="UQ")uq=v}
           ok=(cr!=""&&ur!=""); name=$1 }
 NR%4==2 { sq=$0 }
 NR%4==0 { if(ok){ print name"\n"cr ur"\n+\n"cq uq > r1; print name"\n"sq"\n+\n"$0 > r2 } }'
wc -l $SCR/R1.fq | head -1
pigz -p 8 -c $SCR/R1.fq > fastq/R1.fastq.gz; pigz -p 8 -c $SCR/R2.fq > fastq/R2.fastq.gz
/usr/bin/time -v -o encode.time.txt $W/pipeline-benchmark/bin/singlet-campaign encode --reads fastq/R1.fastq.gz fastq/R2.fastq.gz -o $L.1fq -v > encode.log 2>&1 || true
tail -15 encode.log; ls -la
