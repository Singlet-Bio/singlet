#!/bin/bash
#SBATCH --job-name=mtl-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch fetch.sbatch <SRR> <name> <sra|ena>
# MAESTER (Miller et al. 2022, GSE182685). 10x 3' scRNA runs carry R1 (CB+UMI) and I1 as technical reads that ENA
# leaves out, so they come from SRA (fasterq-dump --include-technical); MAESTER runs come from ENA (R1 + R2 present).
# Output: $W/data/maester/<name>/<name>_R1.fastq.gz, <name>_R2.fastq.gz
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$W/envs/sratools/bin:$PATH
SRR=$1; NAME=$2; SRC=$3
D=$W/data/maester/$NAME; mkdir -p $D; cd $D
[ -e done ] && { echo "already done"; ls -la; exit 0; }
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
if [ "$SRC" = sra ]; then
  export HOME=$SCR   # vdb-config writes to $HOME
  prefetch --max-size 200G -O $SCR $SRR
  fasterq-dump --split-files --include-technical -e 16 -t $SCR/tmp -O $SCR $SCR/$SRR
  ls -la $SCR
  for i in 1 2 3; do head -2 $SCR/${SRR}_$i.fastq | tail -1 | awk -v i=$i '{print "read " i " length " length($0)}'; done
  # readTypes TBT: _1 = R1 (CB+UMI, 28 bp), _2 = R2 (cDNA), _3 = I1 (sample index, not needed)
  pigz -p 16 -c $SCR/${SRR}_1.fastq > ${NAME}_R1.fastq.gz
  pigz -p 16 -c $SCR/${SRR}_2.fastq > ${NAME}_R2.fastq.gz
else
  B=https://ftp.sra.ebi.ac.uk/vol1/fastq/${SRR:0:6}/0${SRR: -2}/$SRR
  for r in 1 2; do
    U=$B/${SRR}_$r.fastq.gz
    SZ=$(curl -sIL $U | awk 'tolower($1)=="content-length:"{s=$2+0} END{print s}'); N=16; CH=$(( (SZ+N-1)/N ))
    echo "$U size=$SZ"; mkdir -p $SCR/parts
    for i in $(seq 0 $((N-1))); do
      a=$((i*CH)); b=$((a+CH-1)); [ $b -ge $SZ ] && b=$((SZ-1))
      ( until [ "$(stat -c %s $SCR/parts/p$i 2>/dev/null || echo 0)" = "$((b-a+1))" ]; do rm -f $SCR/parts/p$i; curl -sL --retry 3 -r $a-$b -o $SCR/parts/p$i $U || sleep 10; done ) &
    done; wait
    cat $(for i in $(seq 0 $((N-1))); do echo $SCR/parts/p$i; done) > ${NAME}_R$r.fastq.gz; rm -rf $SCR/parts
    gzip -t ${NAME}_R$r.fastq.gz 2>/dev/null && echo "R$r gzip ok" || echo "R$r gzip FAILED"
  done
fi
set +o pipefail
for r in 1 2; do zcat ${NAME}_R$r.fastq.gz | head -2 | tail -1 | awk -v r=$r '{print "R" r " length " length($0)}'; done
touch done; ls -la
