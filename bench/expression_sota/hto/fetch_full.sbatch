#!/bin/bash
#SBATCH --job-name=hto-fetchfull
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:55:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Amendment 3: the whole SRR8281306 RNA library (788 M read pairs) for genotype truth (the first 200 M pairs give ~72 SNPs per cell,
# too few for vireo to see doublets). 24 parallel Range requests per mate; parts are concatenated (valid multi-member gzip).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/expression-sota/hto/data/stoeckius2018; mkdir -p $D/fastq_full/parts; cd $D/fastq_full
E=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR828/006/SRR8281306
NP=24
for r in 1 2; do
  U=$E/SRR8281306_$r.fastq.gz
  SZ=$(curl -sIL $U | awk 'tolower($1)=="content-length:"{s=$2+0} END{print s}'); CH=$(( (SZ+NP-1)/NP ))
  echo "R$r size=$SZ chunk=$CH"
  for i in $(seq 0 $((NP-1))); do
    a=$((i*CH)); b=$((a+CH-1)); [ $b -ge $SZ ] && b=$((SZ-1))
    ( until [ "$(stat -c %s parts/r${r}_$i 2>/dev/null || echo 0)" = "$((b-a+1))" ]; do rm -f parts/r${r}_$i; curl -sL --retry 3 -r $a-$b -o parts/r${r}_$i $U || sleep 10; done ) &
  done
done
wait
for r in 1 2; do cat $(for i in $(seq 0 $((NP-1))); do echo parts/r${r}_$i; done) > rna_R$r.fastq.gz; done
rm -rf parts; ls -l
