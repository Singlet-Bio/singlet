#!/bin/bash
#SBATCH --job-name=hto-fetch
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Independent HTO set I1: Stoeckius et al. 2018 (GEO GSE108313, public data, cite the paper).
# RNA reads of the PBMC 8-donor hashing library (SRR8281306, ENA): the first 28% of the compressed bytes of each mate (about 200 M
# read pairs of 788 M) by 16 parallel Range requests (one ENA connection gives ~2 MB/s), then cut at 200 M records.
# The authors' deposited tables come from geo_fetch.sbatch (Range requests as well).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH
D=$W/expression-sota/hto/data/stoeckius2018; mkdir -p $D/fastq/parts; cd $D/fastq
E=https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR828/006/SRR8281306
N=200000000; NP=16
for r in 1 2; do
  U=$E/SRR8281306_$r.fastq.gz
  SZ=$(curl -sIL $U | awk 'tolower($1)=="content-length:"{s=$2+0} END{print s}'); PRE=$(( SZ * 28 / 100 )); CH=$(( (PRE+NP-1)/NP ))
  echo "R$r size=$SZ prefix=$PRE chunk=$CH"
  for i in $(seq 0 $((NP-1))); do
    a=$((i*CH)); b=$((a+CH-1)); [ $b -ge $PRE ] && b=$((PRE-1))
    ( until [ "$(stat -c %s parts/r${r}_$i 2>/dev/null || echo 0)" = "$((b-a+1))" ]; do rm -f parts/r${r}_$i; curl -sL --retry 3 -r $a-$b -o parts/r${r}_$i $U || sleep 10; done ) &
  done
done
wait
for r in 1 2; do
  cat $(for i in $(seq 0 $((NP-1))); do echo parts/r${r}_$i; done) | pigz -dc 2>/dev/null | head -n $((4*N)) | pigz -p 4 > rna_R$r.fastq.gz
done
rm -rf parts; ls -l
for r in 1 2; do echo "R$r records: $(( $(pigz -dc rna_R$r.fastq.gz | wc -l) / 4 ))"; pigz -dc rna_R$r.fastq.gz | head -4; done
