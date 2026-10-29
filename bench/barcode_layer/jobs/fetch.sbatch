#!/bin/bash
#SBATCH --job-name=bl-fetch-@WHAT@
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Barcode-layer phase 1c prep, step 0: real read subsets for the STAR comparison; @WHAT@ is substituted per submission
# (pbmcR1, pbmcR2, dropseq, bd, splitseq, indrop). The first N spots of each run, read names rewritten to r<i>.
# 10x pbmc1k_v3 comes from the local copy of the public Cell Ranger FASTQs; the other classes are streamed from ENA (public) inside the job.
set -uo pipefail
WHAT=@WHAT@
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; D=$W/data/barcode-layer; mkdir -p $D
export PATH=$W/envs/pbench/bin:$PATH
trap 'kill $(jobs -p) 2>/dev/null' EXIT INT TERM
echo "node=$(hostname) what=$WHAT $(date -Is)"
rename() { awk 'NR%4==1{print "@r" (NR-1)/4; next} {print}'; }   # keeps +, seq, qual
ena() { # name srr nreads
  local name=$1 srr=$2 n=$3 num=${2#SRR} dir
  case ${#num} in 6) dir="";; 7) dir="/00${num: -1}";; 8) dir="/0${num: -2}";; *) dir="/${num: -3}";; esac
  local base=ftp.sra.ebi.ac.uk/vol1/fastq/SRR${num:0:3}$dir/$srr
  mkdir -p $D/$name
  for m in 1 2; do
    [ -s $D/$name/M$m.fq ] && continue
    set +o pipefail
    curl -sf --retry 6 --retry-delay 30 --retry-all-errors --max-time 1500 https://$base/${srr}_$m.fastq.gz | gzip -dc 2>/dev/null | head -n $((4*n)) | rename > $D/$name/M$m.fq.tmp
    set -o pipefail
    [ -s $D/$name/M$m.fq.tmp ] && mv $D/$name/M$m.fq.tmp $D/$name/M$m.fq || echo "FAILED $name mate $m ($base)"
  done
  echo "== $name $srr: $(( $(wc -l < $D/$name/M1.fq) / 4 )) / $(( $(wc -l < $D/$name/M2.fq) / 4 )) reads"
  for m in 1 2; do sed -n 2p $D/$name/M$m.fq; awk 'NR%4==2{print length($0)}' $D/$name/M$m.fq | head -3000 | sort -n | uniq -c | sort -k1nr | head -3; done
}
case $WHAT in
  pbmcR1|pbmcR2)
    m=${WHAT#pbmc}; N=12000000; mkdir -p $D/pbmc1k_v3
    gzip -dc $PB/data/pbmc1k_v3/pbmc_1k_v3_fastqs/pbmc_1k_v3_S1_L001_${m}_001.fastq.gz 2>/dev/null | head -n $((4*N)) | rename > $D/pbmc1k_v3/$m.fq.tmp
    mv $D/pbmc1k_v3/$m.fq.tmp $D/pbmc1k_v3/$m.fq; echo "pbmc1k_v3 $m: $(( $(wc -l < $D/pbmc1k_v3/$m.fq) / 4 )) reads"; sed -n 1,4p $D/pbmc1k_v3/$m.fq;;
  dropseq)  ena dropseq  SRR5747459  6000000;;
  bd)       ena bd       SRR12364726 6000000;;
  splitseq) ena splitseq SRR9900121 6000000;;
  splitseq_swap) [ -e $D/splitseq/M1.fq ] && ! [ -e $D/splitseq_parse ] && mv $D/splitseq $D/splitseq_parse; ena splitseq SRR9900121 6000000;;
  indrop)   ena indrop   SRR15409581 900000;;
  ena_all)  rm -f $D/*/M?.fq.tmp; ena dropseq SRR5747459 6000000; ena bd SRR12364726 6000000; ena splitseq SRR31302021 6000000; ena indrop SRR15409581 900000;;
esac
ls -la $D/*; echo "done $(date -Is)"
