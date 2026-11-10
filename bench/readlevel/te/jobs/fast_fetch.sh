#!/bin/bash
# Faster B7 fetches inside the running oracles1 allocation (srun --overlap): EBI serves ~0.7 MB/s per connection, so fetch in
# 16 parallel byte ranges. mESC R2 in full (then replaces the slow resume and marks mesc_2c DONE); A375 nanopore: the first
# ~6.5 GB of the ENA FASTQ (te_bench uses the first 5 M reads only), cut at a whole record.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/data/readlevel/te; N=${NCONN:-16}
pget() {   # URL OUT TOTAL_BYTES
  local url=$1 out=$2 tot=$3 step=$(( ($3 + N - 1) / N )) pids=() i
  for i in $(seq 0 $((N-1))); do
    local s=$((i*step)) e=$(( (i+1)*step - 1 )); [ $e -ge $tot ] && e=$((tot-1))
    ( for t in 1 2 3 4 5; do
        have=$(stat -c %s $out.p$i 2>/dev/null || echo 0); want=$((e-s+1)); [ $have -ge $want ] && exit 0
        curl -sS --retry 5 -r $((s+have))-$e "$url" >> $out.p$i && continue
        sleep 5
      done; [ $(stat -c %s $out.p$i) -eq $((e-s+1)) ] ) & pids+=($!)
  done
  local rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done; [ $rc = 0 ] || return 1
  cat $(for i in $(seq 0 $((N-1))); do echo $out.p$i; done) > $out.tmp && rm -f $out.p* && mv $out.tmp $out
}
echo "start $(date -Is)"
# mESC R2 (10,917,220,724 bytes per ENA)
M=$D/mesc_2c; R2URL=http://ftp.sra.ebi.ac.uk/vol1/fastq/ERR398/006/ERR3981386/ERR3981386_2.fastq.gz
if [ ! -f $M/DONE ]; then
  pget $R2URL $M/par_R2.fastq.gz 10917220724 || { echo "mesc R2 failed"; exit 1; }
  gzip -t $M/par_R2.fastq.gz && echo "mesc R2 gzip ok" || { echo "mesc R2 gzip BAD"; exit 1; }
  pkill -u $USER -f "ERR3981386_2.fastq.gz" ; sleep 2
  mv -f $M/par_R2.fastq.gz $M/ERR3981386_2.fastq.gz && ls -l $M > $M/DONE; echo "mesc_2c DONE $(date -Is)"
fi
# A375 nanopore head (SRR34318914, 69 GB in total)
O=$D/a375_ont; mkdir -p $O
if [ ! -s $O/SRR34318914_head5M.fastq.gz ]; then
  OURL=http://ftp.sra.ebi.ac.uk/vol1/fastq/SRR343/014/SRR34318914/SRR34318914_1.fastq.gz
  pget $OURL $O/head.part.gz 6500000000 || { echo "ont failed"; exit 1; }
  ( zcat $O/head.part.gz 2>/dev/null | head -n 20000000 | gzip -1 > $O/SRR34318914_head5M.fastq.gz.tmp ) ; rm -f $O/head.part.gz
  n=$(zcat $O/SRR34318914_head5M.fastq.gz.tmp | wc -l); echo "ont lines $n"
  [ $n -eq 20000000 ] && mv $O/SRR34318914_head5M.fastq.gz.tmp $O/SRR34318914_head5M.fastq.gz && ls -l $O > $O/DONE
fi
echo "end $(date -Is)"
