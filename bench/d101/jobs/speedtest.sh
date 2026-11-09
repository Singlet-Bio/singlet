#!/bin/bash
# 30 s throughput probes for one GSE120221 BAM from the node (run as a step inside the node job).
ENV=/mnt/projects/debruinz_project/debruinz/singlet/envs/pbench; export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
S=SRR7881407; B=possorted_genome_bam_H.bam; E=ftp.sra.ebi.ac.uk/vol1/run/SRR788/$S/bam/$B
T=$(mktemp -d /tmp/debruinz-speed-XXXX); trap 'rm -rf $T' EXIT
probe() { local n=$1 u=$2; timeout 30 curl -fsS -r 0-4000000000 -o $T/x $u 2>/dev/null; echo "$n $(( $(stat -c %s $T/x 2>/dev/null || echo 0) / 30000000 )) MB/s"; rm -f $T/x; }
echo "aria2c: $(command -v aria2c || echo none)"; echo "srapath: $(srapath $S 2>&1 | head -3 | tr '\n' ' ')"
probe ena-https https://$E
probe ena-ftp ftp://$E
for k in 1 2; do probe sra-pub-src-$k https://sra-pub-src-$k.s3.amazonaws.com/$S/$B.1; done
# 8 parallel ranges of 1 GB each, 30 s
for i in 0 1 2 3 4 5 6 7; do ( timeout 30 curl -fsS -r $((i*1000000000))-$((i*1000000000+999999999)) -o $T/p$i https://$E 2>/dev/null ) & done; wait
echo "ena-https-8ranges $(( $(cat $T/p* | wc -c) / 30000000 )) MB/s"
