#!/bin/bash
# Singlet on a plate protocol (Smart-seq2): one sample per cell, so one encode + run per SRA run, sequentially in one task.
# STAR keeps the genome in shared memory across cells (--genomeLoad LoadAndKeep through Singlet's STAR pass-through is not
# assumed; the per-cell runs simply reuse the page cache). Output: <O>/cells/<run>/ plus time.txt for the whole loop.
#   tools/singlet-plate.sh <dataset> <outdir>        (exec'd by run.sbatch with common.sh sourced)
set -euo pipefail
ID=$1; O=$2; D=$BP/data/$ID
mkdir -p $O/cells
/usr/bin/time -v -o $O/time.txt bash -c '
  for r in $(cut -f3 '"$D"'/libs.tsv | xargs -n1 dirname | sort -u); do
    run=$(basename $r); out='"$O"'/cells/$run; mkdir -p $out
    reads=$(ls $r/*_R1_*.fastq.gz $r/*_R2_*.fastq.gz 2>/dev/null)
    '"$SINGLET_BIN"' encode --reads $reads -o '"$SCR"'/$run.1fq > $out/encode.log 2>&1
    '"$SINGLET_BIN"' '"$SCR"'/$run.1fq --ref-base '"$REF"' --genome-dir '"$GENOME"' --exons '"$GTF"' \
      --out-prefix $out/ --threads '"$T"' --pipeline --output-format mtx > $out/run.log 2>&1 || echo "$run failed" >> '"$O"'/failed.txt
    rm -f '"$SCR"'/$run.1fq
  done'
ls $O/cells | wc -l
