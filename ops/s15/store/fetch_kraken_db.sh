#!/bin/bash
# S15 store (Slurm job): the open Kraken2 PlusPF index (archaea, bacteria, viruses, plasmids, human, UniVec, fungi, protozoa) from the Ben Langmead index collection, k2_pluspf_20261022 (96 GB).
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/data/kraken_db; U=https://genome-idx.s3.amazonaws.com/kraken/k2_pluspf_20261022.tar.gz; mkdir -p $D/k2_pluspf_20261022; T=$D/k2_pluspf_20261022.tar.gz
SZ=$(curl -sIL $U | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}' | tail -1)
for try in $(seq 1 200); do [ "$(stat -c %s $T 2> /dev/null || echo 0)" = "$SZ" ] && break; curl -sL -C - --speed-time 120 --speed-limit 20000 --retry 5 --retry-delay 30 -o $T $U || sleep 20; done
[ "$(stat -c %s $T)" = "$SZ" ] || { echo "download incomplete"; exit 5; }
tar -xzf $T -C $D/k2_pluspf_20261022 && rm -f $T && ls -l $D/k2_pluspf_20261022 && date -I > $D/k2_pluspf_20261022/DOWNLOADED && echo "[kraken-db] ok"
