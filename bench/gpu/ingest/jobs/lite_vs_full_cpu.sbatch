#!/bin/bash
#SBATCH --job-name=w11-ing-litecpu
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: decode cost of a LOCAL full-quality .sra versus the LOCAL SRA Lite of the same run (SRR21102188), first 4 M spots, same flags.
# Both objects are fetched from NCBI's own host with 8 ranged streams (anonymous, read-only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/jobs/w11/ingest; OUT=$D/results/litecpu_$SLURM_JOB_ID; mkdir -p $OUT
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
SW=$W/pipeline-benchmark/bin/singlet-w3
echo "== node $(hostname) $(date -Is)"
fetch() { # name url size
  mkdir -p $S/$1; local P=$(( $3/8+1 )) pids=""; local t0=$(date +%s.%N)
  for i in 0 1 2 3 4 5 6 7; do curl -s --retry 3 -r $((i*P))-$((i*P+P-1)) -o $S/$1/part$i "$2" & pids="$pids $!"; done; wait $pids
  cat $S/$1/part? > $S/$1.sra; rm -rf $S/$1; local t1=$(date +%s.%N)
  echo "FETCH $1 bytes=$(stat -c %s $S/$1.sra) expect=$3 secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$3 / ($t1 - $t0) / 1000000" | bc -l)"
}
python3 - > $S/urls.txt <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=SRR21102188&filetype=run&accept-alternate-locations=yes"))
for f in d["result"][0]["files"]:
    for l in f["locations"]:
        if l["service"] == ("ncbi" if f["name"].endswith(".lite") else "s3"): print("lite" if f["name"].endswith(".lite") else "full", l["link"], f["size"]); break
PY
cat $S/urls.txt
while read name url size; do fetch $name $url $size; done < $S/urls.txt
for name in full lite; do
  /usr/bin/time -v $SW download $S/$name.sra -o $S/$name.1fq --vdb-threads 4 --max-reads 4000000 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/${name}_time.txt
  echo "LOCAL_DECODE $name $(grep -E 'User time|System time|Elapsed' $OUT/${name}_time.txt | tr -s '\t' ' ' | tr '\n' ';')"
  grep -E "1fq-profile\]   (VDB read|Convert|Writer|Protocol)" $OUT/${name}_time.txt | cut -c1-120
  ls -l $S/$name.1fq
  $SW decode $S/$name.1fq --r1 /dev/stdout --r2 /dev/null --threads 4 2>/dev/null | md5sum | sed "s/^/MD5 $name r1 /"
  $SW decode $S/$name.1fq --r2 /dev/stdout --r1 /dev/null --threads 4 2>/dev/null | md5sum | sed "s/^/MD5 $name r2 /"
done
echo "== done $(date -Is)"
