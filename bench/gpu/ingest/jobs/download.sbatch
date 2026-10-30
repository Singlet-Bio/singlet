#!/bin/bash
#SBATCH --job-name=w11-ing-dl3
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:50:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only (no GPU gres). Read-only public access: NCBI S3 (sra-pub-run-odp and the SRA Lite objects), and the existing `singlet download`
# (SRA -> .1fq). Nothing is written to Cloudflare; reads stay in job /tmp. (Earlier versions 789879 and 789903 hung in `wait`, which also
# waited for the tee process substitution; this one has neither.)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/jobs/w11/ingest; OUT=$D/results/dl3_$SLURM_JOB_ID; mkdir -p $OUT
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
echo "== node $(hostname) $(date -Is)"; uptime
SW=$W/pipeline-benchmark/bin/singlet-w3
echo "== S3 ranged-read scaling (40 s windows, SRR21102188 .sra, bytes actually received)"
URL=https://sra-pub-run-odp.s3.amazonaws.com/sra/SRR21102188/SRR21102188
for N in 1 4 8; do
  t0=$(date +%s.%N); pids=""
  for i in $(seq 0 $((N-1))); do curl -s --max-time 40 -r $((300000000 + i*200000000))-$((300000000 + i*200000000 + 199999999)) -o /dev/null -w "%{size_download}\n" "$URL" > $S/n$i.txt & pids="$pids $!"; done
  wait $pids
  t1=$(date +%s.%N); tot=$(cat $S/n*.txt | awk '{s+=$1} END{print s}'); rm -f $S/n*.txt
  echo "NET streams=$N bytes=$tot secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$tot / ($t1 - $t0) / 1000000" | bc -l)"
done
echo "== SRA Lite: fetch SRR21102188.lite (2.17 GB) with 8 ranged streams, then compare the reads it yields with the full .sra"
python3 - > $S/lite_url.txt <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=SRR21102188&filetype=run&accept-alternate-locations=yes"))
for f in d["result"][0]["files"]:
    if f["name"].endswith(".lite"):
        for l in f["locations"]:
            if l["service"] == "s3": print(l["link"], f["size"]); raise SystemExit
PY
cat $S/lite_url.txt
LURL=$(cut -d' ' -f1 $S/lite_url.txt); LSZ=$(cut -d' ' -f2 $S/lite_url.txt)
mkdir -p $S/lite; t0=$(date +%s.%N); P=$((LSZ/8+1)); pids=""
for i in 0 1 2 3 4 5 6 7; do curl -s --retry 3 -r $((i*P))-$((i*P+P-1)) -o $S/lite/part$i "$LURL" & pids="$pids $!"; done; wait $pids
t1=$(date +%s.%N); cat $S/lite/part? > $S/lite/SRR21102188.sra; rm -f $S/lite/part?
echo "LITE_DL bytes=$(stat -c %s $S/lite/SRR21102188.sra) expect=$LSZ secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$LSZ / ($t1 - $t0) / 1000000" | bc -l)"
/usr/bin/time -v $SW download $S/lite/SRR21102188.sra -o $S/lite.1fq --vdb-threads 4 --max-reads 4000000 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/lite_time.txt
grep -E "Elapsed|User time|System time" $OUT/lite_time.txt | cut -c1-200
/usr/bin/time -v $SW download SRR21102188 -o $S/full_acc.1fq --vdb-threads 4 --max-reads 4000000 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/fullacc_time.txt
grep -E "Elapsed|User time|System time" $OUT/fullacc_time.txt | cut -c1-200
ls -l $S/lite.1fq $S/full_acc.1fq
for f in lite full_acc; do $SW decode $S/$f.1fq --r1 $S/$f.r1 --r2 $S/$f.r2 --threads 4 2>/dev/null; echo "MD5 $f r1=$(md5sum < $S/$f.r1 | cut -c1-32) r2=$(md5sum < $S/$f.r2 | cut -c1-32) bytes=$(stat -c %s $S/$f.r2)"; done
rm -rf $S/lite $S/*.r1 $S/*.r2 $S/lite.1fq $S/full_acc.1fq
echo "== singlet download SRR21102187 FULL (61.1 M spots, 2.80 GB .sra); timeout 25 min"
t0=$(date +%s.%N)
timeout 1500 /usr/bin/time -v $SW download SRR21102187 -o $S/full.1fq --vdb-threads 4 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/download_full_time.txt
echo "full download rc=$? wall=$(echo "$(date +%s.%N) - $t0" | bc)"
grep -E "Elapsed|Maximum resident|User time|System time|Percent of CPU|Done|Output|Protocol|Reads" $OUT/download_full_time.txt | cut -c1-300; ls -l $S/full.1fq
echo "== done $(date -Is)"
