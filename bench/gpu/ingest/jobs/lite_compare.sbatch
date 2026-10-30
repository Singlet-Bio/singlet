#!/bin/bash
#SBATCH --job-name=w11-ing-litecmp
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: SRA Lite (served anonymously by NCBI's own host; the S3 and GCS copies are requester-pays): download speed, and whether the
# reads it yields (first 4 M spots of SRR21102188) equal those from the full-quality .sra (same flags as job 789920's full_acc run).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
D=$W/jobs/w11/ingest; OUT=$D/results/litecmp_$SLURM_JOB_ID; mkdir -p $OUT
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
SW=$W/pipeline-benchmark/bin/singlet-w3
echo "== node $(hostname) $(date -Is)"
python3 - > $S/lite_url.txt <<'PY'
import json, urllib.request
d = json.load(urllib.request.urlopen("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=SRR21102188&filetype=run&accept-alternate-locations=yes"))
for f in d["result"][0]["files"]:
    if f["name"].endswith(".lite"):
        for l in f["locations"]:
            if l["service"] == "ncbi": print(l["link"], f["size"]); raise SystemExit
PY
cat $S/lite_url.txt
LURL=$(cut -d' ' -f1 $S/lite_url.txt); LSZ=$(cut -d' ' -f2 $S/lite_url.txt)
for N in 1 8; do
  mkdir -p $S/lite; t0=$(date +%s.%N); P=$((LSZ/N+1)); pids=""
  for i in $(seq 0 $((N-1))); do curl -s --retry 3 -r $((i*P))-$((i*P+P-1)) -o $S/lite/part$i "$LURL" & pids="$pids $!"; done; wait $pids
  t1=$(date +%s.%N); cat $S/lite/part* > $S/lite/SRR21102188.sra; rm -f $S/lite/part*
  echo "LITE_DL streams=$N bytes=$(stat -c %s $S/lite/SRR21102188.sra) expect=$LSZ secs=$(echo "$t1 - $t0" | bc) MBps=$(echo "$LSZ / ($t1 - $t0) / 1000000" | bc -l)"
  [ $N = 1 ] && rm -f $S/lite/SRR21102188.sra
done
/usr/bin/time -v $SW download $S/lite/SRR21102188.sra -o $S/lite.1fq --vdb-threads 4 --max-reads 4000000 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/lite_time.txt
grep -E "Elapsed|User time|System time|Output|Done" $OUT/lite_time.txt | cut -c1-200
ls -l $S/lite.1fq
$SW decode $S/lite.1fq --r1 $S/lite.r1 --r2 $S/lite.r2 --threads 4 2>/dev/null
echo "MD5 lite r1=$(md5sum < $S/lite.r1 | cut -c1-32) r2=$(md5sum < $S/lite.r2 | cut -c1-32) bytes=$(stat -c %s $S/lite.r2)"
echo "REFERENCE (job 789920, same run, full-quality .sra via accession): r1=c83c80479b30c89385c346a36d5f5d00 r2=d1719db905a61836643834df84257fbd bytes=774888890"
echo "== done $(date -Is)"
