#!/bin/bash
#SBATCH --job-name=w11-ing-cpupipe
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: the whole CPU-node stage for one public run, from the public source to the new format, in one job.
#   fetch SRA Lite (NCBI host, 8 ranged streams) -> `singlet download` of the local file (.1fq) -> enc_1fg (.1fg, zstd 3, 32K-pair chunks, CRC)
#   -> CPU-engine identity check (host decode of the .1fg vs the .1fq library decode) -> upload dry run.
# ACC defaults to SRR21102188 (60.7 M spots, 5' v2). Reads live only in job /tmp.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif
D=$W/jobs/w11/ingest; M8=$W/jobs/w11/m8; BIN=$D/bin2; ACC=${ACC:-SRR21102188}; LEVEL=${LEVEL:-3}
OUT=$D/results/cpupipe_${ACC}_$SLURM_JOB_ID; mkdir -p $OUT
S=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $S; trap 'rm -rf $S' EXIT
SW=$W/pipeline-benchmark/bin/singlet-w3
ctr() { apptainer exec -B $W --env LD_LIBRARY_PATH=$M8/lib $IMG "$@"; }
echo "== node $(hostname) $(date -Is) acc $ACC"
python3 - $ACC > $S/url.txt <<'PY'
import json, sys, urllib.request
d = json.load(urllib.request.urlopen("https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=%s&filetype=run&accept-alternate-locations=yes" % sys.argv[1]))
for f in d["result"][0]["files"]:
    if f["name"].endswith(".lite"):
        for l in f["locations"]:
            if l["service"] == "ncbi": print(l["link"], f["size"]); raise SystemExit
PY
LURL=$(cut -d' ' -f1 $S/url.txt); LSZ=$(cut -d' ' -f2 $S/url.txt); cat $S/url.txt
T0=$(date +%s.%N)
mkdir -p $S/p; P=$((LSZ/8+1)); pids=""
for i in 0 1 2 3 4 5 6 7; do curl -s --retry 3 -r $((i*P))-$((i*P+P-1)) -o $S/p/part$i "$LURL" & pids="$pids $!"; done; wait $pids
cat $S/p/part? > $S/$ACC.sra; rm -rf $S/p; T1=$(date +%s.%N)
echo "STAGE fetch_lite bytes=$(stat -c %s $S/$ACC.sra) secs=$(echo "$T1 - $T0" | bc) MBps=$(echo "$LSZ / ($T1 - $T0) / 1000000" | bc -l)"
/usr/bin/time -v $SW download $S/$ACC.sra -o $S/$ACC.1fq --vdb-threads 4 --whitelist-dir $W/pipeline-benchmark/whitelists 2> $OUT/download_time.txt
T2=$(date +%s.%N)
echo "STAGE decode_local_to_1fq secs=$(echo "$T2 - $T1" | bc) $(grep -E 'User time|System time' $OUT/download_time.txt | tr -s '\t' ' ' | tr '\n' ';') out_bytes=$(stat -c %s $S/$ACC.1fq)"
grep -E "1fq-profile\] (Total|  VDB read|  Convert|  Writer|  Protocol)|Done:" $OUT/download_time.txt | cut -c1-160
rm -f $S/$ACC.sra
ctr $BIN/enc_1fg --in $S/$ACC.1fq --out $S/$ACC.1fg --codec zstd --level $LEVEL --threads 4 --crc 1 --chunk-pairs 32768 | tee $OUT/enc.json | cut -c1-900
T3=$(date +%s.%N)
echo "STAGE encode_1fg secs=$(echo "$T3 - $T2" | bc) out_bytes=$(stat -c %s $S/$ACC.1fg)"
ctr $BIN/dec_1fg --fg $S/$ACC.1fg --fq $S/$ACC.1fq --threads 4 | tee $OUT/cpu_compat.json | cut -c1-400
python3 $D/src/upload_dryrun.py $S/$ACC.1fg --accession $ACC --bucket DRYRUN-NO-BUCKET --part-mib 64 --out $OUT/manifest_dryrun.json > /dev/null 2> $OUT/dryrun.stderr; cat $OUT/dryrun.stderr; python3 -c "
import json; m = json.load(open('$OUT/manifest_dryrun.json')); print('DRYRUN', m['object'], 'parts', m['upload']['parts'], 'network_calls', m['network_calls'])"
echo "== done $(date -Is)"
