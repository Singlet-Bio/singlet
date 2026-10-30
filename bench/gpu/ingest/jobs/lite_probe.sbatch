#!/bin/bash
#SBATCH --job-name=w11-ing-lite
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# CPU-only: where does SRA Lite live and can it be fetched anonymously? (read-only HEAD / small ranged GETs)
set -uo pipefail
curl -s "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=SRR21102188&filetype=run&accept-alternate-locations=yes" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d['result'][0]['files']:
    print('FILE', f['name'], f['size'])
    for l in f['locations']: print('  LOC', l.get('service'), l.get('region'), l.get('link'), l.get('payRequired'), l.get('ceRequired'))
"
for u in $(curl -s "https://locate.ncbi.nlm.nih.gov/sdl/2/retrieve?acc=SRR21102188&filetype=run&accept-alternate-locations=yes" | python3 -c "
import sys, json
d = json.load(sys.stdin)
for f in d['result'][0]['files']:
    if f['name'].endswith('.lite'):
        for l in f['locations']: print(l['link'])
"); do
  echo "== $u"; curl -s -r 0-199 -o /tmp/probe.bin -w "http=%{http_code} size=%{size_download} time=%{time_total}\n" "$u"; head -c 300 /tmp/probe.bin | strings | head -n 5
done
