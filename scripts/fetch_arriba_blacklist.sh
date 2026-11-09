#!/bin/bash
# Fetch Arriba's published GRCh38 blacklist (Arriba v2.5.1, MIT license: data/fusion/arriba/LICENSE) for the chimeric table.
# The file is 55 MB, so it is fetched from the Arriba release instead of being kept in git; the checksum pins the exact bytes.
#   scripts/fetch_arriba_blacklist.sh <ref-base>/pileup_indices/fusion
# Citation: Uhrig S, Ellermann J, Walther T, et al. Accurate and efficient detection of gene fusions from RNA sequencing data.
#           Genome Research 31:448-460 (2021). doi:10.1101/gr.257246.119
set -euo pipefail
OUT=${1:?output directory}
NAME=blacklist_hg38_GRCh38_v2.5.1.tsv.gz
SHA=be45975b72c8fb07b717e8b7086f6708c219e9f57c905716a71c0c0b8bc43217
URL=https://github.com/suhrig/arriba/releases/download/v2.5.1/arriba_v2.5.1.tar.gz
mkdir -p "$OUT"
if [ -s "$OUT/$NAME" ] && echo "$SHA  $OUT/$NAME" | sha256sum -c --quiet; then echo "present: $OUT/$NAME"; exit 0; fi
curl -sSfL "$URL" | tar -xzO "arriba_v2.5.1/database/$NAME" > "$OUT/$NAME.part"
echo "$SHA  $OUT/$NAME.part" | sha256sum -c --quiet
mv "$OUT/$NAME.part" "$OUT/$NAME"
echo "fetched: $OUT/$NAME"
