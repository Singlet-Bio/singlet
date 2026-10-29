#!/usr/bin/env bash
# usage: tools/idea2_batch.sh logfile sample member cfg1 cfg2 ...   (cfg = "ENV=val ENV=val")
cd "$(dirname "$0")/.."
log="$1"; sample="$2"; member="$3"; shift 3
for cfg in "$@"; do tools/idea2_mrun.sh "$cfg" $sample $member >> "$log" 2>&1; done
echo "BATCH-DONE" >> "$log"
