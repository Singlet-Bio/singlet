#!/usr/bin/env bash
# Final decode-timing pass under bench/compression/results/.timing_lock (protocol: bench/compression/AGENT_RULES.md).
# Waits for the lock (retry every 30 s, give up after 15 minutes; a lock older than 20 minutes is stale and removed), runs the grid, releases.
set -u
cd "$(dirname "$0")/../.."
LOCK=bench/compression/results/.timing_lock
R=bench/compression/results/m_w1
C=bench/compression/corpus
got=0
for i in $(seq 1 ${PZ2_LOCK_TRIES:-30}); do
  if mkdir "$LOCK" 2>/dev/null; then got=1; break; fi
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then rmdir "$LOCK" 2>/dev/null; continue; fi
  sleep 30
done
if [ "$got" != 1 ]; then echo "could not take the timing lock within the retry window" > $R/timing_final.status; exit 1; fi
echo "W1 final timing $(date)" > "$LOCK/owner.txt"
load() { powershell -NoProfile -Command "(Get-CimInstance Win32_Processor).LoadPercentage"; }
OUT=$R/timing_final.tsv
echo "# $(date) before: load=$(load)%" > $OUT
./bench/pz2/bin/bench_pz2.exe time --reps 7 --threads 1,2,4 $C/tcell_5p/exon_counts.1pz $C/brain_sn_v3_deep/exon_counts.1pz $C/pbmc_v2_typical/exon_counts.1pz >> $OUT 2>/dev/null
echo "# $(date) after: load=$(load)%" >> $OUT
: > $R/chunk_decode_speed.txt
for N in 128 256 512 1024 4096; do
  ./bench/pz2/bin/bench_pz2.exe time --reps 7 --isas avx512 --threads 1 --chunk $N $C/tcell_5p/exon_counts.1pz 2>/dev/null | awk -F'\t' -v n=$N 'NR>1 && $9=="native" {printf "chunk_cols %5s: native %s ms  %s Mnnz/s (core %s GHz)\n", n, $11, $13, $14}' >> $R/chunk_decode_speed.txt
done
echo "# $(date) end: load=$(load)%" >> $R/chunk_decode_speed.txt
rm -f "$LOCK/owner.txt"
rmdir "$LOCK"
echo done > $R/timing_final.status
