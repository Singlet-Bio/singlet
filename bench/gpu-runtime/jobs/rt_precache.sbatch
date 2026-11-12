#!/bin/bash
#SBATCH -A debruinz -p short-gpu -w g051 -c 4 --mem=320G --time=00:30:00 -J rt-precache -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# rt-precache: fills the node's page cache with TARGET GB of file pages (reads large input files to /dev/null), so that the rt-node job
# that follows starts on a node whose free memory is mostly page cache, as a node that has processed samples for hours is. The pages
# stay cached after this job ends. Job 812144 (v1.3.8, 4 in flight per GPU) stalled in direct compaction on a node that started with
# 224 GB of page cache; job 812150 ran clean from 135 GB with SINGLET_NOINIT_HUGEPAGE=0: this job removes that difference.
W=/mnt/projects/debruinz_project/debruinz/singlet; TARGET=${TARGET:-225}
fc() { awk '/^Cached/{c=$2} /^Shmem:/{s=$2} END{printf "%d", (c-s)/1048576}' /proc/meminfo; }
echo "node=$(hostname) target=$TARGET GB; before: file cache $(fc) GB"; free -g | head -2
t=$(date +%s)
for f in $(ls -S $W/data/gpu-deep/*/*.1fq $W/pipeline-benchmark/data/*/*.1fq 2>/dev/null); do
  [ $(fc) -ge $TARGET ] && break
  cat "$f" > /dev/null
done
echo "after $(( $(date +%s) - t )) s: file cache $(fc) GB"; free -g | head -2
