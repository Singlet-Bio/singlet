#!/bin/bash
# t2-cgsamp.sh <out.tsv> [markers.tsv]: once a second, the job cgroup's memory and the node's, beside a gpu-pool (pooladmit task, throughput track 2).
# The job-level cgroup is found as avail_gb() in gpu_pool.cpp does it (the ancestor named job_<id> of /proc/self/cgroup).
# out.tsv columns: epoch(s, ms)  memory.max(B, "max" if none)  memory.current(B)  anon(B)  file(B)  shmem(B)  MemFree(kB)  Cached(kB)  Shmem(kB)  events(max oom oom_kill)  pressure_full_avg10
# header line: "# cg=<path>". Optional second argument: markers.tsv gets one line "epoch item marker" the first second a pool item's log
# ($POOLDIR/<item>.log, items 0..MARK_N-1) shows "[engine_gpu] level " (marker "level") or "[allsite] async: waited " (marker "waited").
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
OUT=$1; MK=${2:-}
echo "# cg=$CG" > "$OUT"
declare -A seenL seenW
while :; do
  t=$EPOCHREALTIME; t=${t/,/.}
  mx=$(cat $CG/memory.max 2>/dev/null); cu=$(cat $CG/memory.current 2>/dev/null)
  st=$(awk '$1=="anon"{a=$2} $1=="file"{f=$2} $1=="shmem"{s=$2} END{printf "%s\t%s\t%s", a, f, s}' $CG/memory.stat 2>/dev/null)
  mi=$(awk '$1=="MemFree:"{f=$2} $1=="Cached:"{c=$2} $1=="Shmem:"{s=$2} END{printf "%s\t%s\t%s", f, c, s}' /proc/meminfo)
  ev=$(awk '$1=="max"{m=$2} $1=="oom"{o=$2} $1=="oom_kill"{k=$2} END{printf "%s %s %s", m, o, k}' $CG/memory.events 2>/dev/null)
  pr=$(awk '/^full/{sub("avg10=","",$2); print $2}' $CG/memory.pressure 2>/dev/null)
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "$mx" "$cu" "$st" "$mi" "$ev" "$pr" >> "$OUT"
  if [ -n "$MK" ] && [ -n "${POOLDIR:-}" ]; then
    for ((i = 0; i < ${MARK_N:-0}; i++)); do
      f=$POOLDIR/$i.log; [ -s "$f" ] || continue
      if [ -z "${seenL[$i]:-}" ] && grep -a -q -F '[engine_gpu] level ' "$f"; then seenL[$i]=1; echo -e "$t\t$i\tlevel" >> "$MK"; fi
      if [ -z "${seenW[$i]:-}" ] && grep -a -q -F '[allsite] async: waited ' "$f"; then seenW[$i]=1; echo -e "$t\t$i\twaited" >> "$MK"; fi
    done
  fi
  sleep 1
done
