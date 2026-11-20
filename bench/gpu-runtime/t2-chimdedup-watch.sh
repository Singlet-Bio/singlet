#!/bin/bash
# t2-chimdedup-watch.sh <outdir-of-the-item> <results-dir> <label>   (chimdedup task, throughput track 2; from t2-chimpass-watch.sh)
# Beside one gpu-pool item run with --chimeric: once a second, from the moment the item's chimeric block shows its first file
# (<outdir>/chim_dedup.mate1 = stage A of the deduplicated pass, or <outdir>/chim_Log.out = the second STAR pass without dedup)
# until the caller kills it:
#   <results>/<label>.watch.tsv   epoch  state  pid  cpu_s(user+sys of the pass's STAR child, all threads)  nthreads  VmRSS_kB  VmHWM_kB  RssAnon_kB  RssFile_kB
#                                 junction_file_bytes  cg_cpu_usec  cg_anon_B  cg_shmem_B  cg_file_B  cg_current_B
#   <results>/<label>.events.tsv  epoch  event   (first sighting of: dedupfile, chim_Log.out (logout), chim_Chimeric.out.junction, chim_Log.final.out, child gone)
#   <results>/<label>.chim_Log.final.out (copy)
# Reads only the job's own files and /proc of the job's own processes. No file under a private/ directory is touched.
dd=$1; O=$2; L=$3
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
W=$O/$L.watch.tsv; E=$O/$L.events.tsv
echo -e "# epoch\tstate\tpid\tcpu_s\tnthreads\tvmrss_kB\tvmhwm_kB\trssanon_kB\trssfile_kB\tjunction_bytes\tcg_cpu_usec\tcg_anon_B\tcg_shmem_B\tcg_file_B\tcg_current_B" > $W
: > $E
pid=""; ev() { [ -e "$O/.ev.$L.$1" ] && return; : > "$O/.ev.$L.$1"; echo -e "$t\t$1" >> $E; }
hz=$(getconf CLK_TCK)
cp_i=0
while :; do
  t=$EPOCHREALTIME; t=${t/,/.}
  [ -e "$dd/chim_dedup.mate1" ] && ev dedupfile
  if [ -e "$dd/chim_Log.out" ]; then ev logout; fi
  if [ -e "$dd/chim_Log.out" ] || [ -e "$dd/chim_dedup.mate1" ]; then
    if [ -z "$pid" ] && [ -e "$dd/chim_Log.out" ]; then
      for p in $(find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u); do
        ls -l /proc/$p/fd 2>/dev/null | grep -q 'chim_Log.out' && { pid=$p; break; }
      done
    fi
    state=run; cpu=; nth=; rss=; hwm=; ran=; rfl=
    if [ -n "$pid" ]; then
      if [ -d /proc/$pid ]; then
        read -r cpu nth < <(sed 's/.*) //' /proc/$pid/stat 2>/dev/null | awk '{printf "%.2f %d", ($12+$13)/'$hz', $18}')
        rss=$(awk '$1=="VmRSS:"{print $2}' /proc/$pid/status 2>/dev/null); hwm=$(awk '$1=="VmHWM:"{print $2}' /proc/$pid/status 2>/dev/null)
        ran=$(awk '$1=="RssAnon:"{print $2}' /proc/$pid/status 2>/dev/null); rfl=$(awk '$1=="RssFile:"{print $2}' /proc/$pid/status 2>/dev/null)
      else
        state=gone; ev childgone
      fi
    fi
    jb=$(stat -c %s "$dd/chim_Chimeric.out.junction" 2>/dev/null); [ -n "$jb" ] && ev junction
    cu=$(awk '$1=="usage_usec"{print $2}' $CG/cpu.stat 2>/dev/null)
    st=$(awk '$1=="anon"{a=$2} $1=="file"{f=$2} $1=="shmem"{s=$2} END{printf "%s\t%s\t%s", a, s, f}' $CG/memory.stat 2>/dev/null)
    cur=$(cat $CG/memory.current 2>/dev/null)
    echo -e "$t\t$state\t$pid\t$cpu\t$nth\t$rss\t$hwm\t$ran\t$rfl\t$jb\t$cu\t$st\t$cur" >> $W
    cp_i=$((cp_i + 1))
    if [ -e "$dd/chim_Log.final.out" ]; then ev finalout; [ $((cp_i % 2)) = 0 ] && cp "$dd/chim_Log.final.out" "$O/$L.chim_Log.final.out" 2>/dev/null; fi
  fi
  sleep 1
done
