#!/bin/bash
# hostcut round 2, Step A: per process of the item (the cgroup's processes whose comm starts sg- / star- / gpu- / singlet), from the line
# "mapDistinct returned" in $1/pool0/0.log for $DUR s (170), every ~2 s: RssAnon/RssFile/RssShmem and the mappings with Rss >= 0.5 GiB (name, perms, size, rss).
# usage: t2-smaps.sh <result dir> <cgroup dir>; output $1/smaps.tsv: epoch pid comm what ... (what: M mapping | T totals)
O=$1; CG=$2
until grep -aq "mapDistinct returned" $O/pool0/0.log 2>/dev/null; do sleep 1; done
end=$(( $(date +%s) + ${DUR:-170} ))
while [ "$(date +%s)" -lt "$end" ]; do
  t=$(date +%s.%N)
  for p in $(find $CG -name cgroup.procs -exec cat {} + 2>/dev/null | sort -u); do
    c=$(cat /proc/$p/comm 2>/dev/null); case "$c" in sg-*|star-*|gpu-*|singlet*) ;; *) continue;; esac
    awk -v t=$t -v p=$p -v c="$c" '
      /^[0-9a-f]+-[0-9a-f]+ /{nm=$6; if (nm=="") nm="[anon]"; perms=$2; next}
      /^Size:/{sz=$2; next}
      /^Rss:/{rss=$2; if (rss>=524288) printf "%s %s %s M %s %s size_kB=%d rss_kB=%d\n", t,p,c,perms,nm,sz,rss; tot[nm=="[anon]"?"anon-map":(nm ~ /memfd|shm|deleted/ ? "shm-map" : "file-map")]+=rss; next}
      END{for (k in tot) printf "%s %s %s T %s rss_kB=%d\n", t,p,c,k,tot[k]}' /proc/$p/smaps >> $O/smaps.tsv 2>/dev/null
    grep -h "^VmRSS\|^RssAnon\|^RssFile\|^RssShmem" /proc/$p/status 2>/dev/null | tr -s ' \n' ' ' | awk -v t=$t -v p=$p -v c="$c" '{print t, p, c, "S", $0}' >> $O/smaps.tsv
  done
  sleep 1
done
