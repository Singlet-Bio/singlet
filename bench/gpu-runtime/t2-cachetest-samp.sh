#!/bin/bash
# t2-cachetest-samp.sh <out.tsv>: once a second, the node's memory and reclaim state beside a gpu-pool (throughput track 2, cachetest).
# columns: epoch  MemFree Cached Shmem Active(file) Inactive(file) Dirty Writeback SReclaimable (kB)  psi some avg10, psi full avg10 (node,
# /proc/pressure/memory)  compact_stall pgscan_direct pgsteal_direct pgscan_kswapd allocstall(sum of allocstall*) thp_fault_fallback
# cgroup (job level): anon file shmem memory.current (B)  cgroup psi full avg10  memory.events oom_kill
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#')
OUT=$1
echo "# cg=$CG" > "$OUT"
echo -e "#epoch\tMemFree_kB\tCached_kB\tShmem_kB\tActiveFile_kB\tInactiveFile_kB\tDirty_kB\tWriteback_kB\tSReclaimable_kB\tpsi_some10\tpsi_full10\tcompact_stall\tpgscan_direct\tpgsteal_direct\tpgscan_kswapd\tallocstall\tthp_fault_fallback\tcg_anon_B\tcg_file_B\tcg_shmem_B\tcg_current_B\tcg_psi_full10\tcg_oom_kill" >> "$OUT"
while :; do
  t=$EPOCHREALTIME; t=${t/,/.}
  awk -v t="$t" '
    FILENAME=="/proc/meminfo"{ if($1=="MemFree:")mf=$2; if($1=="Cached:")ca=$2; if($1=="Shmem:")sh=$2; if($1=="Active(file):")af=$2; if($1=="Inactive(file):")ifl=$2; if($1=="Dirty:")di=$2; if($1=="Writeback:")wb=$2; if($1=="SReclaimable:")sr=$2 }
    FILENAME=="/proc/vmstat"{ if($1=="compact_stall")cs=$2; if($1=="pgscan_direct")pd=$2; if($1=="pgsteal_direct")ps=$2; if($1=="pgscan_kswapd")pk=$2; if($1~/^allocstall/)as+=$2; if($1=="thp_fault_fallback")tf=$2 }
    FILENAME=="/proc/pressure/memory"{ v=$2; sub("avg10=","",v); if($1=="some")so=v; if($1=="full")fu=v }
    FILENAME ~ /memory.stat$/{ if($1=="anon")an=$2; if($1=="file")fi=$2; if($1=="shmem")cs2=$2 }
    FILENAME ~ /memory.current$/{ cu=$1 }
    FILENAME ~ /memory.pressure$/{ v=$2; sub("avg10=","",v); if($1=="full")cf=v }
    FILENAME ~ /memory.events$/{ if($1=="oom_kill")ok=$2 }
    END{ printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", t,mf,ca,sh,af,ifl,di,wb,sr,so,fu,cs,pd,ps,pk,as,tf,an,fi,cs2,cu,cf,ok }' \
    /proc/meminfo /proc/vmstat /proc/pressure/memory $CG/memory.stat $CG/memory.current $CG/memory.pressure $CG/memory.events >> "$OUT" 2>/dev/null
  sleep 1
done
