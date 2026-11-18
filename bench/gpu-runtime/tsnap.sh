#!/bin/bash
# tsnap.sh <out> <seconds> [interval]: every interval s, one line per thread on the node: pid tid comm utime+stime (ticks)
out=$1; dur=$2; iv=${3:-2}; end=$(( $(date +%s) + dur ))
while [ $(date +%s) -lt $end ]; do
  echo "S $(date +%s.%N | cut -c1-13)" >> $out
  awk '{ n=split(FILENAME,a,"/"); tid=a[n-1]; pid=a[3]; s=index($0,"("); e=0; for(i=length($0);i>0;i--) if(substr($0,i,1)==")"){e=i;break}
         c=substr($0,s+1,e-s-1); gsub(/ /,"_",c); split(substr($0,e+2),f," "); if (f[12]+f[13] > 0) print pid, tid, c, f[12]+f[13] }' /proc/[0-9]*/task/[0-9]*/stat 2>/dev/null >> $out
  sleep $iv
done
