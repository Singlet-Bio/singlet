O=$1
cat $O/engine.c*.log | grep -a "engine_gpu: steps (s):" | sed -E 's/.*steps \(s\): //; s/; total.*//' | tr ',' '\n' | awk '{ n[$1]++; s[$1]+=$2; if(!($1 in o)) { o[$1]=++k; nm[k]=$1 } } END{ for(i=1;i<=k;i++) if (s[nm[i]]/n[nm[i]]>=0.3) printf "%-20s %6.2f\n", nm[i], s[nm[i]]/n[nm[i]] }'
echo "--- holds"
cat $O/engine.c*.log | grep -a "ENGINE_GPU: device [0-9]" | sed -E 's/.*waited ([0-9.e+-]+) s for it, held it ([0-9.]+) s \(E20 split: ([0-9.]+) s to the lanes. end, released ([0-9.]+) s for the host lane and the records, waited ([0-9.]+) s of it to take it again, ([0-9.]+) s for the records.*/\1 \2 \3 \4 \5 \6/' | awk '{w+=$1;h+=$2;a+=$3;r+=$4;rw+=$5;b+=$6;n++} END{printf "n=%d wait %.1f  held %.1f (hold1 %.1f, released %.1f incl relock wait %.1f, hold2 %.1f)\n", n,w/n,h/n,a/n,r/n,rw/n,b/n}'
grep -a -h "P3-MEM-03 reads phase" $O/engine.c0.5.log | cut -c1-500
grep -a -h "E12 device -> host copies\|E12 host lane\|upload .* dedup .* GPU mapping" $O/engine.c0.5.log | cut -c1-300
