R=/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-runtime/results
for a in S N; do D=$R/t2-chimshare-pool-819120-$a
 awk -F'\t' -v arm=$a 'FILENAME==ARGV[1]{ if($3=="level") lv[$2]=$1; next }
  { t=$1; it=$2; if(it==-1){ if(sh==0) sh=$5/1048576; next } if(it<0) next; s[t","it]+=($5+$7)/1048576; ts[t]=1; ids[it]=1 }
  END{ for(k in s){ split(k,a,","); t=a[1]; it=a[2]; h=s[k]-sh; if(h<0)h=0; if(h>mx[it])mx[it]=h; if(t>=lv[it] && h>ms[it])ms[it]=h }
       for(it in ids) printf "OLD CODE on 819120 arm %s item %d: max held %.2f GiB, after the level line %.2f GiB (pool RssAnon %.2f)\n", arm, it, mx[it], ms[it], sh }' $D/markers.tsv $D/pool0/memsamp.tsv | sort
done
