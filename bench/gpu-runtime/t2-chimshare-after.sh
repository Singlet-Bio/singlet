J=${1:?job}
R=/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-runtime/results
for a in S N P; do F=$R/t2-chimshare-pool-$J-$a/pool0/memsamp-held.tsv; [ -s $F ] || continue
 awk -F'\t' -v arm=$a '/^#/{next} { it=$2; h=$3; o=$4; if(h>mx[it])mx[it]=h; if(o>mo[it])mo[it]=o; if($8==0){ if(h>ms[it])ms[it]=h; if(o>mso[it])mso[it]=o } if($7==1){ k[it]++; if(h>mk[it])mk[it]=h; if(o>mko[it])mko[it]=o } ids[it]=1 }
  END{ for(it in ids) printf "NEW CODE %s arm %s item %d: held (gate figure) max %.2f, after the level line %.2f GiB; old formula on the same samples: max %.2f, after the level line %.2f; samples with kept=1: %d (held max %.2f, old formula %.2f)\n", FILENAME, arm, it, mx[it], ms[it], mo[it], mso[it], k[it], mk[it], mko[it] }' $F | sed "s#$R/##" | sort
done
for a in S N P; do F=$R/t2-chimshare-pool-$J-$a/pool0/memsamp-held.tsv; [ -s $F ] || continue
 awk -F'\t' -v arm=$a '/^#/{next} { if($7==0){ n++; o=($4<0)?0:$4; d=$3-o; if(d<0)d=-d; if(d>1e-3) bad++ } else { kn++ } } END{ printf "arm %s: rows with kept=0: %d, rows where held differs from the old formula (clamped): %d; rows with kept=1: %d\n", arm, n, bad+0, kn+0 }' $F
done
