W=/mnt/projects/debruinz_project/debruinz/singlet/jobs/gpu-runtime/results
win() { f=$1; shift; awk -F'\t' -v lab="$1" -v a=$2 -v b=$3 '/^#/{next} {t=$1; v=($4+$6)/1073741824; if(t>=a && t<=b){ if(n==0){f0=v;mx=v}; if(v>mx)mx=v; n++ } if(t<=a) s=v} END{printf "%s: anon+shmem at pass start %.1f, max in window %.1f, rise %.2f GiB (%d samples)\n", lab, s, mx, mx-s, n}' $f; }
S=$W/t2-chimshare-pool-819120-S/cg1hz.tsv; N=$W/t2-chimshare-pool-819120-N/cg1hz.tsv
win $S "S gsm" 1795174960 1795175332; win $S "S d300_5p" 1795174929 1795175227; win $S "S pbmc5p_v1" 1795174872 1795174945
win $N "N gsm" 1795175642 1795176064; win $N "N d300_5p" 1795175609 1795175949; win $N "N pbmc5p_v1" 1795175552 1795175641
for f in $S $N; do awk -F'\t' '/^#/{next} {v=($4+$6)/1073741824; c=$3/1073741824; if(v>m)m=v; if(c>mc)mc=c} END{printf "%s: max anon+shmem %.1f GiB, max memory.current %.1f GiB\n", FILENAME, m, mc}' $f; done
