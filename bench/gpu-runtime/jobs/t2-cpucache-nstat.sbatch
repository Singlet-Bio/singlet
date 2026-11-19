#!/bin/bash
#SBATCH -A debruinz -p short-gpu --exclude=g050,g051,g052 -c 6 --mem=24G --time=02:00:00 -J t2-cpucache-nstat -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# t2-cpucache: how many N does a cDNA read of a library have: decode the .1fq (singlet decode, the cDNA read to a pipe) and count, per read,
# the N bases (histogram), the reads with N per block of 500 000, the position of the first N and whether every N is in one run at the 3' end.
# usage: SAMPLE=d100s_3p BIN=<singlet-gpu> sbatch t2-cpucache-nstat.sbatch   (SAMPLE: key under data/gpu-deep or pipeline-benchmark/data)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
S=${SAMPLE:?}; BIN=${BIN:?}
case $S in d300_3p|d100s_3p|d500s_3p) FQ=$W/data/gpu-deep/$S/$S.1fq ;; /*) FQ=$S ;; *) FQ=$W/pipeline-benchmark/data/$S/$S.1fq ;; esac
echo "node=$(hostname) fq=$FQ bin=$BIN"
$BIN decode $FQ --r1 /dev/null --r2 /dev/stdout --threads 4 2>/dev/null | awk '
NR%4==2 { r++; L=length($0); s=$0; n=gsub(/N/,"N",s); tot_len+=L;
  h[n>=40?40:n]++; b=int((r-1)/500000);
  if (n>0) { wn[b]++; nn++; sumN+=n; f=index($0,"N"); l=length($0)-match($0,/N+$/)+1;
    if (match($0,/N+$/)) tail++; if (n==L) allN++; if (n==1) one++;
    pos[f>=150?150:f]++ } }
END { printf "reads=%d mean_len=%.1f reads_with_N=%d (%.3f%%) mean_N_per_Nread=%.2f allN=%d single_N=%d tail_run_N=%d\n", r, tot_len/r, nn, 100*nn/r, (nn?sumN/nn:0), allN, one, tail;
  printf "hist_N_count:"; for (i=0;i<=40;i++) if (h[i]>0) printf " %d:%d", i, h[i]; printf "\n";
  printf "blocks_with_N(500k reads each):"; for (b=0;b<=int((r-1)/500000);b++) printf " %d", wn[b]+0; printf "\n";
  printf "first_N_pos:"; for (i=1;i<=150;i++) if (pos[i]>0) printf " %d:%d", i, pos[i]; printf "\n" }'
