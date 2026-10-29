#!/bin/bash
#SBATCH --job-name=bl-fqcheck
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# curl --retry restarts a cut ENA stream from byte 0 inside the same pipe, so a mate can hold a mid-record seam followed by duplicate reads.
# Find the first malformed record of each mate (header '@', '+' line, equal sequence and quality lengths, ACGTN only) and cut both mates of a run
# to the same number of reads before that point.
set -uo pipefail
D=/mnt/projects/debruinz_project/debruinz/singlet/data/barcode-layer
first_bad() { awk 'NR%4==1{ if (substr($0,1,1)!="@") {print (NR-1)/4; exit} } NR%4==2{s=$0; if ($0 !~ /^[ACGTN]+$/) {print (NR-2)/4; exit}} NR%4==3{ if ($0!="+") {print (NR-3)/4; exit} } NR%4==0{ if (length($0)!=length(s)) {print (NR-4)/4; exit} }' $1; }
for c in dropseq bd splitseq_parse indrop; do
  a=$(( $(wc -l < $D/$c/M1.fq) / 4 )); b=$(( $(wc -l < $D/$c/M2.fq) / 4 ))
  f1=$(first_bad $D/$c/M1.fq); f2=$(first_bad $D/$c/M2.fq); f1=${f1:-$a}; f2=${f2:-$b}
  n=$a; [ $b -lt $n ] && n=$b; [ $f1 -lt $n ] && n=$f1; [ $f2 -lt $n ] && n=$f2
  echo "$c: M1 $a reads (first bad ${f1}), M2 $b reads (first bad ${f2}) -> keep $n"
  for m in 1 2; do head -n $((4*n)) $D/$c/M$m.fq > $D/$c/M$m.cut && mv $D/$c/M$m.cut $D/$c/M$m.fq; done
  for m in 1 2; do echo "  M$m: $(( $(wc -l < $D/$c/M$m.fq) / 4 )) reads, first_bad now: $(first_bad $D/$c/M$m.fq)"; done
done
echo done
