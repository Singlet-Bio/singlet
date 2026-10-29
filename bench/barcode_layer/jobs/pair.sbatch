#!/bin/bash
#SBATCH --job-name=bl-pair
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:25:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# ENA streams can end early, so the two mates of a run may hold different numbers of reads (each is a prefix of its run). Cut both mates to the
# same number of complete reads, dropping the last record of each (it may be cut mid-record), and report read lengths.
set -uo pipefail
D=/mnt/projects/debruinz_project/debruinz/singlet/data/barcode-layer
for c in dropseq bd splitseq indrop; do
  a=$(( $(wc -l < $D/$c/M1.fq) / 4 )); b=$(( $(wc -l < $D/$c/M2.fq) / 4 )); n=$(( (a < b ? a : b) - 1 ))
  echo "$c: M1 $a reads, M2 $b reads -> $n"
  for m in 1 2; do head -n $((4*n)) $D/$c/M$m.fq > $D/$c/M$m.cut && mv $D/$c/M$m.cut $D/$c/M$m.fq; done
  for m in 1 2; do echo "  M$m lengths:"; awk 'NR%4==2{print length($0)}' $D/$c/M$m.fq | head -5000 | sort -n | uniq -c | sort -k1nr | head -3 | sed 's/^/    /'; done
  awk 'NR%4==1 && NR<=8' $D/$c/M1.fq | head -2; tail -n 4 $D/$c/M1.fq | head -2 | cut -c1-80
done
echo done
