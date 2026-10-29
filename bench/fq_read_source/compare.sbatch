#!/bin/bash
#SBATCH --job-name=fqrs-cmp
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu,scavenger
#SBATCH --requeue
#SBATCH --nice=100
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
#   sbatch --export=ALL,PAIRS="SET:tagA:tagB SET:tagA:tagB ..." compare.sbatch
# For each pair of runs of one sample: bench/star_accel/out_equiv.py on out/ (sha256 per file, JSON minus timings and
# paths), STAR's Log.final.out read counts, and for every differing count matrix the value-level change (matrix_diff.py).
# Result: $W/jobs/fq-read-source/cmp/<SET>__<tagA>__<tagB>.txt
set -u
W=/mnt/projects/debruinz_project/debruinz/singlet; R=$W/jobs/fq-read-source/runs; C=$W/jobs/fq-read-source/cmp; mkdir -p $C
SRC=$W/src/singlet-py/perf__fq-read-source
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
for p in $PAIRS; do
  IFS=: read SET A B <<< "$p"
  F=$C/${SET}__${A}__${B}.txt
  {
    echo "== $SET: $A vs $B"
    in_runtime python $SRC/bench/star_accel/out_equiv.py $R/$SET/$A/out $R/$SET/$B/out
    for k in "Number of input reads" "Uniquely mapped reads number" "Number of reads mapped to multiple loci" \
             "Number of splices: Total" "Mismatch rate per base" "Average mapped length"; do
      printf "%-45s %s | %s\n" "$k" "$(grep -m1 "$k" $R/$SET/$A/out/star_Log.final.out | cut -f2)" \
             "$(grep -m1 "$k" $R/$SET/$B/out/star_Log.final.out | cut -f2)"
    done
    in_runtime python $SRC/bench/fq_read_source/matrix_diff.py $R/$SET/$A/out $R/$SET/$B/out
    # private/: identical or not, file by file (contents never printed)
    for f in $(cd $R/$SET/$A/private 2>/dev/null && find . -type f | sort); do
      cmp -s $R/$SET/$A/private/$f $R/$SET/$B/private/$f && s=same || s=DIFF; echo "private $f $s"
    done | awk '{n[$3]++} $3=="DIFF"{print "  " $0} END{print "  private files: " n["same"]+0 " identical, " n["DIFF"]+0 " differ"}'
  } > $F 2>&1
  tail -40 $F
done
