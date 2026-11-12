#!/bin/bash
#SBATCH -A debruinz -p short-cpu -c 8 --mem=64G --time=02:00:00 -J hla-probe -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hla-probe: what HLA typing costs per stage and how its calls depend on the number of candidate reads.
# Per sample of SAMPLES: (1) the prefilter alone (`singlet hla --defer`): wall, CPU, candidate reads and their stored size;
# (2) T1K on the first 1/1, 1/2, ... 1/64 of the candidate reads (`singlet hla --pending` on a copy of pending/ cut to the first N
# reads): wall, CPU per T1K program (a 1 s sampler of their processes), T1K's own candidate count, and the genotype table;
# (3) each cut's calls against the full run's: genes whose allele pair (first listed allele of each call, 2 fields) is the same.
# Prints counts and times only, never alleles (HLA types are genotypes); the tables stay under $O (owner-only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference
BIN=${BIN:-$W/jobs/gpu-pr2/bin2/rel138-e0824b21/singlet-gpu}; TK=${TK:-$REF/hla/t1k}; T=${SLURM_CPUS_PER_TASK:-8}
SAMPLES=${SAMPLES:-"d300_3p pbmc1k_v3 pbmc5p_v1"}; CUTS=${CUTS:-"1 2 4 8 16 32 64"}
O=$W/jobs/gpu-runtime/results/hla-probe-$SLURM_JOB_ID; mkdir -p $O; chmod 700 $O; umask 077
SCR=/tmp/$USER-hlaprobe-$SLURM_JOB_ID; mkdir -p $SCR; trap 'kill $(jobs -p) 2>/dev/null; rm -rf $SCR' EXIT
export PATH=$W/envs/pbench/bin:$PATH LD_LIBRARY_PATH=$W/envs/pbench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} TMPDIR=$SCR
R=$O/results.txt
echo "node=$(hostname) job=$SLURM_JOB_ID threads=$T bin=$BIN t1k=$(readlink -f $TK) $(date -Is)" | tee $R
[ -x $BIN ] && [ -x $TK/bin/genotyper ] || { echo "missing binary or T1K" | tee -a $R; exit 2; }
input() { local s=$1; for f in $W/data/gpu-deep/$s/$s.1fq $W/pipeline-benchmark/data/$s/$s.1fq; do [ -s $f ] && { echo $f; return; }; done; }
sampler() {  # file : every 1 s the CPU ticks of T1K's programs (per pid the last value seen)
  while :; do for p in $(pgrep -u $USER -x 'fastq-extractor|genotyper|analyzer' 2>/dev/null); do
      s=$(cat /proc/$p/stat 2>/dev/null) || continue; c=${s#*(}; c=${c%)*}; set -- ${s##*) }; echo "$p $c $(( ${12} + ${13} ))"; done >> $1; sleep 1; done; }
stage_cpu() { awk '{ v[$1" "$2]=$3 } END{ for (k in v) { split(k,a," "); s[a[2]]+=v[k]/100 } printf "fastq-extractor %.0f, genotyper %.0f, analyzer %.0f core-s", s["fastq-extractor"], s["genotyper"], s["analyzer"] }' $1; }
tm() { awk '/Elapsed/{w=$NF} /User time/{u=$NF} /System time/{s=$NF} /Maximum resident/{m=$NF} END{ n=split(w,a,":"); ws=(n==3? a[1]*3600+a[2]*60+a[3] : a[1]*60+a[2]); printf "wall %.0f s, CPU %.0f core-s, peak RSS %.1f GB", ws, u+s, m/1048576 }' $1; }
for s in $SAMPLES; do
  F=$(input $s); [ -n "$F" ] || { echo "no input for $s" | tee -a $R; continue; }
  D=$SCR/$s; mkdir -p $D/full
  echo "######## $s ($(stat -c %s $F) bytes of .1fq)" | tee -a $R
  /usr/bin/time -v -o $D/pre.time $BIN hla --defer --t1k $TK --out $D/full --threads $T $F > $D/pre.log 2>&1
  P=$D/full/private/hla/pending
  [ -s $P/manifest.json ] || { echo "  prefilter failed: $(tail -2 $D/pre.log | cut -c1-200)" | tee -a $R; continue; }
  kept=$(awk -F'[:,]' '/"reads_kept"/{gsub(/ /,"",$2); print $2}' $P/manifest.json); scanned=$(awk -F'[:,]' '/"reads_scanned"/{gsub(/ /,"",$2); print $2}' $P/manifest.json)
  echo "  prefilter: $kept of $scanned reads kept; $(tm $D/pre.time); candidate_reads.fq.gz $(stat -c %s $P/candidate_reads.fq.gz 2>/dev/null) bytes" | tee -a $R
  [ "${kept:-0}" -gt 0 ] || continue
  zcat $P/candidate_reads.fq.gz > $D/cand.fq
  mkdir -p $O/$s
  for c in $CUTS; do
    n=$(( kept / c )); [ $n -ge 1000 ] || continue
    X=$D/c$c; mkdir -p $X/private/hla/pending
    head -n $(( 4 * n )) $D/cand.fq | gzip -1 > $X/private/hla/pending/candidate_reads.fq.gz
    sed -E "s/(\"reads_kept\": )[0-9]+/\1$n/" $P/manifest.json > $X/private/hla/pending/manifest.json
    : > $D/cpu.$c; sampler $D/cpu.$c & SM=$!
    SINGLET_HLA_KEEP_TMP=1 /usr/bin/time -v -o $D/t1k.$c.time $BIN hla --pending $X/private/hla/pending --t1k $TK --threads $T > $D/t1k.$c.log 2>&1; rc=$?
    kill $SM 2>/dev/null; wait $SM 2>/dev/null
    tmpd=$(ls -d $SCR/singlet-hla-* 2>/dev/null | head -1)
    nc=$( [ -n "$tmpd" ] && cat $tmpd/sample_candidate*.fq 2>/dev/null | wc -l ); rm -rf $SCR/singlet-hla-*
    cp $X/private/hla/hla_genotype.tsv $O/$s/genotype.c$c.tsv 2>/dev/null
    echo "  first 1/$c ($n reads): rc=$rc; $(tm $D/t1k.$c.time); $(stage_cpu $D/cpu.$c); T1K's own candidates $(( ${nc:-0} / 4 ))" | tee -a $R
    rm -rf $X
  done
  python3 - $O/$s $CUTS <<'PY' | tee -a $R
import os, sys
d = sys.argv[1]; cuts = sys.argv[2:]
def calls(p):
    out = {}
    if not os.path.exists(p): return None
    for ln in open(p):
        c = ln.rstrip("\n").split("\t")
        if len(c) < 5: continue
        al = []
        for i in (2, 5):
            if i < len(c) and c[i] not in (".", ""):
                q = float(c[i + 2]) if i + 2 < len(c) and c[i + 2] not in ("", ".") else -1
                a = c[i].split(",")[0]
                f = a.split("*")[-1].split(":")
                al.append((a.split("*")[0] + "*" + ":".join(f[:2]), q))
        out[c[0]] = al
    return out
full = calls(os.path.join(d, "genotype.c1.tsv"))
if full is None: print("  no full genotype table"); sys.exit(0)
main = ["HLA-A", "HLA-B", "HLA-C", "HLA-DQB1", "HLA-DRB1"]
conf = lambda g, t: sorted(a for a, q in t.get(g, []) if q > 0)
print("  full run: %d genes with a call of quality > 0; of the five main genes %d" % (sum(1 for g in full if conf(g, full)), sum(1 for g in main if conf(g, full))))
for c in cuts[1:]:
    t = calls(os.path.join(d, "genotype.c%s.tsv" % c))
    if t is None: continue
    genes = [g for g in full if conf(g, full)]
    same = sum(1 for g in genes if conf(g, t) == conf(g, full))
    m = [g for g in main if conf(g, full)]
    sm = sum(1 for g in m if conf(g, t) == conf(g, full))
    print("  first 1/%s against the full run (2-field calls of quality > 0): %d of %d genes the same; main five: %d of %d" % (c, same, len(genes), sm, len(m)))
PY
done
echo "done $(date -Is)" | tee -a $R
