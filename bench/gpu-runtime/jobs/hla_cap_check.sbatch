#!/bin/bash
#SBATCH -A debruinz -p short-cpu -c 8 --mem=64G --time=02:00:00 -J hla-cap -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hla-cap: the cap on the HLA candidate reads (--max-reads / --hla-max-reads / SINGLET_HLA_MAX_READS) on 5' libraries.
# Per sample: A `singlet hla` without the cap; B with --max-reads CAP; C the deferred route with the cap (--defer --max-reads, then
# hla --pending). Reports: B = C byte for byte (genotype, allele table, VCF); B against A by 2-field calls of quality > 0 (the five
# main loci and every gene); wall and CPU of each; the stored candidate reads' size. Counts and times only, never alleles.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet
BIN=${BIN:?BIN: the build with --max-reads}; TK=${TK:-$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b}; T=${SLURM_CPUS_PER_TASK:-8}
SAMPLES=${SAMPLES:-"pbmc5p_v1 d300_5p"}; CAP=${CAP:-500000}
O=$W/jobs/gpu-runtime/results/hla-cap-$SLURM_JOB_ID; umask 077; mkdir -p $O; chmod 700 $O
SCR=/tmp/$USER-hlacap-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$W/envs/pbench/bin:$PATH LD_LIBRARY_PATH=$W/envs/pbench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} TMPDIR=$SCR
R=$O/results.txt
echo "node=$(hostname) job=$SLURM_JOB_ID threads=$T bin=$BIN cap=$CAP $(date -Is)" | tee $R
input() { local s=$1; for f in $W/data/gpu-deep/$s/$s.1fq $W/pipeline-benchmark/data/$s/$s.1fq; do [ -s $f ] && { echo $f; return; }; done; }
tm() { awk '/Elapsed/{w=$NF} /User time/{u=$NF} /System time/{s=$NF} END{ n=split(w,a,":"); ws=(n==3? a[1]*3600+a[2]*60+a[3] : a[1]*60+a[2]); printf "wall %.0f s, CPU %.0f core-s", ws, u+s }' $1; }
for s in $SAMPLES; do
  F=$(input $s); [ -n "$F" ] || { echo "no input for $s" | tee -a $R; continue; }
  D=$SCR/$s; mkdir -p $D/a $D/b $D/c $O/$s
  echo "######## $s" | tee -a $R
  /usr/bin/time -v -o $D/a.time $BIN hla --t1k $TK --out $D/a --threads $T $F > $D/a.log 2>&1; ra=$?
  /usr/bin/time -v -o $D/b.time $BIN hla --t1k $TK --out $D/b --threads $T --max-reads $CAP $F > $D/b.log 2>&1; rb=$?
  /usr/bin/time -v -o $D/c1.time $BIN hla --t1k $TK --out $D/c --threads $T --defer --max-reads $CAP $F > $D/c1.log 2>&1; rc1=$?
  pb=$(stat -c %s $D/c/private/hla/pending/candidate_reads.fq.gz 2>/dev/null)
  /usr/bin/time -v -o $D/c2.time $BIN hla --pending $D/c/private/hla/pending --t1k $TK --threads $T > $D/c2.log 2>&1; rc2=$?
  echo "  A no cap:        rc=$ra; $(tm $D/a.time); stored candidate reads $(stat -c %s $D/a/private/hla/hla_candidate_reads.fq.gz 2>/dev/null) bytes" | tee -a $R
  echo "  B cap $CAP:   rc=$rb; $(tm $D/b.time); stored candidate reads $(stat -c %s $D/b/private/hla/hla_candidate_reads.fq.gz 2>/dev/null) bytes" | tee -a $R
  echo "  C deferred, cap: rc=$rc1/$rc2; prefilter $(tm $D/c1.time); typing $(tm $D/c2.time); pending reads ${pb:-?} bytes" | tee -a $R
  grep -a -h "candidate reads: 1 in" $D/b.log $D/c1.log | sort -u | cut -c1-160 | sed 's/^/  /' | tee -a $R
  grep -o '"prefilter": {[^}]*}' $D/b/private/hla/hla_summary.json | cut -c1-300 | sed 's/^/  B summary /' | tee -a $R
  same=1; for f in hla_genotype.tsv hla_allele.tsv hla_allele.vcf; do cmp -s $D/b/private/hla/$f $D/c/private/hla/$f || same=0; done
  cmp -s $D/b/private/hla/hla_candidate_reads.fq.gz $D/c/private/hla/hla_candidate_reads.fq.gz || same=0
  echo "  B and C (direct and deferred, both capped): $([ $same = 1 ] && echo "byte-identical tables and candidate reads" || echo DIFFERENT)" | tee -a $R
  cp $D/a/private/hla/hla_genotype.tsv $O/$s/genotype.nocap.tsv 2>/dev/null; cp $D/b/private/hla/hla_genotype.tsv $O/$s/genotype.cap.tsv 2>/dev/null
  python3 - $O/$s <<'PY' | tee -a $R
import os, sys
d = sys.argv[1]
def calls(p):
    out = {}
    for ln in open(p):
        c = ln.rstrip("\n").split("\t")
        if len(c) < 8: continue
        al = []
        for i in (2, 5):
            if c[i] not in (".", ""):
                try: q = float(c[i + 2])
                except ValueError: q = -1
                a = c[i].split(",")[0]; f = a.split("*")[-1].split(":")
                if q > 0: al.append(a.split("*")[0] + "*" + ":".join(f[:2]))
        out[c[0]] = sorted(al)
    return out
a, b = calls(os.path.join(d, "genotype.nocap.tsv")), calls(os.path.join(d, "genotype.cap.tsv"))
main = ["HLA-A", "HLA-B", "HLA-C", "HLA-DQB1", "HLA-DRB1"]
g = [x for x in a if a[x]]; m = [x for x in main if a.get(x)]
print("  B against A (2-field calls of quality > 0): main five %d of %d the same; all genes %d of %d" % (
    sum(1 for x in m if b.get(x) == a[x]), len(m), sum(1 for x in g if b.get(x) == a[x]), len(g)))
PY
  rm -rf $D
done
echo "done $(date -Is)" | tee -a $R
