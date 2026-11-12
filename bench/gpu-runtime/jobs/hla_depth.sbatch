#!/bin/bash
#SBATCH -A debruinz -p short-cpu -c 16 --mem=64G --time=03:00:00 -J hla-depth -o /mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# hla-depth: HLA typing accuracy against the number of candidate reads, on the 20 Oetjen 10x 3' libraries with Scisco typing
# (bench/preprint/hla: truth_compiled_nih_3p_genotypes.csv, the launch configuration's saved candidate reads of each library).
# Per library and cut c of CUTS: every c-th of its candidate reads (SUB=stride, the default: what a run 1/c as deep would have kept)
# typed by `singlet hla --pending` with the campaign's T1K and the K=2 prior, scored as score_oetjen.py scores
# SUB=prefix takes the first 1/c instead. That is not a shallower run for these libraries: job 813050 typed 90 of 90 loci from the
# first half of the reads and none from the first quarter, so their reads are not in sequencer order (the first quarter holds the
# reads of one gene), and a cap that keeps the first N candidate reads would be wrong for such inputs.
# (loci A B C DQB1 DRB1, 1-field and 2-field, matched alleles of truth alleles). Prints counts and times only, never alleles;
# the genotype tables stay under $O (owner-only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; H=$W/bench/preprint/hla; CODE=$H/code
BIN=${BIN:-$W/jobs/gpu-pr2/bin2/rel138-e0824b21/singlet-gpu}; TK=${TK:-$W/jobs/gpu-campaign/agents/hla-speed/t1k-p3b}; T=${SLURM_CPUS_PER_TASK:-16}
CUTS=${CUTS:-"1 2 4 8 16 32 64"}; SUB=${SUB:-stride}; LIBS=${LIBS:-$(awk -F'\t' 'NR>1{print $1}' $CODE/libs.tsv | tr '\n' ' ')}
O=$W/jobs/gpu-runtime/results/hla-depth-$SLURM_JOB_ID; umask 077; mkdir -p $O; chmod 700 $O
SCR=/tmp/$USER-hladepth-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export PATH=$W/envs/pbench/bin:$PATH LD_LIBRARY_PATH=$W/envs/pbench/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} TMPDIR=$SCR
R=$O/results.txt
echo "node=$(hostname) job=$SLURM_JOB_ID threads=$T bin=$BIN t1k=$TK cuts='$CUTS' sub=$SUB $(date -Is)" | tee $R
: > $O/runs.tsv
for d in $LIBS; do
  C=$H/launch_cfg/$d/private/hla/hla_candidate_reads.fq.gz; S=$H/launch_cfg/$d/private/hla/hla_summary.json
  [ -s $C ] && [ -s $S ] || { echo "LIB $d: no saved candidate reads" | tee -a $R; continue; }
  scanned=$(grep -o '"reads_scanned": [0-9]*' $S | head -1 | grep -o '[0-9]*$'); kept=$(grep -o '"reads_kept": [0-9]*' $S | head -1 | grep -o '[0-9]*$')
  zcat $C > $SCR/cand.fq; n0=$(( $(wc -l < $SCR/cand.fq) / 4 ))
  [ "$n0" = "$kept" ] || echo "LIB $d: $n0 candidate reads in the file, the summary says $kept" | tee -a $R
  mkdir -p $O/$d
  for c in $CUTS; do
    n=$(( n0 / c )); [ $n -ge 1000 ] || continue
    X=$SCR/x; rm -rf $X; mkdir -p $X/private/hla/pending
    if [ $SUB = prefix ]; then head -n $(( 4 * n )) $SCR/cand.fq; else awk -v c=$c -v n=$n 'int((NR - 1) / 4) % c == 0 && int((NR - 1) / 4) / c < n' $SCR/cand.fq; fi | gzip -1 > $X/private/hla/pending/candidate_reads.fq.gz
    printf '{\n  "format": "singlet-hla-pending-1",\n  "t1k_dir": "%s",\n  "cdna_is_r1": false,\n  "donor_assignments": "",\n  "min_cells": 2,\n  "reads_scanned": %s,\n  "reads_kept": %s,\n  "n_kmers": 229078,\n  "n_alleles": 30322,\n  "seconds_prefilter": 0,\n  "candidate_bytes": %s\n}\n' \
      "$TK" $(( scanned / c )) $n $(stat -c %s $X/private/hla/pending/candidate_reads.fq.gz) > $X/private/hla/pending/manifest.json
    /usr/bin/time -v -o $SCR/time.txt $BIN hla --pending $X/private/hla/pending --t1k $TK --threads $T > $SCR/t1k.log 2>&1; rc=$?
    cpu=$(awk '/User time/{u=$NF} /System time/{s=$NF} END{printf "%.0f", u+s}' $SCR/time.txt)
    [ $rc = 0 ] && cp $X/private/hla/hla_genotype.tsv $O/$d/genotype.c$c.tsv
    echo -e "$d\t$c\t$(( scanned / c ))\t$n\t$rc\t$cpu" >> $O/runs.tsv
  done
  echo "LIB $d: $scanned read pairs, $n0 candidate reads; cuts done: $(awk -F'\t' -v d=$d '$1==d && $5==0' $O/runs.tsv | wc -l)" | tee -a $R
done
python3 - $CODE $O <<'PY' | tee -a $R
import collections, csv, os, sys
code, out = sys.argv[1], sys.argv[2]
sys.path.insert(0, code)
from score_oetjen import LOCI, norm, alts, pair_match, read_genotype
libs = {r["donor"]: r for r in csv.DictReader(open(os.path.join(code, "libs.tsv")), delimiter="\t")}
truth = {r["sample"]: r for r in csv.DictReader(open(os.path.join(code, "truth_compiled_nih_3p_genotypes.csv"))) if r["genotyper"] == "Ground truth"}
agg = collections.defaultdict(lambda: collections.defaultdict(float))
for ln in open(os.path.join(out, "runs.tsv")):
    d, c, scanned, n, rc, cpu = ln.rstrip("\n").split("\t")
    tr = truth.get(libs[d]["compiled_id"]) or truth.get(d)
    if tr is None or rc != "0": continue
    g = read_genotype(os.path.join(out, d, "genotype.c%s.tsv" % c)) or {}
    a = agg[int(c)]
    a["libs"] += 1; a["pairs"] += int(scanned); a["cand"] += int(n); a["cpu"] += float(cpu)
    for l in LOCI:
        if not tr[l + "_1"] or tr[l + "_1"] == "NA" or tr[l + "_2"] == "NA": continue
        call = g.get(l, [None, None])
        for lvl, nf in (("f1", 1), ("f2", 2)):
            a[lvl] += pair_match([alts(tr[l + "_1"], nf), alts(tr[l + "_2"], nf)], [norm(call[0], nf), norm(call[1], nf)])
        a["tot"] += 2
        a["typed"] += 1 if l in g else 0
        a["loci"] += 1
print("cut   libs  mean M pairs  mean candidates  loci typed   1-field accuracy     2-field accuracy     T1K core-s per library")
for c in sorted(agg):
    a = agg[c]
    print("1/%-3d %4d  %12.1f  %15.0f  %4d/%-4d    %4d/%-4d = %.3f    %4d/%-4d = %.3f    %8.0f" % (
        c, a["libs"], a["pairs"] / a["libs"] / 1e6, a["cand"] / a["libs"], a["typed"], a["loci"],
        a["f1"], a["tot"], a["f1"] / a["tot"] if a["tot"] else 0, a["f2"], a["tot"], a["f2"] / a["tot"] if a["tot"] else 0, a["cpu"] / a["libs"]))
PY
echo "done $(date -Is)" | tee -a $R
