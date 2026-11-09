#!/bin/bash
# Frequency-prior test for 3' single-cell HLA typing (D-101). T1K on 3' reads picks rare alleles (IPD-IMGT/HLA records them in 1-4
# cells, mostly unconfirmed) over the common allele of the same group, because 3' reads do not cover the exons that separate them.
# This restricts T1K's reference to alleles recorded in >= K cells (IPD-IMGT/HLA 3.66.0 Allele_status.txt, the release of the index)
# and reruns T1K, as the pipeline does (run-t1k -u ... --preset hla), on each library's saved candidate reads.
# Usage (a step inside the node job): bash hla_prior_test.sh "K1 K2 ..." LIB=DIR ...   (DIR holds private/hla/hla_candidate_reads.fq.gz)
# Output: $W/jobs/d101/hlaprior/K<k>/<LIB>/sample_genotype.tsv (owner-only).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; REF=/mnt/projects/debruinz_project/cellarium/reference; ENV=$W/envs/pbench
T1K=$REF/hla/t1k; O=$W/jobs/d101/hlaprior; KS=$1; shift; T=${TASK_CPUS:-4}
export PATH=$ENV/bin:$PATH LD_LIBRARY_PATH=$ENV/lib
mkdir -p $O; chmod 700 $O
[ -s $O/Allele_status.txt ] || curl -fsSL -o $O/Allele_status.txt https://raw.githubusercontent.com/ANHIG/IMGTHLA/3660/Allele_status.txt
grep -m1 '^# version' $O/Allele_status.txt
for K in $KS; do
  F=$O/hlaidx_rna_seq.K$K.fa
  [ -s $F ] || python3 - $T1K/hlaidx/hlaidx_rna_seq.fa $O/Allele_status.txt $K > $F <<'EOF'
import sys
fa, status, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
cells = {}
for line in open(status):
    if line.startswith('#') or line.startswith('Allele,'): continue
    f = line.rstrip('\n').split(',')
    cells[f[0]] = int(f[1]) if f[1].isdigit() else 0
keep = kept = total = 0
for line in open(fa):
    if line.startswith('>'):
        name = line[1:].split()[0]
        total += 1
        a = name[4:] if name.startswith('HLA-') else name
        # alleles outside HLA's numbering (KIR, unnamed) or missing from the status file stay in
        keep = cells.get(a, k) >= k
        kept += keep
    if keep: sys.stdout.write(line)
print(f"K={k}: kept {kept} of {total} alleles", file=sys.stderr)
EOF
  for arg in "$@"; do
    L=${arg%%=*}; D=${arg#*=}; R=$O/K$K/$L; mkdir -p $R
    [ -s $R/sample_genotype.tsv ] && continue
    perl $T1K/bin/run-t1k -u $D/private/hla/hla_candidate_reads.fq.gz --preset hla -f $F -t $T -o sample --od $R > $R/t1k.log 2>&1 &
    while [ $(jobs -r | wc -l) -ge 6 ]; do sleep 2; done
  done
  wait
done
ls $O/K*/ | head -40
