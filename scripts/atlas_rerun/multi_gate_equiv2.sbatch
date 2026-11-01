#!/bin/bash
#SBATCH --job-name=s14-equiv2
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14: identity checks of the multi_gate runs. Env: P1 (part-1 gate dir with keep/single = 16 threads, keep/multi = 12 threads concurrent), P2 (part-2 gate dir with keep/single12).
# For every sample in P2/keep/single12: out_equiv (with scipy, so matrices are compared too) against the part-1 multi output and the 16-thread single output;
# then the files whose WRITE ORDER is nondeterministic (STAR Unmapped mates, nonhost_reads.fq) as multisets of records, and read_stats.tsv / pileup_stats.json line diffs.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/pbench/bin:$PATH LC_ALL=C
P1=${P1:?}; P2=${P2:?}; EQ=$P2/repo/bench/star_accel/out_equiv.py
python3 -c "import scipy; print('scipy', scipy.__version__)"
for g in $(ls $P2/keep/single12); do
  A=$P2/keep/single12/$g/out
  for other in "12thr concurrent (part-1 multi):$P1/keep/multi/$g/out" "16thr single (part-1):$P1/keep/single/$g/out"; do
    echo "=== $g: single 12 thr vs ${other%%:*}"; python3 $EQ $A ${other#*:} 2>&1 | grep -v "^    [ab]:" | cut -c1-220 | head -60
  done
done
echo "=== multiset of records (order-free) single12 vs part-1 multi"
for g in $(ls $P2/keep/single12); do
  for f in star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    a=$(sort -S 16G $P2/keep/single12/$g/out/$f | sha256sum | cut -c1-16); b=$(sort -S 16G $P1/keep/multi/$g/out/$f | sha256sum | cut -c1-16); echo "$g $f sorted: $a $b $([ $a = $b ] && echo SAME || echo DIFF)"
  done
  f=nonhost_reads.fq
  a=$(paste - - - - < $P2/keep/single12/$g/out/$f | sort -S 16G | sha256sum | cut -c1-16); b=$(paste - - - - < $P1/keep/multi/$g/out/$f | sort -S 16G | sha256sum | cut -c1-16); echo "$g $f records: $a $b $([ $a = $b ] && echo SAME || echo DIFF)"
  diff $P2/keep/single12/$g/out/read_stats.tsv $P1/keep/multi/$g/out/read_stats.tsv | head -8
done
