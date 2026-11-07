#!/bin/bash
#SBATCH --job-name=mtfix-run
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile the caller harness against the fixed headers and the pre-fix headers, run {old,new} x {q0,q20}, evaluate.
#   q0  = any MAPQ (what the pre-fix engine counted), q20 = MAPQ>=20 (the fixed engine)
# Inputs (put by clipper.py into $J): inc.tgz (include/singlet), old/ (pre-fix mt_heteroplasmy.h, mt_event_caller.h),
# mt_caller_bench.cpp, mt_eval2.py, common.py.  Outputs: $O/<run>/ (large), $J/results/*.json (small, pulled back).
set -eu
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/modality-bench/genotype_pool; J=$W/jobs/modfix-mt; O=$D/mt_fix; S=$D/singlet_k8
E=$W/envs/pbench; CXX=$E/bin/x86_64-conda-linux-gnu-g++; PY=$W/envs/gpool/venv/bin/python
GENOME=/mnt/projects/debruinz_project/cellarium/reference/GRCh38-2024-A/star_2.7.11b
cd $J
rm -rf inc && mkdir inc && tar xzf inc.tgz -C inc
mkdir -p results bin
INC="-I$J/inc/include -I$J/inc/include/singlet/pileup -I$E/include"
LIBS="-L$E/lib -lzstd -lz -Wl,-rpath,$E/lib -lpthread"
$CXX -std=c++20 -O2 -w $INC mt_caller_bench.cpp -o bin/bench_new $LIBS
$CXX -std=c++20 -O2 -w -DMTFIX_OLD -I$J/old $INC mt_caller_bench.cpp -o bin/bench_old $LIBS
echo "compiled"

run() {  # run LABEL BINARY COUNTS VCFSUB
  local label=$1 bin=$2 counts=$3 vcfsub=$4
  rm -rf $O/$label
  /usr/bin/time -v -o $O/$label.time.txt $J/bin/$bin $O/pile.$counts.bin $O/pile.assign.tsv $GENOME $O/$label 2>&1 | tail -12
  grep -E "Elapsed|Maximum res" $O/$label.time.txt
  $PY $J/mt_eval2.py $D/truth/GSE96583_batch2.total.tsne.df.tsv.gz $S $D/ref_k8/mt_pileup.npz $O/$label $O/cells_all.txt $O/$label/$vcfsub $J/results/$label.json --label $label > $O/$label.eval.log 2>&1 || tail -20 $O/$label.eval.log
  du -sh $O/$label
}
run old_q0  bench_old q0  mt
run old_q20 bench_old q20 mt
run new_q0  bench_new q0  private/mt
run new_q20 bench_new q20 private/mt
run new_q20umi bench_new q20umi private/mt   # molecule-level counts (UMI consensus per cell/position)

# donor-level homoplasmic threshold sweep (ambient chrM RNA from the other donors dilutes the allele fraction of true
# donor-defining variants to ~0.75-0.9 in a pool; the truth haplotypes use AF>=0.9)
for af in 90 85 75 70; do
  MTFIX_HAP_AF=0.$af run new_q20_hap$af bench_new q20 private/mt
done
ls -la $J/results
