#!/bin/bash
#SBATCH --job-name=e4m-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E4 after the merge with gpu/fable-engine (docs/gpu-engine/fable/E4-order.md, "Merge and M5_NOCLK"); template:
# bench/fable/engine/fable_cc.sbatch. Builds the m5 harness of BRANCH (default gpu/fable/order) in the flavours of FLAV
# ("name:flags|...", built in parallel) -> $W/jobs/fable/order/merge/bin/<name>, and per flavour the ptxas -v log and a
# register table of the kernels (regs, stack, spill stores, spill loads, kernel) in $W/jobs/fable/order/merge/results.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; BARE=$W/repos/singlet-py.git; BR=${BRANCH:-gpu/fable/order}
FLAV=${FLAV:-"m5:|m5n:-DM5_NOCLK"}
O=$W/jobs/fable/order/merge; mkdir -p $O/bin $O/results
SCR=/tmp/$USER-e4mcc-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/fable/e4m && git -C $BARE archive refs/fable/e4m bench/gpu/m5 bench/fable | tar -x -C $SCR
SHA=$(git -C $BARE rev-parse --short=10 refs/fable/e4m); echo "commit $SHA ($BR) on $(hostname) $(date -Is)"; echo $SHA > $O/bin/commit
cd $SCR/bench/gpu/m5
IFS='|' read -ra FL <<< "$FLAV"
for f in "${FL[@]}"; do
  N=${f%%:*}; X=${f#*:}
  ( T0=$(date +%s)
    apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v $X -DM5_GIT="\"$SHA\"" \
      -o $O/bin/$N.new m5.cu > $O/results/ptxas-$N-$SHA.txt 2>&1 && mv $O/bin/$N.new $O/bin/$N && echo "[$N ok] $X in $(( $(date +%s) - T0 )) s" \
      || { echo "[$N FAILED] $X"; grep -E "error" $O/results/ptxas-$N-$SHA.txt | head -20; } ) &
done
wait
for f in "${FL[@]}"; do
  N=${f%%:*}; P=$O/results/ptxas-$N-$SHA.txt; R=$O/results/regs-$N-$SHA.tsv
  awk '/Compiling entry function/ { f = $0; sub(/.*function ./, "", f); sub(/. for .*/, "", f); st = ss = sl = "-" }
       /bytes stack frame/ { st = $1; ss = $5; sl = $9 }
       /Used [0-9]+ registers/ { match($0, /Used [0-9]+ registers/); print substr($0, RSTART + 5, RLENGTH - 15) "\t" st "\t" ss "\t" sl "\t" f }' $P > $SCR/r.tsv
  paste <(cut -f1-4 $SCR/r.tsv) <(cut -f5 $SCR/r.tsv | apptainer exec $IMG c++filt -p) | grep -E "k_p1|k_p2|k_p3|k_lane|k_key|k_cost" > $R
  echo "== registers $N ($R)"; sed -E 's/Caps<160u?, 4u?, 32u?, 16u?, 16u?, 256u?, 8u?, 8u?, 8u?, 640u?>/A/; s/Caps<160u?, 4u?, 32u?, 64u?, 8u?, 1024u?, 8u?, 8u?, 32u?, 2048u?>/M/;
    s/Caps<160u?, 8u?, 128u?, 128u?, 50u?, 4096u?, 20u?, 32u?, 64u?, 4096u?>/B/' $R | sort -t$'\t' -k5
done
ls -la $O/bin/; echo "== done $(date)"
