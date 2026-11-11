#!/bin/bash
#SBATCH --job-name=e1-cc
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E1 profile (docs/gpu-engine/fable/E1-profile.md): build from branch gpu/fable/profile with CUDA 13.3
#   m5  : the harness as is (k_p1/k_p2/k_p3 unchanged; only the --mono-budget knob added)  -> ncu, nsys, timing, sweeps
#   m5t : the same with -DM5_TAIL (bench/fable/profile/tail.cuh: per-thread start/end of k_p1/k_p3) -> batch tails
# Sources are kept per commit under $E/src/<sha> so that -lineinfo paths stay valid for ncu's source view.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; E=$W/jobs/fable/profile
mkdir -p $E/bin $E/results $E/src
BARE=$W/repos/singlet-py.git
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/profile:refs/fable/profile || { echo "fetch failed"; exit 1; }
SHA=$(git -C $BARE rev-parse --short=8 refs/fable/profile); echo "commit $SHA"
S=$E/src/$SHA; rm -rf $S; mkdir -p $S && git -C $BARE archive refs/fable/profile bench/gpu/m5 bench/fable/profile | tar -x -C $S
R=$E/results/build-$SHA; mkdir -p $R
apptainer exec $IMG nvcc --version | tail -2
cc() { local out=$1; shift
  apptainer exec -B $W $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xptxas -v "$@" \
    -o $E/bin/$out.new $S/bench/gpu/m5/m5.cu > $R/ptxas_$out.raw 2>&1; }
BUILD=,${BUILD:-m5,m5t},; P1=; P2=
[[ $BUILD == *,m5,* ]] && { cc m5 & P1=$!; }; [[ $BUILD == *,m5t,* ]] && { cc m5t -DM5_TAIL & P2=$!; }
X1=9; X2=9; [ -n "$P1" ] && { wait $P1; X1=$?; }; [ -n "$P2" ] && { wait $P2; X2=$?; }
echo "nvcc exit m5=$X1 m5t=$X2"
for b in m5 m5t; do
  apptainer exec $IMG bash -c "c++filt < $R/ptxas_$b.raw" > $R/ptxas_$b.txt 2>/dev/null || cp $R/ptxas_$b.raw $R/ptxas_$b.txt
  grep -iE "error" $R/ptxas_$b.txt | head -5
done
[ $X1 -eq 0 ] && mv $E/bin/m5.new $E/bin/m5 && echo $SHA > $E/bin/m5.commit
[ $X2 -eq 0 ] && mv $E/bin/m5t.new $E/bin/m5t && echo $SHA > $E/bin/m5t.commit
# registers / spills of every kernel instantiation (entry functions)
for b in m5 m5t; do
  echo "== $b"
  awk '/Compiling entry function/ {f=$0; sub(/.*entry function .?/, "", f); sub(/. for .sm_90.*/, "", f)}
       /bytes stack frame/ {sf=$0}
       /Used [0-9]+ registers/ && f != "" {r=$0; sub(/.*Used /, "", r); sub(/ registers.*/, "", r); gsub(/^ +/, "", sf); print r " regs | " sf " | " f; f=""}' $R/ptxas_$b.txt \
    | grep -E "k_p1|k_p2|k_p3|k_lane|k_cost" | sed -E 's/\(m5::Params.*//' | sort -t'|' -k3 > $R/regs_$b.txt
  cat $R/regs_$b.txt
done
# demangled kernel names as ncu/nsys will print them (for the --kernel-name regexes)
apptainer exec $IMG bash -c "cuobjdump --dump-elf-symbols $E/bin/m5 2>/dev/null | grep -oE '_Z[0-9]+k_p[123][A-Za-z0-9_]+' | sort -u | cu++filt" > $R/kernel_names.txt 2>&1
head -12 $R/kernel_names.txt | cut -c1-200
ls -la $E/bin
