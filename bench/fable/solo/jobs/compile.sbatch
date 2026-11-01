#!/bin/bash
#SBATCH --job-name=e9-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E9: compile the Solo driver (bench/fable/solo/solo_test.cu + src/engine_gpu/solo_gpu.cu) from gpu/fable/solo with the CUDA 13.3
# image. Output: $W/jobs/fable/solo/bin/solo_test (+ .commit). No data is read.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
E9=$W/jobs/fable/solo; mkdir -p $E9/bin
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/solo:refs/fable/solo
SHA=$(git -C $BARE rev-parse --short=8 refs/fable/solo); echo "commit $SHA"
SCR=/tmp/$USER-e9c-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/hts; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/fable/solo src/engine_gpu include/singlet bench/fable/solo | tar -x -C $SCR/src
ln -s $ENV/include/htslib $SCR/hts/htslib
S=$SCR/src
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -Xcompiler -Wno-format -I$S/src/engine_gpu -I$S/include -I$SCR/hts \
  --extended-lambda -o $SCR/solo_test $S/bench/fable/solo/solo_test.cu $S/src/engine_gpu/solo_gpu.cu -lgomp 2>&1 | head -80
[ -x $SCR/solo_test ] || { echo "compile FAILED"; exit 1; }
cp $SCR/solo_test $E9/bin/solo_test.new && mv $E9/bin/solo_test.new $E9/bin/solo_test && echo $SHA > $E9/bin/solo_test.commit
echo "compiled $SHA in $(( $(date +%s) - T0 )) s"
