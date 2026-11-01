#!/bin/bash
#SBATCH --job-name=e6-compile
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --nice=10000
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# E6: compile the records driver (bench/fable/records/records_test.cu + src/engine_gpu/records_gpu.cu) from gpu/fable/records with the CUDA 13.3
# image. Output: $W/jobs/fable/records/bin/records_test (+ .commit). No data is read.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-13-devel.sif; ENV=$W/envs/pbench; BARE=$W/repos/singlet-py.git
E6=$W/jobs/fable/records; mkdir -p $E6/bin
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/gpu/fable/records:refs/fable/records
SHA=$(git -C $BARE rev-parse --short=8 refs/fable/records); echo "commit $SHA"
SCR=/tmp/$USER-e6c-$SLURM_JOB_ID; mkdir -p $SCR/src $SCR/hts; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/fable/records src/engine_gpu include/singlet bench/fable/records | tar -x -C $SCR/src
ln -s $ENV/include/htslib $SCR/hts/htslib
S=$SCR/src
T0=$(date +%s)
apptainer exec -B $W -B $SCR $IMG nvcc -O3 -std=c++17 -arch=sm_90 -fmad=false -lineinfo -Xcompiler -fopenmp -I$S/src/engine_gpu -I$S/include -I$SCR/hts \
  -o $SCR/records_test $S/bench/fable/records/records_test.cu $S/src/engine_gpu/records_gpu.cu 2>&1 | head -80
[ -x $SCR/records_test ] || { echo "compile FAILED"; exit 1; }
cp $SCR/records_test $E6/bin/records_test.new && mv $E6/bin/records_test.new $E6/bin/records_test && echo $SHA > $E6/bin/records_test.commit
echo "compiled $SHA in $(( $(date +%s) - T0 )) s"
