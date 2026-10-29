#!/bin/bash
#SBATCH --job-name=ar-build
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem,debug-cpu,short-cpu
#SBATCH --nice=10000
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile bench/atac_reducers/atac_reducers_test.cu (no GPU needed to compile) with nvcc 12.4 from the CUDA devel image; the binary goes to
# $W/pipeline-benchmark/bin/atac_reducers_test. Code arrives by git: the branch is fetched by URL into a private ref and archived into scratch.
#   BR=<branch> (default claude/gpu-atac-reducers-270cf8)
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BR=${BR:-claude/gpu-atac-reducers-270cf8}; URL=$(git -C $BARE config remote.origin.url)
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT INT TERM
git -C $BARE fetch -q "$URL" +refs/heads/$BR:refs/w11/ar-build && mkdir -p $SCR/src && git -C $BARE archive refs/w11/ar-build | tar -x -C $SCR/src
echo "commit $(git -C $BARE rev-parse --short refs/w11/ar-build)"
cat > $SCR/cc.sh <<EOS
cd $SCR/src && export PATH=/usr/local/cuda/bin:\$PATH
nvcc -std=c++17 -O3 -arch=sm_90 -fmad=false -ccbin /usr/bin/g++ -lineinfo -Isrc/engine_gpu -Iinclude bench/atac_reducers/atac_reducers_test.cu -o $SCR/atac_reducers_test 2>&1 | grep -v "^\$" | head -150
echo "nvcc rc=\${PIPESTATUS[0]}"
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/cc.sh
if [ -x $SCR/atac_reducers_test ]; then cp $SCR/atac_reducers_test $W/pipeline-benchmark/bin/atac_reducers_test.new && mv $W/pipeline-benchmark/bin/atac_reducers_test.new $W/pipeline-benchmark/bin/atac_reducers_test && echo "installed"; fi
