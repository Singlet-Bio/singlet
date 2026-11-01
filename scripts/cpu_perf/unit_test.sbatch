#!/bin/bash
#SBATCH --job-name=cpuperf-unit
#SBATCH --account=debruinz
#SBATCH --partition=scavenger
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Compile and run header-only C++ tests from BRANCH (default claude/cpu-perf-c66d3f): TESTS="test_umi_deep_roots test_umi_store_scale ..."
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; ENV=$W/envs/pbench; IMG=$W/images/cuda-12.4.1-devel-ubuntu22.04.sif; BARE=$W/repos/singlet-py.git
BR=${BRANCH:-claude/cpu-perf-c66d3f}; TESTS=${TESTS:-test_umi_deep_roots test_umi_store_scale test_umi_dedup test_molecule_labels}
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BR:refs/cpuperf/unit
SCR=/tmp/$USER-$SLURM_JOB_ID; mkdir -p $SCR/src; trap 'rm -rf $SCR' EXIT
git -C $BARE archive refs/cpuperf/unit | tar -x -C $SCR/src; echo "commit $(git -C $BARE rev-parse refs/cpuperf/unit)"
cat > $SCR/run.sh <<EOS
set -eu
export PATH=$ENV/bin:\$PATH LD_LIBRARY_PATH=$ENV/lib
CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
for t in $TESTS; do
  \$CXX -std=c++17 -O2 -I$SCR/src/include -I$SCR/src/tests/cpp -I$SCR/src/src $SCR/src/tests/cpp/\$t.cpp -o $SCR/\$t -L$ENV/lib -Wl,-rpath,$ENV/lib 2>&1 | head -30
  echo "== \$t"; $SCR/\$t | tail -3
done
EOS
apptainer exec -B $W -B $SCR $IMG bash $SCR/run.sh
