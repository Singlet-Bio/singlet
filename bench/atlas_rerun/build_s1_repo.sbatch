#!/bin/bash
#SBATCH --job-name=s6-build-s1
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# A built copy of a synced singlet-py worktree for scripts/atlas_rerun/s1_producer.py --repo: the compiled readers
# singlet._pz and singlet._pz2 sit next to the Python sources (the source worktrees are not built).
#   sbatch --export=ALL,SRCB=bb__prod__campaign build_s1_repo.sbatch      -> $W/jobs/s6/s1repo
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s6/s1repo
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
rm -rf $D; mkdir -p $D
S=$W/src/singlet-py/${SRCB:-bb__prod__campaign}
cp -r $S/python $S/src $S/scripts $S/setup.py $D/
cp $S/pyproject.toml $S/MANIFEST.in $S/README.md $D/ 2>/dev/null || true
cd $D && in_runtime python setup.py build_ext --inplace > build.log 2>&1 || { tail -30 build.log; exit 1; }
ls python/singlet/_pz*.so python/singlet/_pz2*.so || find . -name "_pz*.so"
