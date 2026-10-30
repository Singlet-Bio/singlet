#!/bin/bash
#SBATCH --job-name=s6-build-s1
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:40:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# A copy of a synced singlet-py worktree plus the compiled readers, for scripts/atlas_rerun/s1_producer.py --repo: the
# singlet._pz and singlet._pz2 sit next to the Python sources (the source worktrees are not built).
#   sbatch --export=ALL,SRCB=bb__prod__campaign build_s1_repo.sbatch      -> $W/jobs/s6/s1repo
set -euo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; D=$W/jobs/s6/s1repo
source $W/src/singlet-agents/feat__cell-phenotype/hpc/clipper/in_runtime.sh
rm -rf $D; mkdir -p $D
S=$W/src/singlet-py/${SRCB:-bb__prod__campaign}
cp -r $S/python $S/src $S/include $S/scripts $S/setup.py $D/
cp $S/pyproject.toml $S/MANIFEST.in $S/README.md $D/ 2>/dev/null || true
# The runtime venv has no setuptools/distutils (python 3.12), so install them to a temp dir for the build only. The readers must
# come from the same source as the Python code: another worktree's .so has a different _pz2.encode signature.
cd $D
in_runtime python -m pip install -q --target $D/bt setuptools wheel pybind11 numpy > pip.log 2>&1 || { tail -5 pip.log; exit 1; }
in_runtime env PYTHONPATH=$D/bt SETUPTOOLS_USE_DISTUTILS=local python setup.py build_ext --inplace > build.log 2>&1 || { tail -30 build.log; exit 1; }
PYTHONPATH=$D/python in_runtime python -c "import singlet._pz, singlet._pz2; print('readers ok')"
