#!/bin/bash
#SBATCH --job-name=adt-env
#SBATCH --account=debruinz
#SBATCH --partition=debug-bigmem
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Conda env for the ADT modality: SRA/ENA download, counting, the method benchmark (kallisto|bustools kite,
# alevin-fry) and a C++ compiler for the adt_counter.h harness. Also fetches this branch into its own worktree.
set -euo pipefail
WORK=/mnt/projects/debruinz_project/debruinz/singlet
ENV=$WORK/envs/adt
BRANCH=claude/adt-modality-0637a6
SRC=$WORK/src/singlet-py/${BRANCH//\//__}
export CONDA_PKGS_DIRS=$WORK/cache/conda_pkgs
export HOME=$WORK/cache/condahome; mkdir -p $HOME $CONDA_PKGS_DIRS
source /etc/profile.d/lmod.sh; module load miniforge3

# own branch only, own worktree (never sync.sh)
git -C $WORK/repos/singlet-py.git fetch origin "+refs/heads/$BRANCH:refs/heads/$BRANCH"
if [ -d $SRC ]; then git -C $SRC checkout -q --detach "$BRANCH"; git -C $SRC reset -q --hard "$BRANCH";
else git -C $WORK/repos/singlet-py.git worktree add --detach $SRC "$BRANCH"; fi
git -C $SRC log --oneline -1

if [ ! -x $ENV/bin/python ]; then rm -rf $ENV.tmp; mamba create -y -p $ENV.tmp -c conda-forge -c bioconda --strict-channel-priority \
  python=3.12 sra-tools pigz numpy scipy pandas pyarrow anndata h5py matplotlib zstandard requests \
  dnaio isa-l pybind11 setuptools kallisto bustools alevin-fry salmon simpleaf gxx_linux-64 cmake make zlib zstd && mv $ENV.tmp $ENV; fi
$ENV/bin/pip install -q kb-python==0.29.1 || $ENV/bin/pip install -q kb-python
$ENV/bin/python -c "import numpy, scipy, pandas, anndata, dnaio, h5py; print('py ok', numpy.__version__, dnaio.__version__)"
$ENV/bin/fasterq-dump --version | head -2; $ENV/bin/kallisto version; $ENV/bin/bustools version; $ENV/bin/alevin-fry --version
$ENV/bin/kb --version || true

# the compiled extensions of this branch (singlet._pz decodes legacy GEX matrices); jobs use PYTHONPATH=$SRC/python
export CC=$ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$ENV/bin/x86_64-conda-linux-gnu-g++
cd $SRC && CPATH=$ENV/include LIBRARY_PATH=$ENV/lib $ENV/bin/python setup.py build_ext --inplace -q 2>&1 | tail -5
PYTHONPATH=$SRC/python LD_LIBRARY_PATH=$ENV/lib $ENV/bin/python -c "import singlet._pz, singlet._pz2; print('ext ok')"
