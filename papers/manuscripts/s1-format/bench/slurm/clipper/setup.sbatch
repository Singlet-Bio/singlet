#!/bin/bash
# Build the runtime for one commit, then run the unit tests and the synthetic smoke test. Rerun after environment.yml or
# the singlet package changes; it only rebuilds what changed.
#SBATCH --job-name=s1-setup
#SBATCH --account=debruinz
#SBATCH --partition=short-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/s1paper/logs/%x-%j.out
set -euo pipefail
# The clone has to exist before common.sh (which lives in it) can be sourced.
S1=/mnt/projects/debruinz_project/debruinz/singlet/s1paper
SRC=$S1/src/${S1_SHA:?set S1_SHA}
mkdir -p "$S1/src"
(
  flock 9
  if [ ! -d "$SRC" ]; then
    git clone -q https://github.com/Singlet-Bio/singlet.git "$SRC.part"
    git -C "$SRC.part" checkout -q "$S1_SHA"
    mv "$SRC.part" "$SRC"
  fi
) 9>"$S1/src/.lock"
source "/mnt/projects/debruinz_project/debruinz/singlet/s1paper/src/${S1_SHA:?set S1_SHA}/papers/manuscripts/s1-format/bench/slurm/clipper/common.sh"
s1_checkout

set +u
source /etc/profile.d/lmod.sh
module load miniconda3/25.5.1
eval "$(conda shell.bash hook)"
set -u
# S1_ENV picks the env (default $S1/env); with S1_ENV_FROM a new env starts as a clone of that one, so only what
# environment.yml changed is solved again (S1_ENV=$S1/env-gpu S1_ENV_FROM=$S1/env for the timed jobs).
if [ ! -x "$S1_ENV/bin/python" ] && [ -n "${S1_ENV_FROM:-}" ]; then
  conda create -q -y -p "$S1_ENV" --clone "$S1_ENV_FROM"
fi
if [ ! -x "$S1_ENV/bin/python" ]; then
  conda env create -q -p "$S1_ENV" -f "$BENCH/environment.yml"
elif ! cmp -s "$BENCH/environment.yml" "$S1_ENV/.environment.yml"; then
  conda env update -q -p "$S1_ENV" -f "$BENCH/environment.yml"
fi
cp "$BENCH/environment.yml" "$S1_ENV/.environment.yml"
s1_activate
# CUDA PyTorch: conda-forge's CUDA builds need the __cuda_arch virtual package, which the cluster's conda (25.5.1) does not
# provide, so with S1_TORCH_INDEX set (the timed env) the conda CPU build is replaced by the wheel from that index.
if [ -n "${S1_TORCH_INDEX:-}" ] && ! python -c "import sys, torch; sys.exit(torch.version.cuda is None)"; then
  for p in pytorch libtorch; do
    if conda list -p "$S1_ENV" "^$p\$" | grep -q "^$p "; then conda remove -q -y -p "$S1_ENV" --force "$p"; fi
  done
  T="torch==${S1_TORCH_VERSION:-2.13.0+cu129}"
  pip install -q "$T" --index-url "$S1_TORCH_INDEX" --extra-index-url https://pypi.org/simple
  # an env update reinstalls the conda build over the wheel, and removing it takes shared files with it: rewrite them
  pip install -q --force-reinstall --no-deps "$T" --index-url "$S1_TORCH_INDEX"
fi
python -c "import torch; print('torch', torch.__version__, 'cuda build', torch.version.cuda)"

Rscript -e 'if (!requireNamespace("BPCells", quietly = TRUE)) remotes::install_github("bnprks/BPCells/r", upgrade = "never")'
Rscript -e 'library(BPCells); cat("BPCells", as.character(packageVersion("BPCells")), "\n")'

# The singlet package at this commit (builds singlet._pz2). --no-deps: the env already pins the dependencies.
pip install -q --no-deps --no-build-isolation "$SRC"
python -c "import singlet, singlet._pz2; print('singlet', getattr(singlet, '__version__', '?'))"
R CMD INSTALL --no-test-load "$SRC/r" >"$TMPDIR/r-install.log" 2>&1 \
  && echo "singlet R package installed" || { tail -20 "$TMPDIR/r-install.log"; echo "singlet R package failed (needed by eco only)"; }
s1_cpp

conda env export -p "$S1_ENV" --no-builds >"$S1/results/environment.lock$( [ "$S1_ENV" = "$S1/env" ] || echo ".$(basename "$S1_ENV")").yml"

cd "$BENCH"
python -m pytest -q tests/test_floor.py
python tests/smoke.py
s1b env
echo "setup ok"
