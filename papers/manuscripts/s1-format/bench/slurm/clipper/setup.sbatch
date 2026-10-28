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
if [ ! -x "$S1/env/bin/python" ]; then
  conda env create -q -p "$S1/env" -f "$BENCH/environment.yml"
elif ! cmp -s "$BENCH/environment.yml" "$S1/env/.environment.yml"; then
  conda env update -q -p "$S1/env" -f "$BENCH/environment.yml"
fi
cp "$BENCH/environment.yml" "$S1/env/.environment.yml"
s1_activate

Rscript -e 'if (!requireNamespace("BPCells", quietly = TRUE)) remotes::install_github("bnprks/BPCells/r", upgrade = "never")'
Rscript -e 'library(BPCells); cat("BPCells", as.character(packageVersion("BPCells")), "\n")'

# The singlet package at this commit (builds singlet._pz2). --no-deps: the env already pins the dependencies.
pip install -q --no-deps --no-build-isolation "$SRC"
python -c "import singlet, singlet._pz2; print('singlet', getattr(singlet, '__version__', '?'))"
R CMD INSTALL --no-test-load "$SRC/r" >"$TMPDIR/r-install.log" 2>&1 \
  && echo "singlet R package installed" || { tail -20 "$TMPDIR/r-install.log"; echo "singlet R package failed (needed by eco only)"; }
CXX=g++ bash "$BENCH/cpp/build.sh"

conda env export -p "$S1/env" --no-builds >"$S1/results/environment.lock.yml"

cd "$BENCH"
python -m pytest -q tests/test_floor.py
python tests/smoke.py
s1b env
echo "setup ok"
