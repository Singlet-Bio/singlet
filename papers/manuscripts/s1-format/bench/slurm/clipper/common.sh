# Sourced by the Clipper jobs in this folder (GVSU Clipper, account debruinz). Everything lives under $S1, outside
# $WORK/src (which the lab's sync job prunes), and every job runs one pinned commit of Singlet-Bio/singlet:
#   $S1/src/<sha>/      clone at that commit (code arrives by git, never from a laptop)
#   $S1/env/            conda env from bench/environment.yml, plus BPCells and the singlet package (untimed stages)
#   $S1/env-gpu/        the same env cloned, with the CUDA build of PyTorch (timed stages, GPU and CPU nodes)
#   $S1/data/           S1BENCH_DATA: raw downloads, canonical matrices, format outputs (not committed)
#   $S1/results/        per-stage TSVs and env.json (copied back to the repository after the run)
#   $S1/generated/      report output
#   $S1/logs/           job logs
# Submit with S1_SHA set, e.g. sbatch --export=ALL,S1_SHA=<sha> ... (see README.md in this folder).

WORK=/mnt/projects/debruinz_project/debruinz/singlet
S1=$WORK/s1paper
SHA=${S1_SHA:?set S1_SHA to the commit to run}
SRC=$S1/src/$SHA
BENCH=$SRC/papers/manuscripts/s1-format/bench
S1_ENV=${S1_ENV:-$S1/env}   # the timed jobs use $S1/env-gpu (a clone with the CUDA build of PyTorch)
export S1BENCH_DATA=$S1/data
export CONDA_PKGS_DIRS=$S1/cache/conda-pkgs
export PIP_CACHE_DIR=$S1/cache/pip
export XDG_CACHE_HOME=$S1/cache/xdg
export TMPDIR=/tmp/$USER-${SLURM_JOB_ID:-x}
mkdir -p "$S1"/{src,data,results,generated,logs,cache} "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

# Clone the commit once; array tasks that start together wait on the lock instead of cloning twice.
s1_checkout() {
  (
    flock 9
    if [ ! -d "$SRC" ]; then
      git clone -q https://github.com/Singlet-Bio/singlet.git "$SRC.part"
      git -C "$SRC.part" checkout -q "$SHA"
      mv "$SRC.part" "$SRC"
    fi
  ) 9>"$S1/src/.lock"
  echo "source $(git -C "$SRC" log --oneline -1)"
}

s1_activate() {
  set +u   # lmod and the conda activation scripts read unset variables
  source /etc/profile.d/lmod.sh
  module load miniconda3/25.5.1
  eval "$(conda shell.bash hook)"
  conda activate "$S1_ENV"
  set -u
  export PYTHONPATH="$BENCH"
}

# cpp/decode_bench lives in the clone, so each commit builds its own (once; array tasks wait on the lock).
s1_cpp() {
  (
    flock 9
    [ -x "$BENCH/cpp/decode_bench" ] || CXX=g++ bash "$BENCH/cpp/build.sh"
  ) 9>"$S1/src/.lock-cpp-$SHA"
}

# python -m s1bench with results and generated output under $S1 (data comes from S1BENCH_DATA). S1_RESULTS points a
# cross-check run at a results directory of its own.
s1b() {
  (cd "$BENCH" && python -m s1bench --set "results_dir=${S1_RESULTS:-$S1/results}" --set "generated_dir=$S1/generated" "$@")
}
export PATH="$S1/bin:$PATH"   # toxiproxy-server (netem.sbatch installs it)

# rclone (the S3 store of the network tiers) comes from the cluster's module system.
s1_netem_tools() {
  set +u
  source /etc/profile.d/lmod.sh
  module load rclone/1.74.3
  set -u
}

echo "job ${SLURM_JOB_ID:-?} task ${SLURM_ARRAY_TASK_ID:--} on $(hostname) $(date -Is)"
