#!/bin/bash
#SBATCH --job-name=bp-sync
#SBATCH --account=debruinz
#SBATCH --partition=debug-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/bench-public/logs/%x-%j.out
#   sbatch sync_src.sbatch [BRANCH] [SHA] [DIR]
# Checks out the benchmark branch of Singlet-Bio/singlet into $BP/src/DIR (default: singlet); the performance harness and the
# capability modules run from there (jobs find it through BP_SRC, default $BP/src/singlet). Clipper never receives code from the
# laptop: it comes from GitHub, pinned to a commit.
set -eu
BR=${1:-bench/public-perf}; WANT=${2:-}; DIR=${3:-singlet}
BP=/mnt/projects/debruinz_project/debruinz/singlet/bench-public
mkdir -p $BP/src $BP/logs; cd $BP/src
[ -d $DIR/.git ] || git clone -q --filter=blob:none git@github.com:Singlet-Bio/singlet.git $DIR
cd $DIR
git fetch -q origin "$BR"
git checkout -q -f ${WANT:-FETCH_HEAD}
git log --oneline -1
