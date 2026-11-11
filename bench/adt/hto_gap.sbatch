#!/bin/bash
#SBATCH --job-name=hto-gap
#SBATCH --account=debruinz
#SBATCH --partition=short-bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# Hashtag caller: leave-one-dataset-out selection with the current criterion, then before/after scores against the published callers.
# BRANCH below; development data only (Howitt et al. 2023 Zenodo 8304003: BAL batches and the cell-line mix).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; BARE=$W/repos/singlet-py.git; BRANCH=fix/hto-selection-criterion
OUT=$W/jobs/hto/out; mkdir -p $OUT
git -C $BARE fetch -q "$(git -C $BARE config remote.origin.url)" +refs/heads/$BRANCH:refs/hto/head
C=$(git -C $BARE rev-parse refs/hto/head); echo "commit $C"
T=$(mktemp -d /tmp/hto.XXXX); trap 'rm -rf $T' EXIT
git -C $BARE archive $C include bench/adt | tar -x -C $T
export PATH=$W/envs/pbench/bin:$PATH
g++ -O2 -std=c++17 -I$T/include -o $T/hto_demux_cli $T/bench/adt/hto_demux_cli.cpp || exit 3
PY=$W/envs/adt-bench/bin/python; Z=$W/data/adt_bench/zenodo_8304003; A=$W/release-gate4/artifacts/hto
$PY -P $T/bench/adt/hto_lodo.py --zenodo $Z --out $OUT/selection.json | tail -c 1500
$PY -P $T/bench/adt/hto_gap_score.py --zenodo $Z --exe $T/hto_demux_cli --artifacts $A --selection $OUT/selection.json --out $OUT/gap_score.json
