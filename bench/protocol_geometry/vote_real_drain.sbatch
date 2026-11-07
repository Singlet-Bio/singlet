#!/bin/bash
#SBATCH --job-name=s14-voteDrain
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# GPU drain of a sample that fetch_worker (vote_real_samples.sbatch) left in a stage dir. Env (required): STAGE (a stage dir with ready/<GSM>), BIN_SRC (immutable build dir $W/builds/<commit8>
# or another <dir>-<commit> holding bin/singlet-gpu and commit), CFG (the config.json of that run). The build is copied under /tmp keeping its pinned layout, with ../whitelists.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; export PATH=$W/envs/bpub/bin:$W/envs/pbench/bin:$PATH HOME=$W/cache/condahome
STAGE=${STAGE:?}; BIN_SRC=${BIN_SRC:?}; CFG=${CFG:?}; REPO=${REPO:?}
P=/tmp/$USER-$SLURM_JOB_ID; trap 'rm -rf $P' EXIT; C8=$(cut -c1-8 $BIN_SRC/commit); mkdir -p $P/$C8/bin
cp $BIN_SRC/bin/singlet-gpu $P/$C8/bin/ 2>/dev/null || cp $BIN_SRC/singlet-gpu $P/$C8/bin/; cp $BIN_SRC/commit $P/$C8/commit; ln -s $W/pipeline-benchmark/whitelists $P/$C8/whitelists
BIN=$P/$C8/bin/singlet-gpu; echo "engine $(cat $P/$C8/commit | cut -c1-12) sha $(sha256sum $BIN | cut -c1-16)"
python3 - <<PY
import json
c = json.load(open("$CFG")); c["binary"]["path"] = "$BIN"; c["clusters"]["clipper"]["binary"] = "$BIN"; c["clusters"]["clipper"]["scratch"] = "/tmp"
json.dump(c, open("$P/config.json", "w"))
PY
export SINGLET_BIN=$BIN SINGLET_WHITELIST_DIR=$W/pipeline-benchmark/whitelists
python3 -u $REPO/scripts/atlas_rerun/drain_worker.py --stage $STAGE --engine gpu --config $P/config.json --threads 12 --idle-exit-min 2 --max-samples 1 --mem-gb 190 2>&1 | cut -c1-300 | tail -60
echo "--- events"; cat $(python3 -c "import json;print(json.load(open('$P/config.json'))['clusters']['clipper']['root'])")/inbox/*.json 2>/dev/null | cut -c1-900 | tail -5
for d in ready claimed failed; do echo "$d: $(ls $STAGE/$d | tr '\n' ' ')"; done
