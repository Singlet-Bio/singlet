#!/bin/bash
#SBATCH --job-name=s14-multigate3
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14 gate, part 3: is the deep sample's thread-count dependence (12 vs 16 threads: 4820 vs 4838 cells) the .1fg2 route or the engine? GSM7656149 on the FASTQ route
# (.1fq) through the single-sample worker at 12 and at 16 threads -> $K/fq12, $K/fq16, then out_equiv of fq12 vs fq16 and against the .1fg2 runs of parts 1 and 2.
# Env: P1 (part-1 gate dir), P2 (part-2 gate dir), BIN_SRC, EXPECT_COMMIT as before.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; PERF=$W/data/atlas-rerun/perf
BIN_SRC=${BIN_SRC:-$W/jobs/s14/bin_bbf6497d}; EXPECT=${EXPECT_COMMIT:-bbf6497d}; BRANCH=${BRANCH:-claude/drain-multisample-52667f}
T=$W/jobs/s14/gate3_$SLURM_JOB_ID; K=$T/keep; mkdir -p $T $K/fq12 $K/fq16
SCR=/tmp/$USER-s14-$SLURM_JOB_ID; mkdir -p $SCR; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=7200 SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference PATH=$W/envs/pbench/bin:$PATH SINGLET_GPU_PILEUP=on
# the engine under test: its commit is asserted, and a private copy on node-local disk is run, so nobody rebuilding the shared directory can change it mid-gate
case "$(cat $BIN_SRC/commit 2>/dev/null)" in $EXPECT*) ;; *) echo "ABORT: $BIN_SRC/commit is '$(cat $BIN_SRC/commit 2>/dev/null)', expected $EXPECT"; exit 3;; esac
cp $BIN_SRC/singlet-gpu $SCR/singlet-gpu; BIN=$SCR/singlet-gpu; BSHA=$(sha256sum $BIN | cut -c1-16)
echo "== node $(hostname) $(date -Is) engine commit $(cat $BIN_SRC/commit) sha256 $BSHA"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CG=/sys/fs/cgroup$(cut -d: -f3 /proc/self/cgroup | head -1 | sed -E 's#(/job_[0-9]+).*#\1#'); echo "cgroup $CG memory.max=$(cat $CG/memory.max 2>/dev/null) df /tmp: $(df -BG --output=size,avail /tmp | tail -1)"
cd $T && git clone -q --depth 1 -b $BRANCH git@github.com:Singlet-Bio/singlet.git repo 2>&1 | tail -1; cd $T/repo && git log --oneline -1
SRC=$T/repo/scripts/atlas_rerun

python3 - <<PY
import json
c = json.load(open("$SRC/config.json")); k = c["clusters"]["clipper"]
k.update(root="$T/root", whitelist_dir="$SINGLET_WHITELIST_DIR", scratch="/tmp"); c["binary"]["path"] = "$BIN"
json.dump(c, open("$T/config.json", "w"))
PY
cat > $T/keep.sh <<'SH'
#!/bin/bash
# publish hook: keep the outputs for the comparison (real publish is not exercised here)
mkdir -p "$1/$2" && cp -r "$3" "$1/$2/"
SH
chmod +x $T/keep.sh



P1=${P1:-$W/jobs/s14/gate_795047}; P2=${P2:-$W/jobs/s14/gate2_795246}
for th in 12 16; do
  st=$T/stage$th; mkdir -p $st; python3 - $st <<'PY'
import json, os, sys
stage = sys.argv[1]; W = "/mnt/projects/debruinz_project/debruinz/singlet"
for d in ("todo", "fetching", "incoming", "ready", "claimed", "failed", "deleting"): os.makedirs(os.path.join(stage, d), exist_ok=True)
d = os.path.join(stage, "ready", "GSM7656149"); os.makedirs(d)
os.symlink(W + "/data/atlas-rerun/perf/GSM7656149/sample.1fq", d + "/sample.1fq")
json.dump({"pairs": 263400000, "bytes": 1, "runs": ["sample"], "gse": "GSE_S14", "organism": "Homo sapiens"}, open(d + "/manifest.json", "w"))
PY
  echo "== .1fq route, $th threads"
  python3 -u $SRC/drain_worker.py --stage $st --engine gpu --config $T/config.json --threads $th --scratch /tmp --mem-gb 0 --idle-exit-min 0.3 \
      --publish $T/keep.sh $K/fq$th {gsm} {out} > $T/drain$th.log 2>&1; cut -c1-200 $T/drain$th.log
done
export PATH=$W/envs/pbench/bin:$PATH
EQ=$T/repo/bench/star_accel/out_equiv.py
cmp3() { echo "=== $1"; python3 $EQ $2 $3 2>&1 | grep -v "^    [ab]:" | cut -c1-200 | head -${4:-14}; }
cmp3 "fq 12 thr vs fq 16 thr (threads, FASTQ route)" $K/fq12/GSM7656149/out $K/fq16/GSM7656149/out
cmp3 "fq 12 thr vs fg2 12 thr (route at 12 threads)" $K/fq12/GSM7656149/out $P2/keep/single12/GSM7656149/out
cmp3 "fq 16 thr vs fg2 16 thr (route at 16 threads)" $K/fq16/GSM7656149/out $P1/keep/single/GSM7656149/out
echo "gate3 done $(date -Is); results in $T"
