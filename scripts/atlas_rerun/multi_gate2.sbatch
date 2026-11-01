#!/bin/bash
#SBATCH --job-name=s14-multigate2
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14 gate, part 2 (one H100 job, same pinned engine as multi_gate.sbatch; read its header for the harness):
#   C  identity at EQUAL threads: the single-sample worker at 12 threads on the deep .1fg2 sample (GSM7656149) and one small FASTQ-route sample -> $K/single12,
#      to be compared with the multi worker's outputs of part 1 (12 threads, concurrent) and the 16-thread single outputs. Tells thread-count effects from concurrency effects.
#   D  throughput at steady state: multi_drain.py on a larger queue (13 samples, 1.1 G pairs, publish dry-run), 8 slots x 12 threads, with a cgroup sampler of anon+shmem,
#      page cache and the admitted estimate; the single-sample rate is the 16-thread worker's 0.144 M pairs/s per GPU of part 1 (per-sample rates 0.11-0.17, size independent).
# Env: BIN_SRC, EXPECT_COMMIT as in part 1; SKIP_C=1, SKIP_D=1.
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; PERF=$W/data/atlas-rerun/perf
BIN_SRC=${BIN_SRC:-$W/jobs/s14/bin_bbf6497d}; EXPECT=${EXPECT_COMMIT:-bbf6497d}; BRANCH=${BRANCH:-claude/drain-multisample-52667f}
T=$W/jobs/s14/gate2_$SLURM_JOB_ID; K=$T/keep; mkdir -p $T $K/single12
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
mkdir -p "$1/$2" && cp -r "$3" "$1/$2/" && cp "$(dirname "$3")/singlet.log" "$(dirname "$3")/time.txt" "$1/$2/" 2>/dev/null; true
SH
chmod +x $T/keep.sh


mkstage() {   # $1 stage dir, $2 set: "C" (identity subset) or "D" (steady-state queue)
python3 - "$1" "$2" <<'PY'
import json, os, sys, time
stage, qset = sys.argv[1:3]; W = "/mnt/projects/debruinz_project/debruinz/singlet"; PB = W + "/pipeline-benchmark/data"; PERF = W + "/data/atlas-rerun/perf"
ref = os.path.getsize(PB + "/pbmc1k_v3/pbmc1k_v3.1fq") / 66.6e6
P1 = (PB + "/pbmc1k_v3/pbmc1k_v3.1fq", 66.6e6); P5 = (PB + "/pbmc5p_v1/pbmc5p_v1.1fq", None)
Q = {"C": [("GSM7656149", PERF + "/GSM7656149/sample.1fg2", 263.4e6), ("PBMC1KA", ) + P1],
     "D": [("GSM7036403", PERF + "/GSM7036403/sample.1fq", 111.6e6), ("PBMC1KA",) + P1, ("PBMC1KB",) + P1, ("GSM6470208", PERF + "/GSM6470208/sample.1fq", 121.8e6),
           ("PBMC1KC",) + P1, ("GSM7656149", PERF + "/GSM7656149/sample.1fg2", 263.4e6), ("PBMC1KD",) + P1, ("PBMC5PA",) + P5, ("PBMC1KE",) + P1,
           ("PBMC1KF",) + P1, ("PBMC5PB",) + P5, ("PBMC1KG",) + P1, ("PBMC1KH",) + P1]}[qset]
for d in ("todo", "fetching", "incoming", "ready", "claimed", "failed", "deleting"):
    os.makedirs(os.path.join(stage, d), exist_ok=True)
t = time.time() - 1000
for i, (gsm, path, pairs) in enumerate(Q):
    d = os.path.join(stage, "ready", gsm); os.makedirs(d)
    fg2 = path.endswith(".1fg2")
    os.symlink(path, os.path.join(d, "sample.1fg2" if fg2 else "sample.1fq"))
    man = {"pairs": int(pairs or os.path.getsize(path) / ref), "bytes": os.path.getsize(path), "runs": ["sample"], "gse": "GSE_S14", "organism": "Homo sapiens"}
    if fg2: man.update(format="1fg2", files=["sample.1fg2"])
    json.dump(man, open(d + "/manifest.json", "w"))
    os.utime(d, (t + i, t + i))
PY
}

sampler() {   # $1 tag: time, cgroup anon+shmem, page cache, memory.current, /tmp used, GPU util, every 5 s
  while true; do
    an=$(awk '$1=="anon"{a=$2} $1=="shmem"{s=$2} $1=="file"{f=$2} END{print a+0, s+0, f+0}' $CG/memory.stat)
    echo "$(date +%s) $(cat $CG/memory.current) $an $(df -B1 --output=used /tmp | tail -1 | tr -d ' ') $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" >> $T/$1.mon; sleep 5
  done
}

if [ -z "${SKIP_C:-}" ]; then
  echo "== C: single-sample worker at 12 threads (deep .1fg2 + one small), equal threads to the multi worker"; mkstage $T/stageC C
  python3 -u $SRC/drain_worker.py --stage $T/stageC --engine gpu --config $T/config.json --threads 12 --scratch /tmp --mem-gb 0 --accept-1fg2 --idle-exit-min 0.3 \
      --publish $T/keep.sh $K/single12 {gsm} {out} > $T/drainC.log 2>&1
  cat $T/drainC.log | cut -c1-200
fi
if [ -z "${SKIP_D:-}" ]; then
  echo "== D: multi-sample worker on the 13-sample steady-state queue (8 x 12 threads, publish dry-run)"; mkstage $T/stageD D; mv $T/root $T/rootC 2>/dev/null; sampler D & SP=$!
  python3 -u $SRC/multi_drain.py --stage $T/stageD --config $T/config.json --concurrency 8 --threads 12 --scratch /tmp --mem-gb 370 --route auto --accept-1fg2 --idle-exit-min 0.3 \
      --publish dry-run > $T/drainD.log 2>&1; kill $SP
  cat $T/drainD.log | cut -c1-250
fi

echo "== report"
python3 - "$T" <<'PY'
import glob, json, os, sys
T = sys.argv[1]
def ev(root): return [json.load(open(f)) for f in sorted(glob.glob(root + "/inbox/*.json"))]
for tag, root in (("C (single, 12 thr)", T + "/rootC"), ("D (multi, steady state)", T + "/root")):
    if not os.path.isdir(root + "/inbox"): continue
    e = ev(root); run = [x["t"] for x in e if x["event"] == "running"]; done = [x for x in e if x["event"] == "done"]
    if not done: continue
    wall = max(x["t"] for x in done) - min(run); pairs = sum(x["pairs"] for x in done)
    print("%s: %d done, %d failed/requeued, wall %.0f s, %.1f M pairs (manifest) -> %.3f M pairs/s per GPU" % (tag, len(done), sum(1 for x in e if x["event"] in ("failed", "requeued")), wall, pairs / 1e6, pairs / 1e6 / wall))
    for x in sorted(done, key=lambda x: x["gsm"]):
        print("   %-11s %6.1f M  compute %6.0f s  child peak %s GB (est cgroup share %s)  concurrent at start %s" % (x["gsm"], x["pairs"] / 1e6, x["compute_s"], x.get("peak_rss_gb"), x.get("est_peak_gb"), x.get("concurrent")))
if os.path.exists(T + "/D.mon"):
    r = [l.split() for l in open(T + "/D.mon") if len(l.split()) == 7]
    g = lambda i: max(int(x[i]) for x in r) / 1e9
    print("D cgroup peaks: memory.current %.1f GB, anon+shmem %.1f GB, page cache %.1f GB; /tmp growth %.1f GB; GPU util mean %.0f%%" % (
        g(1), max((int(x[2]) + int(x[3])) for x in r) / 1e9, g(4), (max(int(x[5]) for x in r) - min(int(x[5]) for x in r)) / 1e9, sum(float(x[6]) for x in r) / len(r)))
PY
[ "$(sha256sum $BIN | cut -c1-16)" = "$BSHA" ] || echo "BINARY CHANGED"
echo "engine commit $(cat $BIN_SRC/commit) sha256 $BSHA"
echo "gate2 done $(date -Is); results in $T"
