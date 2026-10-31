#!/bin/bash
#SBATCH --job-name=s14-multigate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14 gate of the multi-sample GPU drain worker, ONE H100 job (28 cores, 370 GB, as production). The same mixed queue of real staged samples
# (deep GSM7656149 263 M, GSM6470208 122 M, GSM7036403 112 M, pbmc1k_v3 x3 66.6 M, pbmc5p_v1) is drained twice from identical stages (symlinked .1fq):
#   A  drain_worker.py   one sample at a time, 16 threads (the existing single-sample worker)  -> $K/single/<GSM>
#   B  multi_drain.py    admission by estimated memory/scratch, 8 slots x 12 threads, --genome-shared on  -> $K/multi/<GSM>
# then every sample's outputs are compared with bench/star_accel/out_equiv.py (B vs A) and the report gives pairs/s per GPU, peak cgroup memory,
# peak /tmp use and the admission log. Env: BRANCH (default claude/drain-multisample-52667f), BIN_SRC (dir with singlet-gpu + commit; default jobs/s14/bin_bbf6497d), EXPECT_COMMIT, SKIP_A=1 (reuse $K/single).
set -uo pipefail
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark; PERF=$W/data/atlas-rerun/perf
BIN_SRC=${BIN_SRC:-$W/jobs/s14/bin_bbf6497d}; EXPECT=${EXPECT_COMMIT:-bbf6497d}; BRANCH=${BRANCH:-claude/drain-multisample-52667f}
T=$W/jobs/s14/gate_$SLURM_JOB_ID; K=$T/keep; mkdir -p $T $K/single $K/multi
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

mkstage() {   # $1 stage dir: ready/<GSM>/{manifest.json, sample.1fq -> data}; oldest first = the order below
python3 - "$1" <<'PY'
import json, os, sys, time
stage = sys.argv[1]; W = "/mnt/projects/debruinz_project/debruinz/singlet"; PB = W + "/pipeline-benchmark/data"; PERF = W + "/data/atlas-rerun/perf"
ref = os.path.getsize(PB + "/pbmc1k_v3/pbmc1k_v3.1fq") / 66.6e6          # bytes per pair of the 3' v3 reference sample
Q = [("GSM7036403", PERF + "/GSM7036403/sample.1fq", 111.6e6), ("GSM6470208", PERF + "/GSM6470208/sample.1fq", 121.8e6),
     ("GSM7656149", PERF + "/GSM7656149/sample.1fg2", 263.4e6),
     ("PBMC1KA", PB + "/pbmc1k_v3/pbmc1k_v3.1fq", 66.6e6), ("PBMC1KB", PB + "/pbmc1k_v3/pbmc1k_v3.1fq", 66.6e6),
     ("PBMC1KC", PB + "/pbmc1k_v3/pbmc1k_v3.1fq", 66.6e6), ("PBMC5P", PB + "/pbmc5p_v1/pbmc5p_v1.1fq", None)]
for d in ("todo", "fetching", "incoming", "ready", "claimed", "failed", "deleting"):
    os.makedirs(os.path.join(stage, d), exist_ok=True)
t = time.time() - 1000
for i, (gsm, path, pairs) in enumerate(Q):
    d = os.path.join(stage, "ready", gsm); os.makedirs(d)
    fg2 = path.endswith(".1fg2")     # the deep sample is on the .1fg2 route (#126), the rest on the FASTQ route
    os.symlink(path, os.path.join(d, "sample.1fg2" if fg2 else "sample.1fq"))
    p = int(pairs or os.path.getsize(path) / ref)
    man = {"pairs": p, "bytes": os.path.getsize(path), "runs": ["sample"], "gse": "GSE_S14", "organism": "Homo sapiens"}
    if fg2: man.update(format="1fg2", files=["sample.1fg2"])
    json.dump(man, open(d + "/manifest.json", "w"))
    os.utime(d, (t + i, t + i))
PY
}

sampler() {   # $1 tag: cgroup memory.current, /tmp used and GPU util every 5 s
  while true; do echo "$(date +%s) $(cat $CG/memory.current 2>/dev/null) $(df -B1 --output=used /tmp | tail -1 | tr -d ' ') $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" >> $T/$1.mon; sleep 5; done
}

if [ -z "${SKIP_A:-}" ]; then
  echo "== A: single-sample worker (drain_worker.py, 16 threads)"; mkstage $T/stageA; sampler A & SP=$!
  python3 -u $SRC/drain_worker.py --stage $T/stageA --engine gpu --config $T/config.json --threads 16 --scratch /tmp --mem-gb 0 --accept-1fg2 --idle-exit-min 0.3 \
      --publish $T/keep.sh $K/single {gsm} {out} > $T/drainA.log 2>&1; kill $SP; mv $T/root $T/rootA
  tail -12 $T/drainA.log | cut -c1-220
fi
echo "== B: multi-sample worker (multi_drain.py, 8 x 12 threads, --genome-shared on)"; mkstage $T/stageB; sampler B & SP=$!
python3 -u $SRC/multi_drain.py --stage $T/stageB --config $T/config.json --concurrency 8 --threads 12 --scratch /tmp --mem-gb 370 --route auto --accept-1fg2 --idle-exit-min 0.3 \
    --publish $T/keep.sh $K/multi {gsm} {out} > $T/drainB.log 2>&1; kill $SP
cat $T/drainB.log | cut -c1-260
for s in $(ls $K/multi 2>/dev/null); do ls /dev/shm | head -3; break; done

echo "== out_equiv B vs A"
RC=0
for g in $(ls $K/multi); do
  if [ -d $K/single/$g ]; then
    echo "-- $g"; python3 $T/repo/bench/star_accel/out_equiv.py $K/single/$g/out $K/multi/$g/out 2>&1 | tail -6 || RC=1
  else echo "-- $g: no single-sample reference"; RC=1; fi
done
echo "-- replicates PBMC1KA/B/C of B among themselves"
for g in PBMC1KB PBMC1KC; do python3 $T/repo/bench/star_accel/out_equiv.py $K/multi/PBMC1KA/out $K/multi/$g/out 2>&1 | tail -2; done

echo "== report"
python3 - "$T" "$K" <<'PY'
import glob, json, os, re, sys
T, K = sys.argv[1:3]
def ev(root):
    out = []
    for f in sorted(glob.glob(root + "/inbox/*.json")):
        out.append(json.load(open(f)))
    return out
def reads(d):
    n = 0
    for f in glob.glob(d + "/**/Log.final.out", recursive=True):
        m = re.search(r"Number of input reads \|\s*(\d+)", open(f).read())
        n += int(m.group(1)) if m else 0
    return n
def mon(tag):
    rows = [l.split() for l in open("%s/%s.mon" % (T, tag)) if len(l.split()) == 4]
    return max(int(r[1]) for r in rows) / 1e9, (max(int(r[2]) for r in rows) - min(int(r[2]) for r in rows)) / 1e9, sum(float(r[3]) for r in rows) / len(rows)
for tag, root, kd in (("A", T + "/rootA", K + "/single"), ("B", T + "/root", K + "/multi")):
    if not os.path.isdir(root + "/inbox"): continue
    e = ev(root); run = [x["t"] for x in e if x["event"] == "running"]; done = [x for x in e if x["event"] == "done"]
    wall = max(x["t"] for x in done) - min(run)
    pairs = sum(reads(kd + "/" + x["gsm"]) for x in done) or sum(x["pairs"] for x in done)
    pk, tmpgb, util = mon(tag)
    print("%s: %d done, %d failed/requeued events, wall %.0f s, %.1f M pairs -> %.3f M pairs/s per GPU; peak cgroup memory %.1f GB; /tmp growth %.1f GB; GPU util mean %.0f%%" % (
        tag, len(done), sum(1 for x in e if x["event"] in ("failed", "requeued")), wall, pairs / 1e6, pairs / 1e6 / wall, pk, tmpgb, util))
    for x in sorted(done, key=lambda x: x["gsm"]):
        print("   %-11s %6.1f M pairs  compute %6.0f s  peak_rss %s GB (est %s)  scratch %.1f GB  concurrent %s" % (
            x["gsm"], x["pairs"] / 1e6, x["compute_s"], x.get("peak_rss_gb"), x.get("est_peak_gb"), (x.get("scratch_bytes") or 0) / 1e9, x.get("concurrent")))
PY
[ "$(sha256sum $BIN | cut -c1-16)" = "$BSHA" ] || { echo "BINARY CHANGED"; RC=1; }
echo "engine commit $(cat $BIN_SRC/commit) sha256 $BSHA (same binary for A and B)"
echo "gate rc=$RC $(date -Is); results in $T"
