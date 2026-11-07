#!/bin/bash
#SBATCH --job-name=s14-multigate
#SBATCH --account=debruinz
#SBATCH --partition=short-gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --cpus-per-task=28
#SBATCH --mem=370G
#SBATCH --time=03:59:00
#SBATCH --output=/mnt/projects/debruinz_project/debruinz/singlet/logs/%x-%j.out
# S14 gate of the multi-sample GPU drain worker: ONE H100 job (28 cores, 370 GB, as production), one pinned engine, real staged samples (symlinked .1fq / .1fg2).
#   queue M (mixed, 7 samples, 745 M pairs): deep GSM7656149 (.1fg2 route, 263 M), GSM6470208 122 M, GSM7036403 112 M, pbmc1k_v3 x3 (66.6 M), pbmc5p_v1 (48 M)
#   queue D (steady state, 13 samples, 1.1 G pairs): M plus five more pbmc1k_v3 / pbmc5p_v1 copies
#   B    multi_drain.py on M      8 slots x 12 threads, --genome-shared on, outputs kept          -> $K/multi
#   D    multi_drain.py on D      same, publish dry-run: throughput at steady state
#   A12  drain_worker.py on M     one sample at a time, 12 threads (EQUAL threads to B), outputs kept -> $K/single12   (identity reference)
#   A16  drain_worker.py on M     one sample at a time, 16 threads = the existing production worker: the throughput baseline (SKIP_A16=1 skips it)
# then out_equiv of B vs A12 per sample (write-order files also as sorted record multisets), and the report: pairs/s per GPU, peak anon+shmem / page cache of
# the cgroup, /tmp use, GPU utilisation, and THE PATH CHECK: every sample must have pileup_threads == 1 (stream / GPU pileup). The engine is run from a private
# bin/ directory with ../whitelists beside it: singlet resolves <binary dir>/../whitelists, and without it a run silently takes the sorted-BAM pileup
# (pileup_threads = thread count), a different and slower path (S8). The workers refuse to start without it (common.require_whitelists).
# Env: BIN_SRC (dir with singlet-gpu + commit; required, no default), EXPECT_COMMIT (required), BRANCH (default claude/drain-multisample-52667f), SKIP_A16.
set -uo pipefail
# the engine must be an immutable, identified build: <dir>-<commit>/bin/<binary> (or <dir>/<binary>) beside a `commit` file starting with that commit (never a shared mutable directory)
pin() { local r d c; r=$(readlink -f "$1"); d=$(dirname "$r"); [ "$(basename $d)" = bin ] && d=$(dirname $d); c=$(basename $d); c=${c##*[-_]}; [[ $c =~ ^[0-9a-f]{7,40}$ ]] && [ -x "$r" ] && case "$(cat $d/commit 2>/dev/null)" in $c*) return 0;; esac; echo "ABORT: $1 is not a pinned build (<dir>-<commit>/bin/<binary> with a commit file starting with the commit)"; exit 3; }
W=/mnt/projects/debruinz_project/debruinz/singlet; PB=$W/pipeline-benchmark
BIN_SRC=${BIN_SRC:?set BIN_SRC=$W/builds/<commit8> (immutable build dir with bin/singlet-gpu and commit)}; EXPECT=${EXPECT_COMMIT:?set EXPECT_COMMIT=<commit8 of the build>}; pin "$BIN_SRC/bin/singlet-gpu"; BRANCH=${BRANCH:-claude/drain-multisample-52667f}
T=$W/jobs/s14/gate4_$SLURM_JOB_ID; K=$T/keep; mkdir -p $T $K/multi $K/single12 $K/single16
SCR=/tmp/$USER-s14-$SLURM_JOB_ID; mkdir -p $SCR/$EXPECT/bin; trap 'rm -rf $SCR' EXIT
export SINGLET_WHITELIST_DIR=$PB/whitelists SINGLET_STALL_TIMEOUT=7200 SINGLET_REF_BASE=/mnt/projects/debruinz_project/cellarium/reference SINGLET_GPU_PILEUP=on
export PATH=$W/envs/pbench/bin:$PATH
case "$(cat $BIN_SRC/commit 2>/dev/null)" in $EXPECT*) ;; *) echo "ABORT: $BIN_SRC/commit is '$(cat $BIN_SRC/commit 2>/dev/null)', expected $EXPECT"; exit 3;; esac
cp $BIN_SRC/bin/singlet-gpu $SCR/$EXPECT/bin/singlet-gpu; cp $BIN_SRC/commit $SCR/$EXPECT/commit; ln -s $PB/whitelists $SCR/$EXPECT/whitelists; BIN=$SCR/$EXPECT/bin/singlet-gpu; pin "$BIN"; BSHA=$(sha256sum $BIN | cut -c1-16)
echo "== node $(hostname) $(date -Is) engine commit $(cat $BIN_SRC/commit) sha256 $BSHA; whitelists next to the binary: $(ls $SCR/$EXPECT/whitelists/ | head -3 | tr '\n' ' ')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
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
# publish hook: keep the outputs (and the engine log) for the comparison; real publish is not exercised here
mkdir -p "$1/$2" && cp -r "$3" "$1/$2/" && cp "$(dirname "$3")/singlet.log" "$(dirname "$3")/time.txt" "$1/$2/" 2>/dev/null; true
SH
cat > $T/keep_small.sh <<'SH'
#!/bin/bash
# publish hook for the baseline: only what the path check needs
mkdir -p "$1/$2/out" && cp "$3/pileup_stats.json" "$3/summary.json" "$1/$2/out/" && cp "$(dirname "$3")/singlet.log" "$(dirname "$3")/time.txt" "$1/$2/" 2>/dev/null; true
SH
chmod +x $T/keep.sh $T/keep_small.sh

mkstage() {   # $1 stage dir, $2 queue "M" or "D"; oldest first = the order below
python3 - "$1" "$2" <<'PY'
import json, os, sys, time
stage, qset = sys.argv[1:3]; W = "/mnt/projects/debruinz_project/debruinz/singlet"; PB = W + "/pipeline-benchmark/data"; PERF = W + "/data/atlas-rerun/perf"
ref = os.path.getsize(PB + "/pbmc1k_v3/pbmc1k_v3.1fq") / 66.6e6          # bytes per pair of the 3' v3 reference sample
P1 = (PB + "/pbmc1k_v3/pbmc1k_v3.1fq", 66.6e6); P5 = (PB + "/pbmc5p_v1/pbmc5p_v1.1fq", None)
M = [("GSM7036403", PERF + "/GSM7036403/sample.1fq", 111.6e6), ("GSM6470208", PERF + "/GSM6470208/sample.1fq", 121.8e6),
     ("GSM7656149", PERF + "/GSM7656149/sample.1fg2", 263.4e6), ("PBMC1KA",) + P1, ("PBMC1KB",) + P1, ("PBMC1KC",) + P1, ("PBMC5P",) + P5]
D = M + [("PBMC1KD",) + P1, ("PBMC5PA",) + P5, ("PBMC1KE",) + P1, ("PBMC1KF",) + P1, ("PBMC5PB",) + P5, ("PBMC1KG",) + P1]
Q = {"M": M, "D": D}[qset]
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

sampler() {   # $1 tag: time, memory.current, anon, shmem, page cache, /tmp used, GPU util (7 fields) every 5 s
  while true; do
    an=$(awk '$1=="anon"{a=$2} $1=="shmem"{s=$2} $1=="file"{f=$2} END{print a+0, s+0, f+0}' $CG/memory.stat)
    echo "$(date +%s) $(cat $CG/memory.current) $an $(df -B1 --output=used /tmp | tail -1 | tr -d ' ') $(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)" >> $T/$1.mon; sleep 5
  done
}

run_multi() {   # $1 tag, $2 queue, $3.. publish hook words
  local tag=$1 q=$2; shift 2
  echo "== $tag: multi_drain.py on queue $q (8 slots x 12 threads, --genome-shared on)"; mkstage $T/stage$tag $q; sampler $tag & local sp=$!
  python3 -u $SRC/multi_drain.py --stage $T/stage$tag --config $T/config.json --concurrency 8 --threads 12 --scratch /tmp --mem-gb 370 --route auto --accept-1fg2 \
      --idle-exit-min 0.3 --publish "$@" > $T/drain$tag.log 2>&1; kill $sp
  cut -c1-250 $T/drain$tag.log; mv $T/root $T/root$tag
}
run_single() {  # $1 tag, $2 threads, $3.. publish hook words
  local tag=$1 th=$2; shift 2
  echo "== $tag: drain_worker.py on queue M (one sample at a time, $th threads)"; mkstage $T/stage$tag M; sampler $tag & local sp=$!
  python3 -u $SRC/drain_worker.py --stage $T/stage$tag --engine gpu --config $T/config.json --threads $th --scratch /tmp --mem-gb 0 --accept-1fg2 --idle-exit-min 0.3 \
      --publish "$@" > $T/drain$tag.log 2>&1; kill $sp
  cut -c1-200 $T/drain$tag.log; mv $T/root $T/root$tag
}

run_multi B M $T/keep.sh $K/multi {gsm} {out}
run_multi D D dry-run
run_single A12 12 $T/keep.sh $K/single12 {gsm} {out}
[ -z "${SKIP_A16:-}" ] && run_single A16 16 $T/keep_small.sh $K/single16 {gsm} {out}

echo "== out_equiv: multi (B, 12 threads, concurrent) vs single (A12, 12 threads)"
PY3=$W/envs/pbench/bin/python3; EQ=$T/repo/bench/star_accel/out_equiv.py; RC=0
for g in $(ls $K/multi); do
  echo "-- $g"; $PY3 $EQ $K/single12/$g/out $K/multi/$g/out 2>&1 | grep -v "^    [ab]:" | cut -c1-220 | head -30
  for f in star_Unmapped.out.mate1 star_Unmapped.out.mate2; do
    a=$(LC_ALL=C sort -S 8G $K/single12/$g/out/$f | sha256sum | cut -c1-12); b=$(LC_ALL=C sort -S 8G $K/multi/$g/out/$f | sha256sum | cut -c1-12); [ $a = $b ] && echo "   $f as a multiset: same" || { echo "   $f as a multiset: DIFFERENT"; RC=1; }
  done
  a=$(paste - - - - < $K/single12/$g/out/nonhost_reads.fq | LC_ALL=C sort -S 8G | sha256sum | cut -c1-12); b=$(paste - - - - < $K/multi/$g/out/nonhost_reads.fq | LC_ALL=C sort -S 8G | sha256sum | cut -c1-12)
  [ $a = $b ] && echo "   nonhost_reads.fq as a multiset of records: same" || { echo "   nonhost_reads.fq as a multiset: DIFFERENT"; RC=1; }
done

echo "== report"
python3 - "$T" "$K" <<'PY'
import glob, json, os, sys
T, K = sys.argv[1:3]
def ev(root): return [json.load(open(f)) for f in sorted(glob.glob(root + "/inbox/*.json"))]
def pt(d):
    try: return json.load(open(d + "/pileup_stats.json")).get("pileup_threads")
    except Exception: return None
bad = 0
for tag, kd in (("B", K + "/multi"), ("D", None), ("A12", K + "/single12"), ("A16", K + "/single16")):
    root = T + "/root" + tag
    if not os.path.isdir(root + "/inbox"): continue
    e = ev(root); run = [x["t"] for x in e if x["event"] == "running"]; done = [x for x in e if x["event"] == "done"]
    if not done: print(tag, "no sample finished"); continue
    wall = max(x["t"] for x in done) - min(run); pairs = sum(x["pairs"] for x in done)
    print("%s: %d done, %d failed/requeued events, wall %.0f s, %.1f M pairs -> %.3f M pairs/s per GPU" % (tag, len(done), sum(1 for x in e if x["event"] in ("failed", "requeued")), wall, pairs / 1e6, pairs / 1e6 / wall))
    for x in sorted(done, key=lambda x: x["gsm"]):
        p = x.get("pileup_threads") if x.get("pileup_threads") is not None else (pt(kd + "/" + x["gsm"] + "/out") if kd else None)
        if p not in (1, None): bad += 1
        print("   %-11s %6.1f M  compute %6.0f s  child peak %s GB (est private %s)  concurrent %s  pileup_threads %s%s" % (
            x["gsm"], x["pairs"] / 1e6, x["compute_s"], x.get("peak_rss_gb"), x.get("est_peak_gb"), x.get("concurrent"), p, "" if p in (1, None) else "   <-- SORTED-BAM PATH"))
    f = T + "/" + tag + ".mon"
    if os.path.exists(f):
        r = [l.split() for l in open(f) if len(l.split()) == 7]
        if r:
            g = lambda i: max(int(x[i]) for x in r) / 1e9
            print("   cgroup peaks: memory.current %.1f GB, anon+shmem %.1f GB, page cache %.1f GB; /tmp growth %.1f GB; GPU util mean %.0f%%" % (
                g(1), max(int(x[2]) + int(x[3]) for x in r) / 1e9, g(4), (max(int(x[5]) for x in r) - min(int(x[5]) for x in r)) / 1e9, sum(float(x[6]) for x in r) / len(r)))
print("PATH CHECK:", "all samples ran the stream / GPU pileup (pileup_threads == 1)" if not bad else "%d samples took the sorted-BAM pileup" % bad)
PY
[ "$(sha256sum $BIN | cut -c1-16)" = "$BSHA" ] || { echo "BINARY CHANGED"; RC=1; }
echo "engine commit $(cat $BIN_SRC/commit) sha256 $BSHA (the same binary for every part)"
echo "gate rc=$RC $(date -Is); results in $T"
